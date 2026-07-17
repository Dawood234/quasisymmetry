from __future__ import annotations

import argparse
import json
import time
from functools import cache, reduce
from math import comb
from pathlib import Path
from typing import Any, Callable, Union
from uuid import uuid4

import numpy as np
import scipy
import scipy.optimize
import scipy.sparse.linalg

try:
    import ffsim
    import openfermion as of
    import openfermionpyscf  # noqa: F401 — registers openfermion pyscf integration
    import pyscf
    import pyscf.fci
    import pyscf.tools.fcidump  # noqa: F401
    import fcidump_openfermion  # noqa: F401
    from chemistry import fcidump_data, load_moldata
    from src.bs import beam  # noqa: F401
    from src.decoupled_energy import (
        best_sector,
        make_decoupled_energy_cost,
        make_fixed_sector_energy_cost,
        make_clifford_decoupled_energy_cost,
        make_clifford_fixed_sector_energy_cost,
        optimize_with_sector_switching,
        optimize_with_clifford_sector_switching,
        scan_clifford_sector_energies,
    )
    from src.clifford_sectors import (
        load_symmetry_manifest,
        prepare_clifford_context,
        z_symmetries_from_parity_matrix,
    )
    from src.sector_utils import symmetry_sectors
    from src.state_utils import get_cisd_gs, get_fci_state_openfermion  # noqa: F401
except ImportError as exc:
    _FCI_STACK_ERROR = exc
    ffsim = None
    of = None
    pyscf = None
    fcidump_data = None
    load_moldata = None
    best_sector = None
    make_decoupled_energy_cost = None
    make_fixed_sector_energy_cost = None
    make_clifford_decoupled_energy_cost = None
    make_clifford_fixed_sector_energy_cost = None
    optimize_with_sector_switching = None
    optimize_with_clifford_sector_switching = None
    scan_clifford_sector_energies = None
    load_symmetry_manifest = None
    prepare_clifford_context = None
    z_symmetries_from_parity_matrix = None
    symmetry_sectors = None
else:
    _FCI_STACK_ERROR = None

from src.workflow_cli import (
    add_optimize_workflow_args,
    optimize_cost_engine,
    print_workflow_banner,
)
from src.orbital_rotation import n_params, params_to_U, resolve_orbital_rotation


def commutator_cost(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray,
                    pairs=None) -> Callable:
    def f(x: np.ndarray) -> float:
        U = x_to_rotation(x, moldata.norb, pairs)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                  norb=moldata.norb, nelec=moldata.nelec)
        total_nc = 0
        h_on_rotated_state = h @ rotated_state # compute once for all symmetries
        for s in symmetries:
            term1 = h @ (s @ rotated_state)
            term2 = s @ h_on_rotated_state
            commutator_on_state = term1 - term2
            total_nc += np.vdot(commutator_on_state, commutator_on_state).real
        return total_nc
    return f

def variance_cost(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray,
                    pairs=None) -> Callable:
    def f(x: np.ndarray) -> float:
        U = x_to_rotation(x, moldata.norb, pairs)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        total_var = 0
        for s in symmetries:
            total_var += 1 - ((rotated_state.T.conj() @ s @ rotated_state)**2).real
        return total_var
    return f

"""Cost function measuring summed variances of general symmetry operators for orbital-rotated reference state. Supports arbitrary symmetries with a fast path for parity-only cases reproducing variance_cost exactly."""
def variance_cost_general(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray, only_parities=False) -> Callable:
    def f(x: np.ndarray) -> float:
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        total_var = 0
        for s in symmetries:
            if only_parities: # use simplified expression
                total_var += 1 - ((rotated_state.T.conj() @ s @ rotated_state)**2).real
            else:
                total_var += (reference_state.T.conj() @ (s @ (s @ reference_state)) - (reference_state.T.conj() @ (s @ reference_state)) ** 2).real
        return total_var
    return f

"""Cost function based on eigenvalue equation residuals, measuring how well rotated state satisfies eval equation for each symmetry operator.
Similar in spirit as variance-based, but more efficient for
parities using only_parities=True.

Key differences from variance-based cost function:
- additional arg evals: list of symmetry eigenvalues, to be precomputed by rounding <psi|S_i|psi>
- parity-specific path (only_parities=True): instead of maximizing |<psi|U^dag S_i|psi>|^2,
maximize (minimize) <psi|U^dag S_i|psi> for evals_i = +1 (-1).
"""
def eval_eq_cost(symmetries: list, evals: list,
                    reference_state: np.ndarray, norb:int, nelec:int, only_parities=False) -> Callable:
    if len(symmetries) != len(evals):
        raise ValueError("len(evals) must match len(symmetries)")
    def f(x):
        U = x_to_rotation(x, norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     norb,
                                                     nelec)
        total = 0
        for i in range(len(symmetries)):
            if only_parities: # use simplified expression
                total += 2 * (1 - evals[i] * np.vdot(rotated_state, symmetries[i] @ rotated_state).real)
            else:
                vec = symmetries[i] @ rotated_state - evals[i] * rotated_state
                total += np.vdot(vec, vec).real 
        return total
    return f

@cache
def parities(norb: int, nelec: tuple[int, int]) -> list[scipy.sparse.linalg.LinearOperator]:
    local_parities = []
    for i in range(norb):
        s_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): -2,
                (): 1
            } # FermionOperator for alpha spin  1 - a_{i} a^\dagger_{i}, AKA, local parity
        )
        s_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): -2,
                (): 1
            } # FermionOperator for beta spin  1 - b_{i} b^\dagger_{i}, AKA, local parity
        )
        s = s_alpha * s_beta
        local_parities.append(ffsim.linear_operator(s, norb, nelec))
    return local_parities


def parity_matrix_to_quasisymmetries(parity_matrix: np.ndarray,
                                     norb: int,
                                     nelec: tuple[int, int]) -> list[scipy.sparse.linalg.LinearOperator]:
    local_parities = parities(norb, nelec)
    if len(parity_matrix) == 0: # needed when dealing with parities and numbers together
        return([])
    if parity_matrix.shape[1] == norb:
        operators = []
        for i in range(parity_matrix.shape[0]): # rows
            relevant_parities = [local_parities[j] for j in range(norb)
                                 if parity_matrix[i][j] == 1]
            quasisymmetry = reduce(lambda a, b: a @ b,
                relevant_parities
            )
            operators.append(quasisymmetry)
        return operators
    elif parity_matrix.shape[1] == norb * 2:
        operators = []
        for i in range(parity_matrix.shape[0]):
            current_ops = []
            for j in range(norb):
                if parity_matrix[i][2 * j] == parity_matrix[i][2 * j + 1] == 1:
                    current_ops.append(local_parities[j])
                elif parity_matrix[i][2 * j] == 1:
                    s_alpha = ffsim.FermionOperator(
                        {
                            (ffsim.cre_a(j), ffsim.des_a(j)): -2,
                            (): 1
                        }
                    )
                    current_ops.append(ffsim.linear_operator(s_alpha, norb, nelec))
                elif parity_matrix[i][2 * j + 1] == 1:
                    s_beta = ffsim.FermionOperator(
                        {
                            (ffsim.cre_b(j), ffsim.des_b(j)): -2,
                            (): 1
                        }
                    )
                    current_ops.append(ffsim.linear_operator(s_beta, norb, nelec))

            quasisymmetry = reduce(lambda a, b: a @ b,
                                   current_ops
                                   )
            operators.append(quasisymmetry)
        return operators
    else:
        raise ValueError("shape[1] must be norb or 2 * norb")


def x_to_rotation(x: np.ndarray, norb: int, pairs=None) -> np.ndarray:
    return params_to_U(x, norb, pairs)


def get_fci(dumpdata: dict, flatten: bool = True) -> tuple[float, np.ndarray]:
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 10000 
    cisolver.conv_tol = 1e-10
    e_fci, fcivec = cisolver.kernel(
        dumpdata["H1"],
        dumpdata["H2"],
        dumpdata["NORB"],
        dumpdata["NELEC"],
        ecore=dumpdata["ECORE"],
    )
    if not cisolver.converged:
        import warnings
        warnings.warn(
            f"FCI didn't converge (conv_tol={cisolver.conv_tol}); using best available result.",
            RuntimeWarning,
            stacklevel=2,
        )
    if flatten:
        return e_fci, np.array(fcivec.reshape((-1,)), dtype="complex")
    else:
        return e_fci, fcivec


def expand_state(mol: of.MolecularData, ci: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
    """Given a pyscf/ffsim representation of a CI state, expand it into a (2**n_qubits, 1) vector"""
    norb = mol.n_orbitals
    n_qubits = 2 * norb
    dim = 1 << n_qubits

    nelec = int(mol.n_electrons)
    spin = int(mol.multiplicity - 1)   # = n_alpha - n_beta
    n_alpha = (nelec + spin) // 2
    n_beta = nelec - n_alpha

    alpha_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_alpha), dtype=np.int64)
    beta_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_beta), dtype=np.int64)

    expected_shape = (len(alpha_strings), len(beta_strings))
    if ci.shape != expected_shape:
        raise ValueError(
            f"Unexpected CI tensor shape {ci.shape}; expected {expected_shape}."
        )


    def occupied_orbitals(det, norb):
        return [p for p in range(norb) if (det >> p) & 1]

    def pyscf_det_to_openfermion_index_and_phase(alpha_det, beta_det, norb):
        """
        PySCF CI basis is organized by separate alpha/beta strings.
        We embed into OpenFermion spin-orbital order:
            [a0, b0, a1, b1, ..., a_{norb-1}, b_{norb-1}]
        and build the basis index using qubit 0 as the leftmost tensor factor.

        Returns
        -------
        basis_index : int
        phase : +1 or -1
        """
        occ_alpha = occupied_orbitals(alpha_det, norb)
        occ_beta = occupied_orbitals(beta_det, norb)

        # Fermionic sign from reordering:
        # starting from all-alpha then all-beta ordering
        # into interleaved spin-orbital ordering.
        inversions = 0
        for p in occ_alpha:
            for q in occ_beta:
                if q < p:
                    inversions += 1
        phase = -1.0 if (inversions % 2) else 1.0

        # Build OpenFermion computational-basis index.
        # qubit 0 is the leftmost tensor factor, so it contributes to the
        # most-significant bit of the basis index.
        idx = 0
        for p in occ_alpha:
            q = 2 * p
            idx |= (1 << (n_qubits - 1 - q))
        for p in occ_beta:
            q = 2 * p + 1
            idx |= (1 << (n_qubits - 1 - q))

        return idx, phase

    rows = []
    data = []
    for ia, alpha_det in enumerate(alpha_strings):
        for ib, beta_det in enumerate(beta_strings):
            coeff = ci[ia, ib]
            if abs(coeff) > threshold:
                idx, phase = pyscf_det_to_openfermion_index_and_phase(
                    int(alpha_det), int(beta_det), norb
                )
                rows.append(idx)
                data.append(phase * coeff)

    if rows:
        cols = np.zeros(len(rows), dtype=np.int64)
        state = scipy.sparse.csr_matrix(
            (np.asarray(data, dtype=np.complex128), (np.asarray(rows), cols)),
            shape=(dim, 1),
            dtype=np.complex128,
        )
    else:
        state = scipy.sparse.csr_matrix((dim, 1), dtype=np.complex128)
    return state.todense()


def callback(intermediate_result) -> None:
    print(time.strftime("%a, %d %b %Y %H:%M:%S",
                        time.localtime()), end=" ")
    # SciPy < 1.11 passes the parameter vector; newer versions pass OptimizeResult.
    if hasattr(intermediate_result, "fun"):
        print("{0:4.6f}".format(intermediate_result.fun))
    else:
        print("(iter)")


def parse_sector_label(text):
    """Parse a command-line sector label such as ``0,1,0``."""
    if not text.strip():
        raise ValueError("empty sector label")
    return tuple(int(part.strip()) for part in text.split(","))


def comm_sq_exp_fast(sym_ops: list[of.QubitOperator], H: Any, state: np.ndarray, n_qubits: int, verbose: bool = False) -> Union[float, complex]:
    """
    Compute sum_k <state| ( i[H, S_k] )^2 |state> efficiently.

    Parameters
    ----------
    sym_ops : list[QubitOperator]
        Symmetry operators (Pauli products).
    H : sparse operator
        Hamiltonian.
    state : np.ndarray
        State vector.
    n_qubits : int

    Returns
    -------
    float or complex
    """

    psi = np.asarray(state)

    # Reused for every symmetry operator
    Hpsi = H @ psi

    total = 0.0 + 0.0j
    for sym in sym_ops:
        S = of.get_sparse_operator(sym, n_qubits).tocsr()

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))  # delta = i[H,S]|psi>

        # <psi| (i[H,S])^2 |psi> = || delta ||^2
        total += np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp



# ─────────────────────────────────────────────────────────────────────────────
# New: arbitrary operator support via of_to_ffsim
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT ALREADY WORKS with any LinearOperator:
#   commutator_cost      — accepts list[LinearOperator], no restrictions
#   x_to_rotation        — pure math, no operator knowledge
#   callback             — pure math
#
# WHAT IS QUBIT-ONLY AND INCOMPATIBLE WITH ffsim LinearOperators:
#   comm_sq_exp_fast     — calls of.get_sparse_operator; pass it a LinearOperator and it breaks
#   expand_state         — expands into 2^(2*norb) Fock space; only needed for the JW path
#
# To add a new symmetry type:
#   1. Express it as an of.FermionOperator  →  fermion_op_to_linop(op, norb, nelec)
#   2. Or build ffsim.FermionOperator directly  →  ffsim.linear_operator(op, norb, nelec)
#   3. Pass the resulting LinearOperator in the symmetries list to commutator_cost
#   That's it. commutator_cost needs no changes.


def of_to_ffsim(op: of.FermionOperator) -> ffsim.FermionOperator:
    """
    Remap of.FermionOperator to ffsim.FermionOperator.
    OF: mode 2p -> cre/des_a(p),  mode 2p+1 -> cre/des_b(p).
    Pure index remap — no Jordan-Wigner, no 2^(2*norb) matrix.
    """
    terms = {}
    for term, coeff in op.terms.items():
        ffsim_ops = []
        for mode, action in term:
            p, spin = divmod(mode, 2)
            if action == 1:
                ffsim_ops.append(ffsim.cre_a(p) if spin == 0 else ffsim.cre_b(p))
            else:
                ffsim_ops.append(ffsim.des_a(p) if spin == 0 else ffsim.des_b(p))
        terms[tuple(ffsim_ops)] = coeff
    return ffsim.FermionOperator(terms)
 
 
def fermion_op_to_linop(
    op: of.FermionOperator,
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """of.FermionOperator -> LinearOperator on the ffsim FCI subspace. No qubits."""
    return ffsim.linear_operator(of_to_ffsim(op), norb, nelec)

 
def fci_reference(moldata: ffsim.MolecularData) -> np.ndarray:
    h = ffsim.linear_operator(moldata.hamiltonian, norb=moldata.norb, nelec=moldata.nelec)
    _, v = scipy.sparse.linalg.eigsh(h, k=1, which="SA")
    return v[:, 0].astype(complex)
 
 
def hf_reference(moldata: ffsim.MolecularData) -> np.ndarray:
    return ffsim.hartree_fock_state(moldata.norb, moldata.nelec).astype(complex)
 
 
def optimize_fcidump(
    input_path: str,
    symmetry_op: "of.FermionOperator | list[of.FermionOperator] | list[scipy.sparse.linalg.LinearOperator]",
    reference_fn: "Callable[[ffsim.MolecularData], np.ndarray]",
    output_path: "str | None" = None,
    x0: "np.ndarray | None" = None,
    method: str = "L-BFGS-B",
    maxiter: int = 500,
    verbose: bool = False,
    cost: str = "NC",
    pairs=None,
) -> scipy.optimize.OptimizeResult:
    """
    Load FCIDUMP, minimise cost(H(U), S, |psi(U)>), write rotated FCIDUMP.
 
    symmetry_op  : of.FermionOperator, list[of.FermionOperator], or list[LinearOperator].
    reference_fn : Callable[[ffsim.MolecularData], np.ndarray]
                   Built-ins: fci_reference, hf_reference.
                   Custom:    lambda md: your_solver(md, your_initial_vector)
    cost         : "NC" (commutator, default) or "variance".
    output_path  : write rotated FCIDUMP here; None to skip.
    """
    print(f"optimize_fcidump: input_path={input_path}, output_path={output_path}, "
          f"symmetry_op={type(symmetry_op)}, reference_fn={reference_fn.__name__}, cost={cost}, method={method}, maxiter={maxiter}")
    
    moldata = load_moldata(input_path)
    norb, nelec = moldata.norb, moldata.nelec
 
    state = reference_fn(moldata)
 
    if isinstance(symmetry_op, of.FermionOperator):
        sym_linops = [fermion_op_to_linop(symmetry_op, norb, nelec)]
    elif symmetry_op and isinstance(symmetry_op[0], of.FermionOperator):
        sym_linops = [fermion_op_to_linop(op, norb, nelec) for op in symmetry_op]
    else:
        sym_linops = list(symmetry_op)  # already LinearOperators
 
    if cost == "NC":
        f = commutator_cost(moldata, sym_linops, state, pairs=pairs)
    elif cost == "variance":
        f = variance_cost(moldata, sym_linops, state, pairs=pairs)
    else:
        raise ValueError("cost must be 'NC' or 'variance'")
 
    x0 = np.zeros(n_params(norb, pairs)) if x0 is None else x0
 
    if verbose:
        print(f"NC before: {f(x0):.6e}")
 
    res = scipy.optimize.minimize(f, x0, method=method,
                                  options={"maxiter": maxiter},
                                  callback=callback if verbose else None)
    if verbose:
        print(f"NC after:  {res.fun:.6e}  ({res.message})")
 
    if output_path is not None:
        U_opt = x_to_rotation(res.x, norb, pairs)
        rh    = moldata.hamiltonian.rotated(U_opt)
        ffsim.MolecularData(
            atom=moldata.atom, basis=moldata.basis, spin=moldata.spin, nelec=nelec,
            hf_energy=moldata.hf_energy, norb=norb, core_energy=moldata.core_energy,
            one_body_integrals=rh.one_body_tensor,
            two_body_integrals=rh.two_body_tensor,
        ).to_fcidump(output_path)
 
    return res
 
 
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Orbital optimization of approximate symmetries. "
            "Use --reference to choose the cost wavefunction "
            "(fci/hf → ffsim CI costs; dmrg → Block2 MPS costs). "
            "Use --orbital_rotation irrep for point-group-restricted "
            "packing (needs a --point_group Hamiltonian). "
            "See --help epilog for examples."
        ),
    )
    parser.add_argument("molpath")
    parser.add_argument("parity", nargs="?", default=None,
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--seniority", action="store_true")
    add_optimize_workflow_args(parser)
    parser.add_argument("--multiply_bond_dim", type=int, default=None,
                        help="bond dimension for MPO-MPS multiplies (--reference dmrg)")
    parser.add_argument("--multiply_sweeps", type=int, default=8,
                        help="sweeps per MPO-MPS multiply (--reference dmrg)")
    parser.add_argument(
        "--cost_function",
        choices=("NC", "variance", "decoupled", "fixed_sector", "switching_sector"),
        default="NC",
    )
    parser.add_argument(
        "--sector_backend",
        choices=("determinant", "clifford"),
        default="determinant",
        help="sector representation used by the energy objectives (CI path only)",
    )
    parser.add_argument(
        "--symmetry_manifest",
        default=None,
        help="ordered Z-product symmetry manifest for the Clifford backend",
    )
    parser.add_argument("--x0", default=None)
    parser.add_argument(
        "--fixed_sector",
        default=None,
        help="sector label for fixed_sector mode, e.g. 0,1,0. If omitted, use the best initial sector.",
    )
    parser.add_argument(
        "--optimizer_maxiter",
        type=int,
        default=100,
        help="maximum L-BFGS-B iterations per optimization stage",
    )
    parser.add_argument(
        "--maxiter",
        dest="optimizer_maxiter",
        type=int,
        default=argparse.SUPPRESS,
        help="alias for --optimizer_maxiter",
    )
    parser.add_argument(
        "--sector_switch_maxiter",
        type=int,
        default=2,
        help="maximum sector switches for switching_sector mode",
    )
    parser.add_argument("--sector_bond_dim", type=int, default=100)
    parser.add_argument("--sector_sweeps", type=int, default=6)
    parser.add_argument("--sector_penalty", type=float, default=30.0)
    parser.add_argument(
        "--sector_gradient",
        choices=("analytic", "finite_difference"),
        default="analytic",
        help=(
            "gradient for DMRG switching-sector orbital optimization; "
            "analytic uses one sector solve plus Block2 RDM contractions"
        ),
    )
    parser.add_argument("--min_dominant_sectors", type=int, default=8)
    parser.add_argument("--max_dominant_sectors", type=int, default=16)
    parser.add_argument(
        "--reference_sweeps",
        type=int,
        default=20,
        help="sweeps used to prepare the parent DMRG reference",
    )
    parser.add_argument(
        "--objective_store",
        default=None,
        help="scratch directory for switching-sector optimizer MPS files",
    )
    parser.add_argument(
        "--restart_state",
        default=None,
        help="JSON restart state for switching-sector DMRG optimization",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume switching-sector optimization from --restart_state",
    )
    parser.add_argument(
        "--no_cleanup_optimizer_mps",
        action="store_true",
        help="keep all temporary optimizer MPS files instead of pruning old tags",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None)
    parser.add_argument("--output_fcidump", default=None,
                        help="write rotated FCIDUMP here")
    parser.add_argument("--orbene_npy", default=None,
                        help="save rotated orbital energies (h1e diagonal) to this .npy file")

    args = parser.parse_args()
    print_workflow_banner(
        "optimize",
        args.reference,
        cost_function=args.cost_function,
        bond_dim=args.bond_dim if args.reference == "dmrg" else None,
        orbital_rotation=args.orbital_rotation,
    )

    rotation_pairs = None
    rotation_irreps = None

    # ── MPS-native path ───────────────────────────────────────────────────────
    if args.reference == "dmrg":
        from src.dmrg_costs import MultiplyConfig, build_dmrg_orbital_costs
        from src.dmrg_solver import DMRGConfig, solve_or_load_ground_state
        from src.dmrg_decoupled_energy import (
            load_json,
            make_context,
            optimize_with_dmrg_sector_switching,
            rotated_solver,
            rotation_key,
            screen_sector_labels,
        )

        if args.cost_function not in ("NC", "variance", "switching_sector"):
            parser.error(
                "--reference dmrg supports NC, variance, or switching_sector"
            )
        if args.sector_backend != "determinant":
            parser.error(
                "--sector_backend clifford requires --reference fci or hf"
            )
        if args.seniority:
            parser.error("--reference dmrg requires a parity matrix (not --seniority)")
        if args.parity is None:
            parser.error("--reference dmrg requires a parity matrix file")
        if args.orbene_npy is not None:
            parser.error("--orbene_npy is not supported with --reference dmrg")

        parity_matrix = np.atleast_2d(np.loadtxt(args.parity, dtype=int))
        store_dir = args.wavefunction_dir
        if store_dir is None:
            store_dir = str(Path("wavefunctions") / Path(args.molpath).stem)

        costs, dmrg_result, solver = build_dmrg_orbital_costs(
            args.molpath,
            parity_matrix,
            store_dir=store_dir,
            config=DMRGConfig(
                max_bond_dim=args.bond_dim,
                n_sweeps=args.reference_sweeps,
            ),
            multiply=MultiplyConfig(
                bond_dim=args.multiply_bond_dim,
                n_sweeps=args.multiply_sweeps,
            ),
            reuse=True,
            n_threads=args.n_threads,
        )
        rotation_pairs, rotation_irreps = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, solver.n_sites
        )
        costs.pairs = rotation_pairs
        n_free = n_params(solver.n_sites, None)
        n_sym = n_params(solver.n_sites, rotation_pairs)
        print(
            f"orbital_rotation={args.orbital_rotation}: "
            f"N_free={n_free}, N_sym={n_sym}"
            + (f", reduced={n_free - n_sym}" if rotation_pairs is not None else "")
        )
        print("DMRG reference energy: {0:4.6f}".format(dmrg_result.energy))
        print("wavefunction store: {}".format(dmrg_result.store_dir))

        x0 = (
            np.loadtxt(args.x0)
            if args.x0
            else np.zeros(n_params(solver.n_sites, rotation_pairs))
        )
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))

        p = Path(args.molpath)
        if args.outname:
            outname = args.outname
        else:
            outname = (
                "OO_" + p.parts[-1] + "_"
                + time.strftime("%Y%m%d_%H%M%S", time.localtime())
                + "_" + str(uuid4())[:6] + ".json"
            )
        Path(outname).parent.mkdir(parents=True, exist_ok=True)

        switching_history = []
        screened_labels = []
        sector_weights = {}
        restart_state = None
        objective_cache = None

        if args.cost_function in ("NC", "variance"):
            f = costs.cost_function(args.cost_function)
            cost_before = f(x0)
            print("before optimization: {0:4.6f}".format(cost_before))
            t_start = time.time()
            if args.optimizer_maxiter > 0:
                res = scipy.optimize.minimize(
                    f,
                    x0,
                    method="L-BFGS-B",
                    options={"maxiter": args.optimizer_maxiter},
                    callback=callback if args.verbose else None,
                )
                elapsed = time.time() - t_start
            else:
                res = scipy.optimize.OptimizeResult(
                    x=x0,
                    fun=cost_before,
                    success=False,
                    nit=0,
                    nfev=0,
                    message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
                )
                elapsed = 0.0
        else:
            objective_store = Path(
                args.objective_store
                or (Path(outname).parent / (Path(outname).stem + "_optimizer_mps"))
            )
            objective_store.mkdir(parents=True, exist_ok=True)
            restart_state = Path(
                args.restart_state or (str(outname) + ".restart.json")
            )
            objective_cache = objective_store / "objective_cache.json"

            resume_label = None
            if args.resume and restart_state.exists():
                saved = load_json(restart_state, {})
                if saved.get("rotation") is not None:
                    x0 = np.asarray(saved["rotation"], dtype=float)
                if saved.get("sector") is not None:
                    resume_label = tuple(int(value) for value in saved["sector"])
                print("resuming from", restart_state, flush=True)

            if np.max(np.abs(x0)) < 1.0e-15:
                screening_solver = solver
                screening_tag = dmrg_result.mps_tag
            else:
                screening_solver = rotated_solver(
                    solver, x0, rotation_pairs, objective_store, args.n_threads
                )
                screening_tag = "SCREEN_" + rotation_key(x0)
                screening_result = solve_or_load_ground_state(
                    screening_solver,
                    config=DMRGConfig(
                        max_bond_dim=args.bond_dim,
                        n_sweeps=args.reference_sweeps,
                        mps_tag=screening_tag,
                    ),
                    reuse=True,
                )
                screening_tag = screening_result.mps_tag

            screened_labels, sector_weights = screen_sector_labels(
                screening_solver,
                parity_matrix,
                reference_tag=screening_tag,
                minimum=args.min_dominant_sectors,
                maximum=args.max_dominant_sectors,
            )
            if resume_label is not None and resume_label not in screened_labels:
                screened_labels.insert(0, resume_label)
                screened_labels = screened_labels[: args.max_dominant_sectors]
            print("screened sectors:", screened_labels, flush=True)

            context = make_context(
                solver,
                parity_matrix,
                rotation_pairs,
                objective_store,
                objective_cache,
                bond_dim=args.sector_bond_dim,
                sweeps=args.sector_sweeps,
                penalty=args.sector_penalty,
                n_threads=args.n_threads,
                cleanup_mps=not args.no_cleanup_optimizer_mps,
            )
            t_start = time.time()
            res, switching_history, initial_scan = (
                optimize_with_dmrg_sector_switching(
                    context,
                    screened_labels,
                    x0,
                    maxiter=args.optimizer_maxiter,
                    max_switches=args.sector_switch_maxiter,
                    state_path=restart_state,
                    callback=callback if args.verbose else None,
                    initial_label=resume_label,
                    use_analytic_gradient=(args.sector_gradient == "analytic"),
                )
            )
            elapsed = time.time() - t_start
            cost_before = float(initial_scan[0][0])

        print(res.message)
        print("optimized: {0:4.12f}".format(res.fun))

        if args.output_fcidump is not None:
            moldata = load_moldata(args.molpath)
            rh = moldata.hamiltonian.rotated(
                x_to_rotation(res.x, moldata.norb, rotation_pairs)
            )
            ffsim.MolecularData(
                atom=moldata.atom, basis=moldata.basis, spin=moldata.spin,
                nelec=moldata.nelec, hf_energy=moldata.hf_energy,
                norb=moldata.norb, core_energy=moldata.core_energy,
                one_body_integrals=rh.one_body_tensor,
                two_body_integrals=rh.two_body_tensor,
            ).to_fcidump(args.output_fcidump)

        out_data = {
            "reference": "dmrg",
            "backend": optimize_cost_engine("dmrg"),
            "cost_before": float(cost_before),
            "cost_after": float(res.fun),
            "converged": bool(res.success),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "njev": int(getattr(res, "njev", 0)),
            "elapsed": float(elapsed),
            "message": str(res.message),
            "dmrg_energy": float(dmrg_result.energy),
            "wavefunction_dir": str(dmrg_result.store_dir),
            "rotation": res.x.tolist(),
            "orbital_rotation": args.orbital_rotation,
            "parity": args.parity,
            "screened_sector_labels": [list(label) for label in screened_labels],
            "sector_weights": sector_weights,
            "switching_history": switching_history,
            "selected_sector": (
                list(getattr(res, "sector_label", ()))
                if args.cost_function == "switching_sector"
                else None
            ),
            "restart_state": None if restart_state is None else str(restart_state),
            "objective_cache": (
                None if objective_cache is None else str(objective_cache)
            ),
            "sector_gradient": args.sector_gradient,
        }
        if rotation_irreps is not None:
            out_data["irreps"] = np.asarray(rotation_irreps, dtype=int).tolist()
        np.save(str(outname) + ".rotation.npy", np.asarray(res.x, dtype=float))
        with open(outname, "w") as fp:
            json.dump(vars(args) | out_data, fp, indent=2)
        print("results written to", outname)
        raise SystemExit(0)

    if _FCI_STACK_ERROR is not None:
        raise SystemExit(
            f"--reference {args.reference} requires pyscf/ffsim ({_FCI_STACK_ERROR}). "
            "Use --reference dmrg on FCIDUMP files without those packages, "
            "or install the FCI stack / use optimize_dmrg.py."
        )

    moldata  = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    rotation_pairs, rotation_irreps = resolve_orbital_rotation(
        args.orbital_rotation, args.molpath, moldata.norb
    )
    n_free = n_params(moldata.norb, None)
    n_sym = n_params(moldata.norb, rotation_pairs)
    print(
        f"orbital_rotation={args.orbital_rotation}: "
        f"N_free={n_free}, N_sym={n_sym}"
        + (f", reduced={n_free - n_sym}" if rotation_pairs is not None else "")
    )

    # ── symmetries ────────────────────────────────────────────────────────────
    symmetry_manifest = (
        load_symmetry_manifest(args.symmetry_manifest)
        if args.symmetry_manifest
        else None
    )
    if args.seniority:
        sym_of = of.FermionOperator()
        for p in range(moldata.norb):
            # local seniority operator: n_{p,alpha} + n_{p,beta} - 2 n_{p,alpha} n_{p,beta}
            a, b = 2 * p, 2 * p + 1
            sym_of += of.FermionOperator(f"{a}^ {a}", 1.0)
            sym_of += of.FermionOperator(f"{b}^ {b}", 1.0)
            sym_of += of.FermionOperator(((a, 1), (b, 1), (b, 0), (a, 0)), -2.0)
        symmetry_op = sym_of
        sym_linops  = [fermion_op_to_linop(sym_of, moldata.norb, moldata.nelec)]
    elif args.parity is not None:
        parity_matrix = np.atleast_2d(np.loadtxt(args.parity, dtype=int))
        symmetry_op   = parity_matrix_to_quasisymmetries(parity_matrix,
                                                          moldata.norb,
                                                          moldata.nelec)
        sym_linops    = symmetry_op
    elif symmetry_manifest is not None:
        parity_matrix = np.atleast_2d(
            np.asarray(symmetry_manifest["parity_matrix"], dtype=int)
        )
        symmetry_op = parity_matrix_to_quasisymmetries(
            parity_matrix, moldata.norb, moldata.nelec
        )
        sym_linops = symmetry_op
    else:
        parser.error("supply a parity matrix file or --seniority")

    clifford_symmetries = None
    if args.sector_backend == "clifford":
        if args.seniority:
            parser.error("Clifford backend currently requires Z-product symmetries")
        if symmetry_manifest is not None:
            clifford_symmetries = symmetry_manifest["symmetries"]
        else:
            clifford_symmetries = z_symmetries_from_parity_matrix(
                parity_matrix, moldata.norb
            )

    # ── reference state (CI / ffsim path) ─────────────────────────────────────
    if args.reference == "fci":
        _, state = get_fci(dumpdata)
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("reference must be fci or hf on the CI path")
    # Wrap as callable; optimize_fcidump will call reference_fn(moldata) internally.
    # Using a lambda that returns the already-computed state avoids running the
    # solver a second time.
    reference_fn = lambda md: state

    # ── cost before optimization ──────────────────────────────────────────────
    x0 = (
        np.loadtxt(args.x0)
        if args.x0
        else np.zeros(n_params(moldata.norb, rotation_pairs))
    )
    switching_history = None

    if args.cost_function == "NC":
        f = commutator_cost(moldata, sym_linops, state, pairs=rotation_pairs)
    elif args.cost_function == "variance":
        f = variance_cost(moldata, sym_linops, state, pairs=rotation_pairs)
    elif args.cost_function == "decoupled":
        if args.parity is None and symmetry_manifest is None:
            parser.error("decoupled cost requires a parity matrix")
        if args.sector_backend == "clifford":
            f, clifford_context = make_clifford_decoupled_energy_cost(
                moldata, clifford_symmetries, pairs=rotation_pairs
            )
        else:
            sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)
            f = make_decoupled_energy_cost(moldata, sectors, pairs=rotation_pairs)
    elif args.cost_function == "fixed_sector":
        if args.parity is None and symmetry_manifest is None:
            parser.error("fixed_sector cost requires a parity matrix")
        if args.sector_backend == "clifford":
            clifford_context = prepare_clifford_context(
                clifford_symmetries, moldata.norb, moldata.nelec
            )
            if args.fixed_sector is None:
                initial_energy, sector_label, _ = scan_clifford_sector_energies(
                    moldata, clifford_context, x0, pairs=rotation_pairs
                )[0]
                print("Selected initial fixed sector:", sector_label)
                print("Initial fixed-sector energy: {0:4.12f}".format(initial_energy))
            else:
                sector_label = parse_sector_label(args.fixed_sector)
            f = make_clifford_fixed_sector_energy_cost(
                moldata, clifford_context, sector_label, pairs=rotation_pairs
            )
        else:
            sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)
            if args.fixed_sector is None:
                initial_energy, sector_label, _ = best_sector(
                    moldata, sectors, x0, pairs=rotation_pairs
                )
                print("Selected initial fixed sector:", sector_label)
                print("Initial fixed-sector energy: {0:4.12f}".format(initial_energy))
            else:
                sector_label = parse_sector_label(args.fixed_sector)
            if sector_label not in sectors:
                raise ValueError(
                    f"sector {sector_label} is not present in this determinant space"
                )
            f = make_fixed_sector_energy_cost(
                moldata, sectors[sector_label], pairs=rotation_pairs
            )
    elif args.cost_function == "switching_sector":
        if args.parity is None and symmetry_manifest is None:
            parser.error("switching_sector cost requires a parity matrix")
        if args.sector_backend == "clifford":
            clifford_context = prepare_clifford_context(
                clifford_symmetries, moldata.norb, moldata.nelec
            )
            initial_energy, initial_label, _ = scan_clifford_sector_energies(
                moldata, clifford_context, x0, pairs=rotation_pairs
            )[0]
        else:
            sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)
            initial_energy, initial_label, _ = best_sector(
                moldata, sectors, x0, pairs=rotation_pairs
            )
        print("Initial switching sector:", initial_label)
        f = None
    else:
        raise ValueError("unknown cost function")

    cost_before = initial_energy if args.cost_function == "switching_sector" else f(x0)
    print("before optimization: {0:4.6f}".format(cost_before))

    # ── optimise ──────────────────────────────────────────────────────────────
    t_start = time.time()
    if args.optimizer_maxiter > 0:
        if args.cost_function in ("NC", "variance"):
            res = optimize_fcidump(
                input_path=args.molpath,
                symmetry_op=symmetry_op,
                reference_fn=reference_fn,
                output_path=args.output_fcidump,
                x0=x0,
                method="L-BFGS-B",
                maxiter=args.optimizer_maxiter,
                verbose=args.verbose,
                cost=args.cost_function,
                pairs=rotation_pairs,
            )
        elif args.cost_function == "switching_sector":
            if args.sector_backend == "clifford":
                res, switching_history = optimize_with_clifford_sector_switching(
                    moldata,
                    clifford_symmetries,
                    x0,
                    maxiter=args.optimizer_maxiter,
                    max_switches=args.sector_switch_maxiter,
                    callback=callback if args.verbose else None,
                    pairs=rotation_pairs,
                )
            else:
                res, switching_history = optimize_with_sector_switching(
                    moldata,
                    sectors,
                    x0,
                    maxiter=args.optimizer_maxiter,
                    max_switches=args.sector_switch_maxiter,
                    callback=callback if args.verbose else None,
                    pairs=rotation_pairs,
                )
            for i, step in enumerate(switching_history):
                print(
                    "switch step {0}: {1} -> {2}; optimized={3:4.12f}; rescanned={4:4.12f}".format(
                        i,
                        step["start_sector"],
                        step["best_sector_after_rescan"],
                        step["optimized_energy"],
                        step["best_energy_after_rescan"],
                    )
                )
        else:
            res = scipy.optimize.minimize(
                f,
                x0,
                method="L-BFGS-B",
                options={"maxiter": args.optimizer_maxiter},
                callback=callback if args.verbose else None,
            )

        if args.cost_function not in ("NC", "variance") and args.output_fcidump is not None:
            U_opt = x_to_rotation(res.x, moldata.norb, rotation_pairs)
            rh = moldata.hamiltonian.rotated(U_opt)
            ffsim.MolecularData(
                atom=moldata.atom, basis=moldata.basis, spin=moldata.spin, nelec=moldata.nelec,
                hf_energy=moldata.hf_energy, norb=moldata.norb, core_energy=moldata.core_energy,
                one_body_integrals=rh.one_body_tensor,
                two_body_integrals=rh.two_body_tensor,
            ).to_fcidump(args.output_fcidump)

        elapsed = time.time() - t_start
        print(res.message)
        print("optimized: {0:4.6f}".format(res.fun))
    else:
        print("Optimizer maxiter = 0, returning data for canonical orbitals")
        res = scipy.optimize.OptimizeResult()
        res.x = x0
        res.fun = cost_before
        res.success = False
        res.nit = 0
        res.nfev = 0
        elapsed = 0
        res.message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"

    p = Path(args.molpath)
    if args.outname:
        outname = args.outname
    else:
        outname = ("OO_" +
                   p.parts[-1] + "_" +
                   time.strftime("%Y%m%d_%H%M%S", time.localtime())
                   + "_" + str(uuid4())[:6] + ".json")

    out_data = {}
    out_data["reference"] = args.reference
    out_data["backend"] = optimize_cost_engine(args.reference)
    out_data["cost_before"] = cost_before
    out_data["cost_after"] = res.fun
    out_data["converged"] = res.success
    out_data["nit"] = res.nit
    out_data["nfev"] = res.nfev
    out_data["elapsed"] = elapsed
    out_data["message"] = res.message
    if switching_history is not None:
        out_data["switching_history"] = switching_history
    out_data["rotation"] = res.x.tolist()
    out_data["orbital_rotation"] = args.orbital_rotation
    if rotation_irreps is not None:
        out_data["irreps"] = np.asarray(rotation_irreps, dtype=int).tolist()

    full_output = vars(args) | out_data

    with open(outname, "a") as fp:
        json.dump(full_output, fp, indent=2)

    # ── orbene_npy (optional) ─────────────────────────────────────────────────
    if args.orbene_npy:
        # Generalized Fock matrix diagonal in the rotated orbital basis.
        # See comments in the original __main__ block for the full rationale.
        from pyscf.fci import rdm as fci_rdm
        norb, nelec = moldata.norb, moldata.nelec
        state_2d = np.asarray(state).real.reshape(
            comb(norb, nelec[0]), comb(norb, nelec[1])
        )
        U_opt = x_to_rotation(res.x, norb, rotation_pairs)
        rh    = moldata.hamiltonian.rotated(U_opt)
        rdm1a = fci_rdm.make_rdm1_spin1(
            fname='FCImake_rdm1a',
            cibra=state_2d, ciket=state_2d, norb=norb, nelec=nelec
        )
        rdm1b = fci_rdm.make_rdm1_spin1(
            fname='FCImake_rdm1b',
            cibra=state_2d, ciket=state_2d, norb=norb, nelec=nelec
        )
        rdm1_rot = U_opt.T @ (rdm1a + rdm1b) @ U_opt
        h2       = rh.two_body_tensor
        J        = np.einsum('pqrs,rs->pq', h2, rdm1_rot)
        K        = np.einsum('psrq,rs->pq', h2, rdm1_rot)
        fock_rot = rh.one_body_tensor + J - 0.5 * K
        np.save(args.orbene_npy, np.diag(fock_rot).real)
