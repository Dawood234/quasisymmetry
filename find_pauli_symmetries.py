import argparse
import json
import numpy as np
import time
import ffsim
import scipy
import pyscf
import pyscf.fci
import openfermion as of
import openfermionpyscf

from typing import Callable
from math import comb
from functools import cache, reduce
from itertools import combinations
from pathlib import Path

from chemistry import load_moldata, fcidump_data

from external_imports import get_cisd_gs, get_hf_occ, get_hf_wfn
from external_imports import beam_search_symmetries, BeamSearch_Symmetries
from external_imports import mask_to_qubit_operator
from external_imports import variance, molecular_data_from_fcidump


from optimize_symmetries import get_fci, expand_state, comm_sq_exp_fast
from src.clifford_sectors import save_symmetry_manifest


def expanded_spin_parity_matrix(parity_matrix):
    """Expand spatial pair-parity rows to interleaved spin-orbital rows."""
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    expanded = np.zeros(
        (parity_matrix.shape[0], 2 * parity_matrix.shape[1]), dtype=int
    )
    expanded[:, 0::2] = parity_matrix
    expanded[:, 1::2] = parity_matrix
    return expanded


def run_dmrg_senquart_selection(args):
    """Select seniority/quartet generators without building a full CI state."""
    from src.dmrg_costs import (
        MultiplyConfig,
        build_dmrg_orbital_costs,
        commutator_scores_by_row,
        parity_expectations_by_row,
    )
    from src.dmrg_solver import DMRGConfig
    from src.dmrg_symmetry_selection import (
        assign_candidate_scores,
        candidate_matrix,
        canonical_row_space,
        selected_parity_matrix,
        selection_to_json,
        select_independent_candidates,
        seniority_quartet_candidates,
    )
    from src.orbital_rotation import n_params, resolve_orbital_rotation
    from src.clifford_sectors import z_symmetries_from_parity_matrix

    if not args.senquart:
        raise ValueError("--reference dmrg currently requires --senquart")

    from pathlib import Path
    molpath = Path(args.molpath)
    if molpath.suffix == ".chk":
        dumpdata = fcidump_data(str(molpath))
        norb = int(np.asarray(dumpdata["H1"]).shape[0])
    else:
        from src.dmrg_solver import Block2DMRGSolver

        probe = Block2DMRGSolver.from_fcidump(
            molpath,
            store_dir=Path(args.wavefunction_dir) / "input_probe",
            n_threads=args.n_threads,
            save_integrals=False,
        )
        norb = int(probe.n_sites)

    candidates = seniority_quartet_candidates(norb)
    rows = candidate_matrix(candidates)
    pairs, irreps = resolve_orbital_rotation(
        args.orbital_rotation, str(molpath), norb
    )
    x = (
        np.asarray(np.load(args.rotation), dtype=float)
        if args.rotation and str(args.rotation).endswith(".npy")
        else np.loadtxt(args.rotation)
        if args.rotation
        else np.zeros(n_params(norb, pairs))
    )
    x = np.atleast_1d(np.asarray(x, dtype=float))

    costs, reference, _ = build_dmrg_orbital_costs(
        str(molpath),
        rows,
        store_dir=args.wavefunction_dir,
        config=DMRGConfig(
            max_bond_dim=args.bond_dim,
            n_sweeps=args.n_sweeps,
        ),
        multiply=MultiplyConfig(
            bond_dim=args.multiply_bond_dim,
            n_sweeps=args.multiply_sweeps,
        ),
        reuse=not args.no_reuse,
        n_threads=args.n_threads,
        pairs=pairs,
    )
    scores = commutator_scores_by_row(costs, x)
    scored = assign_candidate_scores(candidates, scores)
    selected = select_independent_candidates(scored, args.target_rank)
    parity_matrix = selected_parity_matrix(selected)
    expectations = parity_expectations_by_row(costs, x, parity_matrix)
    signs = np.where(expectations < 0.0, -1.0, 1.0)
    spin_matrix = expanded_spin_parity_matrix(parity_matrix)
    symmetries = [
        float(sign) * symmetry
        for sign, symmetry in zip(
            signs, z_symmetries_from_parity_matrix(parity_matrix, norb)
        )
    ]

    np.savetxt(args.parity_output, parity_matrix, fmt="%d")
    metadata = {
        "candidate_family": "seniority_plus_quartet",
        "reference": "dmrg",
        "reference_energy": float(reference.energy),
        "selection_score": "state_specific_noncommutativity",
        "target_rank": int(args.target_rank),
        "gf2_rank": int(args.target_rank),
        "selected_row_space": [list(row) for row in canonical_row_space(parity_matrix)],
        "selected": selection_to_json(selected),
        "selected_expectations": expectations.tolist(),
        "selected_signs": signs.astype(int).tolist(),
        "rotation": x.tolist(),
        "orbital_rotation": args.orbital_rotation,
        "irreps": None if irreps is None else np.asarray(irreps, dtype=int).tolist(),
        "mps_store": str(reference.store_dir),
    }
    save_symmetry_manifest(
        args.symmetry_manifest, symmetries, spin_matrix, metadata=metadata
    )

    selection_path = args.selection_output
    if selection_path is None:
        selection_path = str(Path(args.symmetry_manifest).with_suffix(".selection.json"))
    with Path(selection_path).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "quasisymmetry.dmrg_selection",
                "version": 1,
                "molpath": str(molpath),
                "norb": norb,
                "candidate_count": len(candidates),
                "target_rank": int(args.target_rank),
                "selected": selection_to_json(selected),
                "parity_matrix": parity_matrix.tolist(),
                "metadata": metadata,
            },
            handle,
            indent=2,
        )

    print("DMRG reference energy:", reference.energy)
    print("candidate count:", len(candidates))
    print("selected GF(2) rank:", args.target_rank)
    for item in selected:
        print(item["label"], item["score"])
    print("parity matrix written to", args.parity_output)
    print("symmetry manifest written to", args.symmetry_manifest)
    print("selection data written to", selection_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--cost_function", default="NC")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None,
                        help="Name of the output file. If none specified, a time stamp will be used.")
    parser.add_argument("--senquart", action="store_true")
    parser.add_argument("--parity_output", default="parity_matrix.txt",
                        help="output path for the legacy parity matrix")
    parser.add_argument("--symmetry_manifest", default="symmetry_manifest.json",
                        help="output path for ordered signed Pauli symmetries")
    parser.add_argument(
        "--target_rank",
        type=int,
        default=None,
        help="number of independent generators to retain",
    )
    parser.add_argument(
        "--rotation",
        default=None,
        help="saved orbital-rotation parameters used when rescoring DMRG candidates",
    )
    parser.add_argument(
        "--orbital_rotation",
        choices=("full", "irrep"),
        default="full",
        help="packing used by --rotation (irrep requires a symmetry-adapted checkpoint)",
    )
    parser.add_argument(
        "--wavefunction_dir",
        default=None,
        help="Block2 MPS storage used by --reference dmrg",
    )
    parser.add_argument("--bond_dim", type=int, default=500)
    parser.add_argument("--n_sweeps", type=int, default=20)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--multiply_bond_dim", type=int, default=None)
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument(
        "--selection_output",
        default=None,
        help="JSON file for ordered candidates, scores, and selected GF(2) row space",
    )

    args = parser.parse_args()

    if args.reference == "dmrg":
        if args.target_rank is None:
            parser.error("--reference dmrg requires --target_rank")
        if args.wavefunction_dir is None:
            args.wavefunction_dir = str(
                Path("wavefunctions") / (Path(args.molpath).stem + "_selection")
            )
        run_dmrg_senquart_selection(args)
        raise SystemExit(0)

    mol = molecular_data_from_fcidump(args.molpath)


    H = of.get_fermion_operator(mol.get_molecular_hamiltonian())
    n_qubits = of.count_qubits(H)
    qubit_hamiltonian = of.jordan_wigner(H)
    sparse_qubit_op = of.get_sparse_operator(qubit_hamiltonian, n_qubits)

    dumpdata = fcidump_data(args.molpath)
    if args.reference == "fci":
        # e, gs, gs_info = get_fci_state_openfermion(mol)
        e, state = get_fci(dumpdata, flatten=False)
        ref_state = expand_state(mol, state)
    elif args.reference == "hf":
        hf_occ = get_hf_occ(mol.n_electrons, mol.n_orbitals, as_str=True)
        ref_state = get_hf_wfn([int(s) for s in hf_occ])
    elif args.reference == "cisd":
        hf_occ = get_hf_occ(mol.n_electrons, mol.n_orbitals, as_str=True)
        e, ref_state = get_cisd_gs(hf_occ, qubit_hamiltonian, n_qubits, 'wfs', tf='jw')
    else:
        raise ValueError('reference can be fci, cisd, hf')

    if args.cost_function == "NC":
        cost = lambda s_list: comm_sq_exp_fast(s_list, sparse_qubit_op,
                                                          ref_state, n_qubits)
    elif args.cost_function == "variance":
        cost = lambda s_list: variance(s_list, ref_state, n_qubits)
    else:
        raise NotImplementedError()

    beam_score = lambda s: (-1) * cost(s)

    n_sym = args.target_rank if args.target_rank is not None else n_qubits // 2

    if args.senquart:
        seniorities = [(0, 2**(2 * i) + 2**(2 * i + 1)) for i in range(n_qubits // 2)]
        quartets = [(0, s[0][1] + s[1][1]) for s in combinations(seniorities, 2)]

        symmetry_costs = []
        for s in seniorities + quartets:
            symmetry_costs.append(cost([mask_to_qubit_operator(s, n_qubits)]))

        print("Symmetries (not) sorted by their cost value")
        # for i in np.argsort(symmetry_costs):
        for i in range(len(seniorities + quartets)):
            symm_z_mask = (seniorities + quartets)[i][1]
            print(i, format(symm_z_mask, "0" + str(n_qubits) + "b")[::-1],
                  symmetry_costs[i])


        beam_symmetries = beam_search_symmetries(
            qubit_hamiltonian,
            seniorities + quartets,
            target_rank=n_sym,
            n_qubits=n_qubits,
            beam_width=16,
            heavy_core_fraction=0.95,
            initial_generators=None,
            score_func=beam_score
        )

    else:

        beam_symmetries = BeamSearch_Symmetries(qubit_hamiltonian,
                                                     target_rank=n_sym,
                                                     beam_width=16,
                                                     heavy_core_fraction=0.95,
                                                     include_pairwise_products=True,
                                                     pairwise_seed_terms=12,
                                                     seed_with_exact_symmetries=True,
                                                     score_func=beam_score
                                                     )

    parity_matrix = np.zeros((len(beam_symmetries), n_qubits), dtype=int)

    print("Kept symmetries:")
    for i, s in enumerate(beam_symmetries):
        pauli_keys = list(s.terms.keys())
        assert len(pauli_keys) == 1
        key = pauli_keys[0]
        string_letters = "".join([w[1] for w in key])
        pauli_positions = [w[0] for w in key]
        if string_letters.find("X") == -1 and string_letters.find("Y") == -1:
            parity_matrix[i, pauli_positions] = 1
        print(s)
    print("Parity matrix from the Z symmetries")
    print(parity_matrix)
    np.savetxt(args.parity_output, parity_matrix, fmt='%d')
    save_symmetry_manifest(
        args.symmetry_manifest,
        beam_symmetries,
        parity_matrix,
        metadata={
            "molpath": args.molpath,
            "reference": args.reference,
            "cost_function": args.cost_function,
            "candidate_family": "seniority_plus_quartet" if args.senquart else "beam_pauli",
            "selected_set_cost": float(np.real(cost(beam_symmetries))),
            "individual_costs": [float(np.real(cost([symmetry]))) for symmetry in beam_symmetries],
        },
    )
    print("Saved parity matrix to", args.parity_output)
    print("Saved symmetry manifest to", args.symmetry_manifest)
