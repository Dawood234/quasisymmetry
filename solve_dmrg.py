"""Run block2 DMRG on a Hamiltonian and store the wavefunction locally.

This is the DMRG counterpart of the FCI reference used across the pipeline.
The optimized MPS (and optional sector MPSs) are persisted under a store
directory so later stages (``optimize_symmetries.py --reference dmrg``,
``metrics.py --backend dmrg``, notebooks) can reload them without re-solving.

Optional ``--U`` applies orbital-rotation parameters before DMRG. Use
``--orbital_rotation irrep`` when those parameters were optimized in
irrep packing against a symmetry-adapted Hamiltonian.

Examples
--------
Ground state from an FCIDUMP (works without pyscf)::

    python solve_dmrg.py hamiltonians/n2_1.2_ccpvdz_8o8e.FCIDUMP --bond_dim 500

Full sector diagnostics with K and orbital entropies::

    python solve_dmrg.py hamiltonians/sentest_5_d754.FCIDUMP \\
        --parity_matrix parity.txt --decoupled --k_coupled \\
        --entanglement --entropies --reorder fiedler
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np

from src.dmrg_diagnostics import (
    coupled_energy_dmrg,
    decoupled_energy_dmrg,
    entanglement_diagnostic,
    prepare_parity_matrix,
)
from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    rotation_preserves_orbital_symmetries,
    rotate_integrals,
    solve_or_load_ground_state,
)
from src.workflow_cli import add_orbital_rotation_arg


def comma_separated_ints(text: str) -> tuple[int, ...]:
    """Parse a comma-separated DMRG bond-dimension schedule."""
    return tuple(int(value) for value in text.split(",") if value.strip())


def comma_separated_floats(text: str) -> tuple[float, ...]:
    """Parse a comma-separated DMRG noise schedule."""
    return tuple(float(value) for value in text.split(",") if value.strip())


def build_solver(args: argparse.Namespace) -> Block2DMRGSolver:
    """Create the solver from .FCIDUMP (block2-only) or .chk (needs pyscf)."""
    molpath = Path(args.molpath)
    if molpath.suffix == ".chk":
        from chemistry import fcidump_data  # requires pyscf

        dumpdata = fcidump_data(str(molpath))
        base = Block2DMRGSolver.from_dumpdata(
            dumpdata, store_dir=None, n_threads=args.n_threads,
            save_integrals=False,
        )
        h1e, g2e, ecore = base.h1e, base.g2e, base.ecore
        n_elec, spin = base.n_elec, base.spin
    elif molpath.suffix == ".FCIDUMP" or molpath.name.endswith("FCIDUMP"):
        base = Block2DMRGSolver.from_fcidump(
            molpath,
            store_dir=None,
            n_threads=args.n_threads,
            point_group=args.point_group,
            save_integrals=False,
        )
        h1e, g2e, ecore = base.h1e, base.g2e, base.ecore
        n_elec, spin = base.n_elec, base.spin
    else:
        raise ValueError("molpath must be a .chk or FCIDUMP file")

    orbital_symmetries = (
        None if args.disable_point_group else base.orbital_symmetries
    )
    target_irrep = base.target_irrep

    suffix = ""
    if args.U is not None:
        from src.orbital_rotation import params_to_U, resolve_orbital_rotation

        x = np.loadtxt(args.U, comments=["#", "{"])
        pairs, _ = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, h1e.shape[0]
        )
        rotation = params_to_U(x, h1e.shape[0], pairs)
        h1e, g2e = rotate_integrals(h1e, g2e, rotation)
        if not rotation_preserves_orbital_symmetries(
            rotation, orbital_symmetries
        ):
            orbital_symmetries = None
        x_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:8]
        suffix = f"_rot-{x_hash}"
    if args.reorder:
        suffix += f"_{args.reorder}"

    store_dir = args.store_dir
    if store_dir is None:
        store_dir = Path("wavefunctions") / (molpath.stem + suffix)

    return Block2DMRGSolver(
        h1e=h1e, g2e=g2e, ecore=ecore, n_elec=n_elec, spin=spin,
        store_dir=store_dir, n_threads=args.n_threads,
        reorder=args.reorder,
        orbital_symmetries=orbital_symmetries,
        target_irrep=target_irrep,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a Hamiltonian with block2 DMRG and store the MPS locally"
    )
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (.chk or FCIDUMP)")
    parser.add_argument("--U", default=None,
                        help="path to orbital-rotation parameters x "
                             "(same format as metrics.py --U)")
    add_orbital_rotation_arg(parser)
    parser.add_argument("--parity_matrix", default=None,
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--store_dir", default=None,
                        help="wavefunction store directory "
                             "(default: wavefunctions/<molname>)")
    parser.add_argument("--bond_dim", type=int, default=250)
    parser.add_argument("--n_sweeps", type=int, default=20)
    parser.add_argument("--energy_tol", type=float, default=1.0e-8)
    parser.add_argument("--davidson_threshold", type=float, default=1.0e-10)
    parser.add_argument(
        "--bond_dims",
        type=comma_separated_ints,
        default=(),
        help="optional comma-separated bond dimension for every sweep",
    )
    parser.add_argument(
        "--noises",
        type=comma_separated_floats,
        default=(),
        help="optional comma-separated noise schedule",
    )
    parser.add_argument(
        "--twosite_to_onesite",
        type=int,
        default=None,
        help="number of initial 2-site sweeps before switching to 1-site DMRG",
    )
    parser.add_argument(
        "--dmrg_iprint",
        type=int,
        default=0,
        help="Block2 sweep verbosity; use 1 for cluster progress",
    )
    parser.add_argument("--mps_tag", default="GS")
    parser.add_argument(
        "--initial_mps_tag",
        default=None,
        help="warm-start from this tag in the same --store_dir",
    )
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--penalty", type=float, default=30.0,
                        help="sector penalty strength (Hartree)")
    parser.add_argument("--decoupled", action="store_true",
                        help="run sector-resolved DMRG for E_decoupled "
                             "(requires --parity_matrix)")
    parser.add_argument("--k_coupled", action="store_true",
                        help="PT-screened coupled-energy K via DMRG sector "
                             "excited states (implies --decoupled)")
    parser.add_argument("--states_per_sector", type=int, default=5,
                        help="DMRG roots per sector for --k_coupled")
    parser.add_argument("--max_sectors", type=int, default=16,
                        help="max sectors to solve in the decoupled diagnostic")
    parser.add_argument("--entanglement", action="store_true",
                        help="print bipartite entanglement across orbital cuts")
    parser.add_argument("--entropies", action="store_true",
                        help="print 1-orbital entropies and mutual information")
    parser.add_argument("--reorder", choices=("fiedler", "gaopt"), default=None,
                        help="reorder orbitals before DMRG (parity matrix is "
                             "remapped automatically)")
    parser.add_argument(
        "--point_group",
        default="c1",
        help="point group for FCIDUMP orbital labels (e.g. d2h); "
             "checkpoint files supply this automatically",
    )
    parser.add_argument(
        "--disable_point_group",
        action="store_true",
        help="ignore checkpoint/FCIDUMP orbital irreps for a C1 comparison",
    )
    parser.add_argument("--no_reuse", action="store_true",
                        help="re-solve even if a stored wavefunction exists")
    parser.add_argument("--outname", default=None,
                        help="results file (default: <store_dir>/result.txt)")
    args = parser.parse_args()

    if args.k_coupled:
        args.decoupled = True
    if args.decoupled and args.parity_matrix is None:
        parser.error("--decoupled/--k_coupled requires --parity_matrix")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    solver = build_solver(args)
    outname = args.outname or (solver.store_dir / "result.txt")

    lines: list[str] = []

    def report(message: str) -> None:
        print(message)
        lines.append(message)

    report(str(vars(args)))
    report(f"store {solver.store_dir}")
    report(f"norb {solver.n_sites} nelec {solver.n_elec} spin {solver.spin}")
    report(f"orbital_permutation {list(solver.orbital_permutation)}")
    report(
        "orbital_symmetries "
        + (
            "C1"
            if solver.orbital_symmetries is None
            else str(list(solver.orbital_symmetries))
        )
    )
    report(f"target_irrep {solver.target_irrep}")

    config = DMRGConfig(
        max_bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        energy_tol=args.energy_tol,
        davidson_threshold=args.davidson_threshold,
        mps_tag=args.mps_tag,
        bond_dims=args.bond_dims,
        noises=args.noises,
        twosite_to_onesite=args.twosite_to_onesite,
        iprint=args.dmrg_iprint,
    )
    result = solve_or_load_ground_state(
        solver,
        config=config,
        reuse=not args.no_reuse,
        initial_mps_tag=args.initial_mps_tag,
    )
    report(f"E_DMRG {result.energy:.10f} (bond_dim {args.bond_dim})")

    parity_matrix = None
    if args.parity_matrix is not None:
        parity_matrix = np.atleast_2d(np.loadtxt(args.parity_matrix, dtype=int))
        parity = prepare_parity_matrix(solver, parity_matrix)
        reference_mps = solver.get_mps(result.mps_tag)
        expectations = solver.symmetry_expectations(parity, ket=reference_mps)
        report(f"symmetry expectations <S_k> {np.round(expectations, 6)}")

    if args.decoupled:
        decoupled = decoupled_energy_dmrg(
            solver,
            parity_matrix,
            result.energy,
            config=config,
            penalty=args.penalty,
            max_sectors=args.max_sectors,
            reference_tag=result.mps_tag,
        )
        for label, energy in decoupled.sector_energies.items():
            report(f"sector {label}: E = {energy:.10f}")
        report(
            f"E_decoupled {decoupled.e_decoupled:.10f} "
            f"(sector {decoupled.best_sector})"
        )
        report(f"dE {decoupled.dE:.10f}")
        report(f"K = 1: {decoupled.k_equals_one}")

        if args.k_coupled and not decoupled.k_equals_one:
            coupled = coupled_energy_dmrg(
                solver,
                parity_matrix,
                result.energy,
                decoupled.e_decoupled,
                sector_labels=list(decoupled.sector_energies.keys()),
                nroots=args.states_per_sector,
                penalty=args.penalty,
                config=config,
            )
            report(f"E_coupled {coupled.e_coupled:.10f}")
            report(f"K {coupled.k}")
            report(f"converged {coupled.converged}")
            for key in coupled.chosen:
                report(str(key))
        elif args.k_coupled:
            report("K 1")

    if args.entanglement or args.entropies:
        ent = entanglement_diagnostic(
            solver, ket=solver.get_mps(result.mps_tag)
        )
        if args.entanglement:
            report(f"bipartite entanglement (nats) {np.round(ent.bipartite, 6)}")
        if args.entropies:
            report(f"orbital entropies {np.round(ent.orbital_s1, 6)}")
            report(
                f"mutual_information_max {float(np.max(ent.mutual_information)):.6f}"
            )

    with open(outname, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"results written to {outname}")


if __name__ == "__main__":
    main()
