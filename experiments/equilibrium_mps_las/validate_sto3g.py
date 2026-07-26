#!/usr/bin/env python3
"""Compare MPS projection/Krylov operations with exact STO-3G determinants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ffsim
import numpy as np

from common import atomic_json, copy_solver_store, label_text
from linear_algebra import canonical_generalized_eigh
from mps_krylov import decoupled_mpo, extend_sector_chain, projector_beam_split


def parse_args():
    """Parse one small-system MPS/exact comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--proxy_mps", required=True, type=Path)
    parser.add_argument("--proxy_tag", default="M100")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--parity", default=None, type=Path)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--bond_dim", type=int, default=200)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weight_tolerance", type=float, default=1.0e-7)
    parser.add_argument("--energy_tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def molecular_hamiltonian_from_solver(solver):
    """Build the matching ffsim Hamiltonian from solver-order integrals."""
    from src.dmrg_solver import restore_g2e

    g2e = restore_g2e(solver.g2e, solver.n_sites)
    return ffsim.MolecularHamiltonian(
        one_body_tensor=np.asarray(solver.h1e),
        two_body_tensor=np.asarray(g2e),
        constant=float(solver.ecore),
    )


def dense_krylov(operator, support, seed, depth, tolerance=1.0e-12):
    """Build an exact determinant-space Krylov basis on one sector support."""
    support = np.asarray(support, dtype=int)
    vector = np.asarray(seed, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    basis = []
    for _ in range(int(depth)):
        for previous in basis:
            vector -= previous * np.vdot(previous, vector)
        norm = np.linalg.norm(vector)
        if norm <= tolerance:
            break
        vector /= norm
        basis.append(vector.copy())
        embedded = np.zeros(operator.shape[0], dtype=np.complex128)
        embedded[support] = vector
        vector = (operator @ embedded)[support]
    return np.column_stack(basis)


def main() -> None:
    """Run projection completeness and fitted-Krylov comparisons."""
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    project_dir = str(args.project_dir.resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from src.dmrg_solver import Block2DMRGSolver
    from src.sector_utils import symmetry_sectors

    store = args.run_dir / "validation_mps"
    if not store.exists():
        copy_solver_store(args.proxy_mps, store, [args.proxy_tag])
    solver = Block2DMRGSolver.load(store, n_threads=args.threads)
    if args.parity is None:
        rank = min(int(args.rank), solver.n_sites)
        parity = np.eye(solver.n_sites, dtype=int)[:rank]
    else:
        parity = np.atleast_2d(np.loadtxt(args.parity, dtype=int))

    split = projector_beam_split(
        solver=solver,
        source_tag=args.proxy_tag,
        parity_matrix=parity,
        work_dir=args.run_dir,
        prefix="STO3G_VALIDATE",
        beam_width=2 ** len(parity),
        bond_dim=args.bond_dim,
        sweeps=args.sweeps,
        tolerance=1.0e-10,
        project_dir=args.project_dir,
        workers=1,
        total_threads=args.threads,
    )
    parent = solver.get_mps(args.proxy_tag)
    parent_ci = solver.to_ci_vector(parent)
    nelec = (
        (solver.n_elec + solver.spin) // 2,
        (solver.n_elec - solver.spin) // 2,
    )
    sectors = symmetry_sectors(parity, solver.n_sites, nelec)
    weight_errors = {}
    for branch in split["branches"]:
        label = tuple(branch["label"])
        exact_weight = float(
            np.vdot(
                parent_ci[np.asarray(sectors[label], dtype=int)],
                parent_ci[np.asarray(sectors[label], dtype=int)],
            ).real
        )
        weight_errors[label_text(label)] = abs(
            exact_weight - float(branch["norm2"])
        )
    maximum_weight_error = max(weight_errors.values(), default=0.0)

    branch = max(split["branches"], key=lambda item: float(item["norm2"]))
    label = tuple(branch["label"])
    h_dec = decoupled_mpo(solver, parity)
    created = extend_sector_chain(
        solver=solver,
        h_dec_mpo=h_dec,
        seed_tag=branch["tag"],
        label=label,
        existing_basis=[],
        additions=2,
        cycle=0,
        work_dir=args.run_dir,
        bond_dim=args.bond_dim,
        sweeps=args.sweeps,
        tolerance=1.0e-10,
    )
    mps_vectors = [
        solver.to_ci_vector(solver.get_mps(item["tag"])) for item in created
    ]
    molecular_hamiltonian = molecular_hamiltonian_from_solver(solver)
    operator = ffsim.linear_operator(
        molecular_hamiltonian, norb=solver.n_sites, nelec=nelec
    )
    mps_h = np.asarray(
        [[np.vdot(left, operator @ right) for right in mps_vectors] for left in mps_vectors]
    )
    mps_s = np.asarray(
        [[np.vdot(left, right) for right in mps_vectors] for left in mps_vectors]
    )
    mps_energy = float(canonical_generalized_eigh(mps_h, mps_s)["energies"][0])

    support = np.asarray(sectors[label], dtype=int)
    seed = parent_ci[support]
    exact_basis = dense_krylov(operator, support, seed, len(created))
    exact_h = np.zeros((exact_basis.shape[1], exact_basis.shape[1]), dtype=complex)
    for column in range(exact_basis.shape[1]):
        embedded = np.zeros(operator.shape[0], dtype=complex)
        embedded[support] = exact_basis[:, column]
        applied = (operator @ embedded)[support]
        exact_h[:, column] = exact_basis.conj().T @ applied
    exact_energy = float(np.linalg.eigvalsh(exact_h)[0])
    energy_error = abs(mps_energy - exact_energy)

    output = {
        "schema": "quasisymmetry.sto3g_mps_exact_validation",
        "version": 1,
        "parity_matrix": parity.tolist(),
        "projector_source_norm2": split["source_norm2"],
        "projector_retained_weight": split["retained_weight"],
        "projector_compression_loss": split["compression_loss"],
        "maximum_sector_weight_error": maximum_weight_error,
        "sector_weight_errors": weight_errors,
        "tested_sector": list(label),
        "krylov_dimension": len(created),
        "mps_krylov_energy": mps_energy,
        "exact_krylov_energy": exact_energy,
        "krylov_energy_error": energy_error,
        "projector_passed": maximum_weight_error <= args.weight_tolerance,
        "krylov_passed": energy_error <= args.energy_tolerance,
    }
    output["passed"] = output["projector_passed"] and output["krylov_passed"]
    atomic_json(args.run_dir / "validation.json", output)
    print(json.dumps(output, indent=2), flush=True)
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
