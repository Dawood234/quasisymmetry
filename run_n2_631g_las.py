"""Run the resumable N2/6-31G LAS selection and optimization workflow.

This driver intentionally stops after orbital optimization and symmetry
reselection.  The current determinant-supported final Clifford/Krylov
evaluator is not suitable for the roughly one-billion-dimensional N2/6-31G
fixed-spin space.  A later MPS-native coupled-sector evaluator can consume the
saved ``final_optimized.json`` without repeating these stages.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from run_h2o_631g_las import (
    atomic_json,
    final_oo_json,
    git_commit,
    load_json,
    optimize_command,
    parse_energy,
    row_space,
    run_stage,
    selection_command,
    write_rotation,
)


PROJECT_DIR = Path(__file__).resolve().parent


def validate_n2_metadata(metadata):
    """Reject checkpoints that are not the intended all-electron N2 problem."""
    expected = {
        "norb": 18,
        "nelec": 14,
        "n_alpha": 7,
        "n_beta": 7,
        "parent_qubits": 36,
        "fixed_spin_dimension": 1_012_766_976,
        "rotation_parameters": 24,
    }
    for key, value in expected.items():
        if int(metadata[key]) != value:
            raise ValueError(
                f"expected N2/6-31G {key}={value}, got {metadata[key]}"
            )
    return metadata


def checkpoint_metadata(path):
    """Read dimensions and D2h orbital-rotation packing from a checkpoint."""
    from chemistry import fcidump_data
    from src.orbital_rotation import n_params, resolve_orbital_rotation

    dump = fcidump_data(str(path))
    norb = int(np.asarray(dump["H1"]).shape[0])
    raw_nelec = dump["NELEC"]
    if isinstance(raw_nelec, (tuple, list, np.ndarray)):
        n_alpha, n_beta = (int(raw_nelec[0]), int(raw_nelec[1]))
        nelec = n_alpha + n_beta
    else:
        nelec = int(raw_nelec)
        n_alpha = (nelec + int(dump.get("MS2", 0))) // 2
        n_beta = nelec - n_alpha
    pairs, irreps = resolve_orbital_rotation("irrep", str(path), norb)
    metadata = {
        "norb": norb,
        "nelec": nelec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "parent_qubits": 2 * norb,
        "fixed_spin_dimension": math.comb(norb, n_alpha)
        * math.comb(norb, n_beta),
        "rotation_parameters": n_params(norb, pairs),
        "orbital_irreps": np.asarray(irreps, dtype=int).tolist(),
    }
    return validate_n2_metadata(metadata)


def parse_args():
    """Controls for the N2 reference, selection, and optimization stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bond", type=float, default=1.0977)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--candidate_workers", type=int, default=8)
    parser.add_argument("--target_rank", type=int, default=7)
    parser.add_argument("--max_macrocycles", type=int, default=3)
    parser.add_argument("--reference_bond_dim", type=int, default=150)
    parser.add_argument("--reference_sweeps", type=int, default=8)
    parser.add_argument("--sector_bond_dim", type=int, default=100)
    parser.add_argument("--sector_sweeps", type=int, default=6)
    parser.add_argument("--sector_penalty", type=float, default=30.0)
    parser.add_argument(
        "--sector_gradient",
        choices=("analytic", "finite_difference"),
        default="analytic",
    )
    parser.add_argument("--optimizer_maxiter", type=int, default=8)
    parser.add_argument("--sector_switch_maxiter", type=int, default=2)
    parser.add_argument("--min_dominant_sectors", type=int, default=8)
    parser.add_argument("--max_dominant_sectors", type=int, default=16)
    parser.add_argument("--multiply_bond_dim", type=int, default=150)
    parser.add_argument("--multiply_sweeps", type=int, default=4)
    return parser.parse_args()


def write_summary(
    run_dir,
    state,
    metadata,
    parent_energy,
    final_optimized,
    final_rotation,
    optimization_selection,
    final_selection,
):
    """Record exactly what was completed and what remains intentionally absent."""
    summary = {
        "system": "N2",
        "basis": "6-31G",
        "geometry": {"r_nn_angstrom": state["config"]["bond"]},
        "metadata": metadata,
        "parent_dmrg_energy": float(parent_energy),
        "reference_bond_dimension": int(
            state["config"]["reference_bond_dim"]
        ),
        "target_rank": int(state["config"]["target_rank"]),
        "nominal_reduced_qubits": int(
            metadata["parent_qubits"] - state["config"]["target_rank"]
        ),
        "completed_macrocycles": int(state.get("completed_macrocycles", 0)),
        "row_space_stable": bool(state.get("row_space_stable", False)),
        "optimization_row_space": [
            list(row) for row in row_space(optimization_selection)
        ],
        "reselected_row_space": [list(row) for row in row_space(final_selection)],
        "final_optimized": str(final_optimized),
        "final_rotation": str(final_rotation),
        "final_coupled_evaluation": "not_run_by_design",
        "git_commit": state["git_commit"],
    }
    atomic_json(run_dir / "staged_summary.json", summary)
    lines = [
        "# N2/6-31G staged LAS workflow",
        "",
        f"- Git commit: `{state['git_commit']}`",
        f"- N-N distance: {state['config']['bond']} Angstrom",
        f"- Orbitals/electrons: {metadata['norb']} / {metadata['nelec']}",
        f"- Fixed-spin determinant dimension: {metadata['fixed_spin_dimension']:,}",
        f"- Parent/nominal reduced qubits: {metadata['parent_qubits']} / {summary['nominal_reduced_qubits']}",
        f"- D2h orbital-rotation parameters: {metadata['rotation_parameters']}",
        f"- Parent DMRG energy: {parent_energy:.12f} Ha",
        f"- Completed macrocycles: {summary['completed_macrocycles']}",
        f"- GF(2) row space stable: {summary['row_space_stable']}",
        f"- Final optimized input: `{final_optimized}`",
        "- Final coupled-sector evaluation: not run by design",
        "",
        "The staged result is ready for a future MPS-native coupled-sector evaluator.",
    ]
    (run_dir / "staged_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    """Run parent DMRG, selection, optimization, and iterative reselection."""
    args = parse_args()
    if args.smoke:
        args.reference_bond_dim = 50
        args.reference_sweeps = 2
        args.sector_bond_dim = 50
        args.sector_sweeps = 2
        args.optimizer_maxiter = 1
        args.sector_switch_maxiter = 0
        args.max_macrocycles = 1
        args.multiply_bond_dim = 50
        args.multiply_sweeps = 2

    job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    run_dir = Path(
        args.run_dir
        or scratch
        / "alris"
        / "quasisymmetry"
        / "n2"
        / "6-31g"
        / f"r_{args.bond:.4f}"
        / f"job_{job_id}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "workflow_state.json"
    state = load_json(
        state_path,
        {
            "schema": "quasisymmetry.n2_631g_staged_workflow",
            "version": 1,
            "run_dir": str(run_dir),
            "git_commit": git_commit(),
            "config": vars(args),
            "stages": {},
        },
    )
    atomic_json(run_dir / "configuration.json", state["config"])
    atomic_json(state_path, state)

    checkpoint = run_dir / f"n2_{args.bond:.4f}_6-31g_d2h.chk"
    run_stage(
        state_path,
        state,
        "01_input",
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "make_pyscf_hamiltonian.py"),
            "n2",
            str(args.bond),
            "--basis",
            "6-31g",
            "--point_group",
            "D2h",
            "--output",
            str(checkpoint),
        ],
        [checkpoint],
        args.resume,
    )
    metadata = checkpoint_metadata(checkpoint)
    atomic_json(run_dir / "input_metadata.json", metadata)
    print(
        "N2/6-31G fixed-spin dimension:",
        f"{metadata['fixed_spin_dimension']:,}",
        flush=True,
    )
    print(
        "This staged driver will stop before determinant-supported final coupling.",
        flush=True,
    )

    parent_store = run_dir / "parent_mps"
    parent_result = run_dir / "parent_reference.txt"
    run_stage(
        state_path,
        state,
        "02_parent_reference",
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "solve_dmrg.py"),
            str(checkpoint),
            "--bond_dim",
            str(args.reference_bond_dim),
            "--n_sweeps",
            str(args.reference_sweeps),
            "--n_threads",
            str(args.cpus),
            "--store_dir",
            str(parent_store),
            "--outname",
            str(parent_result),
        ],
        [parent_result, parent_store / "metadata.json"],
        args.resume,
    )

    selection_dir = run_dir / "selection_0"
    selection_dir.mkdir(parents=True, exist_ok=True)
    command, parity, manifest, selection = selection_command(
        args, checkpoint, None, selection_dir, parent_store
    )
    run_stage(
        state_path,
        state,
        "03_selection_0",
        command,
        [parity, manifest, selection],
        args.resume,
    )

    current_parity = parity
    current_manifest = manifest
    current_selection = selection
    current_rotation = None
    latest_optimize = None
    latest_optimize_parity = None
    latest_optimize_manifest = None
    latest_optimization_selection = None
    stable = False

    for cycle in range(1, args.max_macrocycles + 1):
        cycle_dir = run_dir / f"macrocycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        latest_optimization_selection = current_selection
        command, optimize_json, restart = optimize_command(
            args,
            checkpoint,
            current_parity,
            current_rotation,
            cycle_dir,
            parent_store,
            args.resume,
        )
        run_stage(
            state_path,
            state,
            f"04_optimize_{cycle}",
            command,
            [optimize_json, restart],
            args.resume,
        )
        current_rotation = cycle_dir / "rotation.txt"
        write_rotation(current_rotation, optimize_json)
        latest_optimize = optimize_json
        latest_optimize_parity = current_parity
        latest_optimize_manifest = current_manifest

        post_dir = cycle_dir / "reselection"
        post_dir.mkdir(parents=True, exist_ok=True)
        command, post_parity, post_manifest, post_selection = selection_command(
            args, checkpoint, current_rotation, post_dir, parent_store
        )
        run_stage(
            state_path,
            state,
            f"05_reselection_{cycle}",
            command,
            [post_parity, post_manifest, post_selection],
            args.resume,
        )
        stable = row_space(current_selection) == row_space(post_selection)
        record = {
            "cycle": cycle,
            "input_selection": str(current_selection),
            "optimization_parity": str(latest_optimize_parity),
            "optimization_manifest": str(latest_optimize_manifest),
            "optimized": str(optimize_json),
            "output_selection": str(post_selection),
            "row_space_stable": bool(stable),
        }
        previous = [
            item
            for item in state.get("macrocycles", [])
            if int(item.get("cycle", -1)) != cycle
        ]
        state["macrocycles"] = previous + [record]
        atomic_json(state_path, state)
        current_parity = post_parity
        current_manifest = post_manifest
        current_selection = post_selection
        if stable:
            print(f"GF(2) row space stable after macrocycle {cycle}", flush=True)
            break

    state["row_space_stable"] = bool(stable)
    state["completed_macrocycles"] = len(state.get("macrocycles", []))
    atomic_json(state_path, state)
    required = (
        latest_optimize,
        latest_optimize_parity,
        latest_optimize_manifest,
        latest_optimization_selection,
        current_rotation,
    )
    if any(value is None for value in required):
        raise RuntimeError("no complete optimization/reselection macrocycle")

    final_optimized = run_dir / "final_optimized.json"
    final_oo_json(
        final_optimized,
        latest_optimize,
        checkpoint,
        latest_optimize_parity,
        latest_optimize_manifest,
    )
    final_rotation = run_dir / "final_rotation.txt"
    write_rotation(final_rotation, final_optimized)
    parent_energy = parse_energy(parent_result)
    write_summary(
        run_dir,
        state,
        metadata,
        parent_energy,
        final_optimized,
        final_rotation,
        latest_optimization_selection,
        current_selection,
    )
    print("\nStaged N2 workflow complete:", run_dir, flush=True)
    print(
        "Final coupled-sector evaluation was not launched.",
        flush=True,
    )


if __name__ == "__main__":
    main()
