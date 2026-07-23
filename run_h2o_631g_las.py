"""Run the resumable H2O/6-31G LAS workflow on a cluster node.

The driver keeps symmetry selection, orbital optimization, and final metrics as
separate command-line stages.  Stage metadata are written after every command,
so rerunning with ``--resume`` skips completed work and lets the switching-
sector optimizer continue from its own callback state.
"""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def atomic_json(path, data):
    """Write valid JSON even if the job is interrupted during replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)


def load_json(path, default=None):
    """Load JSON or return a caller-supplied default."""
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def git_commit():
    """Return the local source revision without changing repository state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def maximum_child_rss_mib():
    """Maximum resident memory reported for completed child processes."""
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def stage_complete(state, name, outputs):
    """A stage is reusable only when marked complete and outputs still exist."""
    entry = state.get("stages", {}).get(name, {})
    return entry.get("status") == "complete" and all(Path(p).exists() for p in outputs)


def run_stage(state_path, state, name, command, outputs, resume):
    """Run one subprocess with streamed output and durable stage metadata."""
    outputs = [str(Path(path)) for path in outputs]
    if resume and stage_complete(state, name, outputs):
        print(f"[{name}] already complete; reusing outputs", flush=True)
        return

    log_path = Path(state["run_dir"]) / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "status": "running",
        "command": [str(value) for value in command],
        "outputs": outputs,
        "log": str(log_path),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    state.setdefault("stages", {})[name] = entry
    atomic_json(state_path, state)

    print(f"\n=== {name} ===", flush=True)
    print("command:", " ".join(command), flush=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()

    entry["elapsed_seconds"] = float(time.perf_counter() - started)
    entry["max_child_rss_mib"] = maximum_child_rss_mib()
    entry["return_code"] = int(return_code)
    entry["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if return_code != 0:
        entry["status"] = "failed"
        atomic_json(state_path, state)
        raise RuntimeError(f"stage {name} failed; see {log_path}")
    missing = [path for path in outputs if not Path(path).exists()]
    if missing:
        entry["status"] = "failed"
        entry["missing_outputs"] = missing
        atomic_json(state_path, state)
        raise RuntimeError(f"stage {name} did not create {missing}")
    entry["status"] = "complete"
    atomic_json(state_path, state)


def parse_energy(path):
    """Read the ``E_DMRG`` value written by solve_dmrg.py."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("E_DMRG "):
                return float(line.split()[1])
    raise ValueError(f"E_DMRG not found in {path}")


def row_space(selection_path):
    """Canonical GF(2) row-space signature from a selection JSON file."""
    data = load_json(selection_path)
    rows = data["metadata"]["selected_row_space"]
    return tuple(tuple(int(value) for value in row) for row in rows)


def write_rotation(path, optimize_json):
    """Save the latest optimizer vector in the text format used by the CLIs."""
    data = load_json(optimize_json)
    rotation = np.asarray(data["rotation"], dtype=float)
    np.savetxt(path, rotation)
    return rotation


def validate_checkpoint(path):
    """Check the expected H2O/6-31G dimensions and C2v rotation packing."""
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
        "parent_qubits": 2 * norb,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "fixed_spin_dimension": math.comb(norb, n_alpha) * math.comb(norb, n_beta),
        "rotation_parameters": n_params(norb, pairs),
        "orbital_irreps": np.asarray(irreps, dtype=int).tolist(),
    }
    if norb != 13 or nelec != 10:
        raise ValueError(f"expected 13 orbitals and 10 electrons, got {norb}, {nelec}")
    if metadata["rotation_parameters"] != 28:
        raise ValueError(
            "expected 28 C2v-restricted rotation parameters, got "
            f"{metadata['rotation_parameters']}"
        )
    return metadata


def selection_command(args, checkpoint, rotation, cycle_dir, parent_store):
    """Command for MPS-native NC scoring and rank-seven GF(2) selection."""
    parity = cycle_dir / "parity_matrix.txt"
    manifest = cycle_dir / "symmetry_manifest.json"
    selection = cycle_dir / "selection.json"
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "find_pauli_symmetries.py"),
        str(checkpoint),
        "--reference",
        "dmrg",
        "--senquart",
        "--target_rank",
        str(args.target_rank),
        "--orbital_rotation",
        "irrep",
        "--wavefunction_dir",
        str(parent_store),
        "--bond_dim",
        str(args.reference_bond_dim),
        "--n_sweeps",
        str(args.reference_sweeps),
        "--n_threads",
        str(args.cpus),
        "--candidate_workers",
        str(args.candidate_workers),
        "--multiply_sweeps",
        str(args.multiply_sweeps),
        "--parity_output",
        str(parity),
        "--symmetry_manifest",
        str(manifest),
        "--selection_output",
        str(selection),
    ]
    if args.multiply_bond_dim is not None:
        command.extend(["--multiply_bond_dim", str(args.multiply_bond_dim)])
    if rotation is not None:
        command.extend(["--rotation", str(rotation)])
    return command, parity, manifest, selection


def optimize_command(args, checkpoint, parity, rotation, cycle_dir, parent_store, resume):
    """Command for one switching-sector decoupled-energy macrocycle."""
    output = cycle_dir / "optimized.json"
    restart = cycle_dir / "optimizer_restart.json"
    objective_store = cycle_dir / "optimizer_mps"
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "optimize_symmetries.py"),
        str(checkpoint),
        str(parity),
        "--reference",
        "dmrg",
        "--cost_function",
        "switching_sector",
        "--orbital_rotation",
        "irrep",
        "--bond_dim",
        str(args.reference_bond_dim),
        "--reference_sweeps",
        str(args.reference_sweeps),
        "--sector_bond_dim",
        str(args.sector_bond_dim),
        "--sector_sweeps",
        str(args.sector_sweeps),
        "--sector_penalty",
        str(args.sector_penalty),
        "--sector_gradient",
        str(args.sector_gradient),
        "--min_dominant_sectors",
        str(args.min_dominant_sectors),
        "--max_dominant_sectors",
        str(args.max_dominant_sectors),
        "--sector_switch_maxiter",
        str(args.sector_switch_maxiter),
        "--optimizer_maxiter",
        str(args.optimizer_maxiter),
        "--n_threads",
        str(args.cpus),
        "--wavefunction_dir",
        str(parent_store),
        "--objective_store",
        str(objective_store),
        "--restart_state",
        str(restart),
        "--outname",
        str(output),
    ]
    if rotation is not None:
        command.extend(["--x0", str(rotation)])
    if resume and restart.exists():
        command.append("--resume")
    return command, output, restart


def final_oo_json(path, optimize_json, checkpoint, parity, manifest):
    """Create the final metrics input with the stabilized generator basis."""
    data = load_json(optimize_json)
    data["molpath"] = str(checkpoint)
    data["parity"] = str(parity)
    data["symmetry_manifest"] = str(manifest)
    atomic_json(path, data)


def write_summary(run_dir, state, metadata, reference_results, metrics):
    """Write compact machine-readable and Markdown run conclusions."""
    (low_bond, e_low), (high_bond, e_high) = reference_results
    difference_mha = abs(e_high - e_low) * 1000.0
    reference_converged = difference_mha <= 0.2
    metric_reference = metrics.get("E_reference", metrics.get("E_FCI"))
    if metric_reference is None:
        raise ValueError("final metrics contain no reference energy")
    decoupled_error_mha = (
        None
        if metrics.get("E_decoupled") is None
        else (float(metrics["E_decoupled"]) - float(metric_reference)) * 1000.0
    )
    coupled_error_mha = (
        None
        if metrics.get("E_coupled") is None
        else (float(metrics["E_coupled"]) - float(metric_reference)) * 1000.0
    )
    chemical_accuracy = (
        reference_converged
        and coupled_error_mha is not None
        and abs(coupled_error_mha) <= 1.6
    )
    summary = {
        "system": "H2O",
        "basis": "6-31G",
        "geometry": {"r_oh_angstrom": 0.958, "hoh_angle_degrees": 104.5},
        "metadata": metadata,
        "reference_energies": {
            f"M{low_bond}": e_low,
            f"M{high_bond}": e_high,
        },
        "final_metrics_reference": float(metric_reference),
        "final_metrics_reference_method": metrics.get("reference_method", "FCI"),
        "reference_difference_mHa": difference_mha,
        "reference_converged_0.2_mHa": reference_converged,
        "decoupled_error_mHa": decoupled_error_mha,
        "coupled_error_mHa": coupled_error_mha,
        "K": metrics.get("K"),
        "sector_label_count": metrics.get("sector_label_count"),
        "candidate_state_count": metrics.get("candidate_state_count"),
        "chemical_accuracy_claim_allowed": chemical_accuracy,
        "git_commit": state["git_commit"],
    }
    atomic_json(run_dir / "summary.json", summary)
    lines = [
        "# H2O/6-31G LAS workflow",
        "",
        f"- Git commit: `{state['git_commit']}`",
        f"- Orbitals/electrons: {metadata['norb']} / {metadata['nelec']}",
        f"- Parent/reduced qubits: {metadata['parent_qubits']} / {metadata['parent_qubits'] - state['config']['target_rank']}",
        f"- C2v orbital-rotation parameters: {metadata['rotation_parameters']}",
        f"- Parent M={low_bond} energy: {e_low:.12f} Ha",
        f"- Parent M={high_bond} energy: {e_high:.12f} Ha",
        f"- M={low_bond} to M={high_bond} difference: {difference_mha:.6f} mHa",
        f"- Reference converged at 0.2 mHa: {reference_converged}",
        f"- Decoupled error: {decoupled_error_mha} mHa",
        f"- Coupled error: {coupled_error_mha} mHa",
        f"- Retained sector labels: {metrics.get('sector_label_count')}",
        f"- Candidate sector eigenstates: {metrics.get('candidate_state_count')}",
        f"- Final K: {metrics.get('K')}",
        f"- Chemical-accuracy claim allowed: {chemical_accuracy}",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    """Cluster and smoke-test controls for the staged workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun_final",
        action="store_true",
        help="rerun final selected-sector evaluation while reusing earlier stages",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument(
        "--candidate_workers",
        type=int,
        default=4,
        help="processes used to score independent NC candidates",
    )
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
    parser.add_argument("--final_bond_dims", default="350,500")
    parser.add_argument("--final_sweeps", type=int, default=20)
    parser.add_argument("--roots_per_sector", type=int, default=5)
    return parser.parse_args()


def main():
    """Execute or resume the complete alternating H2O workflow."""
    args = parse_args()
    if args.smoke:
        args.reference_bond_dim = 50
        args.reference_sweeps = 2
        args.sector_bond_dim = 50
        args.sector_sweeps = 2
        args.optimizer_maxiter = 1
        args.sector_switch_maxiter = 0
        args.max_macrocycles = 1
        args.final_bond_dims = "35,50"
        args.final_sweeps = 2
        args.multiply_bond_dim = 50
        args.multiply_sweeps = 2

    job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    run_dir = Path(
        args.run_dir
        or scratch / "alris" / "quasisymmetry" / "h2o" / "6-31g" / f"job_{job_id}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "workflow_state.json"
    state = load_json(
        state_path,
        {
            "schema": "quasisymmetry.h2o_631g_workflow",
            "version": 1,
            "run_dir": str(run_dir),
            "git_commit": git_commit(),
            "config": vars(args),
            "stages": {},
        },
    )
    atomic_json(run_dir / "configuration.json", state["config"])
    atomic_json(state_path, state)

    checkpoint = run_dir / "h2o_0.958_104.5_6-31g_c2v.chk"
    run_stage(
        state_path,
        state,
        "01_input",
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "make_pyscf_hamiltonian.py"),
            "h2o",
            "0.958",
            "--mol_parameter_2",
            "104.5",
            "--basis",
            "6-31g",
            "--point_group",
            "C2v",
            "--output",
            str(checkpoint),
        ],
        [checkpoint],
        args.resume,
    )
    metadata = validate_checkpoint(checkpoint)
    atomic_json(run_dir / "input_metadata.json", metadata)

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
    stable = False

    for cycle in range(1, args.max_macrocycles + 1):
        cycle_dir = run_dir / f"macrocycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
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
        old_records = [
            item for item in state.get("macrocycles", [])
            if int(item.get("cycle", -1)) != cycle
        ]
        state["macrocycles"] = old_records + [record]
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
    if (
        latest_optimize is None
        or latest_optimize_parity is None
        or latest_optimize_manifest is None
        or current_rotation is None
    ):
        raise RuntimeError("no optimization macrocycle completed")

    final_oo = run_dir / "final_optimized.json"
    final_oo_json(
        final_oo,
        latest_optimize,
        checkpoint,
        latest_optimize_parity,
        latest_optimize_manifest,
    )
    final_rotation = run_dir / "final_rotation.txt"
    write_rotation(final_rotation, final_oo)

    final_bonds = [int(value) for value in args.final_bond_dims.split(",")]
    if len(final_bonds) != 2:
        raise ValueError("--final_bond_dims must contain two comma-separated values")
    reference_results = []
    for bond in final_bonds:
        store = run_dir / f"final_parent_M{bond}"
        result_file = run_dir / f"final_parent_M{bond}.txt"
        run_stage(
            state_path,
            state,
            f"06_final_parent_M{bond}",
            [
                sys.executable,
                "-u",
                str(PROJECT_DIR / "solve_dmrg.py"),
                str(checkpoint),
                "--U",
                str(final_rotation),
                "--orbital_rotation",
                "irrep",
                "--bond_dim",
                str(bond),
                "--n_sweeps",
                str(args.final_sweeps),
                "--n_threads",
                str(args.cpus),
                "--store_dir",
                str(store),
                "--outname",
                str(result_file),
            ],
            [result_file, store / "metadata.json"],
            args.resume,
        )
        reference_results.append((bond, store, result_file))

    _, _, final_reference_result = reference_results[-1]
    metrics_json = run_dir / "final_metrics.json"
    selected_work_dir = run_dir / "final_selected_clifford_lanczos"
    print(
        "\nFinal projected evaluation uses selected Clifford sectors and "
        "matrix-free Lanczos.",
        flush=True,
    )
    print(
        "It will not build the complete fixed-spin Hamiltonian matrix and "
        "will checkpoint every completed sector.",
        flush=True,
    )
    run_stage(
        state_path,
        state,
        "07_final_selected_clifford_lanczos",
        [
            sys.executable,
            "-u",
            str(PROJECT_DIR / "selected_clifford_lanczos.py"),
            str(final_oo),
            "--reference_result",
            str(final_reference_result),
            "--work_dir",
            str(selected_work_dir),
            "--max_sectors",
            str(args.max_dominant_sectors),
            "--roots_per_sector",
            str(args.roots_per_sector),
            "--outname",
            str(metrics_json),
        ] + (["--resume"] if args.resume else []),
        [metrics_json],
        args.resume and not args.rerun_final,
    )

    e_low = parse_energy(reference_results[0][2])
    e_high = parse_energy(reference_results[1][2])
    metrics = load_json(metrics_json)
    write_summary(
        run_dir,
        state,
        metadata,
        [(reference_results[0][0], e_low), (reference_results[1][0], e_high)],
        metrics,
    )
    print("\nWorkflow complete:", run_dir, flush=True)


if __name__ == "__main__":
    main()
