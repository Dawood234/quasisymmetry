#!/usr/bin/env python3
"""Build a convergence-certified H2O/6-31G parent DMRG reference.

The workflow is intentionally separate from LAS symmetry selection and orbital
optimization.  It solves the unprojected molecular Hamiltonian, saves every
MPS needed for later LAS calculations, and reports both sweep convergence and
bond-dimension convergence.
"""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def comma_separated_ints(text):
    """Return integers from a comma-separated command-line value."""
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def comma_separated_strings(text):
    """Return nonempty strings from a comma-separated command-line value."""
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_args():
    """Read parent-reference, calibration, and convergence settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--working_store", type=Path, default=None)
    parser.add_argument("--persistent_store", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("production", "calibration", "all"),
        default="production",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--bond_length", type=float, default=0.958)
    parser.add_argument("--bond_angle", type=float, default=104.5)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--point_group", default="C2v")

    parser.add_argument("--symmetry_mode", choices=("sz", "su2"), default="su2")
    parser.add_argument(
        "--ordering",
        choices=("none", "fiedler", "gaopt"),
        default="fiedler",
    )
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--n_mkl_threads", type=int, default=1)
    parser.add_argument("--stack_mem_gb", type=float, default=8.0)

    parser.add_argument("--bond_dims", default="100,200,350,500,750")
    parser.add_argument("--stage_sweeps", default="4,4,6,6,8")
    parser.add_argument(
        "--reverse_bond_dims",
        default="750,750,500,500,350,350,250,250",
    )
    parser.add_argument("--davidson_threshold", type=float, default=1.0e-10)
    parser.add_argument("--m_tolerance_mha", type=float, default=0.2)
    parser.add_argument("--sweep_tolerance_mha", type=float, default=0.02)

    parser.add_argument("--calibration_bond_dim", type=int, default=100)
    parser.add_argument("--calibration_sweeps", type=int, default=4)
    parser.add_argument("--calibration_symmetries", default="su2,sz")
    parser.add_argument("--calibration_orderings", default="none,fiedler,gaopt")
    parser.add_argument("--calibration_threads", default="8,16,32")

    parser.add_argument("--skip_rdms", action="store_true")
    parser.add_argument("--skip_entanglement", action="store_true")
    parser.add_argument("--fci_validation", action="store_true")
    return parser.parse_args()


def print_stage(title):
    """Print a visible, unbuffered progress heading."""
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def command_text(command):
    """Return a readable shell-like command for logs."""
    return " ".join(str(value) for value in command)


def run_command(command, log_path):
    """Run one subprocess while streaming and saving all output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("command:", command_text(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed; see {log_path}")


def sync_directory(source, destination):
    """Incrementally copy a working MPS directory to persistent storage."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve() or not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync is not None:
        subprocess.run(
            [rsync, "-a", f"{source}/", f"{destination}/"],
            check=True,
        )
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def stage_in_store(working_store, persistent_store):
    """Restore persistent MPS files before a resumed node-local run."""
    if persistent_store.exists():
        print(
            f"restoring persistent MPS: {persistent_store} -> {working_store}",
            flush=True,
        )
        sync_directory(persistent_store, working_store)


def package_versions():
    """Record the numerical packages that define this reference."""
    versions = {}
    for name in ("block2", "numpy", "scipy", "pyscf"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_commit():
    """Return the current project commit without requiring Git metadata later."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_configuration(args, run_dir, working_store, persistent_store):
    """Save all inputs before any expensive calculation starts."""
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["working_store"] = str(working_store)
    config["persistent_store"] = str(persistent_store)
    config["git_commit"] = git_commit()
    config["package_versions"] = package_versions()
    config["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = run_dir / "configuration.json"
    if args.resume and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.setdefault("resume_history", []).append(
            {
                "resumed_at": config["created_at"],
                "git_commit": config["git_commit"],
                "package_versions": config["package_versions"],
                "working_store": str(working_store),
                "settings": config,
            }
        )
        path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8"
        )
        return existing
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def ensure_checkpoint(args, run_dir):
    """Build the symmetry-adapted RHF checkpoint once."""
    checkpoint = run_dir / (
        f"h2o_{args.bond_length}_{args.bond_angle}_"
        f"{args.basis.replace('/', '-')}_{args.point_group.lower()}.chk"
    )
    if checkpoint.exists():
        print(f"reusing checkpoint: {checkpoint}", flush=True)
        return checkpoint

    print_stage("Build the C2v-adapted H2O/6-31G checkpoint")
    command = [
        args.python,
        "-u",
        PROJECT_DIR / "make_pyscf_hamiltonian.py",
        "h2o",
        args.bond_length,
        "--mol_parameter_2",
        args.bond_angle,
        "--basis",
        args.basis,
        "--point_group",
        args.point_group,
        "--output",
        checkpoint,
    ]
    run_command(command, run_dir / "logs" / "01_checkpoint.log")
    return checkpoint


def stage_noises(stage_index, sweeps):
    """Use light expansion noise, followed by at least two clean sweeps."""
    if sweeps <= 2:
        return [0.0] * sweeps
    first = 1.0e-4 if stage_index == 0 else 1.0e-5
    second = 1.0e-5 if stage_index == 0 else 1.0e-6
    values = [first, second] + [0.0] * (sweeps - 2)
    return values[:sweeps]


def solve_command(
    args,
    checkpoint,
    store,
    restart_dir,
    result_text,
    result_json,
    tag,
    bond_dims,
    noises,
    initial_tag=None,
    symmetry_mode=None,
    ordering=None,
    n_threads=None,
    save_rdms=None,
    save_entanglement=None,
):
    """Construct one solve_dmrg.py command with explicit reproducible settings."""
    command = [
        args.python,
        "-u",
        PROJECT_DIR / "solve_dmrg.py",
        checkpoint,
        "--store_dir",
        store,
        "--bond_dim",
        max(bond_dims),
        "--n_sweeps",
        len(bond_dims),
        "--energy_tol",
        0.0,
        "--davidson_threshold",
        args.davidson_threshold,
        "--bond_dims",
        ",".join(str(value) for value in bond_dims),
        "--noises",
        ",".join(str(value) for value in noises),
        "--dmrg_iprint",
        1,
        "--mps_tag",
        tag,
        "--symmetry_mode",
        symmetry_mode or args.symmetry_mode,
        "--n_threads",
        n_threads or args.n_threads,
        "--n_mkl_threads",
        args.n_mkl_threads,
        "--stack_mem_gb",
        args.stack_mem_gb,
        "--outname",
        result_text,
        "--result_json",
        result_json,
    ]
    if Path(store).resolve() != Path(restart_dir).resolve():
        command.extend(["--restart_dir", restart_dir])
    selected_ordering = args.ordering if ordering is None else ordering
    if selected_ordering != "none":
        command.extend(["--reorder", selected_ordering])
    if initial_tag is not None:
        command.extend(["--initial_mps_tag", initial_tag])
    if save_rdms is not None:
        command.extend(["--save_rdms", save_rdms])
    if save_entanglement is not None:
        command.extend(["--save_entanglement", save_entanglement])
    return command


def read_result(path):
    """Read one structured solve_dmrg.py result."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_sweep_csv(result, path):
    """Write a compact human-readable sweep history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sweep", "bond_dim", "discarded_weight", "energy_Ha"]
        )
        for item in result.get("sweep_history", []):
            writer.writerow(
                [
                    item["sweep"],
                    item["bond_dim"],
                    item["discarded_weight"],
                    item["energies"][0],
                ]
            )


def run_calibration(args, checkpoint, run_dir):
    """Benchmark spin symmetry, orbital ordering, and thread count at low M."""
    print_stage("Optional low-cost DMRG configuration calibration")
    symmetries = comma_separated_strings(args.calibration_symmetries)
    orderings = comma_separated_strings(args.calibration_orderings)
    thread_counts = comma_separated_ints(args.calibration_threads)
    rows = []

    for symmetry_mode in symmetries:
        for ordering in orderings:
            for n_threads in thread_counts:
                name = f"{symmetry_mode}_{ordering}_t{n_threads}"
                case_dir = run_dir / "calibration" / name
                result_json = case_dir / "result.json"
                if args.resume and result_json.exists():
                    print(f"[calibration] reusing {name}", flush=True)
                else:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    sweeps = args.calibration_sweeps
                    command = solve_command(
                        args,
                        checkpoint,
                        case_dir / "mps",
                        case_dir / "restart",
                        case_dir / "result.txt",
                        result_json,
                        "CALIBRATION",
                        [args.calibration_bond_dim] * sweeps,
                        stage_noises(0, sweeps),
                        symmetry_mode=symmetry_mode,
                        ordering=ordering,
                        n_threads=n_threads,
                    )
                    run_command(
                        command,
                        run_dir / "logs" / f"calibration_{name}.log",
                    )
                result = read_result(result_json)
                rows.append(
                    {
                        "symmetry_mode": symmetry_mode,
                        "ordering": ordering,
                        "n_threads": n_threads,
                        "energy_Ha": float(result["energy"]),
                        "elapsed_seconds": float(result["elapsed_seconds"]),
                        "completed_sweeps": len(
                            result.get("sweep_history", [])
                        ),
                    }
                )

    output = run_dir / "calibration_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "calibration_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(f"calibration summary: {output}", flush=True)


def run_production(
    args,
    checkpoint,
    run_dir,
    working_store,
    persistent_store,
):
    """Run increasing-M warm starts and a reverse discarded-weight schedule."""
    print_stage("Progressive parent-Hamiltonian DMRG convergence")
    bond_dims = comma_separated_ints(args.bond_dims)
    stage_sweeps = comma_separated_ints(args.stage_sweeps)
    if len(bond_dims) != len(stage_sweeps):
        raise ValueError("--bond_dims and --stage_sweeps must have equal lengths")
    if sorted(bond_dims) != bond_dims or len(set(bond_dims)) != len(bond_dims):
        raise ValueError("--bond_dims must be strictly increasing")

    stage_dir = run_dir / "production" / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    restart_dir = persistent_store
    previous_tag = None
    result_paths = []

    for stage_index, (bond_dim, sweeps) in enumerate(
        zip(bond_dims, stage_sweeps)
    ):
        tag = f"M{bond_dim}"
        result_json = stage_dir / f"{tag}.json"
        result_text = stage_dir / f"{tag}.txt"
        result_paths.append(result_json)
        if args.resume and result_json.exists():
            print(f"[production] reusing completed {tag}", flush=True)
        else:
            print_stage(
                f"Parent DMRG stage {stage_index + 1}/{len(bond_dims)}: "
                f"M={bond_dim}, warm start={previous_tag or 'random'}"
            )
            noises = stage_noises(stage_index, sweeps)
            command = solve_command(
                args,
                checkpoint,
                working_store,
                restart_dir,
                result_text,
                result_json,
                tag,
                [bond_dim] * sweeps,
                noises,
                initial_tag=previous_tag,
            )
            run_command(
                command,
                run_dir / "logs" / f"production_{tag}.log",
            )
            sync_directory(working_store, persistent_store)
        result = read_result(result_json)
        write_sweep_csv(result, stage_dir / f"{tag}_sweeps.csv")
        previous_tag = tag

    reverse_bond_dims = comma_separated_ints(args.reverse_bond_dims)
    if reverse_bond_dims:
        if reverse_bond_dims[0] != bond_dims[-1]:
            raise ValueError(
                "--reverse_bond_dims must begin at the largest production M"
            )
        reverse_json = stage_dir / "REVERSE.json"
        if args.resume and reverse_json.exists():
            print("[production] reusing completed reverse schedule", flush=True)
        else:
            print_stage("Reverse two-site sweep for discarded-weight extrapolation")
            command = solve_command(
                args,
                checkpoint,
                working_store,
                restart_dir,
                stage_dir / "REVERSE.txt",
                reverse_json,
                "REVERSE",
                reverse_bond_dims,
                [0.0] * len(reverse_bond_dims),
                initial_tag=previous_tag,
            )
            run_command(
                command,
                run_dir / "logs" / "production_REVERSE.log",
            )
            sync_directory(working_store, persistent_store)
        reverse_result = read_result(reverse_json)
        write_sweep_csv(
            reverse_result, stage_dir / "REVERSE_sweeps.csv"
        )
    else:
        reverse_json = None

    return result_paths, reverse_json, previous_tag


def final_artifacts(
    args,
    checkpoint,
    run_dir,
    working_store,
    persistent_store,
    final_tag,
):
    """Save RDM and entanglement arrays from the final high-M MPS."""
    if args.skip_rdms and args.skip_entanglement:
        return
    artifact_dir = run_dir / "production" / "artifacts"
    result_json = artifact_dir / "final_reuse.json"
    rdm_path = None if args.skip_rdms else artifact_dir / "rdms.npz"
    ent_path = (
        None
        if args.skip_entanglement
        else artifact_dir / "entanglement.npz"
    )
    outputs = [path for path in (rdm_path, ent_path) if path is not None]
    if args.resume and outputs and all(path.exists() for path in outputs):
        print("[artifacts] reusing final RDM/entanglement files", flush=True)
        return

    print_stage("Save reusable observables from the final high-M MPS")
    command = solve_command(
        args,
        checkpoint,
        working_store,
        persistent_store,
        artifact_dir / "final_reuse.txt",
        result_json,
        final_tag,
        [1],
        [0.0],
        save_rdms=rdm_path,
        save_entanglement=ent_path,
    )
    run_command(command, run_dir / "logs" / "final_artifacts.log")
    sync_directory(working_store, persistent_store)


def run_fci_validation(checkpoint, output):
    """Optionally compute an exact fixed-spin PySCF FCI cross-check."""
    from chemistry import fcidump_data
    from pyscf import fci

    data = fcidump_data(str(checkpoint))
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1.0e-10
    electronic, _ = solver.kernel(
        data["H1"],
        data["H2"],
        int(data["NORB"]),
        (
            (int(data["NELEC"]) + int(data.get("MS2", 0))) // 2,
            (int(data["NELEC"]) - int(data.get("MS2", 0))) // 2,
        ),
    )
    total = float(electronic + data["ECORE"])
    output.write_text(
        json.dumps({"energy_Ha": total}, indent=2) + "\n",
        encoding="utf-8",
    )
    return total


def final_energy_from_sweep(result):
    """Return the final root energy from a structured sweep history."""
    history = result.get("sweep_history", [])
    if history:
        return float(history[-1]["energies"][0])
    return float(result["energy"])


def final_discarded_weight(result):
    """Return the final discarded weight, if Block2 reported one."""
    history = result.get("sweep_history", [])
    if not history:
        return None
    return float(history[-1]["discarded_weight"])


def reverse_extrapolation(reverse_result):
    """Fit energy linearly against discarded weight using one point per M."""
    if reverse_result is None:
        return None
    last_for_m = {}
    for item in reverse_result.get("sweep_history", []):
        weight = float(item["discarded_weight"])
        energy = float(item["energies"][0])
        if np.isfinite(weight) and np.isfinite(energy) and weight > 0.0:
            last_for_m[int(item["bond_dim"])] = (weight, energy)
    points = [
        {"bond_dim": bond_dim, "discarded_weight": values[0], "energy_Ha": values[1]}
        for bond_dim, values in sorted(last_for_m.items(), reverse=True)
    ]
    if len(points) < 2:
        return {"available": False, "points": points}
    weights = np.asarray([item["discarded_weight"] for item in points])
    energies = np.asarray([item["energy_Ha"] for item in points])
    slope, intercept = np.polyfit(weights, energies, 1)
    fitted = intercept + slope * weights
    residual = float(np.sqrt(np.mean((energies - fitted) ** 2)))
    return {
        "available": True,
        "points": points,
        "intercept_Ha": float(intercept),
        "slope_Ha_per_discarded_weight": float(slope),
        "rms_fit_residual_Ha": residual,
    }


def build_summary(args, run_dir, result_paths, reverse_path, fci_energy):
    """Assess sweep and bond-dimension convergence and write final reports."""
    stages = []
    for path in result_paths:
        result = read_result(path)
        history = result.get("sweep_history", [])
        sweep_delta_mha = None
        if len(history) >= 2:
            sweep_delta_mha = 1000.0 * abs(
                float(history[-1]["energies"][0])
                - float(history[-2]["energies"][0])
            )
        stages.append(
            {
                "tag": result["mps_tag"],
                "max_bond_dim": int(result["config"]["max_bond_dim"]),
                "energy_Ha": final_energy_from_sweep(result),
                "discarded_weight": final_discarded_weight(result),
                "sweep_delta_mHa": sweep_delta_mha,
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "completed_sweeps": len(history),
            }
        )

    last_two_delta = None
    if len(stages) >= 2:
        last_two_delta = 1000.0 * abs(
            stages[-1]["energy_Ha"] - stages[-2]["energy_Ha"]
        )
    final_sweep_delta = stages[-1]["sweep_delta_mHa"]
    converged_m = (
        last_two_delta is not None
        and last_two_delta <= args.m_tolerance_mha
    )
    converged_sweeps = (
        final_sweep_delta is not None
        and final_sweep_delta <= args.sweep_tolerance_mha
    )

    reverse_result = None if reverse_path is None else read_result(reverse_path)
    extrapolation = reverse_extrapolation(reverse_result)
    summary = {
        "system": "H2O",
        "basis": args.basis,
        "geometry": {
            "r_OH_Angstrom": args.bond_length,
            "angle_HOH_degrees": args.bond_angle,
        },
        "point_group": args.point_group,
        "symmetry_mode": args.symmetry_mode,
        "ordering": args.ordering,
        "stages": stages,
        "convergence": {
            "last_two_M_difference_mHa": last_two_delta,
            "M_tolerance_mHa": args.m_tolerance_mha,
            "final_sweep_difference_mHa": final_sweep_delta,
            "sweep_tolerance_mHa": args.sweep_tolerance_mha,
            "bond_dimension_converged": converged_m,
            "sweeps_converged": converged_sweeps,
            "reference_certified": converged_m and converged_sweeps,
        },
        "discarded_weight_extrapolation": extrapolation,
        "fci_validation_energy_Ha": fci_energy,
    }
    summary_path = run_dir / "parent_reference_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# H2O/{args.basis} Parent DMRG Reference",
        "",
        f"- Point group: `{args.point_group}`",
        f"- Block2 spin symmetry: `{args.symmetry_mode}`",
        f"- Orbital ordering: `{args.ordering}`",
        "",
        "| M | Energy (Ha) | Last discarded weight | Sweep change (mHa) | Time (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in stages:
        weight = item["discarded_weight"]
        sweep_delta = item["sweep_delta_mHa"]
        lines.append(
            f"| {item['max_bond_dim']} | {item['energy_Ha']:.12f} | "
            f"{'n/a' if weight is None else f'{weight:.6e}'} | "
            f"{'n/a' if sweep_delta is None else f'{sweep_delta:.6f}'} | "
            f"{item['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"- Last-$M$ difference: "
            f"{'n/a' if last_two_delta is None else f'{last_two_delta:.6f} mHa'}",
            f"- Final sweep difference: "
            f"{'n/a' if final_sweep_delta is None else f'{final_sweep_delta:.6f} mHa'}",
            f"- Reference certified: `{converged_m and converged_sweeps}`",
        ]
    )
    if extrapolation and extrapolation.get("available"):
        lines.append(
            "- Zero-discarded-weight linear estimate: "
            f"{extrapolation['intercept_Ha']:.12f} Ha"
        )
    if fci_energy is not None:
        lines.append(f"- Optional FCI cross-check: {fci_energy:.12f} Ha")
    markdown_path = run_dir / "parent_reference_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {markdown_path}", flush=True)
    return summary


def main():
    """Run the requested calibration and/or production reference stages."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    persistent_store = (
        args.persistent_store.resolve()
        if args.persistent_store is not None
        else (run_dir / "parent_mps").resolve()
    )
    working_store = (
        args.working_store.resolve()
        if args.working_store is not None
        else persistent_store
    )
    working_store.mkdir(parents=True, exist_ok=True)
    persistent_store.mkdir(parents=True, exist_ok=True)
    if args.resume and working_store != persistent_store:
        stage_in_store(working_store, persistent_store)

    config = write_configuration(
        args, run_dir, working_store, persistent_store
    )
    print(json.dumps(config, indent=2), flush=True)
    checkpoint = ensure_checkpoint(args, run_dir)

    if args.mode in ("calibration", "all"):
        run_calibration(args, checkpoint, run_dir)

    if args.mode in ("production", "all"):
        result_paths, reverse_path, final_tag = run_production(
            args,
            checkpoint,
            run_dir,
            working_store,
            persistent_store,
        )
        final_artifacts(
            args,
            checkpoint,
            run_dir,
            working_store,
            persistent_store,
            final_tag,
        )
        fci_energy = None
        fci_path = run_dir / "fci_validation.json"
        if args.fci_validation:
            if args.resume and fci_path.exists():
                fci_energy = json.loads(
                    fci_path.read_text(encoding="utf-8")
                )["energy_Ha"]
            else:
                print_stage("Optional PySCF FCI cross-check")
                fci_energy = run_fci_validation(checkpoint, fci_path)
        build_summary(
            args, run_dir, result_paths, reverse_path, fci_energy
        )
        sync_directory(working_store, persistent_store)

    print(f"completed parent-reference workflow: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
