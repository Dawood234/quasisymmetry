#!/usr/bin/env python3
"""Build adaptive H2O/6-31G parent DMRG references along a bond curve."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


DEFAULT_BONDS = (
    0.7,
    0.8,
    0.9,
    0.958,
    1.1,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
)
DEFAULT_BOND_DIMS = (100, 200, 350, 500, 750)
DEFAULT_SWEEPS = (4, 4, 6, 6, 8)


def comma_separated_floats(text):
    """Parse a comma-separated list of floating-point values."""
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def comma_separated_ints(text):
    """Parse a comma-separated list of integer values."""
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def geometry_key(bond_length):
    """Return the stable directory name for one O-H distance."""
    return f"r_{bond_length:.4f}".replace(".", "p")


def parse_args():
    """Read one array task or aggregation request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--bond-lengths",
        type=comma_separated_floats,
        default=DEFAULT_BONDS,
    )
    parser.add_argument("--bond-dims", type=comma_separated_ints, default=DEFAULT_BOND_DIMS)
    parser.add_argument("--stage-sweeps", type=comma_separated_ints, default=DEFAULT_SWEEPS)
    parser.add_argument("--angle", type=float, default=104.5)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--point-group", default="C2v")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--stack-mem-gb", type=float, default=8.0)
    parser.add_argument("--m-tolerance-mha", type=float, default=0.2)
    parser.add_argument("--sweep-tolerance-mha", type=float, default=0.02)
    parser.add_argument("--working-root", type=Path)
    parser.add_argument("--reuse-equilibrium-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path):
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_settings(args):
    """Reject inconsistent schedules before submitting expensive work."""
    if len(args.bond_dims) != len(args.stage_sweeps):
        raise ValueError("--bond-dims and --stage-sweeps must have equal lengths")
    if len(args.bond_dims) < 2:
        raise ValueError("at least two bond dimensions are required")
    if tuple(sorted(set(args.bond_dims))) != args.bond_dims:
        raise ValueError("--bond-dims must be strictly increasing")
    driver = args.project_dir / "run_h2o_631g_parent_reference.py"
    if not driver.is_file():
        raise FileNotFoundError(driver)


def summary_is_certified(summary):
    """Return whether both sweep and bond-dimension tests passed."""
    convergence = summary.get("convergence", {})
    return bool(convergence.get("reference_certified", False))


def register_equilibrium_reference(args, bond_length, point_dir):
    """Register an existing equilibrium result without copying its large MPS."""
    source = args.reuse_equilibrium_dir
    if source is None or abs(bond_length - 0.958) > 1.0e-10:
        return False
    source = source.resolve()
    source_summary = source / "parent_reference_summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    summary = read_json(source_summary)
    geometry = summary.get("geometry", {})
    if abs(float(geometry.get("r_OH_Angstrom", -1.0)) - bond_length) > 1.0e-10:
        raise ValueError("reused reference has the wrong O-H bond length")
    if str(summary.get("basis", "")).lower() != args.basis.lower():
        raise ValueError("reused reference has the wrong basis")
    if not summary_is_certified(summary):
        raise ValueError("reused equilibrium reference is not certified")

    point_dir.mkdir(parents=True, exist_ok=True)
    copied = dict(summary)
    copied["reused_from"] = str(source)
    (point_dir / "parent_reference_summary.json").write_text(
        json.dumps(copied, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "bond_length_Angstrom": bond_length,
        "source_directory": str(source),
        "source_mps_directory": str(source / "parent_mps"),
        "source_checkpoint": str(
            source / "h2o_0.958_104.5_6-31g_c2v.chk"
        ),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (point_dir / "reference_reuse.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[reuse] registered certified equilibrium reference from {source}", flush=True)
    return True


def run_parent_stage(args, bond_length, point_dir, stage_count, working_store):
    """Run the cumulative schedule through one candidate maximum bond dimension."""
    bond_dims = args.bond_dims[:stage_count]
    sweeps = args.stage_sweeps[:stage_count]
    command = [
        args.python,
        "-u",
        str(args.project_dir / "run_h2o_631g_parent_reference.py"),
        "--run_dir",
        str(point_dir),
        "--working_store",
        str(working_store),
        "--persistent_store",
        str(point_dir / "parent_mps"),
        "--mode",
        "production",
        "--resume",
        "--bond_length",
        str(bond_length),
        "--bond_angle",
        str(args.angle),
        "--basis",
        args.basis,
        "--point_group",
        args.point_group,
        "--symmetry_mode",
        "su2",
        "--ordering",
        "fiedler",
        "--n_threads",
        str(args.threads),
        "--n_mkl_threads",
        "1",
        "--stack_mem_gb",
        str(args.stack_mem_gb),
        "--bond_dims",
        ",".join(str(value) for value in bond_dims),
        "--stage_sweeps",
        ",".join(str(value) for value in sweeps),
        "--reverse_bond_dims",
        "",
        "--m_tolerance_mha",
        str(args.m_tolerance_mha),
        "--sweep_tolerance_mha",
        str(args.sweep_tolerance_mha),
        "--skip_rdms",
        "--skip_entanglement",
    ]
    print(
        f"[adaptive] testing convergence through M={bond_dims[-1]}: "
        + " ".join(command),
        flush=True,
    )
    subprocess.run(command, cwd=args.project_dir, check=True)


def run_point(args, bond_length):
    """Build one checkpoint and adaptively converge its parent energy."""
    point_dir = (args.output_dir / "points" / geometry_key(bond_length)).resolve()
    summary_path = point_dir / "parent_reference_summary.json"
    if args.resume and summary_path.is_file() and summary_is_certified(read_json(summary_path)):
        print(f"[resume] certified point already complete: {point_dir}", flush=True)
        return
    if register_equilibrium_reference(args, bond_length, point_dir):
        return

    if args.working_root is None:
        working_store = point_dir / "parent_mps"
    else:
        working_store = args.working_root.resolve() / geometry_key(bond_length) / "parent_mps"
    working_store.mkdir(parents=True, exist_ok=True)

    for stage_count in range(2, len(args.bond_dims) + 1):
        run_parent_stage(args, bond_length, point_dir, stage_count, working_store)
        summary = read_json(summary_path)
        convergence = summary["convergence"]
        print(
            f"[adaptive] M={args.bond_dims[stage_count - 1]} "
            f"delta_M={convergence['last_two_M_difference_mHa']:.6f} mHa "
            f"delta_sweep={convergence['final_sweep_difference_mHa']:.6f} mHa "
            f"certified={convergence['reference_certified']}",
            flush=True,
        )
        if summary_is_certified(summary):
            return
    raise RuntimeError(
        f"reference at rOH={bond_length:.4f} A did not converge through "
        f"M={args.bond_dims[-1]}"
    )


def aggregate(args):
    """Collect completed per-geometry references into one curve table."""
    rows = []
    missing = []
    for bond_length in args.bond_lengths:
        point_dir = args.output_dir / "points" / geometry_key(bond_length)
        summary_path = point_dir / "parent_reference_summary.json"
        if not summary_path.is_file():
            missing.append(bond_length)
            continue
        summary = read_json(summary_path)
        if not summary_is_certified(summary):
            missing.append(bond_length)
            continue
        final_stage = summary["stages"][-1]
        rows.append(
            {
                "r_oh_angstrom": bond_length,
                "reference_energy_ha": final_stage["energy_Ha"],
                "final_bond_dimension": final_stage["max_bond_dim"],
                "m_difference_mha": summary["convergence"]["last_two_M_difference_mHa"],
                "sweep_difference_mha": summary["convergence"]["final_sweep_difference_mHa"],
                "reused": bool(summary.get("reused_from")),
                "point_directory": str(point_dir.resolve()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "reference_curve.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "system": "H2O",
        "basis": args.basis,
        "angle_HOH_degrees": args.angle,
        "requested_bond_lengths_Angstrom": list(args.bond_lengths),
        "completed_points": len(rows),
        "missing_bond_lengths_Angstrom": missing,
        "reference_curve_csv": str(csv_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    if missing:
        raise RuntimeError(f"missing or uncertified bond lengths: {missing}")


def main():
    """Run one point or aggregate a completed array."""
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_settings(args)
    if args.aggregate_only:
        aggregate(args)
        return
    if args.task_index is None:
        raise ValueError("--task-index is required unless --aggregate-only is used")
    if not 0 <= args.task_index < len(args.bond_lengths):
        raise IndexError(f"task index {args.task_index} is outside the bond grid")
    bond_length = args.bond_lengths[args.task_index]
    print(
        f"[task {args.task_index}/{len(args.bond_lengths) - 1}] "
        f"H2O rOH={bond_length:.4f} A",
        flush=True,
    )
    run_point(args, bond_length)


if __name__ == "__main__":
    main()
