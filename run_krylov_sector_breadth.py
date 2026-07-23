#!/usr/bin/env python3
"""Compare coupling-seeded Krylov energies as more sectors are retained.

This driver reuses a completed H2O/6-31G workflow directory.  It calls the
existing selected-sector Krylov evaluator for each requested sector count,
shares the per-sector Krylov checkpoints, and writes a compact comparison.
Earlier workflow stages and the 16-sector result are never overwritten.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def parse_counts(text):
    """Return sorted, unique positive sector counts from comma-separated text."""
    counts = sorted(set(int(value) for value in str(text).split(",")))
    if not counts or counts[0] < 2:
        raise ValueError("sector counts must be integers greater than one")
    return counts


def latest_parent_result(run_dir):
    """Find the available final-parent result with the largest bond dimension."""
    candidates = []
    for path in Path(run_dir).glob("final_parent_M*.txt"):
        text = path.stem.removeprefix("final_parent_M")
        if text.isdigit():
            candidates.append((int(text), path))
    if not candidates:
        raise FileNotFoundError("no final_parent_M*.txt result found")
    return max(candidates)[1]


def final_curve_row(data):
    """Extract the largest-depth row from one Krylov metrics file."""
    curve = data.get("coupled_curve", [])
    if not curve:
        raise ValueError("Krylov metrics contain no coupled depth curve")
    return max(curve, key=lambda row: int(row["depth"]))


def result_row(path):
    """Load one metrics file and return the fields used in the comparison."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    final = final_curve_row(data)
    return {
        "sector_count": int(data["sector_label_count"]),
        "selected_leakage_fraction": float(data["selected_leakage_fraction"]),
        "depth": int(final["depth"]),
        "dimension": int(final["dimension"]),
        "energy": float(final["energy"]),
        "error_mHa": float(final["error_mHa"]),
        "converged": bool(final["converged"]),
        "total_seconds": float(data["timings"]["total_seconds"]),
        "path": str(Path(path).resolve()),
    }


def write_json(path, data):
    """Write JSON through a temporary file so interrupted writes are harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_summary(run_dir, rows):
    """Write machine-readable and Markdown sector-breadth comparisons."""
    run_dir = Path(run_dir)
    rows = sorted(rows, key=lambda row: row["sector_count"])
    write_json(run_dir / "krylov_sector_breadth.json", {"results": rows})

    lines = [
        "# Coupling-seeded Krylov sector-breadth scan",
        "",
        "| sectors | leakage captured | depth | K | error (mHa) | time (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sector_count']} | "
            f"{100.0 * row['selected_leakage_fraction']:.6f}% | "
            f"{row['depth']} | {row['dimension']} | "
            f"{row['error_mHa']:.6f} | {row['total_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Errors are relative to the saved final parent DMRG energy.  Each "
            "external sector uses a coupling-seeded Krylov basis, not its "
            "lowest-energy roots.",
        ]
    )
    (run_dir / "krylov_sector_breadth.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def seed_sector_checkpoints(run_dir, work_dir):
    """Copy existing per-sector bases into an isolated breadth work directory."""
    destination = Path(work_dir) / "sectors"
    destination.mkdir(parents=True, exist_ok=True)
    sources = [Path(run_dir) / "final_selected_clifford_krylov" / "sectors"]
    sources.extend(
        sorted(Path(run_dir).glob("final_selected_clifford_krylov_s*/sectors"))
    )

    copied = 0
    for source in sources:
        if not source.exists() or source.resolve() == destination.resolve():
            continue
        for checkpoint in source.glob("sector_*.npz"):
            target = destination / checkpoint.name
            if target.exists():
                continue
            shutil.copy2(checkpoint, target)
            copied += 1
    print(
        f"[breadth] copied {copied} reusable sector checkpoints into "
        f"{destination}",
        flush=True,
    )


def run_count(args, count, reference_result, optimized_json):
    """Run or reuse one selected-sector Krylov calculation."""
    output = args.run_dir / f"final_krylov_metrics_s{count}.json"
    if output.exists() and not args.force:
        print(f"[breadth] reusing completed {count}-sector result: {output}", flush=True)
        return output

    work_dir = args.run_dir / f"final_selected_clifford_krylov_s{count}"
    seed_sector_checkpoints(args.run_dir, work_dir)

    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "selected_clifford_krylov.py"),
        str(optimized_json),
        "--reference_result",
        str(reference_result),
        "--work_dir",
        str(work_dir),
        "--max_sectors",
        str(count),
        "--krylov_depths",
        args.krylov_depths,
        "--krylov_tolerance",
        str(args.krylov_tolerance),
        "--skip_lcu_files",
        "--outname",
        str(output),
        "--resume",
    ]
    anchor_dir = args.run_dir / "final_selected_clifford_lanczos" / "sectors"
    if anchor_dir.exists():
        command.extend(["--anchor_checkpoint_dir", str(anchor_dir)])

    print("\n" + "=" * 78, flush=True)
    print(f"[breadth] START {count} selected sectors", flush=True)
    print("command:", " ".join(command), flush=True)
    print("=" * 78, flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_DIR)
    print(f"[breadth] DONE {count} selected sectors", flush=True)
    return output


def parse_args():
    """Parse controls for the restartable breadth scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--sector_counts", default="24,32")
    parser.add_argument("--krylov_depths", default="1,2,4,8,12,16,24")
    parser.add_argument("--krylov_tolerance", type=float, default=1.0e-12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    """Run requested breadth points and summarize them with the baseline."""
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    counts = parse_counts(args.sector_counts)
    optimized_json = args.run_dir / "final_optimized.json"
    if not optimized_json.exists():
        raise FileNotFoundError(optimized_json)
    reference_result = latest_parent_result(args.run_dir)

    print("run directory:", args.run_dir, flush=True)
    print("parent reference:", reference_result, flush=True)
    print("sector counts:", counts, flush=True)
    print("Krylov depths:", args.krylov_depths, flush=True)
    print("each sector count uses an isolated coupled-matrix directory", flush=True)

    result_paths = []
    baseline = args.run_dir / "final_krylov_metrics.json"
    if baseline.exists():
        result_paths.append(baseline)
    for count in counts:
        result_paths.append(
            run_count(args, count, reference_result, optimized_json)
        )

    rows_by_count = {}
    for path in result_paths:
        row = result_row(path)
        rows_by_count[row["sector_count"]] = row
    rows = list(rows_by_count.values())
    write_summary(args.run_dir, rows)

    print("\nSector-breadth comparison", flush=True)
    for row in sorted(rows, key=lambda item: item["sector_count"]):
        print(
            f"  sectors={row['sector_count']:2d} "
            f"leakage={100.0 * row['selected_leakage_fraction']:.6f}% "
            f"K={row['dimension']:4d} error={row['error_mHa']:.6f} mHa",
            flush=True,
        )
    print("summary:", args.run_dir / "krylov_sector_breadth.md", flush=True)


if __name__ == "__main__":
    main()
