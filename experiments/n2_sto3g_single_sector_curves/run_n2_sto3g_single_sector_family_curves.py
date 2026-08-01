#!/usr/bin/env python3
"""Compute N2/STO-3G one-sector curves using quasisymmetry and ffsim only.

For each geometry and family, this driver enumerates the small fixed-spin
STO-3G space only at the canonical RHF orbitals, selects the unique lowest
sector, freezes that determinant support, and minimizes its ground-state
energy under orbital rotations.  The ordered curve evaluates both an identity
start and, after the first point, a continuation start from the previous
geometry, retaining the lower result.  It intentionally does not import the
separate ``single_sector_oo`` project.

The four compared families are all-even seniority, unrestricted seniority,
CAS(6e,6o), and seniority plus three quartet products.  No coupled LAS energy
is used in these curves.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from math import comb
import os
from pathlib import Path
import sys
import time

import ffsim
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parents[1]
OUTPUT_ROOT = Path(os.environ.get("ALRIS_OUTPUT_ROOT", WORKSPACE_ROOT / "outputs"))
DEFAULT_OUTPUT = (
    OUTPUT_ROOT
    / "quasisymmetry"
    / "n2"
    / "sto-3g"
    / "single_sector_continuation_curves_20260801"
)

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemistry import build_n2_geometry
from src.decoupled_energy import (
    fixed_sector_energy,
    rotated_hamiltonian_linop,
    sector_ground_energy,
)
from src.orbital_rotation import params_to_U
from src.sector_utils import symmetry_sectors


# N2/STO-3G canonical-orbital partition used in the prior comparison.
INACTIVE_1B = (1, 2, 3, 4)
SPECTATOR_1B = INACTIVE_1B
QUARTETS_1B = ((7, 10), (5, 8), (6, 9))

FAMILY_ORDER = ("all-even", "seniority", "cas", "quartet")
FAMILY_LABELS = {
    "all-even": "DOCI / all-even seniority",
    "seniority": "Lowest seniority sector",
    "cas": "CASSCF-like CAS(6e,6o)",
    "quartet": "Seniority + quartets",
}
FAMILY_COLORS = {
    "all-even": "#C84E2F",
    "seniority": "#D99923",
    "cas": "#2676A3",
    "quartet": "#16856B",
}
FAMILY_MARKERS = {
    "all-even": "s",
    "seniority": "D",
    "cas": "o",
    "quartet": "^",
}
CHEMICAL_ACCURACY_MHA = 1.6


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("The N-N distance grid is empty.")
    if len(set(values)) != len(values):
        raise ValueError("The N-N distance grid contains duplicates.")
    return sorted(values)


def parse_families(text: str) -> list[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    invalid = sorted(set(values) - set(FAMILY_ORDER))
    if invalid:
        raise ValueError(f"Unsupported families: {invalid}")
    if not values:
        raise ValueError("At least one family must be selected.")
    return [family for family in FAMILY_ORDER if family in values]


def geometry_key(distance: float) -> str:
    return f"r_{distance:.4f}".replace(".", "p")


def build_molecular_data(geometry, basis: str) -> tuple[object, ffsim.MolecularData]:
    """Run canonical RHF and convert it to ffsim molecular data."""
    from pyscf import gto, scf

    mol = gto.M(
        atom=geometry,
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=False,
    )
    mf = scf.RHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1.0e-11
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge after 300 cycles.")
    print(f"  [RHF] E={energy:.12f} Ha", flush=True)
    return mf, ffsim.MolecularData.from_scf(mf)


def seniority_matrix(norb: int) -> np.ndarray:
    return np.eye(norb, dtype=np.uint8)


def quartet_matrix(norb: int) -> np.ndarray:
    rows = []
    for orbital_1b in SPECTATOR_1B:
        row = np.zeros(norb, dtype=np.uint8)
        row[orbital_1b - 1] = 1
        rows.append(row)
    for first_1b, second_1b in QUARTETS_1B:
        row = np.zeros(norb, dtype=np.uint8)
        row[first_1b - 1] = 1
        row[second_1b - 1] = 1
        rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def cas_matrix(norb: int) -> np.ndarray:
    """Fix alpha and beta occupations of the four inactive orbitals."""
    rows = []
    for orbital_1b in INACTIVE_1B:
        orbital = orbital_1b - 1
        for spin_offset in (0, 1):
            row = np.zeros(2 * norb, dtype=np.uint8)
            row[2 * orbital + spin_offset] = 1
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def family_sectors(family: str, norb: int, nelec: tuple[int, int]) -> dict:
    """Build exact determinant supports for one N2/STO-3G parity family."""
    seniority = symmetry_sectors(seniority_matrix(norb), norb, nelec)
    if family == "all-even":
        label = (0,) * norb
        return {label: seniority[label]}
    if family == "seniority":
        return seniority
    if family == "quartet":
        return symmetry_sectors(quartet_matrix(norb), norb, nelec)
    if family == "cas":
        sectors = symmetry_sectors(cas_matrix(norb), norb, nelec)
        label = (1,) * (2 * len(INACTIVE_1B))
        return {label: sectors[label]}
    raise ValueError(f"Unsupported family: {family}")


def label_text(label: tuple[int, ...]) -> str:
    return "".join(str(int(bit)) for bit in label)


def full_fci_energy(moldata: ffsim.MolecularData) -> float:
    """Obtain the full fixed-spin FCI energy without materializing its matrix."""
    linop = ffsim.linear_operator(
        moldata.hamiltonian,
        norb=moldata.norb,
        nelec=moldata.nelec,
    )
    energy = eigsh(linop, k=1, which="SA", tol=1.0e-11, return_eigenvectors=False)
    return float(np.real(energy[0]))


def scan_initial_sector_energies(
    moldata: ffsim.MolecularData,
    sectors: dict,
    params: np.ndarray,
    *,
    family: str,
) -> list[tuple[float, tuple[int, ...], int]]:
    """Exactly rank canonical-frame sectors while reporting long scans."""
    hamiltonian = rotated_hamiltonian_linop(moldata, params)
    total = len(sectors)
    results = []
    for number, (label, support) in enumerate(sectors.items(), start=1):
        energy = sector_ground_energy(hamiltonian, support)
        results.append((energy, label, len(support)))
        if total > 1 and (number == 1 or number % 25 == 0 or number == total):
            print(
                f"  [{family}] canonical sector scan {number}/{total}; "
                f"current dim={len(support)}",
                flush=True,
            )
    results.sort(key=lambda item: item[0])
    return results


def atomic_json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_candidate(path: Path, item: dict, rotation: np.ndarray) -> dict:
    payload = dict(item)
    payload.pop("rotation", None)
    rotation_path = path.with_name(path.stem + "_rotation.npy")
    np.save(rotation_path, np.asarray(rotation))
    payload["rotation_path"] = str(rotation_path)
    payload["completed_at"] = timestamp()
    atomic_json_dump(path, payload)
    return payload


def load_candidate(path: Path) -> tuple[dict, np.ndarray]:
    payload = load_json(path)
    rotation_path = Path(payload["rotation_path"])
    if not rotation_path.is_file():
        raise FileNotFoundError(rotation_path)
    return payload, np.load(rotation_path)


def load_or_scan_initial_sector(
    *,
    family: str,
    family_dir: Path,
    moldata: ffsim.MolecularData,
    args,
) -> tuple[dict, dict]:
    """Cache the canonical lowest sector and reconstruct its determinant support."""
    scan_path = family_dir / "initial_sector.json"
    settings = {
        "backend": "quasisymmetry_ffsim",
        "family": family,
        "basis": args.basis,
        "scan_rotation": "identity",
    }
    scan = None
    if args.resume and scan_path.is_file():
        candidate = load_json(scan_path)
        if candidate.get("settings") == settings:
            scan = candidate
            print(
                f"  [{family}] canonical sector scan complete; reusing "
                f"{candidate['sector_count']} sectors",
                flush=True,
            )

    sectors = family_sectors(family, moldata.norb, moldata.nelec)
    if scan is None:
        zero_params = np.zeros(moldata.norb * (moldata.norb - 1) // 2)
        started = time.perf_counter()
        initial_energy, initial_label, support_dim = scan_initial_sector_energies(
            moldata, sectors, zero_params, family=family
        )[0]
        scan = {
            "settings": settings,
            "family": family,
            "sector": label_text(initial_label),
            "sector_bits": list(initial_label),
            "support_dim": int(support_dim),
            "initial_energy_ha": float(initial_energy),
            "sector_count": int(len(sectors)),
            "seconds": float(time.perf_counter() - started),
            "completed_at": timestamp(),
        }
        atomic_json_dump(scan_path, scan)

    initial_label = tuple(int(bit) for bit in scan["sector_bits"])
    try:
        support = sectors[initial_label]
    except KeyError as error:
        raise RuntimeError(
            f"Cached sector label {scan['sector']} is not present in the "
            f"current {family} sector partition."
        ) from error
    return scan, support


def selected_family_result(
    *,
    family: str,
    point_dir: Path,
    moldata: ffsim.MolecularData,
    reference_energy: float,
    previous_parameters: np.ndarray | None,
    args,
) -> tuple[dict, np.ndarray]:
    """Optimize the canonical lowest support from identity and continuation starts."""
    family_dir = point_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    selected_rotation_path = family_dir / "selected_rotation.npy"
    settings = {
        "backend": "quasisymmetry_ffsim",
        "optimization_mode": "fixed-initial-lowest-sector",
        "continuation": True,
        "basis": args.basis,
        "maxiter": None if args.no_maxiter else int(args.maxiter),
        "maxfev": args.maxfev,
    }

    scan, support = load_or_scan_initial_sector(
        family=family,
        family_dir=family_dir,
        moldata=moldata,
        args=args,
    )
    print(
        f"  [{family}] initial lowest sector={scan['sector']} "
        f"dim={scan['support_dim']} E={scan['initial_energy_ha']:.12f} Ha",
        flush=True,
    )

    starts = [("identity", None)]
    if previous_parameters is not None:
        starts.append(("continuation", previous_parameters))

    candidates: list[tuple[dict, np.ndarray]] = []
    for start_label, initial_parameters in starts:
        candidate_path = family_dir / f"start_{start_label}.json"
        if args.resume and candidate_path.is_file():
            candidate, rotation = load_candidate(candidate_path)
            if candidate.get("optimization_settings") == settings and candidate.get("success"):
                print(f"  [{family}] {start_label} complete; reusing", flush=True)
                candidates.append((candidate, rotation))
                continue

        print(f"  [{family}] optimizing fixed support from {start_label}", flush=True)
        evaluations = 0

        def objective(params: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            return float(fixed_sector_energy(moldata, support, params))

        options = {
            "maxiter": None if args.no_maxiter else args.maxiter,
            "xtol": 1.0e-4,
            "ftol": 1.0e-8,
        }
        if options["maxiter"] is None:
            options.pop("maxiter")
        if args.maxfev is not None:
            options["maxfev"] = args.maxfev
        zero_params = np.zeros(moldata.norb * (moldata.norb - 1) // 2)
        if initial_parameters is not None:
            # Reuse the optimizer coordinates directly. This avoids taking a
            # matrix logarithm of a finite rotation and preserves the exact
            # parameter convention used by params_to_U.
            initial_params = np.asarray(initial_parameters, dtype=float).copy()
        else:
            initial_params = zero_params
        started = time.perf_counter()
        result = minimize(objective, initial_params, method="Powell", options=options)
        elapsed = time.perf_counter() - started
        final_energy = float(fixed_sector_energy(moldata, support, result.x))
        rotation = params_to_U(result.x, moldata.norb)
        item = {
            "family": family,
            "method": FAMILY_LABELS[family],
            "backend": "quasisymmetry_ffsim",
            "success": bool(result.success),
            "message": str(result.message),
            "energy_ha": final_energy,
            "error_mha_vs_rotated_fci": 1000.0 * (final_energy - reference_energy),
            "sector": scan["sector"],
            "sector_bits": scan["sector_bits"],
            "support_dim": int(scan["support_dim"]),
            "initial_energy_ha": float(scan["initial_energy_ha"]),
            "function_evaluations": int(evaluations),
            "iterations": int(result.nit),
            "seconds": float(elapsed),
            "parameters": np.asarray(result.x).tolist(),
            "initial_guess": start_label,
            "optimization_settings": settings,
            "rotation": rotation,
        }
        candidate = save_candidate(candidate_path, item, rotation)
        candidates.append((candidate, rotation))

    if not candidates:
        raise RuntimeError(f"No completed optimization candidate for {family}.")

    payload, rotation = min(candidates, key=lambda pair: float(pair[0]["energy_ha"]))
    result = dict(payload)
    result["family"] = family
    result["method"] = FAMILY_LABELS[family]
    result["start_label"] = result["initial_guess"]
    result["candidate_results"] = [
        {
            "source": candidate["initial_guess"],
            "energy_ha": candidate["energy_ha"],
            "error_mha": candidate["error_mha_vs_rotated_fci"],
            "sector": candidate["sector"],
            "success": candidate["success"],
            "function_evaluations": candidate["function_evaluations"],
        }
        for candidate, _candidate_rotation in candidates
    ]
    np.save(selected_rotation_path, rotation)
    result["selected_rotation_path"] = str(selected_rotation_path)
    result["selected_at"] = timestamp()
    atomic_json_dump(family_dir / "selected.json", result)
    return result, rotation

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict], families: list[str]) -> list[dict]:
    summaries = []
    for family in families:
        items = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["r_nn_angstrom"],
        )
        errors = [float(row["error_mha"]) for row in items]
        sectors = [row["sector"] for row in items]
        summaries.append(
            {
                "family": family,
                "method": FAMILY_LABELS[family],
                "points": len(items),
                "max_error_mha": max(errors),
                "mean_error_mha": sum(errors) / len(errors),
                "npe_mha": max(errors) - min(errors),
                "chemical_points": sum(error <= CHEMICAL_ACCURACY_MHA for error in errors),
                "dimensions": sorted({int(row["support_dim"]) for row in items}),
                "sector_switches": sum(left != right for left, right in zip(sectors, sectors[1:])),
                "total_optimizer_evaluations": sum(int(row["function_evaluations"]) for row in items),
                "total_optimizer_seconds": sum(float(row["seconds"]) for row in items),
            }
        )
    return summaries


def write_report(output_dir: Path, rows: list[dict], references: list[dict], families: list[str]) -> None:
    summaries = summary_rows(rows, families)
    atomic_json_dump(
        output_dir / "summary.json",
        {
            "updated_at": timestamp(),
            "reference": "Full fixed-spin N2/STO-3G FCI at every geometry",
            "rows": rows,
            "references": references,
            "method_summaries": summaries,
        },
    )
    write_csv(output_dir / "curve_rows.csv", rows)
    write_csv(output_dir / "reference_curve.csv", references)

    lines = [
        "# N2/STO-3G optimized single-sector comparison",
        "",
        "Every point is a fresh molecular calculation. Errors are relative to full fixed-spin STO-3G FCI at the same geometry.",
        "No coupled LAS result is used.",
        "",
        "## Methods",
        "",
        "- `DOCI / all-even seniority`: the fixed seniority-zero sector.",
        "- `Lowest seniority sector`: all compatible seniority labels are evaluated in the canonical frame; only the lowest block is optimized.",
        "- `CASSCF-like CAS(6e,6o)`: four inactive orbitals are doubly occupied and six electrons move in six active orbitals.",
        "- `Seniority + quartets`: three quartet products resolve the sigma and two pi bond-breaking channels while four spectators keep individual seniorities.",
        "- Seniority and quartet sectors are ranked exactly in the canonical frame; only the initially lowest fixed sector is optimized for each family.",
        "",
        "## Curve summary",
        "",
        "| method | points | dimensions | max error (mHa) | mean error (mHa) | NPE (mHa) | chemical points | sector switches | evals | seconds |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        dimensions = ", ".join(str(value) for value in item["dimensions"])
        lines.append(
            f"| {item['method']} | {item['points']} | {dimensions} | "
            f"{item['max_error_mha']:.6f} | {item['mean_error_mha']:.6f} | "
            f"{item['npe_mha']:.6f} | {item['chemical_points']}/{item['points']} | "
            f"{item['sector_switches']} | {item['total_optimizer_evaluations']} | "
            f"{item['total_optimizer_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pointwise optimized sectors",
            "",
            "| rNN (A) | method | sector | dimension | energy (Ha) | error (mHa) | converged | route |",
            "|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['r_nn_angstrom']:.4f} | {row['method']} | `{row['sector']}` | "
            f"{row['support_dim']} | {row['energy_ha']:.12f} | {row['error_mha']:.6f} | "
            f"{row['success']} | {row['selected_start']} |"
        )
    lines.extend(
        [
            "",
            "## Comparison caveat",
            "",
            "These are the natural sectors proposed by each family, so their determinant dimensions are not necessarily equal. The dimension curve must be reported with the energy curves; a strict matched-dimension claim requires a separate capped-sector comparison.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def make_plots(output_dir: Path, rows: list[dict], references: list[dict], families: list[str]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    reference_by_r = {float(row["r_nn_angstrom"]): row for row in references}

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    grid = sorted(reference_by_r)
    ax.plot(
        grid,
        [reference_by_r[value]["fci_energy_ha"] for value in grid],
        color="#202B33",
        linewidth=2.8,
        label="FCI",
        zorder=5,
    )
    for family in families:
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_nn_angstrom"])
        ax.plot(
            [row["r_nn_angstrom"] for row in items],
            [row["energy_ha"] for row in items],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.0,
            markersize=5.5,
            label=FAMILY_LABELS[family],
        )
    ax.set_xlabel("N-N distance (Angstrom)")
    ax.set_ylabel("Electronic energy (Ha)")
    ax.set_title("N2/STO-3G orbital-optimized single-sector energies")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_curves.png", dpi=240)
    fig.savefig(output_dir / "energy_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for family in families:
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_nn_angstrom"])
        ax.plot(
            [row["r_nn_angstrom"] for row in items],
            [max(float(row["error_mha"]), 1.0e-8) for row in items],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.0,
            markersize=5.5,
            label=FAMILY_LABELS[family],
        )
    ax.axhline(
        CHEMICAL_ACCURACY_MHA,
        color="#202B33",
        linestyle="--",
        linewidth=1.5,
        label="Chemical accuracy (1.6 mHa)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("N-N distance (Angstrom)")
    ax.set_ylabel("Single-sector error relative to FCI (mHa)")
    ax.set_title("N2/STO-3G optimized single-sector errors")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_curves.png", dpi=240)
    fig.savefig(output_dir / "error_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for family in families:
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_nn_angstrom"])
        ax.plot(
            [row["r_nn_angstrom"] for row in items],
            [row["support_dim"] for row in items],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.0,
            markersize=5.5,
            label=FAMILY_LABELS[family],
        )
    ax.set_xlabel("N-N distance (Angstrom)")
    ax.set_ylabel("Sector determinant dimension")
    ax.set_title("Retained one-sector dimensions")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "sector_dimensions.png", dpi=240)
    fig.savefig(output_dir / "sector_dimensions.pdf")
    plt.close(fig)


def collect_completed(output_dir: Path, grid: list[float], families: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    references: list[dict] = []
    for distance in grid:
        point_dir = output_dir / "points" / geometry_key(distance)
        reference_path = point_dir / "reference.json"
        if reference_path.is_file():
            references.append(load_json(reference_path))
        for family in families:
            selected_path = point_dir / family / "selected.json"
            if selected_path.is_file():
                item = load_json(selected_path)
                rows.append(
                    {
                        "r_nn_angstrom": float(distance),
                        "family": family,
                        "method": FAMILY_LABELS[family],
                        "sector": item["sector"],
                        "support_dim": int(item["support_dim"]),
                        "energy_ha": float(item["energy_ha"]),
                        "error_mha": float(item["error_mha_vs_rotated_fci"]),
                        "success": bool(item["success"]),
                        "message": item["message"],
                        "function_evaluations": int(item["function_evaluations"]),
                        "iterations": int(item["iterations"]),
                        "seconds": float(item["seconds"]),
                        "selected_start": item["start_label"],
                    }
                )
    rows.sort(key=lambda row: (row["r_nn_angstrom"], FAMILY_ORDER.index(row["family"])))
    references.sort(key=lambda row: row["r_nn_angstrom"])
    return rows, references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        default=(
            "1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,"
            "2.00,2.10,2.20,2.30,2.40,2.50,2.60,2.70,2.80,2.90,3.00"
        ),
        help="Comma-separated N-N distances in Angstrom.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--families", default=",".join(FAMILY_ORDER))
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--no-maxiter", action="store_true")
    parser.add_argument("--maxfev", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Write point checkpoints only; used by independent Slurm array tasks.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Build summaries and plots from an already completed grid.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = parse_grid(args.grid)
    families = parse_families(args.families)
    if args.no_aggregate and args.aggregate_only:
        raise ValueError("--no-aggregate and --aggregate-only are mutually exclusive.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "created_at": timestamp(),
        "script": str(Path(__file__).resolve()),
        "system": "n2",
        "basis": args.basis,
        "grid_angstrom": grid,
        "families": families,
        "family_labels": {family: FAMILY_LABELS[family] for family in families},
        "maxiter": args.maxiter,
        "no_maxiter": bool(args.no_maxiter),
        "maxfev": args.maxfev,
        "backend": "quasisymmetry_ffsim",
        "optimization_mode": "fixed-initial-lowest-sector",
        "continuation": True,
        "reference": "Full fixed-spin FCI recomputed at every geometry",
        "output_dir": str(args.output_dir),
    }
    if args.no_aggregate:
        task_name = geometry_key(grid[0]) if len(grid) == 1 else f"pid_{os.getpid()}"
        atomic_json_dump(
            args.output_dir / "task_configurations" / f"{task_name}.json",
            configuration,
        )
    else:
        atomic_json_dump(args.output_dir / "configuration.json", configuration)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"Grid: {grid}", flush=True)
    print(f"Families: {families}", flush=True)
    if args.dry_run:
        print("Dry run complete; no molecular calculation was started.", flush=True)
        return

    if args.aggregate_only:
        rows, references = collect_completed(args.output_dir, grid, families)
        expected_rows = len(grid) * len(families)
        if len(references) != len(grid) or len(rows) != expected_rows:
            raise RuntimeError(
                "Cannot aggregate an incomplete curve: "
                f"found {len(references)}/{len(grid)} references and "
                f"{len(rows)}/{expected_rows} family results."
            )
        write_report(args.output_dir, rows, references, families)
        make_plots(args.output_dir, rows, references, families)
        completion = {
            "status": "complete",
            "completed_at": timestamp(),
            "points": len(grid),
            "families": len(families),
            "result_rows": len(rows),
        }
        atomic_json_dump(args.output_dir / "completion.json", completion)
        print(f"Report: {args.output_dir / 'summary.md'}", flush=True)
        print(f"Energy plot: {args.output_dir / 'energy_curves.png'}", flush=True)
        print(f"Error plot: {args.output_dir / 'error_curves.png'}", flush=True)
        return

    previous_parameters: dict[str, np.ndarray] = {}
    total_start = time.perf_counter()
    for point_number, distance in enumerate(grid, start=1):
        print("\n" + "=" * 78, flush=True)
        print(
            f"[{timestamp()}] geometry {point_number}/{len(grid)}: rNN={distance:.4f} Angstrom",
            flush=True,
        )
        point_dir = args.output_dir / "points" / geometry_key(distance)
        point_dir.mkdir(parents=True, exist_ok=True)
        mf, moldata = build_molecular_data(build_n2_geometry(distance), args.basis)
        if moldata.norb != 10 or moldata.nelec != (7, 7):
            raise RuntimeError(
                "Expected N2/STO-3G with 10 spatial orbitals and (7,7) electrons; "
                f"got n_orb={moldata.norb}, nelec={moldata.nelec}."
            )
        full_dimension = comb(moldata.norb, moldata.nelec[0]) * comb(moldata.norb, moldata.nelec[1])
        reference_path = point_dir / "reference.json"
        if args.resume and reference_path.is_file():
            reference = load_json(reference_path)
            print(f"  [reference] FCI={reference['fci_energy_ha']:.12f} Ha; reusing", flush=True)
        else:
            started = time.perf_counter()
            fci_energy = full_fci_energy(moldata)
            reference = {
                "r_nn_angstrom": float(distance),
                "basis": args.basis,
                "hf_energy_ha": float(mf.e_tot),
                "fci_energy_ha": float(fci_energy),
                "fixed_spin_dimension": int(full_dimension),
                "backend": "ffsim_sparse_eigsh",
                "seconds": time.perf_counter() - started,
                "completed_at": timestamp(),
            }
            atomic_json_dump(reference_path, reference)
            print(f"  [reference] FCI={fci_energy:.12f} Ha", flush=True)

        for family in families:
            result, _rotation = selected_family_result(
                family=family,
                point_dir=point_dir,
                moldata=moldata,
                reference_energy=float(reference["fci_energy_ha"]),
                previous_parameters=previous_parameters.get(family),
                args=args,
            )
            previous_parameters[family] = np.asarray(result["parameters"], dtype=float)
            print(
                f"  [{family}] sector={result['sector']} dim={result['support_dim']} "
                f"E={result['energy_ha']:.12f} Ha "
                f"error={result['error_mha_vs_rotated_fci']:.6f} mHa",
                flush=True,
            )

        if not args.no_aggregate:
            rows, references = collect_completed(args.output_dir, grid, families)
            write_report(args.output_dir, rows, references, families)
            make_plots(args.output_dir, rows, references, families)

    if args.no_aggregate:
        print(f"Point task complete in {time.perf_counter() - total_start:.2f} s", flush=True)
        return

    rows, references = collect_completed(args.output_dir, grid, families)
    write_report(args.output_dir, rows, references, families)
    make_plots(args.output_dir, rows, references, families)
    elapsed = time.perf_counter() - total_start
    completion = {
        "status": "complete",
        "completed_at": timestamp(),
        "wall_time_s": elapsed,
        "points": len(grid),
        "families": len(families),
        "result_rows": len(rows),
    }
    atomic_json_dump(args.output_dir / "completion.json", completion)
    print("\n" + "=" * 78, flush=True)
    print(f"Complete in {elapsed:.2f} s", flush=True)
    print(f"Report: {args.output_dir / 'summary.md'}", flush=True)
    print(f"Energy plot: {args.output_dir / 'energy_curves.png'}", flush=True)
    print(f"Error plot: {args.output_dir / 'error_curves.png'}", flush=True)

if __name__ == "__main__":
    main()
