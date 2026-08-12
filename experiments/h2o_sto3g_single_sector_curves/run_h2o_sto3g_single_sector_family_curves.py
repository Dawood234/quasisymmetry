#!/usr/bin/env python3
"""Recompute H2O/STO-3G optimized single-sector comparison curves.

This is a quasisymmetry experiment built on the canonical
``single_sector_oo`` implementation.  It compares four independently
orbital-optimized one-sector models against full STO-3G FCI:

* CASSCF-like CAS(4e,4o) number sector;
* all-even seniority-zero/DOCI sector;
* lowest-energy full seniority sector;
* quartet-plus-spectator-seniority sector.

Every geometry and family is checkpointed independently. The seniority and
quartet families are scanned exactly in the canonical frame, and only their
single lowest sector is orbital optimized. An optional second pass refits each
point from cached neighboring rotations and keeps the lowest converged result;
this is a local-minimum/branch-tracking aid, not a coupled-sector calculation.
No previously saved LAS energy is used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parents[1]
OUTPUT_ROOT = Path(os.environ.get("ALRIS_OUTPUT_ROOT", WORKSPACE_ROOT / "outputs"))
DEFAULT_OUTPUT = (
    OUTPUT_ROOT
    / "quasisymmetry"
    / "h2o"
    / "sto-3g"
    / "single_sector_continuation_curves_20260801"
)


def configure_single_sector_package() -> Path:
    """Resolve the separately maintained one-sector package."""
    requested = os.environ.get("SINGLE_SECTOR_OO_DIR")
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.append(REPOSITORY_ROOT.parent / "single_sector_oo")

    if importlib.util.find_spec("single_sector_oo") is None:
        for candidate in candidates:
            if (candidate / "single_sector_oo" / "__init__.py").is_file():
                sys.path.insert(0, str(candidate.resolve()))
                break

    spec = importlib.util.find_spec("single_sector_oo")
    if spec is None or spec.origin is None:
        searched = ", ".join(str(path) for path in candidates)
        raise ModuleNotFoundError(
            "single_sector_oo is required for this experiment. Install it in the "
            "active environment or set SINGLE_SECTOR_OO_DIR to its project root. "
            f"Searched: {searched}"
        )
    return Path(spec.origin).resolve().parent.parent


SINGLE_SECTOR_PROJECT = configure_single_sector_package()

from single_sector_oo.backends.determinants import build_spin_string_basis
from single_sector_oo.backends.integrals import (
    RHFData,
    build_integrals,
    rotate_integrals,
    transform_integrals,
)
from single_sector_oo.backends.rotations import (
    params_to_rotation,
    rotation_to_params,
    zero_rotation_params,
)
from single_sector_oo.backends.systems import build_h2o_geometry
from single_sector_oo.workflows.h2o.run_h2o_sto3g_parity_casscf_generalization import (
    best_row,
    canonical_results,
    diagonalize_support,
    evaluate_sector_family,
    family_groups_and_rank,
)


FAMILY_ORDER = ("all-even", "seniority", "cas", "quartet")
FAMILY_LABELS = {
    "all-even": "DOCI / all-even seniority",
    "seniority": "Lowest seniority sector",
    "cas": "CASSCF-like CAS(4e,4o)",
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

# Fixed-spin FCI values validated by the completed H2O/STO-3G curve at the
# same C2v geometry convention used here (HOH = 104.5 degrees).
H2O_STO3G_FCI_104P5 = {
    0.700: -74.64369313464032,
    0.800: -74.88300179581195,
    0.900: -74.98769269780746,
    0.958: -75.01262894492994,
    1.100: -75.01262458312669,
    1.250: -74.96755510720359,
    1.500: -74.87343608829161,
    1.750: -74.80092772880100,
    2.000: -74.76198842504448,
    2.250: -74.74627744978720,
    2.500: -74.74059586933893,
    2.750: -74.73851248172926,
    3.000: -74.73773981628111,
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def saved_fci_energy(distance: float, basis: str, angle: float) -> float | None:
    """Return a validated H2O reference only for the matching input model."""
    if basis.lower() != "sto-3g" or not np.isclose(angle, 104.5, atol=1.0e-12):
        return None
    key = round(float(distance), 3)
    if not np.isclose(distance, key, atol=1.0e-12):
        return None
    return H2O_STO3G_FCI_104P5.get(key)


def parse_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("The O-H distance grid is empty.")
    if len(set(values)) != len(values):
        raise ValueError("The O-H distance grid contains duplicates.")
    return sorted(values)


def refine_grid(values: list[float], max_step: float | None) -> list[float]:
    """Add evenly spaced points without dropping any requested geometry."""
    if max_step is None:
        return list(values)
    if max_step <= 0.0:
        raise ValueError("--refine-step must be positive.")

    refined: set[float] = set(values)
    for left, right in zip(values, values[1:]):
        intervals = max(1, int(math.ceil((right - left) / max_step)))
        for index in range(intervals + 1):
            refined.add(left + (right - left) * index / intervals)
    return sorted(refined)


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


def build_integrals_with_rhf_fallback(
    geometry,
    basis: str,
    *,
    charge: int,
    multiplicity: int,
):
    """Retry stretched-bond RHF with a larger cycle budget.

    PySCF's default 50-cycle RHF solve can stop before convergence for stretched
    molecules. The fallback changes only the cycle budget and convergence tolerance,
    preserving the same RHF problem and canonical orbital definition.
    """
    try:
        return build_integrals(
            geometry,
            basis,
            charge=charge,
            multiplicity=multiplicity,
        )
    except RuntimeError as error:
        if str(error) != "RHF did not converge.":
            raise

    from pyscf import gto, scf

    print("  [RHF fallback] retrying with max_cycle=300", flush=True)
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.unit = "Angstrom"
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1.0e-11
    hf_energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge after the 300-cycle fallback.")

    print(
        f"  [RHF fallback] converged E={hf_energy:.12f} Ha "
        f"after {getattr(mf, 'cycles', 'unknown')} cycles",
        flush=True,
    )
    rhf = RHFData(
        mol=mol,
        mf=mf,
        n_orb=int(mf.mo_coeff.shape[1]),
        n_elec=int(mol.nelectron),
        n_alpha=int(mol.nelec[0]),
        n_beta=int(mol.nelec[1]),
        mo_occ=np.asarray(mf.mo_occ),
        mo_energy=np.asarray(mf.mo_energy),
    )
    return transform_integrals(rhf, np.eye(rhf.n_orb))


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


def rotation_fingerprint(rotation: np.ndarray | None) -> str:
    """Identify an optimizer seed so resume cannot reuse a stale continuation."""
    if rotation is None:
        return "identity"
    array = np.asarray(rotation, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_candidate(path: Path) -> tuple[dict, np.ndarray]:
    payload = load_json(path)
    rotation_path = Path(payload["rotation_path"])
    if not rotation_path.is_file():
        raise FileNotFoundError(rotation_path)
    return payload, np.load(rotation_path)


def optimize_initial_lowest_sector(
    *,
    family: str,
    base_integrals,
    basis,
    reference_energy: float,
    maxiter: int | None,
    maxfev: int | None,
    initial_rotation: np.ndarray | None,
    start_label: str,
) -> dict:
    """Freeze the initially lowest support and optimize only that sector."""
    final_family, groups, label_to_text, label_to_description, rank = (
        family_groups_and_rank(family, base_integrals, basis)
    )
    initial_rows = evaluate_sector_family(
        family=final_family,
        groups=groups,
        label_to_text=label_to_text,
        label_to_description=label_to_description,
        integrals=base_integrals,
        basis=basis,
        reference_energy=reference_energy,
        rank=rank,
        compute_spectral_range=False,
    )
    initial_row = best_row(initial_rows, final_family)
    if initial_row is None:
        raise RuntimeError(f"No initial sector found for {family}.")

    initial_label = next(
        label for label in groups if label_to_text(label) == initial_row.sector
    )
    fixed_support = np.asarray(groups[initial_label], dtype=np.int64)
    evaluations = 0

    def objective(params: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        rotation = params_to_rotation(params, base_integrals.n_orb)
        rotated = rotate_integrals(base_integrals, rotation)
        energy, _, _, _ = diagonalize_support(
            rotated, basis, fixed_support, compute_spectral_range=False
        )
        return energy

    options = {"xtol": 1.0e-4, "ftol": 1.0e-8}
    if maxiter is not None:
        options["maxiter"] = int(maxiter)
    if maxfev is not None:
        options["maxfev"] = int(maxfev)
    initial_params = (
        zero_rotation_params(base_integrals.n_orb)
        if initial_rotation is None
        else rotation_to_params(initial_rotation)
    )
    started = time.perf_counter()
    result = minimize(
        objective,
        initial_params,
        method="Powell",
        options=options,
    )
    elapsed = time.perf_counter() - started
    rotation = params_to_rotation(result.x, base_integrals.n_orb)
    final_integrals = rotate_integrals(base_integrals, rotation)
    energy, _, assembly_seconds, diagonalization_seconds = diagonalize_support(
        final_integrals, basis, fixed_support, compute_spectral_range=False
    )
    return {
        "family": family,
        "initial_sector": initial_row.sector,
        "initial_energy_ha": initial_row.energy_ha,
        "initial_guess": start_label,
        "success": bool(result.success),
        "message": str(result.message),
        "energy_ha": energy,
        "error_mha_vs_rotated_fci": 1000.0 * (energy - reference_energy),
        "sector": initial_row.sector,
        "support_dim": int(len(fixed_support)),
        "function_evaluations": evaluations,
        "iterations": int(result.nit),
        "seconds": elapsed,
        "assembly_seconds": assembly_seconds,
        "diagonalization_seconds": diagonalization_seconds,
        "parameters": np.asarray(result.x).tolist(),
        "rotation": rotation,
    }


def selected_family_result(
    *,
    family: str,
    point_dir: Path,
    integrals,
    basis,
    reference_energy: float,
    previous_rotation: np.ndarray | None,
    extra_starts: list[tuple[str, np.ndarray]] | None,
    args,
) -> tuple[dict, np.ndarray]:
    family_dir = point_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    selected_path = family_dir / "selected.json"
    selected_rotation_path = family_dir / "selected_rotation.npy"
    settings = {
        "optimization_mode": "fixed-initial-lowest-sector",
        "continuation": True,
        "reference_energy_hartree": float(reference_energy),
        "maxiter": None if args.no_maxiter else int(args.maxiter),
        "maxfev": args.maxfev,
    }

    starts = [("identity", None)]
    if previous_rotation is not None:
        starts.append(("continuation", previous_rotation))
    for start_label, initial_rotation in extra_starts or []:
        if any(existing_label == start_label for existing_label, _ in starts):
            raise ValueError(f"Duplicate optimization start label: {start_label}")
        starts.append((start_label, np.asarray(initial_rotation)))

    candidates: list[tuple[dict, np.ndarray]] = []
    for start_label, initial_rotation in starts:
        candidate_path = family_dir / f"start_{start_label}.json"
        expected_seed = rotation_fingerprint(initial_rotation)
        if args.resume and candidate_path.is_file():
            candidate, rotation = load_candidate(candidate_path)
            cached_seed = candidate.get("initial_rotation_sha256")
            seed_matches = cached_seed == expected_seed or (
                start_label == "identity" and cached_seed is None
            )
            if (
                candidate.get("optimization_settings") == settings
                and candidate.get("success")
                and seed_matches
            ):
                print(f"  [{family}] {start_label} complete; reusing", flush=True)
                candidates.append((candidate, rotation))
                continue

        print(
            f"  [{family}] optimizing initially lowest sector from {start_label}",
            flush=True,
        )
        started = time.perf_counter()
        item = optimize_initial_lowest_sector(
            family=family,
            base_integrals=integrals,
            basis=basis,
            reference_energy=reference_energy,
            maxiter=None if args.no_maxiter else args.maxiter,
            maxfev=args.maxfev,
            initial_rotation=initial_rotation,
            start_label=start_label,
        )
        rotation = np.asarray(item.pop("rotation"))
        item["optimization_settings"] = settings
        item["initial_rotation_sha256"] = expected_seed
        item["driver_wall_time_s"] = time.perf_counter() - started
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
    result["selection_strategy"] = (
        "identity-plus-continuation"
        if not extra_starts
        else "identity-continuation-neighbor-envelope"
    )
    np.save(selected_rotation_path, rotation)
    result["selected_rotation_path"] = str(selected_rotation_path)
    result["selected_at"] = timestamp()
    atomic_json_dump(selected_path, result)
    return result, rotation


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict], families: list[str]) -> list[dict]:
    summaries = []
    for family in families:
        items = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["r_oh_angstrom"],
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
            "reference": "Full fixed-spin H2O/STO-3G FCI at every geometry",
            "rows": rows,
            "references": references,
            "method_summaries": summaries,
        },
    )
    write_csv(output_dir / "curve_rows.csv", rows)
    write_csv(output_dir / "reference_curve.csv", references)

    lines = [
        "# H2O/STO-3G optimized single-sector comparison",
        "",
        "Every point is a fresh molecular calculation. Errors are relative to full fixed-spin STO-3G FCI at the same geometry.",
        "No coupled LAS result is used.",
        "",
        "## Methods",
        "",
        "- `DOCI / all-even seniority`: the fixed seniority-zero sector.",
        "- `Lowest seniority sector`: all compatible seniority labels are evaluated in the canonical frame; only the lowest block is optimized.",
        "- `CASSCF-like CAS(4e,4o)`: three inactive orbitals are doubly occupied and four electrons move in four active orbitals.",
        "- `Seniority + quartets`: two quartet products resolve correlated broken-pair channels while three spectators keep individual seniorities.",
        "- Seniority and quartet sectors are ranked exactly in the canonical frame; only the initially lowest fixed sector is optimized for each family.",
        "- With `--neighbor-refit`, the forward continuation result is refit from cached left/right geometry rotations and the lowest converged candidate is retained pointwise. This envelope pass is unavailable to independent array tasks.",
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
            "| rOH (A) | method | sector | dimension | energy (Ha) | error (mHa) | converged | route | candidates |",
            "|---:|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['r_oh_angstrom']:.4f} | {row['method']} | `{row['sector']}` | "
            f"{row['support_dim']} | {row['energy_ha']:.12f} | {row['error_mha']:.6f} | "
            f"{row['success']} | {row['selected_start']} | {row['candidate_starts']} |"
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
    reference_by_r = {float(row["r_oh_angstrom"]): row for row in references}

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
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_oh_angstrom"])
        ax.plot(
            [row["r_oh_angstrom"] for row in items],
            [row["energy_ha"] for row in items],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.0,
            markersize=5.5,
            label=FAMILY_LABELS[family],
        )
    ax.set_xlabel("O-H distance (Angstrom)")
    ax.set_ylabel("Electronic energy (Ha)")
    ax.set_title("H2O/STO-3G orbital-optimized single-sector energies")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_curves.png", dpi=240)
    fig.savefig(output_dir / "energy_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for family in families:
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_oh_angstrom"])
        ax.plot(
            [row["r_oh_angstrom"] for row in items],
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
    ax.set_xlabel("O-H distance (Angstrom)")
    ax.set_ylabel("Single-sector error relative to FCI (mHa)")
    ax.set_title("H2O/STO-3G optimized single-sector errors")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_curves.png", dpi=240)
    fig.savefig(output_dir / "error_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for family in families:
        items = sorted((row for row in rows if row["family"] == family), key=lambda row: row["r_oh_angstrom"])
        ax.plot(
            [row["r_oh_angstrom"] for row in items],
            [row["support_dim"] for row in items],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.0,
            markersize=5.5,
            label=FAMILY_LABELS[family],
        )
    ax.set_xlabel("O-H distance (Angstrom)")
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
                        "r_oh_angstrom": float(distance),
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
                        "candidate_starts": ";".join(
                            candidate["source"]
                            for candidate in item.get("candidate_results", [])
                        ),
                    }
                )
    rows.sort(key=lambda row: (row["r_oh_angstrom"], FAMILY_ORDER.index(row["family"])))
    references.sort(key=lambda row: row["r_oh_angstrom"])
    return rows, references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        default=(
            "0.70,0.80,0.90,0.958,1.10,1.25,1.50,1.75,2.00,2.25,"
            "2.50,2.75,3.00"
        ),
        help="Comma-separated O-H distances in Angstrom.",
    )
    parser.add_argument("--angle", type=float, default=104.5)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--families", default=",".join(FAMILY_ORDER))
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--no-maxiter", action="store_true")
    parser.add_argument("--maxfev", type=int, default=None)
    parser.add_argument(
        "--refine-step",
        type=float,
        default=None,
        help=(
            "Subdivide each requested grid interval to at most this step in "
            "Angstrom; preserves all explicitly requested points."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--neighbor-refit",
        action="store_true",
        help=(
            "After the forward continuation pass, refit each point from the "
            "cached neighboring rotations and select the lowest result. "
            "Requires a full sequential grid, not --no-aggregate array tasks."
        ),
    )
    parser.add_argument(
        "--neighbor-refit-passes",
        type=int,
        default=1,
        help="Maximum pointwise neighbor-envelope passes (default: 1).",
    )
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
    requested_grid = parse_grid(args.grid)
    grid = refine_grid(requested_grid, args.refine_step)
    families = parse_families(args.families)
    if args.no_aggregate and args.aggregate_only:
        raise ValueError("--no-aggregate and --aggregate-only are mutually exclusive.")
    if args.no_aggregate and args.neighbor_refit:
        raise ValueError(
            "--neighbor-refit requires a full sequential grid; it cannot run "
            "inside an independent --no-aggregate array task."
        )
    if args.neighbor_refit_passes < 1:
        raise ValueError("--neighbor-refit-passes must be at least one.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "created_at": timestamp(),
        "script": str(Path(__file__).resolve()),
        "canonical_single_sector_project": str(SINGLE_SECTOR_PROJECT),
        "system": "h2o",
        "basis": args.basis,
        "angle_deg": args.angle,
        "requested_grid_angstrom": requested_grid,
        "grid_angstrom": grid,
        "refine_step_angstrom": args.refine_step,
        "families": families,
        "family_labels": {family: FAMILY_LABELS[family] for family in families},
        "maxiter": args.maxiter,
        "no_maxiter": bool(args.no_maxiter),
        "maxfev": args.maxfev,
        "optimization_mode": "fixed-initial-lowest-sector",
        "continuation": True,
        "neighbor_refit": bool(args.neighbor_refit),
        "neighbor_refit_passes": int(args.neighbor_refit_passes),
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

    previous_rotations: dict[str, np.ndarray] = {}
    forward_rotations: dict[tuple[float, str], np.ndarray] = {}
    point_contexts: dict[float, tuple[object, object, float]] = {}
    total_start = time.perf_counter()
    for point_number, distance in enumerate(grid, start=1):
        print("\n" + "=" * 78, flush=True)
        print(
            f"[{timestamp()}] geometry {point_number}/{len(grid)}: rOH={distance:.4f} Angstrom",
            flush=True,
        )
        point_dir = args.output_dir / "points" / geometry_key(distance)
        point_dir.mkdir(parents=True, exist_ok=True)
        geometry = build_h2o_geometry(distance, args.angle)
        integrals = build_integrals_with_rhf_fallback(
            geometry,
            args.basis,
            charge=0,
            multiplicity=1,
        )
        if integrals.n_orb != 7 or integrals.n_alpha != 5 or integrals.n_beta != 5:
            raise RuntimeError(
                "Expected H2O/STO-3G with 7 spatial orbitals and (5,5) electrons; "
                f"got n_orb={integrals.n_orb}, nelec=({integrals.n_alpha},{integrals.n_beta})."
            )
        basis = build_spin_string_basis(integrals.n_orb, integrals.n_alpha, integrals.n_beta)
        reference_path = point_dir / "reference.json"
        if args.resume and reference_path.is_file():
            reference = load_json(reference_path)
            print(f"  [reference] FCI={reference['fci_energy_ha']:.12f} Ha; reusing", flush=True)
        else:
            fci_energy = saved_fci_energy(distance, args.basis, args.angle)
            if fci_energy is None:
                canonical, fci_energy, ranks = canonical_results(integrals, basis, False)
                reference = {
                    "r_oh_angstrom": float(distance),
                    "angle_deg": float(args.angle),
                    "basis": args.basis,
                    "hf_energy_ha": float(integrals.mf.e_tot),
                    "fci_energy_ha": float(fci_energy),
                    "fixed_spin_dimension": int(basis.full_dimension),
                    "ranks": ranks,
                    "reference_rows": [
                        asdict(row)
                        for row in canonical
                        if row.family == "reference" or row.family == "number-CAS" or row.selected
                    ],
                    "reference_source": "computed fixed-spin FCI",
                    "completed_at": timestamp(),
                }
            else:
                reference = {
                    "r_oh_angstrom": float(distance),
                    "angle_deg": float(args.angle),
                    "basis": args.basis,
                    "hf_energy_ha": float(integrals.mf.e_tot),
                    "fci_energy_ha": float(fci_energy),
                    "fixed_spin_dimension": int(basis.full_dimension),
                    "reference_source": "embedded validated H2O/STO-3G FCI table",
                    "completed_at": timestamp(),
                }
            atomic_json_dump(reference_path, reference)
            print(
                f"  [reference] FCI={fci_energy:.12f} Ha; "
                f"{reference['reference_source']}",
                flush=True,
            )

        reference_energy = float(reference["fci_energy_ha"])
        point_contexts[distance] = (integrals, basis, reference_energy)

        for family in families:
            result, rotation = selected_family_result(
                family=family,
                point_dir=point_dir,
                integrals=integrals,
                basis=basis,
                reference_energy=reference_energy,
                previous_rotation=previous_rotations.get(family),
                extra_starts=None,
                args=args,
            )
            previous_rotations[family] = rotation
            forward_rotations[(distance, family)] = np.asarray(rotation)
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

    if args.neighbor_refit and not args.no_aggregate:
        print("\n" + "=" * 78, flush=True)
        print(
            f"[{timestamp()}] neighbor-refit/envelope pass "
            f"(up to {args.neighbor_refit_passes} pass(es))",
            flush=True,
        )
        envelope_rotations = dict(forward_rotations)
        for pass_number in range(1, args.neighbor_refit_passes + 1):
            source_rotations = dict(envelope_rotations)
            updated_rotations: dict[tuple[float, str], np.ndarray] = {}
            changed = False
            print(
                f"  [neighbor envelope] pass {pass_number}/"
                f"{args.neighbor_refit_passes}",
                flush=True,
            )
            for point_index, distance in enumerate(grid):
                integrals, basis, reference_energy = point_contexts[distance]
                point_dir = args.output_dir / "points" / geometry_key(distance)
                for family in families:
                    key = (distance, family)
                    neighbor_starts: list[tuple[str, np.ndarray]] = []
                    if point_index > 0:
                        left = (grid[point_index - 1], family)
                        if left in source_rotations:
                            neighbor_starts.append(
                                ("neighbor_previous", source_rotations[left])
                            )
                    if point_index + 1 < len(grid):
                        right = (grid[point_index + 1], family)
                        if right in source_rotations:
                            neighbor_starts.append(
                                ("neighbor_next", source_rotations[right])
                            )
                    if not neighbor_starts:
                        updated_rotations[key] = source_rotations[key]
                        continue

                    result, rotation = selected_family_result(
                        family=family,
                        point_dir=point_dir,
                        integrals=integrals,
                        basis=basis,
                        reference_energy=reference_energy,
                        previous_rotation=source_rotations[key],
                        extra_starts=neighbor_starts,
                        args=args,
                    )
                    updated_rotations[key] = np.asarray(rotation)
                    changed = changed or not np.allclose(
                        updated_rotations[key], source_rotations[key], atol=1.0e-10
                    )
                    print(
                        f"    [{family}] rOH={distance:.4f} "
                        f"selected={result['start_label']} "
                        f"E={result['energy_ha']:.12f} Ha",
                        flush=True,
                    )

            envelope_rotations.update(updated_rotations)
            if not changed:
                print("  [neighbor envelope] stable; stopping early", flush=True)
                break

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
