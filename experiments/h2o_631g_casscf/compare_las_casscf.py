#!/usr/bin/env python3
"""Compare the successful H2O/6-31G LAS run with a CASSCF ladder."""

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_REFERENCE_ENERGY = -76.1208994639


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--las-run", required=True, type=Path)
    parser.add_argument("--casscf-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=DEFAULT_REFERENCE_ENERGY,
        help="common total-energy reference in Ha",
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def label_text(label):
    return "".join(str(int(bit)) for bit in label)


def gf2_rank(matrix):
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[row:, column])
        if not len(pivots):
            continue
        pivot = row + int(pivots[0])
        work[[row, pivot]] = work[[pivot, row]]
        for other in range(work.shape[0]):
            if other != row and work[other, column]:
                work[other] ^= work[row]
        row += 1
        if row == work.shape[0]:
            break
    return row


def fixed_spin_sector_dimensions(parity_matrix, norb, nalpha, nbeta):
    """Count sector determinants without constructing full determinant vectors."""

    def one_spin_counts(nelec):
        counts = Counter()
        for occupied in itertools.combinations(range(norb), nelec):
            occupation = np.zeros(norb, dtype=np.uint8)
            occupation[list(occupied)] = 1
            label = tuple((parity_matrix @ occupation % 2).tolist())
            counts[label] += 1
        return counts

    alpha_counts = one_spin_counts(nalpha)
    beta_counts = one_spin_counts(nbeta)
    dimensions = {}
    for label in itertools.product((0, 1), repeat=parity_matrix.shape[0]):
        dimension = 0
        for alpha_label, alpha_count in alpha_counts.items():
            beta_label = tuple(a ^ s for a, s in zip(alpha_label, label))
            dimension += alpha_count * beta_counts[beta_label]
        dimensions[label] = dimension
    return dimensions


def validate_coupled_matrix(path, expected_dimension, expected_energy):
    matrix = np.load(path)
    if matrix.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f"{path} has shape {matrix.shape}, expected "
            f"({expected_dimension}, {expected_dimension})"
        )
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    ground_energy = float(np.linalg.eigvalsh(matrix)[0])
    if abs(ground_energy - expected_energy) > 1.0e-9:
        raise ValueError(
            f"{path} ground energy {ground_energy} does not match "
            f"recorded energy {expected_energy}"
        )
    return {
        "shape": list(matrix.shape),
        "hermiticity_error": hermiticity_error,
        "ground_energy_Ha": ground_energy,
    }


def write_sector_distribution(path, labels, dimensions, candidates, initial_labels):
    candidate_counts = Counter(tuple(item["label"]) for item in candidates)
    kind_counts = {}
    for item in candidates:
        label = tuple(item["label"])
        kind_counts.setdefault(label, Counter())[item["kind"]] += 1

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sector_label",
                "determinant_dimension",
                "coupled_basis_vectors",
                "anchor_vectors",
                "initial_krylov_vectors",
                "residual_krylov_vectors",
                "present_before_residual_enrichment",
            ],
        )
        writer.writeheader()
        for label in labels:
            kinds = kind_counts.get(label, {})
            writer.writerow(
                {
                    "sector_label": label_text(label),
                    "determinant_dimension": dimensions[label],
                    "coupled_basis_vectors": candidate_counts[label],
                    "anchor_vectors": kinds.get("anchor", 0),
                    "initial_krylov_vectors": kinds.get("krylov", 0),
                    "residual_krylov_vectors": kinds.get("residual_krylov", 0),
                    "present_before_residual_enrichment": label in initial_labels,
                }
            )


def write_comparison_csv(path, casscf_results, las_points):
    rows = []
    for result in casscf_results:
        rows.append(
            {
                "method": f"CASSCF CAS({2 * result['nelecas_alpha']}e,{result['ncas']}o)",
                "stage": result["space"],
                "variational_subspace_dimension": result["determinant_dimension"],
                "energy_Ha": result["casscf_energy_Ha"],
                "error_mHa": result["common_reference_error_mHa"],
                "sector_count": "",
                "determinant_support": result["determinant_dimension"],
            }
        )
    for point in las_points:
        rows.append(
            {
                "method": "LAS",
                "stage": point["stage"],
                "variational_subspace_dimension": point["K"],
                "energy_Ha": point["energy_Ha"],
                "error_mHa": point["error_mHa"],
                "sector_count": point["sector_count"],
                "determinant_support": point["determinant_support"],
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, casscf_results, las_points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    casscf_dimensions = [item["determinant_dimension"] for item in casscf_results]
    casscf_errors = [item["common_reference_error_mHa"] for item in casscf_results]
    las_dimensions = [item["K"] for item in las_points]
    las_errors = [item["error_mHa"] for item in las_points]

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(
        casscf_dimensions,
        casscf_errors,
        marker="s",
        linewidth=2.0,
        color="#b74f2a",
        label="CASSCF active-space CI dimension",
    )
    axis.plot(
        las_dimensions,
        las_errors,
        marker="o",
        linewidth=2.0,
        color="#176b70",
        label="LAS coupled-basis dimension K",
    )
    axis.axhline(1.6, color="#333333", linestyle="--", linewidth=1.2)
    axis.text(5.0, 1.82, "chemical accuracy", fontsize=9, color="#333333")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Variational subspace dimension")
    axis.set_ylabel("Energy error relative to M=750 DMRG (mHa)")
    axis.set_title("H2O/6-31G at equilibrium: LAS and CASSCF")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown(path, summary):
    las = summary["las"]
    casscf = summary["casscf"]
    best = casscf["smallest_chemical_accuracy_space"]
    lines = [
        "# H2O/6-31G LAS versus CASSCF at equilibrium",
        "",
        f"Common reference: `{summary['reference_energy_Ha']:.12f} Ha` "
        "(the certified M=750 parent DMRG energy).",
        "",
        "## Headline comparison",
        "",
        "| Method | Energy (Ha) | Error (mHa) | Explicit final subspace | Additional support |",
        "|---|---:|---:|---:|---:|",
        (
            f"| LAS residual-enriched | {las['final_energy_Ha']:.12f} | "
            f"{las['final_error_mHa']:.6f} | K={las['final_K']:,} | "
            f"{las['final_sector_count']} sectors; "
            f"{las['final_determinant_support']:,} determinants |"
        ),
        (
            f"| CASSCF {best['space']} | {best['energy_Ha']:.12f} | "
            f"{best['error_mHa']:.6f} | "
            f"{best['determinant_dimension']:,} determinants | "
            "one optimized active space |"
        ),
        "",
        "The LAS final diagonalization is "
        f"{summary['comparison']['casscf_to_las_final_dimension_ratio']:.1f}x "
        "smaller than the smallest chemically accurate CASSCF CI problem. "
        "This is not the full cost ratio: each LAS Krylov vector was generated "
        "through Hamiltonian action in determinant spaces, and the retained LAS "
        f"sectors jointly contain {las['final_determinant_support']:,} determinants.",
        "",
        "## LAS progression",
        "",
        "| Stage | K | Sectors | Determinant support | Error (mHa) |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in las["progression"]:
        lines.append(
            f"| {point['stage']} | {point['K']:,} | "
            f"{point['sector_count']} | {point['determinant_support']:,} | "
            f"{point['error_mHa']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## CASSCF ladder",
            "",
            "| Active space | CI dimension | CASSCF error (mHa) | Chemical accuracy |",
            "|---|---:|---:|:---:|",
        ]
    )
    for point in casscf["results"]:
        lines.append(
            f"| {point['space']} | {point['determinant_dimension']:,} | "
            f"{point['common_reference_error_mHa']:.6f} | "
            f"{'yes' if point['common_reference_error_mHa'] <= 1.6 else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Seven generator rows have GF(2) rank {las['generator_gf2_rank']}.",
            f"- The 32-sector support recomputed here is "
            f"{las['initial_determinant_support']:,}, exactly matching the saved run.",
            f"- The final 48 sectors cover "
            f"{100.0 * las['final_support_fraction']:.3f}% of the full fixed-spin space.",
            f"- The 873 x 873 saved matrix is Hermitian to "
            f"{las['final_matrix']['hermiticity_error']:.3e} and reproduces the saved energy.",
            "- The historical LAS file reports 0.197416 mHa against its M=500 "
            "reference. The fair common-reference error used here is 0.211163 mHa.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    las_run = args.las_run.resolve()
    parity_matrix = np.loadtxt(
        las_run / "selection_0" / "parity_matrix.txt", dtype=np.uint8
    )
    krylov = load_json(las_run / "final_krylov_metrics_s32.json")
    residual = load_json(las_run / "final_residual_metrics_s32.json")
    restart = load_json(las_run / "final_residual_enrichment_s32" / "restart.json")
    casscf_summary = load_json(args.casscf_summary.resolve())

    dimensions = fixed_spin_sector_dimensions(parity_matrix, 13, 5, 5)
    full_dimension = sum(dimensions.values())
    initial_labels = {tuple(label) for label in krylov["sector_labels"]}
    final_labels = [tuple(label) for label in residual["sector_labels"]]
    initial_support = sum(dimensions[label] for label in initial_labels)
    final_support = sum(dimensions[label] for label in final_labels)
    if initial_support != krylov["selected_determinant_count"]:
        raise ValueError(
            f"recomputed initial support {initial_support} does not match "
            f"saved value {krylov['selected_determinant_count']}"
        )
    if len(restart["candidates"]) != residual["K"]:
        raise ValueError("candidate manifest length does not match final K")

    initial_matrix = validate_coupled_matrix(
        las_run / "final_residual_enrichment_s32" / "coupled_cycle_0.npy",
        residual["K_source"],
        residual["E_source"],
    )
    final_matrix = validate_coupled_matrix(
        las_run / "final_residual_enrichment_s32" / "coupled_cycle_1.npy",
        residual["K"],
        residual["E_coupled"],
    )

    reference = args.reference_energy
    las_points = [
        {
            "stage": "decoupled anchor",
            "K": 1,
            "sector_count": 1,
            "determinant_support": dimensions[tuple(krylov["anchor_sector"])],
            "energy_Ha": krylov["E_decoupled"],
            "error_mHa": 1000.0 * (krylov["E_decoupled"] - reference),
        },
        {
            "stage": "initial coupled Krylov",
            "K": residual["K_source"],
            "sector_count": residual["sector_count_source"],
            "determinant_support": initial_support,
            "energy_Ha": residual["E_source"],
            "error_mHa": 1000.0 * (residual["E_source"] - reference),
        },
        {
            "stage": "residual enriched",
            "K": residual["K"],
            "sector_count": residual["sector_label_count"],
            "determinant_support": final_support,
            "energy_Ha": residual["E_coupled"],
            "error_mHa": 1000.0 * (residual["E_coupled"] - reference),
        },
    ]

    casscf_results = []
    for item in casscf_summary["results"]:
        result = dict(item)
        result["common_reference_error_mHa"] = 1000.0 * (
            result["casscf_energy_Ha"] - reference
        )
        casscf_results.append(result)
    chemically_accurate = [
        item for item in casscf_results if item["common_reference_error_mHa"] <= 1.6
    ]
    if not chemically_accurate:
        raise ValueError("no CASSCF active space reached chemical accuracy")
    smallest_chemical = min(
        chemically_accurate, key=lambda item: item["determinant_dimension"]
    )

    candidate_counts = Counter(
        tuple(item["label"]) for item in restart["candidates"]
    )
    final_error = las_points[-1]["error_mHa"]
    summary = {
        "schema": "quasisymmetry.h2o_631g_las_casscf_comparison",
        "reference_energy_Ha": reference,
        "chemical_accuracy_mHa": 1.6,
        "sources": {
            "las_run": str(las_run),
            "casscf_summary": str(args.casscf_summary.resolve()),
        },
        "las": {
            "generator_count": int(parity_matrix.shape[0]),
            "generator_gf2_rank": gf2_rank(parity_matrix),
            "possible_sector_count": 2 ** int(parity_matrix.shape[0]),
            "full_fixed_spin_dimension": full_dimension,
            "anchor_sector": label_text(krylov["anchor_sector"]),
            "anchor_sector_dimension": dimensions[tuple(krylov["anchor_sector"])],
            "initial_sector_count": len(initial_labels),
            "initial_determinant_support": initial_support,
            "initial_K": residual["K_source"],
            "initial_matrix": initial_matrix,
            "final_sector_count": len(final_labels),
            "final_determinant_support": final_support,
            "final_support_fraction": final_support / full_dimension,
            "final_K": residual["K"],
            "final_energy_Ha": residual["E_coupled"],
            "final_error_mHa": final_error,
            "historical_M500_reference_energy_Ha": residual["E_reference"],
            "historical_error_mHa": residual["error_mHa"],
            "candidate_count_distribution": {
                str(count): sectors
                for count, sectors in sorted(Counter(candidate_counts.values()).items())
            },
            "candidate_kind_counts": dict(
                Counter(item["kind"] for item in restart["candidates"])
            ),
            "progression": las_points,
            "final_matrix": final_matrix,
        },
        "casscf": {
            "results": casscf_results,
            "smallest_chemical_accuracy_space": {
                "space": smallest_chemical["space"],
                "determinant_dimension": smallest_chemical["determinant_dimension"],
                "energy_Ha": smallest_chemical["casscf_energy_Ha"],
                "error_mHa": smallest_chemical["common_reference_error_mHa"],
            },
        },
        "comparison": {
            "casscf_to_las_final_dimension_ratio": (
                smallest_chemical["determinant_dimension"] / residual["K"]
            ),
            "las_to_casscf_error_ratio": (
                final_error / smallest_chemical["common_reference_error_mHa"]
            ),
            "las_support_to_casscf_dimension_ratio": (
                final_support / smallest_chemical["determinant_dimension"]
            ),
        },
    }

    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_sector_distribution(
        args.output_dir / "las_sector_distribution.csv",
        final_labels,
        dimensions,
        restart["candidates"],
        initial_labels,
    )
    write_comparison_csv(
        args.output_dir / "comparison_points.csv", casscf_results, las_points
    )
    write_plot(
        args.output_dir / "energy_error_vs_dimension.png", casscf_results, las_points
    )
    write_markdown(args.output_dir / "comparison_summary.md", summary)

    print(f"summary: {summary_path}")
    print(
        f"LAS: K={residual['K']}, sectors={len(final_labels)}, "
        f"support={final_support:,}, error={final_error:.6f} mHa"
    )
    print(
        f"CASSCF: {smallest_chemical['space']}, "
        f"dimension={smallest_chemical['determinant_dimension']:,}, "
        f"error={smallest_chemical['common_reference_error_mHa']:.6f} mHa"
    )


if __name__ == "__main__":
    main()
