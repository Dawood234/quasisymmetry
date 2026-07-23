#!/usr/bin/env python3
"""Evaluate selected LAS sectors with Clifford tapering and physical Lanczos.

The parent MPS has already screened the important labels during orbital
optimization.  This script constructs only those fixed-spin sector supports,
solves their low roots without a full Hamiltonian matrix, and builds the final
coupled-state Hamiltonian.  Every sector solve is checkpointed independently.
"""

import argparse
import gzip
import json
import resource
import sys
import time
from pathlib import Path

import ffsim
import numpy as np

from chemistry import CHEMICAL_PRECISION, load_moldata
from src.clifford_sectors import (
    build_clifford_frame,
    load_symmetry_manifest,
    molecular_hamiltonian_to_jw,
    qubit_operator_to_data,
    tapered_operator,
    z_symmetries_from_parity_matrix,
)
from src.coupled_energy_core import one_shot_from_hamiltonian
from src.orbital_rotation import rotation_from_oo_data
from src.selected_sector_lanczos import (
    coupled_candidate_matrix,
    label_text,
    selected_sector_supports,
    solve_selected_sector,
    spin_orbital_parity_matrix,
)


def timestamp():
    """Current local time for progress messages."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rss_mib():
    """Current process resident-memory estimate in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def stage(name):
    """Print a visible stage boundary in unbuffered cluster logs."""
    print("\n" + "=" * 78, flush=True)
    print(f"[{timestamp()}] {name}  RSS={rss_mib():.1f} MiB", flush=True)
    print("=" * 78, flush=True)


def atomic_json(path, data):
    """Write JSON through a temporary file so restart metadata is durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_dmrg_energy(path):
    """Read the E_DMRG line emitted by solve_dmrg.py."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("E_DMRG "):
            return float(line.split()[1])
    raise ValueError(f"E_DMRG not found in {path}")


def selected_labels(input_data, maximum):
    """Use MPS-screened labels, ordered by their saved reference weights."""
    raw_labels = input_data.get("screened_sector_labels", [])
    if not raw_labels:
        raise ValueError("optimized JSON does not contain screened_sector_labels")
    weights = input_data.get("sector_weights", {})
    labels = [tuple(int(bit) for bit in label) for label in raw_labels]
    labels.sort(key=lambda label: -float(weights.get(label_text(label), 0.0)))

    anchor = input_data.get("selected_sector")
    if anchor is not None:
        anchor = tuple(int(bit) for bit in anchor)
        if anchor in labels:
            labels.remove(anchor)
        labels.insert(0, anchor)
    return labels[: int(maximum)]


def validate_anchor_energy(input_data, label, energy, tolerance):
    """Check the exact anchor-sector root against the optimized DMRG value."""
    anchor = input_data.get("selected_sector")
    expected = input_data.get("cost_after")
    if anchor is None or expected is None:
        return
    anchor = tuple(int(bit) for bit in anchor)
    if tuple(label) != anchor:
        return

    difference = abs(float(energy) - float(expected))
    print(
        f"[anchor check] optimized={float(expected):.12f} Ha; "
        f"Lanczos={float(energy):.12f} Ha; "
        f"difference={1000.0 * difference:.6f} mHa",
        flush=True,
    )
    if difference > float(tolerance):
        raise RuntimeError(
            "anchor-sector Lanczos energy does not match the optimized "
            "sector energy; the generator basis or sector labels are "
            "inconsistent"
        )


def load_or_build_sector(path, support, full_operator, full_dimension, args):
    """Reload one completed sector or solve and checkpoint it."""
    path = Path(path)
    if args.resume and path.exists():
        saved = np.load(path, allow_pickle=False)
        saved_support = np.asarray(saved["full_addresses"], dtype=np.int64)
        saved_tolerance = (
            float(saved["tolerance"].item())
            if "tolerance" in saved.files
            else float("inf")
        )
        if np.array_equal(saved_support, support["full_addresses"]):
            energies = np.asarray(saved["energies"])
            vectors = np.asarray(saved["vectors"])
            if (
                len(energies) >= args.roots_per_sector
                and saved_tolerance <= args.lanczos_tolerance
            ):
                print(
                    f"[sector {label_text(support['label'])}] reusing "
                    f"{len(energies)} checkpointed roots from {path}",
                    flush=True,
                )
                return {
                    **support,
                    "energies": energies[: args.roots_per_sector],
                    "vectors": vectors[:, : args.roots_per_sector],
                    "solver": str(saved["solver"].item()),
                    "elapsed_seconds": float(saved["elapsed_seconds"].item()),
                    "matvec_count": int(saved["matvec_count"].item()),
                    "matvec_seconds": float(saved["matvec_seconds"].item()),
                    "reused": True,
                }

    label = label_text(support["label"])
    print(
        f"[sector {label}] START roots={args.roots_per_sector}, "
        f"dimension={support['dimension']:,}, tol={args.lanczos_tolerance:g}",
        flush=True,
    )
    solved = solve_selected_sector(
        full_operator,
        full_dimension,
        support["full_addresses"],
        args.roots_per_sector,
        tolerance=args.lanczos_tolerance,
        maxiter=args.lanczos_maxiter,
        print_every=args.print_every_matvec,
    )
    result = {**support, **solved, "reused": False}
    np.savez_compressed(
        path,
        label=np.asarray(support["label"], dtype=np.uint8),
        residual_indices=support["residual_indices"],
        full_addresses=support["full_addresses"],
        energies=result["energies"],
        vectors=result["vectors"],
        solver=np.asarray(result["solver"]),
        elapsed_seconds=np.asarray(result["elapsed_seconds"]),
        matvec_count=np.asarray(result["matvec_count"]),
        matvec_seconds=np.asarray(result["matvec_seconds"]),
        tolerance=np.asarray(args.lanczos_tolerance),
    )
    print(
        f"[sector {label}] DONE in {result['elapsed_seconds']:.1f} s; "
        f"H actions={result['matvec_count']}; "
        f"energies={np.asarray(result['energies']).tolist()}",
        flush=True,
    )
    print(f"[sector {label}] checkpoint: {path}", flush=True)
    return result


def save_diagonal_lcu(path, frame, physical_label, clifford_label):
    """Construct and gzip one diagonal tapered Pauli LCU."""
    operator = tapered_operator(frame, clifford_label, clifford_label)
    data = {
        "physical_label": list(physical_label),
        "clifford_label": list(clifford_label),
        "n_qubits": int(frame["n_residual_qubits"]),
        "pauli_count": len(operator.terms),
        "lcu_one_norm": float(
            sum(abs(complex(value)) for value in operator.terms.values())
        ),
        "operator": qubit_operator_to_data(operator),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(data, handle)
    return data["pauli_count"], data["lcu_one_norm"]


def parse_args():
    """Command-line controls for selected-sector final evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimized_json")
    parser.add_argument("--reference_result", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--outname", required=True)
    parser.add_argument("--max_sectors", type=int, default=16)
    parser.add_argument("--roots_per_sector", type=int, default=5)
    parser.add_argument("--lanczos_tolerance", type=float, default=1e-9)
    parser.add_argument("--lanczos_maxiter", type=int, default=None)
    parser.add_argument("--print_every_matvec", type=int, default=25)
    parser.add_argument(
        "--anchor_tolerance",
        type=float,
        default=1.0e-2,
        help="maximum optimized-vs-Lanczos anchor energy difference in Ha",
    )
    parser.add_argument("--chemical_accuracy", type=float, default=CHEMICAL_PRECISION)
    parser.add_argument("--tau_pt", type=float, default=1e-12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_lcu_files",
        action="store_true",
        help="skip writing the selected diagonal tapered Pauli LCUs",
    )
    return parser.parse_args()


def main():
    """Run the restartable selected-sector Clifford/Lanczos workflow."""
    args = parse_args()
    total_start = time.perf_counter()
    work_dir = Path(args.work_dir).resolve()
    sector_dir = work_dir / "sectors"
    lcu_dir = work_dir / "diagonal_lcus"
    sector_dir.mkdir(parents=True, exist_ok=True)
    lcu_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work_dir / "progress.json"

    stage("1/7 Load optimized frame and parent DMRG reference")
    input_data = json.loads(Path(args.optimized_json).read_text(encoding="utf-8"))
    reference_energy = parse_dmrg_energy(args.reference_result)
    moldata = load_moldata(input_data["molpath"])
    rotation = rotation_from_oo_data(input_data, moldata.norb)
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    full_operator = ffsim.linear_operator(
        rotated_hamiltonian, norb=moldata.norb, nelec=moldata.nelec
    )
    full_dimension = int(full_operator.shape[0])
    parity_matrix = np.atleast_2d(np.loadtxt(input_data["parity"], dtype=int))
    labels = selected_labels(input_data, args.max_sectors)
    print("system orbitals:", moldata.norb, flush=True)
    print("fixed-spin electrons:", moldata.nelec, flush=True)
    print("fixed-spin CI dimension:", f"{full_dimension:,}", flush=True)
    print("parent DMRG reference energy:", f"{reference_energy:.12f} Ha", flush=True)
    print("selected MPS-screened labels:", [label_text(x) for x in labels], flush=True)

    stage("2/7 Build JW Pauli LCU and one Clifford frame")
    jw_start = time.perf_counter()
    jw_hamiltonian = molecular_hamiltonian_to_jw(
        rotated_hamiltonian, moldata.nelec
    )
    manifest_path = input_data.get("symmetry_manifest")
    if manifest_path:
        manifest = load_symmetry_manifest(manifest_path)
        symmetries = manifest["symmetries"]
        manifest_rows = spin_orbital_parity_matrix(
            manifest["parity_matrix"], moldata.norb
        )
        parity_rows = spin_orbital_parity_matrix(parity_matrix, moldata.norb)
        if not np.array_equal(manifest_rows, parity_rows):
            raise ValueError("symmetry manifest and parity file contain different rows")
    else:
        symmetries = z_symmetries_from_parity_matrix(parity_matrix, moldata.norb)
    frame = build_clifford_frame(
        jw_hamiltonian, symmetries, 2 * moldata.norb
    )
    clifford_seconds = time.perf_counter() - jw_start
    print("parent JW Pauli terms:", len(jw_hamiltonian.terms), flush=True)
    print("selected independent generators:", frame["n_symmetries"], flush=True)
    print(
        "qubits: parent -> residual =",
        frame["n_qubits"],
        "->",
        frame["n_residual_qubits"],
        flush=True,
    )
    print("Clifford/JW construction time:", f"{clifford_seconds:.2f} s", flush=True)

    stage("3/7 Generate only selected fixed-spin sector supports")
    support_start = time.perf_counter()
    supports = selected_sector_supports(
        parity_matrix,
        labels,
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )
    support_seconds = time.perf_counter() - support_start
    selected_dimension = sum(item["dimension"] for item in supports.values())
    print("selected determinant count:", f"{selected_dimension:,}", flush=True)
    print(
        "selected fraction of fixed-spin space:",
        f"{selected_dimension / full_dimension:.6%}",
        flush=True,
    )
    print("support construction time:", f"{support_seconds:.2f} s", flush=True)

    stage("4/7 Construct selected diagonal tapered Pauli LCUs")
    lcu_metadata = {}
    if args.skip_lcu_files:
        print("LCU serialization skipped by --skip_lcu_files", flush=True)
    else:
        for number, label in enumerate(labels, start=1):
            text_label = label_text(label)
            path = lcu_dir / f"sector_{text_label}.json.gz"
            print(
                f"[LCU] {number}/{len(labels)} sector {text_label} START",
                flush=True,
            )
            started = time.perf_counter()
            pauli_count, one_norm = save_diagonal_lcu(
                path,
                frame,
                label,
                supports[label]["clifford_label"],
            )
            lcu_metadata[text_label] = {
                "path": str(path),
                "pauli_count": int(pauli_count),
                "lcu_one_norm": float(one_norm),
            }
            print(
                f"[LCU] sector {text_label} DONE in "
                f"{time.perf_counter() - started:.2f} s; "
                f"Paulis={pauli_count:,}; Lambda={one_norm:.8f}",
                flush=True,
            )

    stage("5/7 Solve low roots of each selected sector with Lanczos")
    sector_results = {}
    for number, label in enumerate(labels, start=1):
        print(
            f"[sector progress] {number}/{len(labels)} labels; "
            f"current={label_text(label)}",
            flush=True,
        )
        checkpoint = sector_dir / f"sector_{label_text(label)}.npz"
        sector_results[label] = load_or_build_sector(
            checkpoint,
            supports[label],
            full_operator,
            full_dimension,
            args,
        )
        validate_anchor_energy(
            input_data,
            label,
            sector_results[label]["energies"][0],
            args.anchor_tolerance,
        )
        atomic_json(
            progress_path,
            {
                "stage": "sector_roots",
                "completed_labels": [
                    label_text(done) for done in sector_results
                ],
                "completed_count": len(sector_results),
                "total_count": len(labels),
                "updated": timestamp(),
            },
        )

    decoupled_energy = min(
        float(result["energies"][0]) for result in sector_results.values()
    )
    print("decoupled energy:", f"{decoupled_energy:.12f} Ha", flush=True)
    print(
        "decoupled error:",
        f"{1000.0 * (decoupled_energy - reference_energy):.6f} mHa",
        flush=True,
    )

    stage("6/7 Build and rank the coupled sector-root Hamiltonian")
    coupled_path = work_dir / "coupled_hamiltonian.npy"
    candidate_path = work_dir / "candidate_metadata.json"
    expected_candidates = [
        {
            "label": list(label),
            "root": int(root),
            "energy": float(result["energies"][root]),
        }
        for label, result in sorted(sector_results.items())
        for root in range(len(result["energies"]))
    ]
    reuse_coupled = False
    if args.resume and coupled_path.exists() and candidate_path.exists():
        saved_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        reuse_coupled = saved_candidates.get("candidates") == expected_candidates
    if reuse_coupled:
        coupled = np.load(coupled_path)
        candidates = saved_candidates["candidates"]
        coupled_seconds = float(saved_candidates.get("elapsed_seconds", 0.0))
        print("reusing coupled Hamiltonian:", coupled_path, flush=True)
    else:
        coupled, raw_candidates, coupled_seconds = coupled_candidate_matrix(
            full_operator, full_dimension, sector_results
        )
        candidates = [
            {
                "label": list(item["label"]),
                "root": int(item["root"]),
                "energy": float(item["energy"]),
            }
            for item in raw_candidates
        ]
        np.save(coupled_path, coupled)
        atomic_json(
            candidate_path,
            {"candidates": candidates, "elapsed_seconds": coupled_seconds},
        )
    hermiticity_error = float(np.max(np.abs(coupled - coupled.conj().T)))
    print("candidate states:", len(candidates), flush=True)
    print("coupled matrix shape:", coupled.shape, flush=True)
    print("coupled Hermiticity error:", f"{hermiticity_error:.3e}", flush=True)
    print("coupled construction time:", f"{coupled_seconds:.2f} s", flush=True)

    selection = one_shot_from_hamiltonian(
        coupled,
        e_exact=reference_energy,
        tol=args.chemical_accuracy,
        tau_pt=args.tau_pt,
        keys=[(tuple(item["label"]), item["root"]) for item in candidates],
    )
    curve = selection.as_curve()
    print("one-shot perturbative starting K:", selection.K_pt, flush=True)
    print("chemical-accuracy K:", selection.K, flush=True)
    print("coupled convergence reached:", selection.converged, flush=True)
    print("coupled energy:", selection.e_coupled, flush=True)
    if selection.e_coupled is not None:
        print(
            "coupled error:",
            f"{1000.0 * (selection.e_coupled - reference_energy):.6f} mHa",
            flush=True,
        )

    stage("7/7 Save final metrics and restart metadata")
    sector_metadata = {}
    for label, result in sector_results.items():
        text_label = label_text(label)
        sector_metadata[text_label] = {
            "dimension": int(result["dimension"]),
            "energies": [float(value) for value in result["energies"]],
            "solver": result["solver"],
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "matvec_count": int(result["matvec_count"]),
            "matvec_seconds": float(result["matvec_seconds"]),
            "reference_weight": float(
                input_data.get("sector_weights", {}).get(text_label, 0.0)
            ),
            "reused": bool(result["reused"]),
        }
        if text_label in lcu_metadata:
            sector_metadata[text_label]["diagonal_lcu"] = lcu_metadata[text_label]

    output = {
        "schema": "quasisymmetry.selected_clifford_lanczos",
        "version": 1,
        "method": "selected Clifford sectors with compiled physical Lanczos action",
        "full_fixed_spin_matrix_constructed": False,
        "all_sector_labels_enumerated": False,
        "optimized_json": str(Path(args.optimized_json).resolve()),
        "reference_method": "DMRG",
        "E_reference": float(reference_energy),
        "E_decoupled": float(decoupled_energy),
        "E_coupled": (
            None if selection.e_coupled is None else float(selection.e_coupled)
        ),
        "dE": float(decoupled_energy - reference_energy),
        "K": selection.K,
        "K_pt": selection.K_pt,
        "converged": bool(selection.converged),
        "coupled_curve": curve,
        "sector_label_count": len(labels),
        "candidate_state_count": len(candidates),
        "sector_labels": [list(label) for label in labels],
        "sector_eigenstates": candidates,
        "sector_results": sector_metadata,
        "n_parent_qubits": int(frame["n_qubits"]),
        "n_tapered_qubits": int(frame["n_residual_qubits"]),
        "qubit_reduction": int(frame["n_symmetries"]),
        "full_fixed_spin_dimension": full_dimension,
        "selected_determinant_count": int(selected_dimension),
        "parent_jw_pauli_count": len(jw_hamiltonian.terms),
        "parent_jw_lcu_one_norm": float(
            sum(abs(complex(value)) for value in jw_hamiltonian.terms.values())
        ),
        "clifford_pauli_count": len(frame["hamiltonian"].terms),
        "clifford_lcu_one_norm": float(
            sum(abs(complex(value)) for value in frame["hamiltonian"].terms.values())
        ),
        "coupled_hermiticity_error": hermiticity_error,
        "timings": {
            "clifford_and_jw_seconds": clifford_seconds,
            "selected_support_seconds": support_seconds,
            "coupled_matrix_seconds": coupled_seconds,
            "total_seconds": float(time.perf_counter() - total_start),
        },
        "args": vars(args),
    }
    atomic_json(args.outname, output)
    atomic_json(
        progress_path,
        {"stage": "complete", "output": str(args.outname), "updated": timestamp()},
    )
    print("final metrics:", Path(args.outname).resolve(), flush=True)
    print("work directory:", work_dir, flush=True)
    print("total elapsed:", f"{output['timings']['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
