#!/usr/bin/env python3
"""Improve a selected-sector Krylov result using its coupled-state residual.

This is a final-energy refinement.  It reuses a completed selected-sector
Krylov calculation, resolves the coupled Ritz residual by LAS sector, and adds
short sector-local Krylov chains seeded by the largest missing residual pieces.
It never builds the complete fixed-spin Hamiltonian matrix.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import ffsim
import numpy as np

from chemistry import CHEMICAL_PRECISION, load_moldata
from selected_clifford_lanczos import parse_dmrg_energy, stage
from src.clifford_sectors import (
    build_clifford_frame,
    load_symmetry_manifest,
    molecular_hamiltonian_to_jw,
    z_symmetries_from_parity_matrix,
)
from src.orbital_rotation import rotation_from_oo_data
from src.selected_sector_lanczos import (
    coupled_ground_residual,
    coupled_krylov_matrix,
    extend_coupled_matrix,
    krylov_candidates,
    label_text,
    residual_seeded_krylov_extension,
    sector_vector_weights,
    selected_sector_supports,
    spin_orbital_parity_matrix,
)


def atomic_json(path, data):
    """Write JSON atomically so a stopped job cannot leave partial metadata."""
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


def atomic_npy(path, array):
    """Write one NumPy array atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, np.asarray(array))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_basis(path, support, basis):
    """Write one sector support and its orthonormal basis atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            full_addresses=np.asarray(support, dtype=np.int64),
            basis=np.asarray(basis, dtype=np.complex128),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def find_anchor_checkpoint(source_data, source_work_dir, anchor):
    """Locate the saved optimized anchor-sector eigenvector."""
    text_label = label_text(anchor)
    candidates = []
    anchor_dir = source_data.get("args", {}).get("anchor_checkpoint_dir")
    if anchor_dir:
        candidates.append(Path(anchor_dir) / f"sector_{text_label}.npz")
    candidates.append(source_work_dir / "anchor" / f"sector_{text_label}.npz")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "anchor checkpoint not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


def load_source_basis(source_data):
    """Load the anchor and all sector Krylov bases from prior metrics."""
    source_work_dir = Path(source_data["args"]["work_dir"])
    anchor = tuple(int(bit) for bit in source_data["anchor_sector"])
    basis_by_sector = {}
    support_by_sector = {}

    anchor_path = find_anchor_checkpoint(source_data, source_work_dir, anchor)
    saved_anchor = np.load(anchor_path, allow_pickle=False)
    anchor_vector = np.asarray(saved_anchor["vectors"][:, 0], dtype=np.complex128)
    anchor_support = np.asarray(saved_anchor["full_addresses"], dtype=np.int64)
    basis_by_sector[anchor] = anchor_vector[:, None]
    support_by_sector[anchor] = anchor_support

    for raw_label in source_data["sector_labels"]:
        label = tuple(int(bit) for bit in raw_label)
        if label == anchor:
            continue
        path = source_work_dir / "sectors" / f"sector_{label_text(label)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        saved = np.load(path, allow_pickle=False)
        depth = int(source_data["sector_results"][label_text(label)]["krylov_depth"])
        basis_by_sector[label] = np.asarray(
            saved["basis"][:, :depth], dtype=np.complex128
        )
        support_by_sector[label] = np.asarray(
            saved["full_addresses"], dtype=np.int64
        )
    return anchor, basis_by_sector, support_by_sector


def source_sector_bases(anchor, basis_by_sector, support_by_sector):
    """Convert saved arrays to the format used by coupled_krylov_matrix."""
    results = {}
    for label in sorted(basis_by_sector):
        if label == anchor:
            continue
        results[label] = {
            "basis": basis_by_sector[label],
            "full_addresses": support_by_sector[label],
        }
    return results


def candidate_metadata(candidates):
    """Return serializable metadata needed to reconstruct candidate vectors."""
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "label": list(candidate["label"]),
                "kind": candidate["kind"],
                "depth": int(candidate.get("depth", 0)),
                "sector_column": int(candidate["sector_column"]),
                "cycle": int(candidate.get("cycle", 0)),
            }
        )
    return rows


def candidates_from_metadata(rows, basis_by_sector, support_by_sector):
    """Reconstruct coupled candidates from saved per-sector bases."""
    candidates = []
    for row in rows:
        label = tuple(int(bit) for bit in row["label"])
        column = int(row["sector_column"])
        candidates.append(
            {
                "label": label,
                "kind": row["kind"],
                "depth": int(row.get("depth", 0)),
                "sector_column": column,
                "cycle": int(row.get("cycle", 0)),
                "support": support_by_sector[label],
                "vector": basis_by_sector[label][:, column],
            }
        )
    return candidates


def save_restart(work_dir, cycle, matrix, candidates, basis_by_sector, support_by_sector, history):
    """Checkpoint a completed enrichment macrocycle."""
    basis_dir = Path(work_dir) / "sector_bases"
    for label in sorted(basis_by_sector):
        atomic_basis(
            basis_dir / f"sector_{label_text(label)}.npz",
            support_by_sector[label],
            basis_by_sector[label],
        )
    matrix_path = Path(work_dir) / f"coupled_cycle_{cycle}.npy"
    atomic_npy(matrix_path, matrix)
    state = {
        "version": 1,
        "completed_cycle": int(cycle),
        "matrix_path": str(matrix_path),
        "candidates": candidate_metadata(candidates),
        "history": history,
    }
    atomic_json(Path(work_dir) / "restart.json", state)


def load_restart(work_dir):
    """Load the most recent complete enrichment checkpoint, if present."""
    restart_path = Path(work_dir) / "restart.json"
    if not restart_path.exists():
        return None
    state = json.loads(restart_path.read_text(encoding="utf-8"))
    basis_by_sector = {}
    support_by_sector = {}
    labels = {tuple(int(bit) for bit in row["label"]) for row in state["candidates"]}
    for label in labels:
        path = Path(work_dir) / "sector_bases" / f"sector_{label_text(label)}.npz"
        saved = np.load(path, allow_pickle=False)
        support_by_sector[label] = np.asarray(
            saved["full_addresses"], dtype=np.int64
        )
        basis_by_sector[label] = np.asarray(saved["basis"], dtype=np.complex128)
    matrix = np.load(state["matrix_path"])
    candidates = candidates_from_metadata(
        state["candidates"], basis_by_sector, support_by_sector
    )
    return state, matrix, candidates, basis_by_sector, support_by_sector


def build_initial_coupled(
    source_work_dir,
    full_operator,
    full_dimension,
    anchor,
    basis_by_sector,
    support_by_sector,
):
    """Reuse the prior coupled matrix or rebuild it from saved sector bases."""
    sector_bases = source_sector_bases(
        anchor, basis_by_sector, support_by_sector
    )
    candidates = krylov_candidates(
        {"label": anchor, "full_addresses": support_by_sector[anchor]},
        basis_by_sector[anchor][:, 0],
        sector_bases,
    )
    coupled_path = source_work_dir / "coupled_krylov_hamiltonian.npy"
    metadata_path = source_work_dir / "candidate_metadata.json"
    expected = [
        {
            "label": list(item["label"]),
            "depth": int(item["depth"]),
            "kind": item["kind"],
        }
        for item in candidates
    ]
    if coupled_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("candidates") == expected:
            print("[initial] reusing coupled matrix:", coupled_path, flush=True)
            return np.load(coupled_path), candidates, 0.0

    print("[initial] rebuilding coupled matrix from source sector bases", flush=True)
    matrix, candidates, seconds = coupled_krylov_matrix(
        full_operator,
        full_dimension,
        {"label": anchor, "full_addresses": support_by_sector[anchor]},
        basis_by_sector[anchor][:, 0],
        sector_bases,
    )
    return matrix, candidates, seconds


def choose_residual_labels(ranking, current_labels, existing_count, new_count):
    """Select the largest residual sectors inside and outside the current set."""
    current = set(current_labels)
    existing = [label for label, _weight in ranking if label in current]
    outside = [label for label, _weight in ranking if label not in current]
    return existing[: int(existing_count)], outside[: int(new_count)]


def residual_summary(ranking, current_labels):
    """Return residual norm inside and outside the currently represented sectors."""
    current = set(current_labels)
    inside = sum(weight for label, weight in ranking if label in current)
    outside = sum(weight for label, weight in ranking if label not in current)
    return float(inside), float(outside)


def parse_args():
    """Parse residual-enrichment controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--source_metrics", default=None, type=Path)
    parser.add_argument("--work_dir", default=None, type=Path)
    parser.add_argument("--outname", default=None, type=Path)
    parser.add_argument("--max_macrocycles", type=int, default=10)
    parser.add_argument("--existing_sectors_per_cycle", type=int, default=16)
    parser.add_argument("--new_sectors_per_cycle", type=int, default=16)
    parser.add_argument("--chain_depth", type=int, default=4)
    parser.add_argument("--max_dimension", type=int, default=2000)
    parser.add_argument("--energy_tolerance_mha", type=float, default=0.0)
    parser.add_argument("--residual_tolerance", type=float, default=1.0e-12)
    parser.add_argument("--chemical_accuracy", type=float, default=CHEMICAL_PRECISION)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    """Run restartable coupled-state residual enrichment."""
    args = parse_args()
    total_start = time.perf_counter()
    run_dir = args.run_dir.resolve()
    source_path = (
        args.source_metrics.resolve()
        if args.source_metrics
        else run_dir / "final_krylov_metrics_s32.json"
    )
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else run_dir / "final_residual_enrichment_s32"
    )
    outname = (
        args.outname.resolve()
        if args.outname
        else run_dir / "final_residual_metrics_s32.json"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    stage("1/6 Load the optimized Hamiltonian and source Krylov result")
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    optimized_json = Path(source_data["optimized_json"])
    input_data = json.loads(optimized_json.read_text(encoding="utf-8"))
    reference_result = source_data["args"]["reference_result"]
    reference_energy = parse_dmrg_energy(reference_result)
    moldata = load_moldata(input_data["molpath"])
    rotation = rotation_from_oo_data(input_data, moldata.norb)
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    full_operator = ffsim.linear_operator(
        rotated_hamiltonian, norb=moldata.norb, nelec=moldata.nelec
    )
    full_dimension = int(full_operator.shape[0])
    parity_matrix = np.atleast_2d(np.loadtxt(input_data["parity"], dtype=int))
    print("source metrics:", source_path, flush=True)
    print("reference energy:", f"{reference_energy:.12f} Ha", flush=True)
    print("fixed-spin dimension:", f"{full_dimension:,}", flush=True)

    stage("2/6 Build one Clifford frame for any newly requested sectors")
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
        symmetries = z_symmetries_from_parity_matrix(
            parity_matrix, moldata.norb
        )
    frame = build_clifford_frame(jw_hamiltonian, symmetries, 2 * moldata.norb)
    print("selected generators:", frame["n_symmetries"], flush=True)
    print(
        "qubits: parent -> residual =",
        frame["n_qubits"],
        "->",
        frame["n_residual_qubits"],
        flush=True,
    )

    stage("3/6 Load or reconstruct the current coupled basis")
    restart = load_restart(work_dir) if args.resume else None
    if restart is not None:
        state, matrix, candidates, basis_by_sector, support_by_sector = restart
        start_cycle = int(state["completed_cycle"]) + 1
        history = list(state["history"])
        anchor = tuple(int(bit) for bit in source_data["anchor_sector"])
        print(
            f"[resume] loaded cycle {state['completed_cycle']} with "
            f"K={len(candidates)}",
            flush=True,
        )
    else:
        anchor, basis_by_sector, support_by_sector = load_source_basis(source_data)
        source_work_dir = Path(source_data["args"]["work_dir"])
        matrix, candidates, build_seconds = build_initial_coupled(
            source_work_dir,
            full_operator,
            full_dimension,
            anchor,
            basis_by_sector,
            support_by_sector,
        )
        start_cycle = 1
        history = []
        save_restart(
            work_dir,
            0,
            matrix,
            candidates,
            basis_by_sector,
            support_by_sector,
            history,
        )
        print(
            f"[initial] K={len(candidates)}, sectors={len(basis_by_sector)}, "
            f"build time={build_seconds:.2f} s",
            flush=True,
        )

    stage("4/6 Enrich from the coupled-state residual")
    stop_reason = "maximum macrocycles reached"
    for cycle in range(start_cycle, int(args.max_macrocycles) + 1):
        cycle_start = time.perf_counter()
        energy_before, _coefficients, _state, residual = coupled_ground_residual(
            full_operator,
            full_dimension,
            matrix,
            candidates,
        )
        error_before = energy_before - reference_energy
        ranking, residual_norm_sq = sector_vector_weights(
            residual,
            parity_matrix,
            moldata.norb,
            moldata.nelec,
        )
        inside_sq, outside_sq = residual_summary(ranking, basis_by_sector)
        existing_labels, new_labels = choose_residual_labels(
            ranking,
            basis_by_sector,
            args.existing_sectors_per_cycle,
            args.new_sectors_per_cycle,
        )
        selected_labels = existing_labels + new_labels

        print("\n" + "-" * 78, flush=True)
        print(
            f"[cycle {cycle}] START K={len(candidates)} "
            f"energy={energy_before:.12f} Ha "
            f"error={1000.0 * error_before:.6f} mHa",
            flush=True,
        )
        print(
            f"[cycle {cycle}] residual norm={np.sqrt(residual_norm_sq):.8e}; "
            f"inside={np.sqrt(inside_sq):.8e}; "
            f"outside={np.sqrt(outside_sq):.8e}",
            flush=True,
        )
        print(
            f"[cycle {cycle}] existing sectors:",
            [label_text(label) for label in existing_labels],
            flush=True,
        )
        print(
            f"[cycle {cycle}] new sectors:",
            [label_text(label) for label in new_labels],
            flush=True,
        )

        if abs(error_before) <= float(args.chemical_accuracy):
            stop_reason = "chemical accuracy reached before enrichment"
            break
        if len(candidates) >= int(args.max_dimension):
            stop_reason = "maximum coupled dimension reached"
            break
        if not selected_labels:
            stop_reason = "no residual sectors above threshold"
            break

        missing_labels = [
            label for label in new_labels if label not in support_by_sector
        ]
        if missing_labels:
            new_supports = selected_sector_supports(
                parity_matrix,
                missing_labels,
                moldata.norb,
                moldata.nelec,
                frame["clifford"],
                frame["n_symmetries"],
            )
            for label in missing_labels:
                support_by_sector[label] = np.asarray(
                    new_supports[label]["full_addresses"], dtype=np.int64
                )
                basis_by_sector[label] = np.zeros(
                    (len(support_by_sector[label]), 0), dtype=np.complex128
                )

        new_candidates = []
        additions = {}
        for label in selected_labels:
            remaining = int(args.max_dimension) - len(candidates) - len(new_candidates)
            if remaining <= 0:
                break
            count = min(int(args.chain_depth), remaining)
            support = support_by_sector[label]
            extension = residual_seeded_krylov_extension(
                full_operator,
                full_dimension,
                support,
                residual[support],
                basis_by_sector[label],
                count,
                tolerance=args.residual_tolerance,
            )
            if extension.shape[1] == 0:
                continue
            first_column = basis_by_sector[label].shape[1]
            basis_by_sector[label] = np.column_stack(
                [basis_by_sector[label], extension]
            )
            additions[label_text(label)] = int(extension.shape[1])
            for offset in range(extension.shape[1]):
                column = first_column + offset
                new_candidates.append(
                    {
                        "label": label,
                        "kind": "residual_krylov",
                        "depth": int(column + 1),
                        "sector_column": int(column),
                        "cycle": int(cycle),
                        "support": support,
                        "vector": basis_by_sector[label][:, column],
                    }
                )

        for label in missing_labels:
            if basis_by_sector[label].shape[1] == 0:
                del basis_by_sector[label]
                del support_by_sector[label]

        if not new_candidates:
            stop_reason = "residual produced no independent basis vectors"
            break

        matrix, candidates, extension_seconds = extend_coupled_matrix(
            full_operator,
            full_dimension,
            matrix,
            candidates,
            new_candidates,
        )
        energy_after, _coefficients, _state, residual_after = coupled_ground_residual(
            full_operator,
            full_dimension,
            matrix,
            candidates,
        )
        error_after = energy_after - reference_energy
        improvement_mha = 1000.0 * (energy_before - energy_after)
        residual_after_norm = float(np.linalg.norm(residual_after))
        row = {
            "cycle": int(cycle),
            "dimension_before": int(len(candidates) - len(new_candidates)),
            "dimension_after": int(len(candidates)),
            "sector_count": int(len(basis_by_sector)),
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "error_mHa": float(1000.0 * error_after),
            "improvement_mHa": float(improvement_mha),
            "residual_norm_before": float(np.sqrt(residual_norm_sq)),
            "residual_norm_after": residual_after_norm,
            "inside_residual_norm_before": float(np.sqrt(inside_sq)),
            "outside_residual_norm_before": float(np.sqrt(outside_sq)),
            "existing_labels": [list(label) for label in existing_labels],
            "new_labels": [list(label) for label in new_labels],
            "vectors_added": int(len(new_candidates)),
            "additions_by_sector": additions,
            "extension_seconds": float(extension_seconds),
            "elapsed_seconds": float(time.perf_counter() - cycle_start),
        }
        history.append(row)
        save_restart(
            work_dir,
            cycle,
            matrix,
            candidates,
            basis_by_sector,
            support_by_sector,
            history,
        )
        print(
            f"[cycle {cycle}] DONE K={len(candidates)} "
            f"sectors={len(basis_by_sector)} "
            f"energy={energy_after:.12f} Ha "
            f"error={1000.0 * error_after:.6f} mHa "
            f"gain={improvement_mha:.6f} mHa",
            flush=True,
        )
        print(
            f"[cycle {cycle}] residual norm after={residual_after_norm:.8e}; "
            f"vectors added={len(new_candidates)}; checkpoint={work_dir}",
            flush=True,
        )

        if abs(error_after) <= float(args.chemical_accuracy):
            stop_reason = "chemical accuracy reached"
            break
        if (
            float(args.energy_tolerance_mha) > 0.0
            and improvement_mha < float(args.energy_tolerance_mha)
        ):
            stop_reason = "energy improvement below tolerance"
            break
        if len(candidates) >= int(args.max_dimension):
            stop_reason = "maximum coupled dimension reached"
            break

    stage("5/6 Evaluate the final residual and convergence")
    final_energy, _coefficients, _state, final_residual = coupled_ground_residual(
        full_operator,
        full_dimension,
        matrix,
        candidates,
    )
    final_error = final_energy - reference_energy
    final_ranking, final_residual_norm_sq = sector_vector_weights(
        final_residual,
        parity_matrix,
        moldata.norb,
        moldata.nelec,
    )
    final_inside_sq, final_outside_sq = residual_summary(
        final_ranking, basis_by_sector
    )
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    converged = bool(abs(final_error) <= float(args.chemical_accuracy))
    print("final energy:", f"{final_energy:.12f} Ha", flush=True)
    print("final error:", f"{1000.0 * final_error:.6f} mHa", flush=True)
    print("final K:", len(candidates), flush=True)
    print("final sector count:", len(basis_by_sector), flush=True)
    print("final residual norm:", f"{np.sqrt(final_residual_norm_sq):.8e}", flush=True)
    print("chemical accuracy reached:", converged, flush=True)
    print("stop reason:", stop_reason, flush=True)

    stage("6/6 Save final enrichment metrics")
    output = {
        "schema": "quasisymmetry.selected_clifford_residual_enrichment",
        "version": 1,
        "method": "iterative coupled-state residual enrichment",
        "source_metrics": str(source_path),
        "optimized_json": str(optimized_json),
        "reference_method": "DMRG",
        "E_reference": float(reference_energy),
        "E_source": float(source_data["E_coupled"]),
        "E_coupled": float(final_energy),
        "error_Ha": float(final_error),
        "error_mHa": float(1000.0 * final_error),
        "K_source": int(source_data["candidate_state_count"]),
        "K": int(len(candidates)),
        "sector_count_source": int(source_data["sector_label_count"]),
        "sector_label_count": int(len(basis_by_sector)),
        "sector_labels": [list(label) for label in sorted(basis_by_sector)],
        "converged": converged,
        "chemical_accuracy_Ha": float(args.chemical_accuracy),
        "stop_reason": stop_reason,
        "residual_norm": float(np.sqrt(final_residual_norm_sq)),
        "inside_residual_norm": float(np.sqrt(final_inside_sq)),
        "outside_residual_norm": float(np.sqrt(final_outside_sq)),
        "coupled_hermiticity_error": hermiticity_error,
        "full_fixed_spin_matrix_constructed": False,
        "history": history,
        "top_final_residual_sectors": [
            {"label": list(label), "weight": float(weight)}
            for label, weight in final_ranking[:16]
        ],
        "timings": {
            "total_seconds": float(time.perf_counter() - total_start),
        },
        "args": {
            **vars(args),
            "run_dir": str(run_dir),
            "source_metrics": str(source_path),
            "work_dir": str(work_dir),
            "outname": str(outname),
        },
    }
    atomic_json(outname, output)
    print("final metrics:", outname, flush=True)
    print("restart state:", work_dir / "restart.json", flush=True)


if __name__ == "__main__":
    main()
