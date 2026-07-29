"""MPS-native seniority/quartet scoring that respects saved orbital ordering."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    atomic_json,
    cap_worker_store_resources,
    copy_solver_store,
    directory_fingerprint,
    load_json,
    map_rows_to_solver_order,
)

CHECKPOINT_SCHEMA = "quasisymmetry.equilibrium_mps_nc_checkpoint"
CHECKPOINT_VERSION = 1


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def payload_fingerprint(payload) -> str:
    """Return a stable SHA-256 digest for JSON-compatible scientific inputs."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_fingerprint(values, dtype) -> str:
    """Fingerprint an array without serializing large numeric payloads to JSON."""
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def scoring_context(
    proxy_mps,
    proxy_tag,
    rows_solver,
    rotation_solver,
    multiply_bond_dim,
    multiply_sweeps,
    multiply_tolerance,
) -> dict:
    """Describe every input that can change a candidate NC score."""
    proxy = directory_fingerprint(proxy_mps, include_contents=False)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": CHECKPOINT_VERSION,
        "kind": "candidate_score_context",
        "proxy_mps_fingerprint": proxy["fingerprint"],
        "proxy_mps_file_count": int(proxy["file_count"]),
        "proxy_mps_total_bytes": int(proxy["total_bytes"]),
        "proxy_tag": str(proxy_tag),
        "rows_fingerprint": array_fingerprint(rows_solver, np.int8),
        "rotation_fingerprint": array_fingerprint(rotation_solver, np.float64),
        "candidate_count": int(len(np.atleast_2d(rows_solver))),
        "multiply_bond_dim": int(multiply_bond_dim),
        "multiply_sweeps": int(multiply_sweeps),
        "multiply_tolerance": float(multiply_tolerance),
    }
    return {**payload, "fingerprint": payload_fingerprint(payload)}


def candidate_checkpoint_path(checkpoint_dir, index) -> Path:
    """Return the unique checkpoint path for one candidate."""
    return Path(checkpoint_dir) / f"candidate_{int(index):04d}.json"


def write_candidate_checkpoint(
    checkpoint_dir,
    index,
    row,
    score,
    context_fingerprint,
    elapsed_seconds,
) -> Path:
    """Atomically persist one completed candidate score."""
    path = candidate_checkpoint_path(checkpoint_dir, index)
    atomic_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "version": CHECKPOINT_VERSION,
            "kind": "candidate_score",
            "status": "complete",
            "candidate_index": int(index),
            "row": np.asarray(row, dtype=int).tolist(),
            "score": float(score),
            "context_fingerprint": str(context_fingerprint),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )
    return path


def read_candidate_checkpoint(
    checkpoint_dir,
    index,
    row,
    context_fingerprint,
):
    """Return a valid saved score or ``None`` when any input differs."""
    path = candidate_checkpoint_path(checkpoint_dir, index)
    if not path.exists():
        return None
    try:
        record = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_row = np.asarray(row, dtype=int).tolist()
    if (
        record.get("schema") != CHECKPOINT_SCHEMA
        or int(record.get("version", -1)) != CHECKPOINT_VERSION
        or record.get("kind") != "candidate_score"
        or record.get("status") != "complete"
        or int(record.get("candidate_index", -1)) != int(index)
        or record.get("row") != expected_row
        or record.get("context_fingerprint") != str(context_fingerprint)
    ):
        return None
    try:
        score = float(record["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(score) or score < 0.0:
        return None
    return score


def candidate_checkpoint_state(checkpoint_dir, rows, context_fingerprint):
    """Collect valid candidate checkpoints and identify missing indices."""
    rows = np.atleast_2d(np.asarray(rows, dtype=int))
    scores = np.full(len(rows), np.nan, dtype=float)
    completed = []
    missing = []
    for index, row in enumerate(rows):
        score = read_candidate_checkpoint(
            checkpoint_dir,
            index,
            row,
            context_fingerprint,
        )
        if score is None:
            missing.append(index)
        else:
            scores[index] = score
            completed.append(index)
    return {
        "scores": scores,
        "completed_indices": completed,
        "missing_indices": missing,
    }


def write_candidate_progress(
    checkpoint_dir,
    context,
    completed_indices,
    missing_indices,
    status,
) -> None:
    """Write a restart manifest reconstructed from atomic candidate records."""
    atomic_json(
        Path(checkpoint_dir) / "candidate_progress.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "version": CHECKPOINT_VERSION,
            "kind": "candidate_progress",
            "status": str(status),
            "context": context,
            "completed_count": len(completed_indices),
            "completed_indices": [int(value) for value in completed_indices],
            "missing_count": len(missing_indices),
            "missing_indices": [int(value) for value in missing_indices],
        },
    )


def sign_checkpoint_path(checkpoint_dir, index) -> Path:
    """Return the checkpoint path for one selected-generator expectation."""
    return Path(checkpoint_dir) / f"generator_{int(index):03d}.json"


def read_sign_checkpoint(
    checkpoint_dir,
    index,
    row,
    context_fingerprint,
):
    """Return a matching saved generator expectation or ``None``."""
    path = sign_checkpoint_path(checkpoint_dir, index)
    if not path.exists():
        return None
    try:
        record = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        record.get("schema") != CHECKPOINT_SCHEMA
        or int(record.get("version", -1)) != CHECKPOINT_VERSION
        or record.get("kind") != "generator_expectation"
        or record.get("status") != "complete"
        or int(record.get("generator_index", -1)) != int(index)
        or record.get("row") != np.asarray(row, dtype=int).tolist()
        or record.get("context_fingerprint") != str(context_fingerprint)
    ):
        return None
    try:
        expectation = float(record["expectation"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(expectation):
        return None
    return expectation


def write_sign_checkpoint(
    checkpoint_dir,
    index,
    row,
    expectation,
    context_fingerprint,
    elapsed_seconds,
) -> None:
    """Atomically persist one selected-generator expectation."""
    atomic_json(
        sign_checkpoint_path(checkpoint_dir, index),
        {
            "schema": CHECKPOINT_SCHEMA,
            "version": CHECKPOINT_VERSION,
            "kind": "generator_expectation",
            "status": "complete",
            "generator_index": int(index),
            "row": np.asarray(row, dtype=int).tolist(),
            "expectation": float(expectation),
            "context_fingerprint": str(context_fingerprint),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def score_rows_in_store(task):
    """Score one independent candidate chunk in an isolated Block2 store."""
    add_project_path(task["project_dir"])
    from src.dmrg_costs import DMRGOrbitalCosts, MultiplyConfig
    from src.dmrg_solver import Block2DMRGSolver

    threads = int(task["threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    solver = Block2DMRGSolver.load(task["store_dir"], n_threads=threads)
    rows = np.atleast_2d(np.asarray(task["rows"], dtype=int))
    rotation = np.asarray(task["rotation"], dtype=float)
    costs = DMRGOrbitalCosts(
        solver,
        rows,
        mps_tag=task["proxy_tag"],
        multiply=MultiplyConfig(
            bond_dim=int(task["multiply_bond_dim"]),
            n_sweeps=int(task["multiply_sweeps"]),
            tol=float(task["multiply_tolerance"]),
        ),
    )
    costs._ensure_eta()
    scores = []
    completed = []
    total = int(task["total_candidates"])
    for local_index, row in enumerate(rows):
        index = int(task["indices"][local_index])
        candidate = index + 1
        print(
            f"[NC worker {task['worker']}] candidate {candidate}/{total}",
            flush=True,
        )
        started = time.perf_counter()
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SCORE_PHI")
        xi = costs._apply_symmetry(row, rotation, costs._eta, "SCORE_XI")
        chi = costs._apply(costs._h_mpo, phi, "SCORE_CHI")
        value = (
            costs.solver.mps_norm2(chi)
            + costs.solver.mps_norm2(xi)
            - 2.0
            * float(np.real(costs.solver.mps_overlap(chi, xi)))
        )
        score = max(0.0, float(value))
        elapsed = time.perf_counter() - started
        write_candidate_checkpoint(
            task["checkpoint_dir"],
            index,
            row,
            score,
            task["context_fingerprint"],
            elapsed,
        )
        scores.append(score)
        completed.append(index)
        atomic_json(
            Path(task["checkpoint_dir"])
            / f"worker_{int(task['worker']):03d}_progress.json",
            {
                "schema": CHECKPOINT_SCHEMA,
                "version": CHECKPOINT_VERSION,
                "kind": "worker_progress",
                "status": "running",
                "worker": int(task["worker"]),
                "last_completed_index": index,
                "completed_indices": completed,
                "context_fingerprint": task["context_fingerprint"],
            },
        )
        print(
            f"[NC checkpoint] candidate {candidate}/{total} saved "
            f"after {elapsed:.1f} s",
            flush=True,
        )
    atomic_json(
        Path(task["checkpoint_dir"])
        / f"worker_{int(task['worker']):03d}_progress.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "version": CHECKPOINT_VERSION,
            "kind": "worker_progress",
            "status": "complete",
            "worker": int(task["worker"]),
            "completed_indices": completed,
            "context_fingerprint": task["context_fingerprint"],
        },
    )
    return {
        "indices": [int(value) for value in task["indices"]],
        "scores": scores,
    }


def parallel_scores(
    project_dir,
    proxy_mps,
    proxy_tag,
    rows_solver,
    rotation_solver,
    worker_root,
    workers,
    total_threads,
    multiply_bond_dim,
    multiply_sweeps,
    multiply_tolerance,
    checkpoint_dir,
) -> dict:
    """Score candidate rows with isolated Block2 stores per process."""
    rows_solver = np.atleast_2d(np.asarray(rows_solver, dtype=int))
    context = scoring_context(
        proxy_mps=proxy_mps,
        proxy_tag=proxy_tag,
        rows_solver=rows_solver,
        rotation_solver=rotation_solver,
        multiply_bond_dim=multiply_bond_dim,
        multiply_sweeps=multiply_sweeps,
        multiply_tolerance=multiply_tolerance,
    )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = candidate_checkpoint_state(
        checkpoint_dir,
        rows_solver,
        context["fingerprint"],
    )
    missing_indices = saved["missing_indices"]
    print(
        f"[NC resume] {len(saved['completed_indices'])}/{len(rows_solver)} "
        f"candidate scores complete; {len(missing_indices)} missing",
        flush=True,
    )
    write_candidate_progress(
        checkpoint_dir,
        context,
        saved["completed_indices"],
        missing_indices,
        "running" if missing_indices else "complete",
    )
    if not missing_indices:
        return {
            "scores": saved["scores"],
            "context": context,
            "reused_count": len(rows_solver),
            "computed_count": 0,
            "checkpoint_dir": str(checkpoint_dir),
        }

    workers = max(
        1,
        min(int(workers), len(missing_indices), int(total_threads)),
    )
    chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.asarray(missing_indices, dtype=int), workers)
        if len(chunk)
    ]
    threads = max(1, int(total_threads) // len(chunks))
    worker_root = Path(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for number, indices in enumerate(chunks, start=1):
        store = worker_root / f"worker_{number}" / "mps"
        if store.exists():
            shutil.rmtree(store)
        copied = copy_solver_store(proxy_mps, store, [proxy_tag])
        cap_worker_store_resources(store, threads)
        print(
            f"[NC setup] worker {number}/{len(chunks)}: "
            f"{threads} threads, {len(indices)} candidates, "
            f"{copied[proxy_tag]} MPS files",
            flush=True,
        )
        tasks.append(
            {
                "project_dir": str(project_dir),
                "store_dir": str(store),
                "proxy_tag": proxy_tag,
                "rows": rows_solver[indices].tolist(),
                "rotation": np.asarray(rotation_solver, dtype=float).tolist(),
                "indices": indices,
                "worker": number,
                "threads": threads,
                "total_candidates": len(rows_solver),
                "multiply_bond_dim": int(multiply_bond_dim),
                "multiply_sweeps": int(multiply_sweeps),
                "multiply_tolerance": float(multiply_tolerance),
                "checkpoint_dir": str(checkpoint_dir),
                "context_fingerprint": context["fingerprint"],
            }
        )
    if len(tasks) == 1:
        results = [score_rows_in_store(tasks[0])]
    else:
        process_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(tasks), mp_context=process_context
        ) as executor:
            results = list(executor.map(score_rows_in_store, tasks))
    del results
    finished = candidate_checkpoint_state(
        checkpoint_dir,
        rows_solver,
        context["fingerprint"],
    )
    write_candidate_progress(
        checkpoint_dir,
        context,
        finished["completed_indices"],
        finished["missing_indices"],
        "complete" if not finished["missing_indices"] else "incomplete",
    )
    if finished["missing_indices"]:
        raise RuntimeError(
            "NC scoring workers returned without checkpoints for candidates "
            f"{[index + 1 for index in finished['missing_indices']]}"
        )
    return {
        "scores": finished["scores"],
        "context": context,
        "reused_count": len(saved["completed_indices"]),
        "computed_count": len(missing_indices),
        "checkpoint_dir": str(checkpoint_dir),
    }


def selected_expectations(
    project_dir,
    proxy_mps,
    proxy_tag,
    rows_solver,
    rotation_solver,
    bond_dim,
    sweeps,
    tolerance,
    threads,
    work_dir,
    checkpoint_dir,
    context_fingerprint,
) -> np.ndarray:
    """Evaluate selected parity expectations in the optimized frame."""
    add_project_path(project_dir)
    from src.dmrg_costs import DMRGOrbitalCosts, MultiplyConfig
    from src.dmrg_solver import Block2DMRGSolver

    rows_solver = np.atleast_2d(np.asarray(rows_solver, dtype=int))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    values = np.full(len(rows_solver), np.nan, dtype=float)
    missing = []
    for index, row in enumerate(rows_solver):
        saved = read_sign_checkpoint(
            checkpoint_dir,
            index,
            row,
            context_fingerprint,
        )
        if saved is None:
            missing.append(index)
        else:
            values[index] = saved
    print(
        f"[generator sign resume] {len(rows_solver) - len(missing)}/"
        f"{len(rows_solver)} complete; {len(missing)} missing",
        flush=True,
    )
    if not missing:
        return values

    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    copy_solver_store(proxy_mps, work_dir, [proxy_tag])
    solver = Block2DMRGSolver.load(work_dir, n_threads=int(threads))
    costs = DMRGOrbitalCosts(
        solver,
        rows_solver,
        mps_tag=proxy_tag,
        multiply=MultiplyConfig(
            bond_dim=int(bond_dim),
            n_sweeps=int(sweeps),
            tol=float(tolerance),
        ),
    )
    for index in missing:
        row = rows_solver[index]
        print(
            f"[generator sign] selected generator {index + 1}/"
            f"{len(costs.parity_matrix)}",
            flush=True,
        )
        started = time.perf_counter()
        applied = costs._apply_symmetry(
            row,
            np.asarray(rotation_solver, dtype=float),
            costs.ket,
            "SIGN_PHI",
        )
        expectation = float(
            np.real(costs.solver.mps_overlap(costs.ket, applied))
        )
        values[index] = expectation
        write_sign_checkpoint(
            checkpoint_dir,
            index,
            row,
            expectation,
            context_fingerprint,
            time.perf_counter() - started,
        )
    return values


def run_selection(
    project_dir,
    checkpoint,
    proxy_mps,
    proxy_tag,
    permutation,
    rotation_solver,
    output_dir,
    target_rank,
    workers,
    total_threads,
    multiply_bond_dim,
    multiply_sweeps,
    multiply_tolerance=1.0e-10,
) -> dict:
    """Score, select, and save rank-independent seniority/quartet generators."""
    add_project_path(project_dir)
    from src.clifford_sectors import (
        save_symmetry_manifest,
        z_symmetries_from_parity_matrix,
    )
    from src.dmrg_symmetry_selection import (
        assign_candidate_scores,
        candidate_matrix,
        canonical_row_space,
        selected_parity_matrix,
        selection_to_json,
        select_independent_candidates,
        seniority_quartet_candidates,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = seniority_quartet_candidates(len(permutation))
    canonical_rows = candidate_matrix(candidates)
    rows_solver = map_rows_to_solver_order(canonical_rows, permutation)
    started = time.perf_counter()
    worker_root = output_dir / "candidate_workers"
    score_result = parallel_scores(
        project_dir=project_dir,
        proxy_mps=proxy_mps,
        proxy_tag=proxy_tag,
        rows_solver=rows_solver,
        rotation_solver=rotation_solver,
        worker_root=worker_root,
        workers=workers,
        total_threads=total_threads,
        multiply_bond_dim=multiply_bond_dim,
        multiply_sweeps=multiply_sweeps,
        multiply_tolerance=multiply_tolerance,
        checkpoint_dir=output_dir / "candidate_scores",
    )
    scores = score_result["scores"]
    elapsed = time.perf_counter() - started
    scored = assign_candidate_scores(candidates, scores)
    selected = select_independent_candidates(scored, target_rank)
    parity = selected_parity_matrix(selected)
    selected_solver_rows = map_rows_to_solver_order(parity, permutation)
    sign_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": CHECKPOINT_VERSION,
        "kind": "generator_expectation_context",
        "candidate_context_fingerprint": score_result["context"]["fingerprint"],
        "selected_rows_fingerprint": array_fingerprint(
            selected_solver_rows,
            np.int8,
        ),
        "rotation_fingerprint": array_fingerprint(
            rotation_solver,
            np.float64,
        ),
        "multiply_bond_dim": int(multiply_bond_dim),
        "multiply_sweeps": int(multiply_sweeps),
        "multiply_tolerance": float(multiply_tolerance),
    }
    sign_context_fingerprint = payload_fingerprint(sign_payload)
    expectations = selected_expectations(
        project_dir=project_dir,
        proxy_mps=proxy_mps,
        proxy_tag=proxy_tag,
        rows_solver=selected_solver_rows,
        rotation_solver=rotation_solver,
        bond_dim=multiply_bond_dim,
        sweeps=multiply_sweeps,
        tolerance=multiply_tolerance,
        threads=total_threads,
        work_dir=output_dir / "sign_worker_mps",
        checkpoint_dir=output_dir / "generator_signs",
        context_fingerprint=sign_context_fingerprint,
    )
    signs = np.where(expectations < 0.0, -1.0, 1.0)
    expanded = np.zeros((len(parity), 2 * parity.shape[1]), dtype=int)
    expanded[:, 0::2] = parity
    expanded[:, 1::2] = parity
    symmetries = [
        float(sign) * symmetry
        for sign, symmetry in zip(
            signs,
            z_symmetries_from_parity_matrix(parity, parity.shape[1]),
        )
    ]
    parity_path = output_dir / "parity_matrix.txt"
    manifest_path = output_dir / "symmetry_manifest.json"
    selection_path = output_dir / "selection.json"
    np.savetxt(parity_path, parity, fmt="%d")
    ordered = sorted(scored, key=lambda item: (item["score"], item["label"]))
    metadata = {
        "candidate_family": "seniority_plus_quartet",
        "reference": "dmrg_proxy_mps",
        "selection_score": "state_specific_noncommutativity",
        "target_rank": int(target_rank),
        "gf2_rank": int(target_rank),
        "selected_row_space": [
            list(row) for row in canonical_row_space(parity)
        ],
        "selected": selection_to_json(selected),
        "selected_expectations": expectations.tolist(),
        "selected_signs": signs.astype(int).tolist(),
        "rotation_matrix_solver_order": np.asarray(
            rotation_solver, dtype=float
        ).tolist(),
        "orbital_permutation": [int(value) for value in permutation],
        "mps_store": str(Path(proxy_mps).resolve()),
        "mps_tag": str(proxy_tag),
        "candidate_workers": int(workers),
        "candidate_worker_threads": max(1, int(total_threads) // int(workers)),
        "candidate_scoring_seconds": float(elapsed),
        "candidate_score_context": score_result["context"],
        "candidate_score_checkpoint_dir": score_result["checkpoint_dir"],
        "candidate_scores_reused": int(score_result["reused_count"]),
        "candidate_scores_computed": int(score_result["computed_count"]),
        "generator_sign_context_fingerprint": sign_context_fingerprint,
    }
    save_symmetry_manifest(
        manifest_path,
        symmetries,
        expanded,
        metadata=metadata,
    )
    output = {
        "schema": "quasisymmetry.equilibrium_mps_selection",
        "version": 1,
        "checkpoint": str(Path(checkpoint).resolve()),
        "norb": len(permutation),
        "candidate_count": len(candidates),
        "target_rank": int(target_rank),
        "selected": selection_to_json(selected),
        "ordered_candidates": selection_to_json(ordered),
        "parity_matrix": parity.tolist(),
        "parity_matrix_solver_order": selected_solver_rows.tolist(),
        "metadata": metadata,
        "outputs": {
            "parity_matrix": str(parity_path),
            "symmetry_manifest": str(manifest_path),
        },
    }
    atomic_json(selection_path, output)
    shutil.rmtree(worker_root, ignore_errors=True)
    shutil.rmtree(output_dir / "sign_worker_mps", ignore_errors=True)
    print(
        f"[selection] {len(candidates)} candidates -> rank {target_rank}; "
        f"row space {metadata['selected_row_space']}",
        flush=True,
    )
    for item in selected:
        print(
            f"[selection] {item['label']} score={item['score']:.12e} "
            f"row={np.asarray(item['row'], dtype=int).tolist()}",
            flush=True,
        )
    return {
        "selection": str(selection_path),
        "parity": str(parity_path),
        "manifest": str(manifest_path),
        "row_space": metadata["selected_row_space"],
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "score_seconds": elapsed,
        "solver_parity": selected_solver_rows,
    }
