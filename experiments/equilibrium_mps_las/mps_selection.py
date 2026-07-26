"""MPS-native seniority/quartet scoring that respects saved orbital ordering."""

from __future__ import annotations

import concurrent.futures
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
    map_rows_to_solver_order,
)


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


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
    total = int(task["total_candidates"])
    for local_index, row in enumerate(rows):
        candidate = int(task["indices"][local_index]) + 1
        print(
            f"[NC worker {task['worker']}] candidate {candidate}/{total}",
            flush=True,
        )
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SCORE_PHI")
        xi = costs._apply_symmetry(row, rotation, costs._eta, "SCORE_XI")
        chi = costs._apply(costs._h_mpo, phi, "SCORE_CHI")
        value = (
            costs.solver.mps_norm2(chi)
            + costs.solver.mps_norm2(xi)
            - 2.0
            * float(np.real(costs.solver.mps_overlap(chi, xi)))
        )
        scores.append(max(0.0, float(value)))
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
) -> np.ndarray:
    """Score candidate rows with isolated Block2 stores per process."""
    rows_solver = np.atleast_2d(np.asarray(rows_solver, dtype=int))
    workers = max(1, min(int(workers), len(rows_solver), int(total_threads)))
    chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.arange(len(rows_solver)), workers)
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
            }
        )
    scores = np.empty(len(rows_solver), dtype=float)
    if len(tasks) == 1:
        results = [score_rows_in_store(tasks[0])]
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(tasks), mp_context=context
        ) as executor:
            results = list(executor.map(score_rows_in_store, tasks))
    for result in results:
        scores[np.asarray(result["indices"], dtype=int)] = np.asarray(
            result["scores"], dtype=float
        )
    return scores


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
) -> np.ndarray:
    """Evaluate selected parity expectations in the optimized frame."""
    add_project_path(project_dir)
    from src.dmrg_costs import DMRGOrbitalCosts, MultiplyConfig
    from src.dmrg_solver import Block2DMRGSolver

    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    copy_solver_store(proxy_mps, work_dir, [proxy_tag])
    solver = Block2DMRGSolver.load(work_dir, n_threads=int(threads))
    costs = DMRGOrbitalCosts(
        solver,
        np.atleast_2d(np.asarray(rows_solver, dtype=int)),
        mps_tag=proxy_tag,
        multiply=MultiplyConfig(
            bond_dim=int(bond_dim),
            n_sweeps=int(sweeps),
            tol=float(tolerance),
        ),
    )
    values = []
    for index, row in enumerate(costs.parity_matrix, start=1):
        print(
            f"[generator sign] selected generator {index}/"
            f"{len(costs.parity_matrix)}",
            flush=True,
        )
        applied = costs._apply_symmetry(
            row,
            np.asarray(rotation_solver, dtype=float),
            costs.ket,
            "SIGN_PHI",
        )
        values.append(
            float(np.real(costs.solver.mps_overlap(costs.ket, applied)))
        )
    return np.asarray(values, dtype=float)


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
    scores = parallel_scores(
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
    )
    elapsed = time.perf_counter() - started
    scored = assign_candidate_scores(candidates, scores)
    selected = select_independent_candidates(scored, target_rank)
    parity = selected_parity_matrix(selected)
    selected_solver_rows = map_rows_to_solver_order(parity, permutation)
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
