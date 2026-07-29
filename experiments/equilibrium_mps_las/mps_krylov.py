"""Restartable Block2 MPS projection, Krylov, and residual enrichment."""

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
    copy_mps_tag,
    copy_solver_store,
    label_text,
    load_json,
    process_rss_mib,
)
from linear_algebra import (
    coupling_is_allowed,
    hamiltonian_transition_signatures,
    retain_projector_candidates,
)


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def mps_tag_exists(store_dir, tag) -> bool:
    """Return whether a reloadable MPS info file exists."""
    return (Path(store_dir) / f"{tag}-mps_info.bin").exists()


def persist_mps(solver, mps) -> None:
    """Persist the tag-specific Block2 MPS info needed after a restart."""
    solver._activate()
    path = Path(solver.store_dir) / f"{mps.info.tag}-mps_info.bin"
    mps.info.save_data(str(path))


def mps_bond_dimension(mps, fallback=200) -> int:
    """Read the maximum MPS bond dimension across Block2 versions."""
    try:
        return max(2, int(mps.info.get_max_bond_dimension()))
    except Exception:
        try:
            return max(2, int(mps.info.bond_dim))
        except Exception:
            return int(fallback)


def operation_key(*parts) -> str:
    """Compact deterministic identifier for a fitted MPS operation."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]
    return digest


def operation_record(work_dir, tag) -> Path:
    """Metadata path for one completed fitted MPS operation."""
    return Path(work_dir) / "operations" / f"{tag}.json"


def operation_record_matches(record, expected) -> bool:
    """Require exact agreement for every input that defines a fitted MPS."""
    return bool(record) and all(
        record.get(key) == value for key, value in expected.items()
    )


def record_fingerprint(record) -> str | None:
    """Return a canonical fingerprint for one fitted-operation record."""
    if not record:
        return None
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def isolated_split_worker(task) -> dict:
    """Split one projector branch inside an isolated Block2 store."""
    add_project_path(task["project_dir"])
    from src.dmrg_solver import Block2DMRGSolver

    threads = int(task["threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = "1"
    solver = Block2DMRGSolver.load(task["store_dir"], n_threads=threads)
    arguments = (
        solver,
        task["parent_tag"],
        np.asarray(task["parity_row"], dtype=int),
        int(task["level"]),
        tuple(task["parent_label"]),
        task["work_dir"],
        task["prefix"],
        int(task["bond_dim"]),
        int(task["sweeps"]),
        float(task["tolerance"]),
    )
    if task["rotation"] is None:
        children = split_branch_diagonal(*arguments)
    else:
        children = split_branch_rotated(
            *arguments[:3],
            np.asarray(task["rotation"], dtype=float),
            *arguments[3:],
        )
    return {
        "store_dir": task["store_dir"],
        "children": children,
    }


def isolated_sector_chain_worker(task) -> dict:
    """Build one sector-local Krylov extension in an isolated Block2 store."""
    add_project_path(task["project_dir"])
    from src.dmrg_solver import Block2DMRGSolver

    threads = int(task["threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = "1"
    solver = Block2DMRGSolver.load(task["store_dir"], n_threads=threads)
    h_dec_mpo = decoupled_mpo(
        solver, np.asarray(task["parity_matrix"], dtype=int)
    )
    created = extend_sector_chain(
        solver=solver,
        h_dec_mpo=h_dec_mpo,
        seed_tag=task["seed_tag"],
        label=tuple(task["label"]),
        existing_basis=task["existing_basis"],
        additions=int(task["additions"]),
        cycle=int(task["cycle"]),
        work_dir=task["work_dir"],
        bond_dim=int(task["bond_dim"]),
        sweeps=int(task["sweeps"]),
        tolerance=float(task["tolerance"]),
        tag_prefix=task.get("tag_prefix", "KR"),
    )
    return {
        "store_dir": task["store_dir"],
        "created": created,
    }


def safe_source_copy(solver, source, prefix):
    """Protect a stored source because Block2 fitting may rewrite its tensors."""
    tag = f"{prefix}_SRC_{operation_key(source.info.tag, time.time_ns())}"
    return source.deep_copy(tag)


def fitted_apply(
    solver,
    mpo,
    source_tag,
    output_tag,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
) -> dict:
    """Fit ``MPO |source>`` and checkpoint the resulting MPS."""
    record_path = operation_record(work_dir, output_tag)
    expected = {
        "operation": "fitted_mpo_application",
        "source_tag": str(source_tag),
        "output_tag": str(output_tag),
        "bond_dim": int(bond_dim),
        "sweeps": int(sweeps),
        "tolerance": float(tolerance),
    }
    if mps_tag_exists(solver.store_dir, output_tag) and record_path.exists():
        saved = load_json(record_path)
        if operation_record_matches(saved, expected):
            return saved
        print(
            f"[MPS apply] stale record for {output_tag}; recomputing",
            flush=True,
        )
    print(
        f"[MPS apply] {source_tag} -> {output_tag}; "
        f"M={bond_dim}, sweeps={sweeps}",
        flush=True,
    )
    started = time.perf_counter()
    source = solver.get_mps(source_tag)
    protected = safe_source_copy(solver, source, output_tag)
    output = solver.apply_mpo(
        mpo,
        ket=protected,
        tag=output_tag,
        bond_dim=int(bond_dim),
        n_sweeps=int(sweeps),
        tol=float(tolerance),
    )
    persist_mps(solver, output)
    norm2 = solver.mps_norm2(output)
    record = {
        **expected,
        "norm2": float(norm2),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_mib": process_rss_mib(),
    }
    atomic_json(record_path, record)
    return record


def fitted_add(
    solver,
    left_tag,
    right_tag,
    left_coefficient,
    right_coefficient,
    output_tag,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
) -> dict:
    """Fit a two-term linear combination and checkpoint it."""
    record_path = operation_record(work_dir, output_tag)
    coefficients = (
        complex(left_coefficient),
        complex(right_coefficient),
    )
    if any(abs(value.imag) > 1.0e-12 for value in coefficients):
        raise ValueError("the current real Block2 backend requires real coefficients")
    expected = {
        "operation": "fitted_mps_addition",
        "left_tag": str(left_tag),
        "right_tag": str(right_tag),
        "left_coefficient": float(coefficients[0].real),
        "right_coefficient": float(coefficients[1].real),
        "output_tag": str(output_tag),
        "bond_dim": int(bond_dim),
        "sweeps": int(sweeps),
        "tolerance": float(tolerance),
    }
    if mps_tag_exists(solver.store_dir, output_tag) and record_path.exists():
        saved = load_json(record_path)
        if operation_record_matches(saved, expected):
            return saved
        print(
            f"[MPS add] stale record for {output_tag}; recomputing",
            flush=True,
        )
    print(
        f"[MPS add] {coefficients[0].real:+.6e}*{left_tag} "
        f"{coefficients[1].real:+.6e}*{right_tag} -> {output_tag}",
        flush=True,
    )
    started = time.perf_counter()
    left = solver.get_mps(left_tag)
    right = solver.get_mps(right_tag)
    target = solver.driver.get_random_mps(
        tag=output_tag, bond_dim=int(bond_dim), nroots=1
    )
    solver.driver.addition(
        target,
        left,
        right,
        mpo_a=float(coefficients[0].real),
        mpo_b=float(coefficients[1].real),
        n_sweeps=int(sweeps),
        tol=float(tolerance),
        bra_bond_dims=[int(bond_dim)] * int(sweeps),
        noises=[0.0] * int(sweeps),
        iprint=0,
    )
    persist_mps(solver, target)
    norm2 = solver.mps_norm2(target)
    record = {
        **expected,
        "norm2": float(norm2),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_mib": process_rss_mib(),
    }
    atomic_json(record_path, record)
    return record


def fitted_scale(
    solver,
    source_tag,
    coefficient,
    output_tag,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
) -> dict:
    """Fit a scalar multiple without relying on in-place tensor scaling."""
    return fitted_add(
        solver,
        source_tag,
        source_tag,
        coefficient,
        0.0,
        output_tag,
        work_dir,
        bond_dim,
        sweeps,
        tolerance,
    )


def normalize_mps(
    solver,
    source_tag,
    output_tag,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
    linear_dependence_tolerance=1.0e-12,
) -> dict | None:
    """Normalize one MPS through a fitted scalar multiplication."""
    source = solver.get_mps(source_tag)
    norm2 = solver.mps_norm2(source)
    if norm2 <= float(linear_dependence_tolerance):
        return None
    record = fitted_scale(
        solver,
        source_tag,
        1.0 / np.sqrt(norm2),
        output_tag,
        work_dir,
        bond_dim,
        sweeps,
        tolerance,
    )
    record["source_norm2"] = float(norm2)
    atomic_json(operation_record(work_dir, output_tag), record)
    return record


def balanced_linear_combination(
    solver,
    tags,
    coefficients,
    output_tag,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
) -> dict:
    """Fit a long MPS sum with a balanced tree of pairwise additions."""
    tags = [str(tag) for tag in tags]
    coefficients = [complex(value) for value in coefficients]
    if len(tags) != len(coefficients) or not tags:
        raise ValueError("one nonempty coefficient list is required for the tags")
    if len(tags) == 1:
        return fitted_scale(
            solver,
            tags[0],
            coefficients[0],
            output_tag,
            work_dir,
            bond_dim,
            sweeps,
            tolerance,
        )
    nodes = list(zip(tags, coefficients))
    level = 0
    while len(nodes) > 1:
        next_nodes = []
        for pair_index in range(0, len(nodes), 2):
            if pair_index + 1 == len(nodes):
                next_nodes.append(nodes[pair_index])
                continue
            (left_tag, left_coefficient), (
                right_tag,
                right_coefficient,
            ) = nodes[pair_index : pair_index + 2]
            is_final = len(nodes) == 2
            tag = (
                output_tag
                if is_final
                else f"{output_tag}_L{level}_{pair_index // 2}"
            )
            fitted_add(
                solver,
                left_tag,
                right_tag,
                left_coefficient,
                right_coefficient,
                tag,
                work_dir,
                bond_dim,
                sweeps,
                tolerance,
            )
            next_nodes.append((tag, 1.0 + 0.0j))
        nodes = next_nodes
        level += 1
    final_tag, coefficient = nodes[0]
    if final_tag != output_tag or not np.isclose(coefficient, 1.0):
        return fitted_scale(
            solver,
            final_tag,
            coefficient,
            output_tag,
            work_dir,
            bond_dim,
            sweeps,
            tolerance,
        )
    return load_json(operation_record(work_dir, output_tag))


def diagonal_projector_mpo(solver, parity_row, bit):
    """Build ``(I + (-1)^bit S)/2`` for one diagonal parity generator."""
    solver._activate()
    builder = solver.driver.expr_builder()
    builder.add_const(0.5)
    sign = 0.5 if int(bit) == 0 else -0.5
    for expression, indices, coefficient in solver._parity_terms(parity_row):
        scaled = sign * float(coefficient)
        if expression == "":
            builder.add_const(scaled)
        else:
            builder.add_term(expression, indices, scaled)
    return solver.driver.get_mpo(builder.finalize(), iprint=0)


def split_branch_diagonal(
    solver,
    parent_tag,
    parity_row,
    level,
    parent_label,
    work_dir,
    prefix,
    bond_dim,
    sweeps,
    tolerance,
) -> list[dict]:
    """Split one MPS branch using diagonal parity projectors."""
    children = []
    for bit in (0, 1):
        label = tuple(parent_label) + (bit,)
        tag = f"{prefix}_L{level}_{label_text(label)}"
        mpo = diagonal_projector_mpo(solver, parity_row, bit)
        record = fitted_apply(
            solver,
            mpo,
            parent_tag,
            tag,
            work_dir,
            bond_dim,
            sweeps,
            tolerance,
        )
        children.append(
            {"label": label, "tag": tag, "norm2": float(record["norm2"])}
        )
    return children


def split_branch_rotated(
    solver,
    parent_tag,
    parity_row,
    rotation,
    level,
    parent_label,
    work_dir,
    prefix,
    bond_dim,
    sweeps,
    tolerance,
) -> list[dict]:
    """Split one branch when ``S`` is represented by rotated-factor MPOs."""
    symmetry_tag = f"{prefix}_S{level}_{label_text(parent_label) or 'root'}"
    symmetry_record = operation_record(work_dir, symmetry_tag)
    if not (mps_tag_exists(solver.store_dir, symmetry_tag) and symmetry_record.exists()):
        parent = solver.get_mps(parent_tag)
        protected = safe_source_copy(solver, parent, symmetry_tag)
        applied = solver.apply_rotated_parity(
            parity_row,
            np.asarray(rotation, dtype=float),
            ket=protected,
            tag=symmetry_tag,
            bond_dim=int(bond_dim),
            n_sweeps=int(sweeps),
            tol=float(tolerance),
        )
        if str(applied.info.tag) != symmetry_tag:
            applied = applied.deep_copy(symmetry_tag)
        persist_mps(solver, applied)
        atomic_json(
            symmetry_record,
            {
                "operation": "rotated_symmetry_application",
                "source_tag": parent_tag,
                "output_tag": symmetry_tag,
                "norm2": solver.mps_norm2(applied),
            },
        )
    children = []
    for bit, symmetry_coefficient in ((0, 0.5), (1, -0.5)):
        label = tuple(parent_label) + (bit,)
        tag = f"{prefix}_L{level}_{label_text(label)}"
        record = fitted_add(
            solver,
            parent_tag,
            symmetry_tag,
            0.5,
            symmetry_coefficient,
            tag,
            work_dir,
            bond_dim,
            sweeps,
            tolerance,
        )
        children.append(
            {"label": label, "tag": tag, "norm2": float(record["norm2"])}
        )
    return children


def split_branches_isolated(
    solver,
    branches,
    parity_row,
    rotation,
    level,
    work_dir,
    prefix,
    bond_dim,
    sweeps,
    tolerance,
    project_dir,
    worker_root,
    workers,
    total_threads,
) -> list[dict]:
    """Split independent MPS branches with one copied Block2 store per worker."""
    workers = max(1, min(int(workers), len(branches), int(total_threads)))
    if workers == 1:
        output = []
        for branch in branches:
            arguments = (
                solver,
                branch["tag"],
                parity_row,
                level,
                tuple(branch["label"]),
                work_dir,
                prefix,
                bond_dim,
                sweeps,
                tolerance,
            )
            if rotation is None:
                output.extend(split_branch_diagonal(*arguments))
            else:
                output.extend(
                    split_branch_rotated(
                        *arguments[:3],
                        rotation,
                        *arguments[3:],
                    )
                )
        return output

    worker_root = Path(worker_root) / f"{prefix}_level_{level}"
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(total_threads) // workers)
    tasks = []
    for index, branch in enumerate(branches):
        task_root = worker_root / f"branch_{index:03d}"
        store = task_root / "mps"
        copy_solver_store(solver.store_dir, store, [branch["tag"]])
        tasks.append(
            {
                "project_dir": str(project_dir),
                "store_dir": str(store),
                "work_dir": str(task_root / "work"),
                "parent_tag": branch["tag"],
                "parent_label": list(branch["label"]),
                "parity_row": np.asarray(parity_row, dtype=int).tolist(),
                "rotation": (
                    None
                    if rotation is None
                    else np.asarray(rotation, dtype=float).tolist()
                ),
                "level": int(level),
                "prefix": prefix,
                "bond_dim": int(bond_dim),
                "sweeps": int(sweeps),
                "tolerance": float(tolerance),
                "threads": threads,
            }
        )
    print(
        f"[projector parallel] level={level}, branches={len(tasks)}, "
        f"workers={workers}, threads/worker={threads}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        results = list(executor.map(isolated_split_worker, tasks))
    output = []
    for result in results:
        for child in result["children"]:
            copy_mps_tag(result["store_dir"], solver.store_dir, child["tag"])
            output.append(child)
    shutil.rmtree(worker_root, ignore_errors=True)
    return output


def projector_beam_split(
    solver,
    source_tag,
    parity_matrix,
    work_dir,
    prefix,
    beam_width,
    bond_dim,
    sweeps,
    tolerance,
    rotation=None,
    project_dir=None,
    worker_root=None,
    workers=1,
    total_threads=1,
    protected_label=None,
) -> dict:
    """Recursively resolve an MPS into selected joint parity sectors."""
    parity = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    work_dir = Path(work_dir)
    source = solver.get_mps(source_tag)
    source_norm2 = solver.mps_norm2(source)
    source_record_fingerprint = record_fingerprint(
        load_json(operation_record(work_dir, source_tag), {})
    )
    result_path = work_dir / f"{prefix}_projector_result.json"
    if result_path.exists():
        saved = load_json(result_path)
        protected_present = (
            protected_label is None
            or any(
                tuple(item["label"]) == tuple(protected_label)
                for item in saved.get("branches", [])
            )
        )
        compatible = (
            saved.get("source_tag") == source_tag
            and int(saved.get("beam_width", -1)) == int(beam_width)
            and int(saved.get("bond_dim", -1)) == int(bond_dim)
            and len(saved.get("levels", [])) == len(parity)
            and protected_present
            and np.isclose(
                float(saved.get("source_norm2", np.nan)),
                float(source_norm2),
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            and (
                saved.get("source_record_fingerprint") is None
                or saved.get("source_record_fingerprint")
                == source_record_fingerprint
            )
        )
        if compatible and all(
            mps_tag_exists(solver.store_dir, item["tag"])
            for item in saved["branches"]
        ):
            print(f"[projector] reusing {result_path}", flush=True)
            return saved
    branches = [{"label": (), "tag": source_tag, "norm2": source_norm2}]
    discarded_beam_weight = 0.0
    compression_loss = 0.0
    levels = []
    for level, row in enumerate(parity, start=1):
        print(
            f"[projector] level {level}/{len(parity)}: "
            f"splitting {len(branches)} retained branches",
            flush=True,
        )
        candidates = []
        split_norm = 0.0
        parent_norm = sum(float(item["norm2"]) for item in branches)
        candidates = split_branches_isolated(
            solver=solver,
            branches=branches,
            parity_row=row,
            rotation=rotation,
            level=level,
            work_dir=work_dir,
            prefix=prefix,
            bond_dim=bond_dim,
            sweeps=sweeps,
            tolerance=tolerance,
            project_dir=project_dir,
            worker_root=worker_root or (work_dir / "projector_workers"),
            workers=workers,
            total_threads=total_threads,
        )
        split_norm = sum(float(item["norm2"]) for item in candidates)
        level_loss = max(0.0, parent_norm - split_norm)
        compression_loss += level_loss
        protected_prefix = (
            None
            if protected_label is None
            else tuple(int(bit) for bit in protected_label[:level])
        )
        kept, pruned = retain_projector_candidates(
            candidates,
            beam_width,
            protected_prefix=protected_prefix,
        )
        pruned_weight = sum(float(item["norm2"]) for item in pruned)
        discarded_beam_weight += pruned_weight
        branches = kept
        level_record = {
            "level": level,
            "generator_index": level - 1,
            "parent_norm2": parent_norm,
            "child_norm2_before_pruning": split_norm,
            "compression_loss": level_loss,
            "pruned_weight": pruned_weight,
            "retained_count": len(kept),
            "retained_weight": sum(float(item["norm2"]) for item in kept),
            "protected_prefix": (
                None
                if protected_prefix is None
                else list(protected_prefix)
            ),
            "protected_prefix_retained": (
                None
                if protected_prefix is None
                else any(
                    tuple(item["label"]) == protected_prefix for item in kept
                )
            ),
        }
        levels.append(level_record)
        atomic_json(work_dir / f"{prefix}_projector_level_{level}.json", {
            **level_record,
            "branches": kept,
        })
        print(
            f"[projector] level {level}: retained={len(kept)}, "
            f"weight={level_record['retained_weight']:.12e}, "
            f"fit_loss={level_loss:.3e}, pruned={pruned_weight:.3e}",
            flush=True,
        )
    result = {
        "source_tag": source_tag,
        "source_norm2": source_norm2,
        "source_record_fingerprint": source_record_fingerprint,
        "branches": branches,
        "retained_weight": sum(float(item["norm2"]) for item in branches),
        "discarded_beam_weight": discarded_beam_weight,
        "compression_loss": compression_loss,
        "levels": levels,
        "beam_width": int(beam_width),
        "bond_dim": int(bond_dim),
        "protected_label": (
            None
            if protected_label is None
            else [int(bit) for bit in protected_label]
        ),
    }
    atomic_json(result_path, result)
    return result


def decoupled_mpo(solver, parity_matrix):
    """Build the term-filtered $H_dec$ MPO without a sector penalty."""
    solver._activate()
    integrals = solver.decoupled_integrals(parity_matrix)
    builder = solver._qc_expr_builder(integrals)
    return solver.driver.get_mpo(builder.finalize(), iprint=0)


def two_pass_orthogonalize(
    solver,
    source_tag,
    basis_tags,
    output_prefix,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
    dependence_tolerance=1.0e-10,
) -> dict | None:
    """Two-pass fitted Gram-Schmidt within one parity sector."""
    current_tag = source_tag
    serial = 0
    for pass_index in range(2):
        for basis_tag in basis_tags:
            current = solver.get_mps(current_tag)
            basis = solver.get_mps(basis_tag)
            denominator = float(np.real(solver.mps_overlap(basis, basis)))
            if denominator <= float(dependence_tolerance):
                continue
            coefficient = solver.mps_overlap(basis, current) / denominator
            if abs(coefficient) <= float(tolerance):
                continue
            tag = f"{output_prefix}_P{pass_index}_{serial}"
            fitted_add(
                solver,
                current_tag,
                basis_tag,
                1.0,
                -coefficient,
                tag,
                work_dir,
                bond_dim,
                sweeps,
                tolerance,
            )
            current_tag = tag
            serial += 1
    normalized_tag = f"{output_prefix}_N"
    result = normalize_mps(
        solver,
        current_tag,
        normalized_tag,
        work_dir,
        bond_dim,
        sweeps,
        tolerance,
        linear_dependence_tolerance=dependence_tolerance,
    )
    if result is None:
        print(
            f"[orthogonalize] rejected dependent direction {source_tag}",
            flush=True,
        )
        return None
    return {
        "tag": normalized_tag,
        "source_tag": source_tag,
        "pre_normalization_norm2": float(result["source_norm2"]),
    }


def extend_sector_chain(
    solver,
    h_dec_mpo,
    seed_tag,
    label,
    existing_basis,
    additions,
    cycle,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
    dependence_tolerance=1.0e-10,
    tag_prefix="KR",
) -> list[dict]:
    """Add a residual seed and sector-local $H_dec$ Krylov actions."""
    label = tuple(int(bit) for bit in label)
    label_name = label_text(label)
    same_sector_tags = [
        item["tag"] for item in existing_basis if tuple(item["label"]) == label
    ]
    created = []
    current_seed = seed_tag
    for step in range(int(additions)):
        if step > 0:
            raw_tag = f"{tag_prefix}_C{cycle}_{label_name}_RAW{step}"
            fitted_apply(
                solver,
                h_dec_mpo,
                current_seed,
                raw_tag,
                work_dir,
                bond_dim,
                sweeps,
                tolerance,
            )
            current_seed = raw_tag
        output_prefix = (
            f"{tag_prefix}_C{cycle}_{label_name}_D{len(same_sector_tags)}"
        )
        orthogonalized = two_pass_orthogonalize(
            solver,
            current_seed,
            same_sector_tags,
            output_prefix,
            work_dir,
            bond_dim,
            sweeps,
            tolerance,
            dependence_tolerance=dependence_tolerance,
        )
        if orthogonalized is None:
            break
        item = {
            "tag": orthogonalized["tag"],
            "label": list(label),
            "kind": "residual_seed" if step == 0 else "sector_krylov",
            "cycle": int(cycle),
            "depth": len(same_sector_tags),
            "pre_normalization_norm2": orthogonalized[
                "pre_normalization_norm2"
            ],
        }
        created.append(item)
        same_sector_tags.append(item["tag"])
        current_seed = item["tag"]
    return created


def extend_sector_chains_isolated(
    solver,
    parity_matrix,
    requests,
    project_dir,
    worker_root,
    workers,
    total_threads,
) -> list[list[dict]]:
    """Run independent sector-local Krylov requests in isolated worker stores."""
    if not requests:
        return []
    workers = max(1, min(int(workers), len(requests), int(total_threads)))
    if workers == 1:
        h_dec_mpo = decoupled_mpo(solver, parity_matrix)
        return [
            extend_sector_chain(
                solver=solver,
                h_dec_mpo=h_dec_mpo,
                seed_tag=request["seed_tag"],
                label=request["label"],
                existing_basis=request["existing_basis"],
                additions=request["additions"],
                cycle=request["cycle"],
                work_dir=request["work_dir"],
                bond_dim=request["bond_dim"],
                sweeps=request["sweeps"],
                tolerance=request["tolerance"],
                tag_prefix=request.get("tag_prefix", "KR"),
            )
            for request in requests
        ]

    worker_root = Path(worker_root)
    shutil.rmtree(worker_root, ignore_errors=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(total_threads) // workers)
    tasks = []
    for index, request in enumerate(requests):
        task_root = worker_root / f"sector_{index:03d}_{label_text(request['label'])}"
        store = task_root / "mps"
        tags = {request["seed_tag"]}
        tags.update(
            item["tag"]
            for item in request["existing_basis"]
            if tuple(item["label"]) == tuple(request["label"])
        )
        copy_solver_store(solver.store_dir, store, tags)
        tasks.append(
            {
                "project_dir": str(project_dir),
                "store_dir": str(store),
                "work_dir": str(task_root / "work"),
                "parity_matrix": np.asarray(parity_matrix, dtype=int).tolist(),
                "seed_tag": request["seed_tag"],
                "label": list(request["label"]),
                "existing_basis": request["existing_basis"],
                "additions": int(request["additions"]),
                "cycle": int(request["cycle"]),
                "bond_dim": int(request["bond_dim"]),
                "sweeps": int(request["sweeps"]),
                "tolerance": float(request["tolerance"]),
                "tag_prefix": request.get("tag_prefix", "KR"),
                "threads": threads,
            }
        )
    print(
        f"[Krylov parallel] sectors={len(tasks)}, workers={workers}, "
        f"threads/worker={threads}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        results = list(executor.map(isolated_sector_chain_worker, tasks))
    output = []
    for result in results:
        for item in result["created"]:
            copy_mps_tag(result["store_dir"], solver.store_dir, item["tag"])
        output.append(result["created"])
    shutil.rmtree(worker_root, ignore_errors=True)
    return output


def matrix_metadata_path(matrix_path) -> Path:
    """Sidecar path recording basis order for an incremental matrix."""
    return Path(matrix_path).with_suffix(".basis.json")


def build_coupled_matrices(
    solver,
    basis,
    full_hamiltonian_mpo,
    transition_signatures,
    matrix_path,
    progress_path,
    resume=True,
    reuse_matrix_paths=(),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Incrementally construct $H_K$ and $S_K$ with exact sector sparsity."""
    matrix_path = Path(matrix_path)
    basis_ids = [
        {
            "tag": item["tag"],
            "label": list(item["label"]),
            "kind": item.get("kind"),
            "cycle": item.get("cycle"),
            "depth": item.get("depth"),
        }
        for item in basis
    ]
    dimension = len(basis)
    hamiltonian = np.full((dimension, dimension), np.nan + 0.0j)
    overlap = np.full((dimension, dimension), np.nan + 0.0j)
    old_metadata = load_json(matrix_metadata_path(matrix_path), {})
    if resume and matrix_path.exists():
        saved = np.load(matrix_path)
        old_basis = old_metadata.get("basis", [])
        old_h = np.asarray(saved["hamiltonian"], dtype=np.complex128)
        old_s = np.asarray(saved["overlap"], dtype=np.complex128)
        extent = min(
            len(old_basis),
            old_h.shape[0],
            old_s.shape[0],
            dimension,
        )
        if extent > 0 and old_basis[:extent] == basis_ids[:extent]:
            hamiltonian[:extent, :extent] = old_h[:extent, :extent]
            overlap[:extent, :extent] = old_s[:extent, :extent]
            print(
                f"[coupled matrix] reused {extent} x {extent} prefix",
                flush=True,
            )
    if np.any(np.isnan(hamiltonian.real)):
        target_by_tag = {
            item["tag"]: (index, item)
            for index, item in enumerate(basis_ids)
        }
        for reuse_path in reuse_matrix_paths:
            reuse_path = Path(reuse_path)
            if reuse_path == matrix_path or not reuse_path.exists():
                continue
            reuse_metadata = load_json(matrix_metadata_path(reuse_path), {})
            reuse_basis = reuse_metadata.get("basis", [])
            saved = np.load(reuse_path)
            reuse_h = np.asarray(saved["hamiltonian"], dtype=np.complex128)
            reuse_s = np.asarray(saved["overlap"], dtype=np.complex128)
            source_by_tag = {
                item["tag"]: (index, item)
                for index, item in enumerate(reuse_basis)
            }
            imported = 0
            for left_tag, (left, left_item) in target_by_tag.items():
                source_left = source_by_tag.get(left_tag)
                if source_left is None or source_left[1] != left_item:
                    continue
                for right_tag, (right, right_item) in target_by_tag.items():
                    source_right = source_by_tag.get(right_tag)
                    if source_right is None or source_right[1] != right_item:
                        continue
                    old_left = source_left[0]
                    old_right = source_right[0]
                    if (
                        not np.isnan(hamiltonian[left, right].real)
                        or not np.isnan(overlap[left, right].real)
                        or old_left >= reuse_h.shape[0]
                        or old_right >= reuse_h.shape[1]
                        or old_left >= reuse_s.shape[0]
                        or old_right >= reuse_s.shape[1]
                        or np.isnan(reuse_h[old_left, old_right].real)
                        or np.isnan(reuse_s[old_left, old_right].real)
                    ):
                        continue
                    hamiltonian[left, right] = reuse_h[old_left, old_right]
                    overlap[left, right] = reuse_s[old_left, old_right]
                    imported += 1
            if imported:
                print(
                    f"[coupled matrix] imported {imported} elements from "
                    f"{reuse_path}",
                    flush=True,
                )
                break
    contractions = 0
    skipped_h = 0
    skipped_s = 0
    for row in range(dimension):
        bra = solver.get_mps(basis[row]["tag"])
        label_row = tuple(basis[row]["label"])
        for column in range(row + 1):
            if not np.isnan(overlap[row, column].real):
                continue
            ket = solver.get_mps(basis[column]["tag"])
            label_column = tuple(basis[column]["label"])
            if label_row == label_column:
                overlap_value = solver.mps_overlap(bra, ket)
                contractions += 1
            else:
                overlap_value = 0.0
                skipped_s += 1
            if coupling_is_allowed(
                label_row, label_column, transition_signatures
            ):
                solver._activate()
                hamiltonian_value = solver.driver.expectation(
                    bra, full_hamiltonian_mpo, ket
                )
                contractions += 1
            else:
                hamiltonian_value = 0.0
                skipped_h += 1
            overlap[row, column] = overlap_value
            overlap[column, row] = np.conjugate(overlap_value)
            hamiltonian[row, column] = hamiltonian_value
            hamiltonian[column, row] = np.conjugate(hamiltonian_value)
        np.savez_compressed(
            matrix_path,
            hamiltonian=hamiltonian,
            overlap=overlap,
        )
        atomic_json(
            matrix_metadata_path(matrix_path),
            {"basis": basis_ids, "completed_rows": row + 1},
        )
        atomic_json(
            progress_path,
            {
                "stage": "coupled_matrix",
                "dimension": dimension,
                "completed_rows": row + 1,
                "contractions": contractions,
                "skipped_hamiltonian_elements": skipped_h,
                "skipped_overlap_elements": skipped_s,
                "rss_mib": process_rss_mib(),
            },
        )
        print(
            f"[coupled matrix] row {row + 1}/{dimension}; "
            f"contractions={contractions}",
            flush=True,
        )
    diagnostics = {
        "contractions": contractions,
        "skipped_hamiltonian_elements": skipped_h,
        "skipped_overlap_elements": skipped_s,
    }
    return hamiltonian, overlap, diagnostics


def validate_fitted_ritz(
    solver,
    ritz_tag,
    expected_energy,
    full_hamiltonian_mpo,
) -> dict:
    """Compare a fitted coupled Ritz MPS with its generalized eigenpair."""
    state = solver.get_mps(ritz_tag)
    norm2 = solver.mps_norm2(state)
    solver._activate()
    numerator = solver.driver.expectation(
        state, full_hamiltonian_mpo, state
    )
    fitted_energy = float(np.real(numerator / norm2))
    return {
        "tag": ritz_tag,
        "norm2": float(norm2),
        "energy": fitted_energy,
        "generalized_energy": float(expected_energy),
        "energy_difference_mha": abs(
            fitted_energy - float(expected_energy)
        )
        * 1000.0,
    }


def make_residual(
    solver,
    full_hamiltonian_mpo,
    state_tag,
    energy,
    prefix,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
) -> dict:
    """Construct $(H-E)|state>$ as a fitted MPS."""
    fidelity = f"M{int(bond_dim)}_S{int(sweeps)}"
    h_tag = f"{prefix}_{fidelity}_HPSI"
    fitted_apply(
        solver,
        full_hamiltonian_mpo,
        state_tag,
        h_tag,
        work_dir,
        bond_dim,
        sweeps,
        tolerance,
    )
    residual_tag = f"{prefix}_{fidelity}_RESIDUAL"
    result = fitted_add(
        solver,
        h_tag,
        state_tag,
        1.0,
        -float(energy),
        residual_tag,
        work_dir,
        bond_dim,
        sweeps,
        tolerance,
    )
    return {
        "tag": residual_tag,
        "norm2": float(result["norm2"]),
        "norm": float(np.sqrt(max(0.0, result["norm2"]))),
    }


def load_or_build_optimized_solver(
    project_dir,
    proxy_mps,
    rotation_solver,
    store_dir,
    threads,
    stack_mem_gb=8.0,
    symmetry_mode="sz",
) -> object:
    """Build the optimized-frame solver in the saved proxy orbital order."""
    add_project_path(project_dir)
    from src.dmrg_solver import (
        Block2DMRGSolver,
        rotate_integrals,
    )

    proxy = Block2DMRGSolver.load(proxy_mps, n_threads=int(threads))
    h1e, g2e = rotate_integrals(
        proxy.h1e,
        proxy.g2e,
        np.asarray(rotation_solver, dtype=float),
    )
    return Block2DMRGSolver(
        h1e=h1e,
        g2e=g2e,
        ecore=proxy.ecore,
        n_elec=proxy.n_elec,
        spin=proxy.spin,
        store_dir=store_dir,
        n_threads=int(threads),
        orbital_symmetries=proxy.orbital_symmetries,
        target_irrep=proxy.target_irrep,
        symmetry_mode=str(symmetry_mode),
        n_mkl_threads=1,
        stack_mem_bytes=int(float(stack_mem_gb) * 1024**3),
    )


def build_transition_signatures(solver, parity_matrix):
    """Build Hamiltonian label-change signatures from the optimized integrals."""
    from src.dmrg_solver import restore_g2e

    return hamiltonian_transition_signatures(
        parity_matrix,
        solver.h1e,
        restore_g2e(solver.g2e, solver.n_sites),
    )
