"""Switching-sector optimization and MPS-native sector screening."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    atomic_json,
    copy_solver_store,
    load_json,
    map_rows_to_solver_order,
)
from mps_krylov import projector_beam_split


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def solver_rotation_pairs(solver):
    """Return intra-irrep rotation pairs in the solver's orbital order."""
    from src.orbital_rotation import irrep_pairs

    if solver.orbital_symmetries is None:
        raise ValueError("irrep-restricted optimization needs orbital symmetries")
    return irrep_pairs(solver.orbital_symmetries)


def determinant_screening_diagnostic(
    solver,
    parity_solver,
    proxy_tag,
    cutoff,
) -> dict:
    """Record norm recovered by the existing dominant-determinant screening."""
    state = solver.get_mps(proxy_tag)
    weighted = solver.dominant_sector_labels(
        parity_solver,
        ket=state,
        cutoff=float(cutoff),
        max_sectors=None,
    )
    return {
        "method": "dominant_determinants_from_proxy_mps",
        "coefficient_cutoff": float(cutoff),
        "recovered_norm": float(sum(weight for _label, weight in weighted)),
        "observed_sector_count": len(weighted),
        "sector_weights": {
            "".join(map(str, label)): float(weight)
            for label, weight in weighted
        },
        "note": (
            "This diagnostic classifies determinants in the saved proxy frame. "
            "The production sector list below uses MPS projector branching so "
            "it remains valid for a nonzero orbital rotation."
        ),
    }


def screen_rotated_sectors(
    project_dir,
    proxy_mps,
    proxy_tag,
    parity_solver,
    rotation_solver,
    output_dir,
    minimum,
    maximum,
    bond_dim,
    sweeps,
    tolerance,
    threads,
    sector_workers,
    determinant_cutoff=1.0e-6,
) -> dict:
    """Screen joint labels by recursively projecting the saved proxy MPS."""
    add_project_path(project_dir)
    from src.dmrg_decoupled_energy import add_neighbour_labels
    from src.dmrg_solver import Block2DMRGSolver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screening_store = output_dir / "screening_mps"
    if not screening_store.exists():
        copy_solver_store(proxy_mps, screening_store, [proxy_tag])
    solver = Block2DMRGSolver.load(screening_store, n_threads=int(threads))
    determinant = determinant_screening_diagnostic(
        solver,
        parity_solver,
        proxy_tag,
        determinant_cutoff,
    )
    split = projector_beam_split(
        solver=solver,
        source_tag=proxy_tag,
        parity_matrix=parity_solver,
        rotation=np.asarray(rotation_solver, dtype=float),
        work_dir=output_dir,
        prefix="SCREEN",
        beam_width=int(maximum),
        bond_dim=int(bond_dim),
        sweeps=int(sweeps),
        tolerance=float(tolerance),
        project_dir=project_dir,
        worker_root=output_dir / "projector_workers",
        workers=sector_workers,
        total_threads=threads,
    )
    weighted = sorted(
        [
            (tuple(item["label"]), float(item["norm2"]))
            for item in split["branches"]
            if float(item["norm2"]) > 1.0e-14
        ],
        key=lambda item: -item[1],
    )
    labels = [label for label, _weight in weighted]
    labels = add_neighbour_labels(labels, len(parity_solver), int(minimum))
    labels = labels[: int(maximum)]
    weight_map = {
        "".join(map(str, label)): float(weight) for label, weight in weighted
    }
    result = {
        "method": "recursive_mps_projector_beam",
        "labels": [list(label) for label in labels],
        "sector_weights": weight_map,
        "retained_projector_weight": float(split["retained_weight"]),
        "source_norm2": float(split["source_norm2"]),
        "discarded_beam_weight": float(split["discarded_beam_weight"]),
        "projector_compression_loss": float(split["compression_loss"]),
        "determinant_screening_diagnostic": determinant,
        "screening_store": str(screening_store),
    }
    atomic_json(output_dir / "sector_screening.json", result)
    print(
        f"[screening] labels={[''.join(map(str, label)) for label in labels]}",
        flush=True,
    )
    print(
        f"[screening] MPS retained={split['retained_weight']:.12f}, "
        f"beam discarded={split['discarded_beam_weight']:.3e}, "
        f"fit loss={split['compression_loss']:.3e}",
        flush=True,
    )
    print(
        f"[screening] determinant norm recovered at cutoff "
        f"{determinant_cutoff:g}: {determinant['recovered_norm']:.12f}",
        flush=True,
    )
    return result


def run_switching_optimization(
    project_dir,
    proxy_mps,
    proxy_tag,
    parity_canonical,
    permutation,
    x0,
    output_dir,
    threads,
    sector_workers,
    screening_minimum,
    screening_maximum,
    screening_bond_dim,
    screening_sweeps,
    screening_tolerance,
    sector_bond_dim,
    sector_sweeps,
    sector_penalty,
    sector_energy_tolerance,
    sector_davidson_threshold,
    sector_twosite_to_onesite,
    optimizer_maxiter,
    sector_switch_maxiter,
    resume,
) -> dict:
    """Run one complete optimize-rescan-switch macrocycle."""
    add_project_path(project_dir)
    from src.dmrg_decoupled_energy import (
        make_context,
        optimize_with_dmrg_sector_switching,
    )
    from src.dmrg_solver import Block2DMRGSolver
    from src.orbital_rotation import params_to_U

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_solver = Block2DMRGSolver.load(proxy_mps, n_threads=int(threads))
    pairs = solver_rotation_pairs(proxy_solver)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (len(pairs),):
        raise ValueError(
            f"rotation has {x0.size} parameters, expected {len(pairs)}"
        )
    parity_solver = map_rows_to_solver_order(parity_canonical, permutation)
    rotation0 = params_to_U(x0, proxy_solver.n_sites, pairs)
    screening = screen_rotated_sectors(
        project_dir=project_dir,
        proxy_mps=proxy_mps,
        proxy_tag=proxy_tag,
        parity_solver=parity_solver,
        rotation_solver=rotation0,
        output_dir=output_dir / "screening",
        minimum=screening_minimum,
        maximum=screening_maximum,
        bond_dim=screening_bond_dim,
        sweeps=screening_sweeps,
        tolerance=screening_tolerance,
        threads=threads,
        sector_workers=sector_workers,
    )
    labels = [tuple(value) for value in screening["labels"]]
    objective_store = output_dir / "optimizer_mps"
    objective_store.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "objective_cache.json"
    restart_path = output_dir / "optimizer_restart.json"
    context = make_context(
        proxy_solver,
        parity_solver,
        pairs,
        objective_store,
        cache_path,
        bond_dim=int(sector_bond_dim),
        sweeps=int(sector_sweeps),
        penalty=float(sector_penalty),
        n_threads=int(threads),
        energy_tol=float(sector_energy_tolerance),
        davidson_threshold=float(sector_davidson_threshold),
        twosite_to_onesite=sector_twosite_to_onesite,
        dmrg_iprint=1,
        cleanup_mps=True,
    )
    initial_label = None
    if resume and restart_path.exists():
        saved = load_json(restart_path)
        if saved.get("rotation") is not None:
            x0 = np.asarray(saved["rotation"], dtype=float)
        if saved.get("sector") is not None:
            initial_label = tuple(int(value) for value in saved["sector"])
        print(f"[optimization] resuming from {restart_path}", flush=True)
    started = time.perf_counter()
    result, switching_history, initial_scan = (
        optimize_with_dmrg_sector_switching(
            context,
            labels,
            x0,
            maxiter=int(optimizer_maxiter),
            max_switches=int(sector_switch_maxiter),
            state_path=restart_path,
            initial_label=initial_label,
            use_analytic_gradient=True,
        )
    )
    rotation = params_to_U(result.x, proxy_solver.n_sites, pairs)
    output = {
        "schema": "quasisymmetry.equilibrium_switching_optimization",
        "version": 1,
        "rotation_parameters_solver_order": np.asarray(
            result.x, dtype=float
        ).tolist(),
        "rotation_matrix_solver_order": rotation.tolist(),
        "orbital_permutation": [int(value) for value in permutation],
        "rotation_pairs_solver_order": [list(pair) for pair in pairs],
        "parity_matrix_canonical": np.asarray(
            parity_canonical, dtype=int
        ).tolist(),
        "parity_matrix_solver_order": parity_solver.tolist(),
        "screened_sector_labels": [list(label) for label in labels],
        "sector_weights": screening["sector_weights"],
        "screening": screening,
        "selected_sector": list(result.sector_label),
        "cost_before": float(initial_scan[0][0]),
        "cost_after": float(result.fun),
        "switching_history": switching_history,
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "objective_evaluations": int(getattr(result, "nfev", 0)),
            "gradient_evaluations": int(getattr(result, "njev", 0)),
            "gradient": "analytic_rdm",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "objective_store": str(objective_store),
        "objective_cache": str(cache_path),
        "restart_state": str(restart_path),
    }
    output_path = output_dir / "optimized.json"
    atomic_json(output_path, output)
    np.savetxt(
        output_dir / "rotation_solver_order.txt",
        np.asarray(result.x, dtype=float),
    )
    print(
        f"[optimization] final sector={''.join(map(str, result.sector_label))}, "
        f"energy={result.fun:.12f} Ha, "
        f"analytic gradient evaluations={getattr(result, 'njev', 0)}",
        flush=True,
    )
    return {**output, "path": str(output_path)}


def remove_screening_scratch(optimization_output) -> None:
    """Delete temporary screening MPSs after their diagnostics are saved."""
    path = optimization_output.get("screening", {}).get("screening_store")
    if path:
        shutil.rmtree(path, ignore_errors=True)
