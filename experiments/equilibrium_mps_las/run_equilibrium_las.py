#!/usr/bin/env python3
"""Run the matched equilibrium H2O/N2 6-31G LAS experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

from clifford_outputs import build_final_clifford_outputs
from common import (
    DEFAULTS,
    SYSTEMS,
    atomic_json,
    load_json,
    map_rotation_to_canonical,
    now,
    process_rss_mib,
    project_git_commit,
    run_subprocess_stage,
    validate_artifacts,
)
from coupled_workflow import (
    run_anchor_schedule,
    run_residual_adaptive_coupling,
)
from mps_selection import run_selection
from orbital_optimization import run_switching_optimization


EXPERIMENT_VERSION = 1


def stage_banner(number, total, title) -> None:
    """Print an unbuffered workflow-stage banner."""
    print("\n" + "=" * 78, flush=True)
    print(f"[{now()}] {number}/{total} {title}  RSS={process_rss_mib():.1f} MiB", flush=True)
    print("=" * 78, flush=True)


def parse_int_list(text) -> list[int]:
    """Parse a comma-separated positive integer list."""
    values = [int(value) for value in str(text).split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def add_arguments(parser) -> None:
    """Add scientific and operational options without a system-specific namespace."""
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--proxy_mps", required=True, type=Path)
    parser.add_argument("--proxy_tag", default=None)
    parser.add_argument("--reference_result", required=True, type=Path)
    parser.add_argument("--optimized_json", default=None, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--project_dir", default=None, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--h2o_validation_summary", default=None, type=Path)
    parser.add_argument("--allow_unvalidated_n2", action="store_true")
    parser.add_argument("--determinant_result", default=None, type=Path)
    parser.add_argument("--skip_determinant_evaluator", action="store_true")
    parser.add_argument("--skip_clifford_outputs", action="store_true")
    parser.add_argument("--stop_after", choices=("validate", "optimize", "anchor", "coupled"))

    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--candidate_workers", type=int, default=None)
    parser.add_argument(
        "--sector_workers",
        type=int,
        default=1,
        help="isolated Block2 projector/Krylov workers; validate serially first",
    )
    parser.add_argument("--target_rank", type=int, default=DEFAULTS["target_rank"])
    parser.add_argument(
        "--max_macrocycles", type=int, default=DEFAULTS["max_macrocycles"]
    )
    parser.add_argument(
        "--macrocycle_energy_tolerance_mha",
        type=float,
        default=DEFAULTS["macrocycle_energy_tolerance_mha"],
    )
    parser.add_argument(
        "--min_dominant_sectors",
        type=int,
        default=DEFAULTS["min_dominant_sectors"],
    )
    parser.add_argument(
        "--max_dominant_sectors",
        type=int,
        default=DEFAULTS["max_dominant_sectors"],
    )
    parser.add_argument(
        "--sector_bond_dim", type=int, default=DEFAULTS["sector_bond_dim"]
    )
    parser.add_argument(
        "--sector_sweeps", type=int, default=DEFAULTS["sector_sweeps"]
    )
    parser.add_argument(
        "--sector_penalty", type=float, default=DEFAULTS["sector_penalty"]
    )
    parser.add_argument("--sector_energy_tolerance", type=float, default=1.0e-6)
    parser.add_argument("--sector_davidson_threshold", type=float, default=1.0e-8)
    parser.add_argument("--sector_twosite_to_onesite", type=int, default=None)
    parser.add_argument(
        "--optimizer_maxiter", type=int, default=DEFAULTS["optimizer_maxiter"]
    )
    parser.add_argument(
        "--sector_switch_maxiter",
        type=int,
        default=DEFAULTS["sector_switch_maxiter"],
    )
    parser.add_argument("--multiply_bond_dim", type=int, default=200)
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument("--multiply_tolerance", type=float, default=1.0e-10)

    parser.add_argument(
        "--anchor_bond_dims",
        type=parse_int_list,
        default=DEFAULTS["anchor_bond_dims"],
    )
    parser.add_argument(
        "--anchor_extra_bond_dim",
        type=int,
        default=DEFAULTS["anchor_extra_bond_dim"],
    )
    parser.add_argument(
        "--anchor_convergence_mha",
        type=float,
        default=DEFAULTS["anchor_convergence_mha"],
    )
    parser.add_argument(
        "--anchor_sweeps", type=int, default=DEFAULTS["anchor_sweeps"]
    )
    parser.add_argument("--anchor_energy_tolerance", type=float, default=1.0e-8)
    parser.add_argument("--anchor_davidson_threshold", type=float, default=1.0e-10)
    parser.add_argument("--anchor_twosite_to_onesite", type=int, default=None)
    parser.add_argument("--block2_stack_mem_gb", type=float, default=8.0)

    parser.add_argument(
        "--projector_beam_width",
        type=int,
        default=DEFAULTS["projector_beam_width"],
    )
    parser.add_argument(
        "--leakage_capture", type=float, default=DEFAULTS["leakage_capture"]
    )
    parser.add_argument(
        "--fit_bond_dim", type=int, default=DEFAULTS["fit_bond_dim"]
    )
    parser.add_argument(
        "--fit_extra_bond_dim",
        type=int,
        default=DEFAULTS["fit_extra_bond_dim"],
    )
    parser.add_argument("--fit_sweeps", type=int, default=DEFAULTS["fit_sweeps"])
    parser.add_argument(
        "--fit_tolerance", type=float, default=DEFAULTS["fit_tolerance"]
    )
    parser.add_argument(
        "--fit_norm_loss_tolerance",
        type=float,
        default=DEFAULTS["fit_norm_loss_tolerance"],
    )
    parser.add_argument(
        "--fit_energy_tolerance_mha",
        type=float,
        default=DEFAULTS["fit_energy_tolerance_mha"],
    )
    parser.add_argument(
        "--initial_krylov_depth",
        type=int,
        default=DEFAULTS["initial_krylov_depth"],
    )
    parser.add_argument(
        "--residual_sectors_per_cycle",
        type=int,
        default=DEFAULTS["residual_sectors_per_cycle"],
    )
    parser.add_argument(
        "--max_enrichment_cycles",
        type=int,
        default=DEFAULTS["max_enrichment_cycles"],
    )
    parser.add_argument(
        "--standard_k_cap", type=int, default=DEFAULTS["standard_k_cap"]
    )
    parser.add_argument(
        "--extended_k_cap", type=int, default=DEFAULTS["extended_k_cap"]
    )
    parser.add_argument(
        "--extended_krylov_depth",
        type=int,
        default=DEFAULTS["extended_krylov_depth"],
    )
    parser.add_argument(
        "--overlap_cutoff", type=float, default=DEFAULTS["overlap_cutoff"]
    )
    parser.add_argument(
        "--chemical_accuracy_mha",
        type=float,
        default=DEFAULTS["chemical_accuracy_mha"],
    )
    parser.add_argument(
        "--energy_change_tolerance_mha",
        type=float,
        default=DEFAULTS["energy_change_tolerance_mha"],
    )

    parser.add_argument("--determinant_max_sectors", type=int, default=48)
    parser.add_argument("--determinant_initial_depth", type=int, default=2)
    parser.add_argument("--determinant_max_dimension", type=int, default=1000)
    parser.add_argument("--determinant_chain_depth", type=int, default=4)
    parser.add_argument("--determinant_max_cycles", type=int, default=10)


def parse_args():
    """Parse the common H2O/N2 experiment interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if args.project_dir is None:
        args.project_dir = (
            Path(__file__).resolve().parents[4] / "projects" / "quasisymmetry"
        )
    return args


def require_dependencies(project_dir) -> None:
    """Fail before calculation if the cluster environment is incomplete."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    missing = []
    for module in ("numpy", "scipy", "pyscf", "pyblock2", "openfermion", "ffsim"):
        try:
            __import__(module)
        except Exception as error:
            missing.append(f"{module}: {error}")
    if missing:
        raise RuntimeError("dependency import checks failed:\n  " + "\n  ".join(missing))
    print("Dependency imports passed", flush=True)


def validate_n2_gate(args) -> dict:
    """Require a passing H2O MPS validation before the production N2 run."""
    if args.system != "n2":
        return {"required": False, "passed": True}
    if args.allow_unvalidated_n2:
        return {"required": True, "passed": False, "overridden": True}
    if args.h2o_validation_summary is None:
        raise ValueError(
            "N2 is gated on H2O validation; provide --h2o_validation_summary "
            "or use --allow_unvalidated_n2 only for a non-production smoke test"
        )
    data = load_json(args.h2o_validation_summary)
    passed = bool(data.get("validation", {}).get("h2o_mps_validation_passed"))
    if not passed:
        raise ValueError("the supplied H2O summary did not pass the MPS validation gate")
    return {
        "required": True,
        "passed": True,
        "summary": str(args.h2o_validation_summary.resolve()),
    }


def selection_result(path) -> dict:
    """Restore the compact return value of a completed selection stage."""
    data = load_json(path)
    return {
        "selection": str(Path(path)),
        "parity": data["outputs"]["parity_matrix"],
        "manifest": data["outputs"]["symmetry_manifest"],
        "row_space": data["metadata"]["selected_row_space"],
        "selected_count": len(data["selected"]),
        "candidate_count": int(data["candidate_count"]),
        "score_seconds": float(data["metadata"]["candidate_scoring_seconds"]),
        "solver_parity": np.asarray(
            data["parity_matrix_solver_order"], dtype=int
        ),
    }


def run_or_reuse_selection(args, artifacts, rotation_solver, output_dir) -> dict:
    """Run MPS-native NC selection or reload its completed JSON."""
    selection_path = Path(output_dir) / "selection.json"
    if args.resume and selection_path.exists():
        print(f"[selection] reusing {selection_path}", flush=True)
        return selection_result(selection_path)
    workers = (
        SYSTEMS[args.system]["candidate_workers"]
        if args.candidate_workers is None
        else int(args.candidate_workers)
    )
    return run_selection(
        project_dir=args.project_dir,
        checkpoint=args.checkpoint,
        proxy_mps=args.proxy_mps,
        proxy_tag=artifacts["proxy_mps"]["selected_tag"],
        permutation=artifacts["proxy_mps"]["orbital_permutation"],
        rotation_solver=rotation_solver,
        output_dir=output_dir,
        target_rank=args.target_rank,
        workers=workers,
        total_threads=args.threads,
        multiply_bond_dim=args.multiply_bond_dim,
        multiply_sweeps=args.multiply_sweeps,
        multiply_tolerance=args.multiply_tolerance,
    )


def initial_rotation(args, artifacts):
    """Return zero parameters and identity in the saved proxy orbital order."""
    project_dir = str(args.project_dir.resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from src.dmrg_solver import Block2DMRGSolver
    from src.orbital_rotation import params_to_U
    from orbital_optimization import solver_rotation_pairs

    solver = Block2DMRGSolver.load(args.proxy_mps, n_threads=args.threads)
    pairs = solver_rotation_pairs(solver)
    expected = SYSTEMS[args.system]["rotation_parameters"]
    if len(pairs) != expected:
        raise ValueError(f"proxy MPS has {len(pairs)} rotation pairs, expected {expected}")
    parameters = np.zeros(len(pairs), dtype=float)
    return parameters, params_to_U(parameters, solver.n_sites, pairs), pairs


def row_space(selection) -> tuple[tuple[int, ...], ...]:
    """Canonical comparable GF(2) row-space key."""
    return tuple(tuple(int(bit) for bit in row) for row in selection["row_space"])


def write_final_optimized(
    args,
    artifacts,
    optimization,
    selection,
    macrocycles,
    stable,
) -> dict:
    """Write the self-contained optimized-frame input used by both evaluators."""
    rotation_solver = np.asarray(
        optimization["rotation_matrix_solver_order"], dtype=float
    )
    rotation_canonical = map_rotation_to_canonical(
        rotation_solver,
        artifacts["proxy_mps"]["orbital_permutation"],
    )
    output = {
        "schema": "quasisymmetry.equilibrium_final_optimized",
        "version": 1,
        "system": args.system,
        "checkpoint": str(args.checkpoint.resolve()),
        "proxy_mps": str(args.proxy_mps.resolve()),
        "proxy_tag": artifacts["proxy_mps"]["selected_tag"],
        "parity": str(Path(selection["parity"]).resolve()),
        "symmetry_manifest": str(Path(selection["manifest"]).resolve()),
        "parity_matrix_canonical": np.atleast_2d(
            np.loadtxt(selection["parity"], dtype=int)
        ).tolist(),
        "parity_matrix_solver_order": np.asarray(
            optimization["parity_matrix_solver_order"], dtype=int
        ).tolist(),
        "rotation_parameters_solver_order": optimization[
            "rotation_parameters_solver_order"
        ],
        "rotation_matrix_solver_order": rotation_solver.tolist(),
        "rotation_matrix_canonical": rotation_canonical.tolist(),
        "orbital_permutation": artifacts["proxy_mps"]["orbital_permutation"],
        "selected_sector": optimization["selected_sector"],
        "decoupled_energy": float(optimization["cost_after"]),
        "row_space": selection["row_space"],
        "row_space_and_energy_stable": bool(stable),
        "macrocycles": macrocycles,
        "optimization_result": optimization["path"],
        "selection_result": selection["selection"],
    }
    output_path = args.run_dir / "final_optimized.json"
    atomic_json(output_path, output)
    np.savetxt(
        args.run_dir / "final_rotation_canonical.txt",
        rotation_canonical,
    )
    return {**output, "path": str(output_path)}


def adapt_existing_optimized(args, artifacts) -> dict:
    """Adapt an existing project or experiment optimization JSON to this workflow."""
    data = load_json(args.optimized_json)
    project_dir = str(args.project_dir.resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    permutation = np.asarray(
        artifacts["proxy_mps"]["orbital_permutation"], dtype=int
    )
    norb = SYSTEMS[args.system]["norb"]
    if data.get("rotation_matrix_solver_order") is not None:
        rotation_solver = np.asarray(
            data["rotation_matrix_solver_order"], dtype=float
        )
        rotation_canonical = np.asarray(
            data.get(
                "rotation_matrix_canonical",
                map_rotation_to_canonical(rotation_solver, permutation),
            ),
            dtype=float,
        )
    else:
        from src.orbital_rotation import rotation_from_oo_data

        rotation_canonical = rotation_from_oo_data(data, norb)
        rotation_solver = rotation_canonical[np.ix_(permutation, permutation)]
    parity_path = data.get("parity")
    manifest_path = data.get("symmetry_manifest")
    if not parity_path or not manifest_path:
        raise ValueError("existing optimized JSON must provide parity and symmetry_manifest")
    parity_canonical = np.atleast_2d(np.loadtxt(parity_path, dtype=int))
    from common import map_rows_to_solver_order

    output = {
        "schema": "quasisymmetry.equilibrium_final_optimized",
        "version": 1,
        "system": args.system,
        "checkpoint": str(args.checkpoint.resolve()),
        "proxy_mps": str(args.proxy_mps.resolve()),
        "proxy_tag": artifacts["proxy_mps"]["selected_tag"],
        "parity": str(Path(parity_path).resolve()),
        "symmetry_manifest": str(Path(manifest_path).resolve()),
        "parity_matrix_canonical": parity_canonical.tolist(),
        "parity_matrix_solver_order": (
            map_rows_to_solver_order(parity_canonical, permutation).tolist()
        ),
        "rotation_parameters_solver_order": data.get(
            "rotation_parameters_solver_order"
        ),
        "rotation_matrix_solver_order": rotation_solver.tolist(),
        "rotation_matrix_canonical": rotation_canonical.tolist(),
        "orbital_permutation": permutation.tolist(),
        "selected_sector": data["selected_sector"],
        "decoupled_energy": float(
            data.get("decoupled_energy", data.get("cost_after", data.get("cost")))
        ),
        "row_space": data.get("row_space"),
        "row_space_and_energy_stable": data.get("row_space_and_energy_stable"),
        "reused_optimized_json": str(args.optimized_json.resolve()),
    }
    output_path = args.run_dir / "final_optimized.json"
    atomic_json(output_path, output)
    np.savetxt(args.run_dir / "final_rotation_canonical.txt", rotation_canonical)
    return {**output, "path": str(output_path)}


def select_and_optimize(args, artifacts, state, state_path) -> dict:
    """Alternate selection and switching-sector optimization for at most three cycles."""
    if args.optimized_json is not None:
        print(f"[optimization] reusing explicit {args.optimized_json}", flush=True)
        return adapt_existing_optimized(args, artifacts)

    parameters, rotation, _pairs = initial_rotation(args, artifacts)
    macrocycles = list(state.get("macrocycles", [])) if args.resume else []
    previous_energy = None
    pending_selection = None
    stable = False
    final_optimization = None
    final_selection = None

    for cycle in range(1, int(args.max_macrocycles) + 1):
        cycle_dir = args.run_dir / f"macrocycle_{cycle}"
        if args.resume:
            completed = next(
                (
                    item
                    for item in macrocycles
                    if int(item.get("cycle", -1)) == cycle
                    and item.get("status") == "complete"
                ),
                None,
            )
        else:
            completed = None
        if completed:
            print(f"[macrocycle {cycle}] reusing completed cycle", flush=True)
            pre = selection_result(completed["pre_selection"])
            post = selection_result(completed["post_selection"])
            optimization = load_json(completed["optimization"])
            optimization["path"] = completed["optimization"]
            parameters = np.asarray(
                optimization["rotation_parameters_solver_order"], dtype=float
            )
            rotation = np.asarray(
                optimization["rotation_matrix_solver_order"], dtype=float
            )
            previous_energy = float(optimization["cost_after"])
            stable = bool(completed["stable"])
            final_optimization, final_selection = optimization, pre
            pending_selection = post
            if stable:
                break
            continue

        print("\n" + "-" * 78, flush=True)
        print(f"[macrocycle {cycle}] selection -> optimization -> reselection", flush=True)
        pre = pending_selection or run_or_reuse_selection(
            args, artifacts, rotation, cycle_dir / "selection_before"
        )
        parity_canonical = np.atleast_2d(np.loadtxt(pre["parity"], dtype=int))
        optimization_path = cycle_dir / "optimization" / "optimized.json"
        if args.resume and optimization_path.exists():
            optimization = load_json(optimization_path)
            optimization["path"] = str(optimization_path)
            print(f"[optimization] reusing {optimization_path}", flush=True)
        else:
            optimization = run_switching_optimization(
                project_dir=args.project_dir,
                proxy_mps=args.proxy_mps,
                proxy_tag=artifacts["proxy_mps"]["selected_tag"],
                parity_canonical=parity_canonical,
                permutation=artifacts["proxy_mps"]["orbital_permutation"],
                x0=parameters,
                output_dir=cycle_dir / "optimization",
                threads=args.threads,
                sector_workers=args.sector_workers,
                screening_minimum=args.min_dominant_sectors,
                screening_maximum=args.max_dominant_sectors,
                screening_bond_dim=args.multiply_bond_dim,
                screening_sweeps=args.multiply_sweeps,
                screening_tolerance=args.multiply_tolerance,
                sector_bond_dim=args.sector_bond_dim,
                sector_sweeps=args.sector_sweeps,
                sector_penalty=args.sector_penalty,
                sector_energy_tolerance=args.sector_energy_tolerance,
                sector_davidson_threshold=args.sector_davidson_threshold,
                sector_twosite_to_onesite=args.sector_twosite_to_onesite,
                optimizer_maxiter=args.optimizer_maxiter,
                sector_switch_maxiter=args.sector_switch_maxiter,
                resume=args.resume,
            )
        parameters = np.asarray(
            optimization["rotation_parameters_solver_order"], dtype=float
        )
        rotation = np.asarray(
            optimization["rotation_matrix_solver_order"], dtype=float
        )
        post = run_or_reuse_selection(
            args, artifacts, rotation, cycle_dir / "selection_after"
        )
        row_stable = row_space(pre) == row_space(post)
        energy = float(optimization["cost_after"])
        energy_change_mha = (
            None
            if previous_energy is None
            else abs(energy - previous_energy) * 1000.0
        )
        energy_stable = (
            energy_change_mha is not None
            and energy_change_mha <= float(args.macrocycle_energy_tolerance_mha)
        )
        stable = row_stable and energy_stable
        record = {
            "cycle": cycle,
            "status": "complete",
            "pre_selection": pre["selection"],
            "optimization": optimization["path"],
            "post_selection": post["selection"],
            "pre_row_space": [list(row) for row in row_space(pre)],
            "post_row_space": [list(row) for row in row_space(post)],
            "row_space_stable": row_stable,
            "decoupled_energy": energy,
            "energy_change_mha": energy_change_mha,
            "energy_stable": energy_stable,
            "stable": stable,
        }
        macrocycles = [
            item for item in macrocycles if int(item.get("cycle", -1)) != cycle
        ]
        macrocycles.append(record)
        macrocycles.sort(key=lambda item: int(item["cycle"]))
        state["macrocycles"] = macrocycles
        atomic_json(state_path, state)
        print(
            f"[macrocycle {cycle}] row_stable={row_stable}, "
            f"dE={energy_change_mha} mHa, converged={stable}",
            flush=True,
        )
        final_optimization, final_selection = optimization, pre
        pending_selection = post
        previous_energy = energy
        if stable:
            break

    if final_optimization is None or final_selection is None:
        raise RuntimeError("no orbital-optimization macrocycle completed")
    return write_final_optimized(
        args,
        artifacts,
        final_optimization,
        final_selection,
        macrocycles,
        stable,
    )


def canonical_parameters(rotation, checkpoint, project_dir) -> tuple[np.ndarray, np.ndarray]:
    """Recover canonical irrep-packed parameters for the determinant evaluator."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from src.orbital_rotation import params_to_U, resolve_orbital_rotation

    rotation = np.asarray(rotation, dtype=float)
    pairs, irreps = resolve_orbital_rotation("irrep", checkpoint, len(rotation))
    generator = scipy.linalg.logm(rotation)
    if np.max(np.abs(generator.imag)) > 1.0e-8:
        raise ValueError("canonical rotation has no stable real logarithm")
    generator = np.asarray(generator.real)
    parameters = np.asarray([generator[i, j] for i, j in pairs], dtype=float)
    reconstructed = params_to_U(parameters, len(rotation), pairs)
    if not np.allclose(reconstructed, rotation, atol=1.0e-7):
        raise ValueError("could not recover canonical irrep rotation parameters")
    return parameters, np.asarray(irreps, dtype=int)


def determinant_compatibility_json(args, optimized) -> Path:
    """Write the legacy input consumed by the successful H2O evaluator."""
    rotation = np.asarray(optimized["rotation_matrix_canonical"], dtype=float)
    parameters, irreps = canonical_parameters(
        rotation, args.checkpoint, args.project_dir
    )
    output = {
        "molpath": str(args.checkpoint.resolve()),
        "parity": optimized["parity"],
        "symmetry_manifest": optimized["symmetry_manifest"],
        "orbital_rotation": "irrep",
        "irreps": irreps.tolist(),
        "rotation": parameters.tolist(),
        "selected_sector": optimized["selected_sector"],
    }
    path = args.run_dir / "determinant_evaluator" / "optimized_compatibility.json"
    atomic_json(path, output)
    return path


def run_h2o_determinant_evaluator(args, optimized, state, state_path) -> Path | None:
    """Run or reuse the successful determinant Krylov/residual comparator."""
    if args.system != "h2o":
        return None
    if args.determinant_result is not None:
        return args.determinant_result.resolve()
    if args.skip_determinant_evaluator:
        return None
    compatibility = determinant_compatibility_json(args, optimized)
    root = args.run_dir / "determinant_evaluator"
    initial = root / "initial_krylov_metrics.json"
    refined = root / "residual_metrics.json"
    command = [
        sys.executable,
        "-u",
        str(args.project_dir / "selected_clifford_krylov.py"),
        str(compatibility),
        "--reference_result",
        str(args.reference_result),
        "--work_dir",
        str(root / "initial_krylov"),
        "--outname",
        str(initial),
        "--max_sectors",
        str(args.determinant_max_sectors),
        "--krylov_depths",
        str(args.determinant_initial_depth),
        "--skip_lcu_files",
    ]
    if args.resume:
        command.append("--resume")
    run_subprocess_stage(
        state_path,
        state,
        "h2o_determinant_initial_krylov",
        command,
        [initial],
        args.resume,
        args.project_dir,
    )
    command = [
        sys.executable,
        "-u",
        str(args.project_dir / "selected_clifford_residual_enrichment.py"),
        "--run_dir",
        str(root),
        "--source_metrics",
        str(initial),
        "--work_dir",
        str(root / "residual_enrichment"),
        "--outname",
        str(refined),
        "--max_macrocycles",
        str(args.determinant_max_cycles),
        "--chain_depth",
        str(args.determinant_chain_depth),
        "--max_dimension",
        str(args.determinant_max_dimension),
    ]
    if args.resume:
        command.append("--resume")
    run_subprocess_stage(
        state_path,
        state,
        "h2o_determinant_residual_enrichment",
        command,
        [refined],
        args.resume,
        args.project_dir,
    )
    return refined


def extract_determinant_metrics(path) -> dict:
    """Read energy/K/sector count across saved determinant evaluator schemas."""
    data = load_json(path)
    energy = None
    for key in ("coupled_energy", "E_coupled", "energy"):
        if data.get(key) is not None:
            energy = float(data[key])
            break
    history = data.get("history") or data.get("cycles") or []
    if energy is None and history:
        energy = float(history[-1].get("energy_after", history[-1].get("energy")))
    if energy is None:
        raise ValueError(f"no determinant coupled energy in {path}")
    k = data.get("K", data.get("coupled_dimension"))
    if k is None and history:
        k = history[-1].get("K_after", history[-1].get("K"))
    sectors = data.get("sector_count", data.get("sector_label_count"))
    if sectors is None and history:
        sectors = history[-1].get("sector_count")
    return {
        "path": str(Path(path).resolve()),
        "energy": energy,
        "K": None if k is None else int(k),
        "sector_count": None if sectors is None else int(sectors),
    }


def certified_determinant_baseline() -> dict:
    """Return the prior successful H2O determinant result for comparison."""
    data = SYSTEMS["h2o"]
    return {
        "source": "certified_prior_successful_determinant_evaluator",
        "energy": float(data["determinant_coupled_energy"]),
        "K": int(data["determinant_coupled_k"]),
        "sector_count": int(data["determinant_sector_count"]),
    }


def write_summary(args, artifacts, optimized, anchor, coupled, determinant) -> dict:
    """Write the final scientific gates and concise Markdown report."""
    reference = float(artifacts["reference_result"]["energy"])
    anchor_error = (float(anchor["decoupled_energy"]) - reference) * 1000.0
    validation = {
        "chemical_accuracy_threshold_mha": float(args.chemical_accuracy_mha),
        "anchor_error_mha": anchor_error,
        "mps_coupled_error_mha": float(coupled["coupled_error_mha"]),
        "mps_chemical_accuracy": bool(coupled["chemical_accuracy"]),
        "variational_monotonicity": bool(coupled["variational_monotonicity"]),
        "hamiltonian_hermiticity_passed": coupled[
            "hamiltonian_hermiticity_error"
        ]
        <= 1.0e-10,
        "overlap_hermiticity_passed": coupled["overlap_hermiticity_error"]
        <= 1.0e-10,
        "anchor_bond_dimension_converged": bool(
            anchor["anchor_converged"]
        ),
    }
    if args.system == "h2o":
        comparator = determinant or certified_determinant_baseline()
        agreement_mha = abs(
            float(coupled["coupled_energy"]) - float(comparator["energy"])
        ) * 1000.0
        validation.update(
            {
                "determinant_comparator": comparator,
                "mps_determinant_agreement_mha": agreement_mha,
                "mps_determinant_agreement_passed": agreement_mha <= 0.2,
                "h2o_mps_validation_passed": bool(coupled["chemical_accuracy"])
                and agreement_mha <= 0.2
                and bool(coupled["variational_monotonicity"])
                and bool(anchor["anchor_converged"]),
            }
        )
    else:
        validation["n2_chemical_accuracy_passed"] = bool(
            coupled["chemical_accuracy"]
        ) and bool(anchor["anchor_converged"])
    output = {
        "schema": "quasisymmetry.equilibrium_las_summary",
        "version": 1,
        "system": args.system,
        "reference_energy": reference,
        "optimized_json": optimized["path"],
        "anchor_summary": str(args.run_dir / "anchor" / "anchor_summary.json"),
        "coupled_summary": str(
            args.run_dir / "mps_coupled" / "coupled_summary.json"
        ),
        "decoupled_energy": float(anchor["decoupled_energy"]),
        "coupled_energy": float(coupled["coupled_energy"]),
        "K": int(coupled["K"]),
        "sector_count": int(coupled["sector_count"]),
        "validation": validation,
        "finished": now(),
    }
    atomic_json(args.run_dir / "summary.json", output)
    lines = [
        f"# {SYSTEMS[args.system]['display_name']}/6-31G Equilibrium LAS",
        "",
        f"- Reference energy: {reference:.12f} Ha",
        f"- Decoupled energy: {anchor['decoupled_energy']:.12f} Ha",
        f"- Decoupled error: {anchor_error:.6f} mHa",
        f"- MPS coupled energy: {coupled['coupled_energy']:.12f} Ha",
        f"- MPS coupled error: {coupled['coupled_error_mha']:.6f} mHa",
        f"- Coupled basis: K={coupled['K']}, sectors={coupled['sector_count']}",
        f"- Chemical accuracy: {coupled['chemical_accuracy']}",
        f"- Anchor bond-dimension converged: {anchor['anchor_converged']}",
        f"- Variational monotonicity: {coupled['variational_monotonicity']}",
    ]
    if args.system == "h2o":
        lines.extend(
            [
                f"- MPS/determinant agreement: "
                f"{validation['mps_determinant_agreement_mha']:.6f} mHa",
                f"- H2O validation gate passed: "
                f"{validation['h2o_mps_validation_passed']}",
            ]
        )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    """Execute the restartable matched equilibrium workflow."""
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.proxy_mps = args.proxy_mps.resolve()
    args.reference_result = args.reference_result.resolve()
    args.run_dir = args.run_dir.resolve()
    args.project_dir = args.project_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "workflow_state.json"
    state = load_json(state_path, {}) if args.resume else {}
    state.setdefault("schema", "quasisymmetry.equilibrium_las_workflow_state")
    state.setdefault("version", EXPERIMENT_VERSION)
    state.setdefault("run_dir", str(args.run_dir))
    state.setdefault("started", now())

    print("System:", args.system, flush=True)
    print("Project:", args.project_dir, flush=True)
    print("Run directory:", args.run_dir, flush=True)
    print("Python:", sys.executable, flush=True)
    print("Threads:", args.threads, flush=True)
    require_dependencies(args.project_dir)

    stage_banner(1, 8, "Validate reusable parent artifacts and N2 gate")
    artifacts = validate_artifacts(
        args.system,
        args.checkpoint,
        args.proxy_mps,
        args.reference_result,
        args.project_dir,
        proxy_tag=args.proxy_tag,
    )
    gate = validate_n2_gate(args)
    provenance = {
        "schema": "quasisymmetry.equilibrium_las_configuration",
        "version": EXPERIMENT_VERSION,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "system": SYSTEMS[args.system],
        "artifacts": artifacts,
        "n2_validation_gate": gate,
        "shared_project_git_commit": project_git_commit(args.project_dir),
        "experiment_files": str(Path(__file__).resolve().parent),
        "created": now(),
    }
    existing = load_json(args.run_dir / "configuration.json", {})
    if args.resume and existing:
        old_artifacts = existing.get("artifacts", {})
        for section, key in (
            ("checkpoint", "sha256"),
            ("proxy_mps", "integrals_sha256"),
            ("reference_result", "sha256"),
        ):
            if old_artifacts.get(section, {}).get(key) != artifacts[section][key]:
                raise ValueError(f"resume artifact fingerprint changed: {section}.{key}")
    atomic_json(args.run_dir / "configuration.json", provenance)
    state["artifacts_validated"] = True
    atomic_json(state_path, state)
    if args.stop_after == "validate":
        return

    stage_banner(2, 8, "MPS-native NC selection and switching-sector optimization")
    optimized = select_and_optimize(args, artifacts, state, state_path)
    state["optimized_json"] = optimized["path"]
    atomic_json(state_path, state)
    if args.stop_after == "optimize":
        return

    stage_banner(3, 8, "High-fidelity decoupled anchor DMRG")
    parity_solver = np.atleast_2d(
        np.asarray(optimized["parity_matrix_solver_order"], dtype=int)
    )
    anchor = run_anchor_schedule(
        project_dir=args.project_dir,
        proxy_mps=args.proxy_mps,
        rotation_solver=np.asarray(
            optimized["rotation_matrix_solver_order"], dtype=float
        ),
        parity_solver=parity_solver,
        anchor_label=optimized["selected_sector"],
        output_dir=args.run_dir / "anchor",
        threads=args.threads,
        bond_dimensions=args.anchor_bond_dims,
        extra_bond_dimension=args.anchor_extra_bond_dim,
        convergence_tolerance_mha=args.anchor_convergence_mha,
        sweeps=args.anchor_sweeps,
        penalty=args.sector_penalty,
        energy_tolerance=args.anchor_energy_tolerance,
        davidson_threshold=args.anchor_davidson_threshold,
        twosite_to_onesite=args.anchor_twosite_to_onesite,
        stack_mem_gb=args.block2_stack_mem_gb,
        resume=args.resume,
    )
    solver = anchor.pop("solver")
    if args.stop_after == "anchor":
        return

    stage_banner(4, 8, "MPS projector beam and coupling-seeded Krylov basis")
    stage_banner(5, 8, "Generalized coupled solve and residual enrichment")
    coupled = run_residual_adaptive_coupling(
        solver=solver,
        parity_solver=parity_solver,
        anchor_tag=anchor["final_tag"],
        anchor_label=optimized["selected_sector"],
        decoupled_energy=anchor["decoupled_energy"],
        reference_energy=artifacts["reference_result"]["energy"],
        output_dir=args.run_dir / "mps_coupled",
        initial_beam_width=args.projector_beam_width,
        minimum_capture=args.leakage_capture,
        fit_bond_dim=args.fit_bond_dim,
        fit_extra_bond_dim=args.fit_extra_bond_dim,
        fit_sweeps=args.fit_sweeps,
        fit_tolerance=args.fit_tolerance,
        fit_norm_loss_tolerance=args.fit_norm_loss_tolerance,
        fit_energy_tolerance_mha=args.fit_energy_tolerance_mha,
        initial_krylov_depth=args.initial_krylov_depth,
        residual_sectors_per_cycle=args.residual_sectors_per_cycle,
        max_cycles=args.max_enrichment_cycles,
        standard_k_cap=args.standard_k_cap,
        extended_k_cap=args.extended_k_cap,
        extended_krylov_depth=args.extended_krylov_depth,
        overlap_cutoff=args.overlap_cutoff,
        chemical_accuracy_mha=args.chemical_accuracy_mha,
        energy_change_tolerance_mha=args.energy_change_tolerance_mha,
        resume=args.resume,
        project_dir=args.project_dir,
        sector_workers=args.sector_workers,
        total_threads=args.threads,
    )
    if args.stop_after == "coupled":
        return

    stage_banner(6, 8, "H2O determinant evaluator baseline")
    determinant_path = run_h2o_determinant_evaluator(
        args, optimized, state, state_path
    )
    determinant = (
        extract_determinant_metrics(determinant_path)
        if determinant_path is not None
        else None
    )

    stage_banner(7, 8, "Final selected-sector Clifford Pauli LCUs")
    if args.skip_clifford_outputs:
        print("[Clifford] explicitly skipped", flush=True)
    else:
        build_final_clifford_outputs(
            project_dir=args.project_dir,
            checkpoint=args.checkpoint,
            rotation_canonical=np.asarray(
                optimized["rotation_matrix_canonical"], dtype=float
            ),
            parity_canonical=np.asarray(
                optimized["parity_matrix_canonical"], dtype=int
            ),
            symmetry_manifest=optimized["symmetry_manifest"],
            coupled_summary=args.run_dir / "mps_coupled" / "coupled_summary.json",
            coupled_matrix_path=args.run_dir
            / "mps_coupled"
            / "coupled_matrices.npz",
            output_dir=args.run_dir / "clifford_lcus",
        )

    stage_banner(8, 8, "Validation gates and final summary")
    summary = write_summary(
        args, artifacts, optimized, anchor, coupled, determinant
    )
    state["status"] = "complete"
    state["finished"] = now()
    state["summary"] = str(args.run_dir / "summary.json")
    atomic_json(state_path, state)
    print("\nCompleted:", args.run_dir / "summary.md", flush=True)
    print(
        f"Final coupled energy = {summary['coupled_energy']:.12f} Ha; "
        f"K={summary['K']}; sectors={summary['sector_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
