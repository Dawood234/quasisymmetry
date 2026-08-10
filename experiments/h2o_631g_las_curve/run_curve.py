#!/usr/bin/env python3
"""Restartable H2O/6-31G fixed-versus-reselected decoupled LAS curve."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EQUILIBRIUM_MODULES = HERE.parent / "equilibrium_mps_las"
LOCAL_PROJECT_ROOT = HERE.parent.parent
for module_dir in (HERE, EQUILIBRIUM_MODULES, LOCAL_PROJECT_ROOT):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from exact_quotient import (  # noqa: E402
    exact_parity_rows,
    exact_sector_dimension_audit,
    quotient_row_space,
    select_quotient_independent,
    selection_diagnostics,
)


DEFAULT_BONDS = (0.7, 0.8, 0.9, 0.958, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
DEFAULT_CASSCF_SPACES = "2e2o,4e4o,6e6o,8e8o,8e10o,8e12o"


def parse_csv_floats(text):
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def parse_csv_ints(text):
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--prepare-fixed-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bond-lengths", type=parse_csv_floats, default=DEFAULT_BONDS)
    parser.add_argument("--target-rank", type=int, default=7)
    parser.add_argument("--proxy-tag", default="M100")
    parser.add_argument("--candidate-workers", type=int, default=10)
    parser.add_argument("--candidate-threads", type=int, default=192)
    parser.add_argument("--multiply-bond-dim", type=int, default=100)
    parser.add_argument("--multiply-sweeps", type=int, default=8)
    parser.add_argument("--dmrg-threads", type=int, default=32)
    parser.add_argument("--sector-bond-dim", type=int, default=100)
    parser.add_argument("--sector-sweeps", type=int, default=6)
    parser.add_argument("--sector-penalty", type=float, default=30.0)
    parser.add_argument("--sector-energy-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--sector-davidson-threshold", type=float, default=1.0e-8)
    parser.add_argument("--screen-minimum", type=int, default=8)
    parser.add_argument("--screen-maximum", type=int, default=16)
    parser.add_argument("--optimizer-maxiter", type=int, default=8)
    parser.add_argument("--sector-switch-maxiter", type=int, default=2)
    parser.add_argument("--max-macrocycles", type=int, default=3)
    parser.add_argument("--macrocycle-energy-tolerance-mha", type=float, default=0.1)
    parser.add_argument("--final-bond-dims", type=parse_csv_ints, default=(100, 200, 350, 500))
    parser.add_argument("--final-sweeps", type=int, default=10)
    parser.add_argument("--final-m-tolerance-mha", type=float, default=0.2)
    parser.add_argument("--casscf-script", type=Path)
    parser.add_argument("--casscf-spaces", default=DEFAULT_CASSCF_SPACES)
    parser.add_argument("--casscf-threads", type=int, default=16)
    parser.add_argument("--skip-casscf", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def geometry_key(bond):
    return f"r_{float(bond):.4f}".replace(".", "p")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def print_stage(text):
    print("\n" + "=" * 78, flush=True)
    print(f"[{now()}] {text}", flush=True)
    print("=" * 78, flush=True)


def resolve_point(parent_root, bond, proxy_tag):
    """Resolve checkpoint, proxy MPS, and authoritative post-DMRG energy."""
    point_dir = Path(parent_root) / "points" / geometry_key(bond)
    if not point_dir.is_dir():
        raise FileNotFoundError(point_dir)
    reuse_path = point_dir / "reference_reuse.json"
    source_dir = point_dir
    if reuse_path.exists():
        reuse = read_json(reuse_path)
        checkpoint = Path(reuse["source_checkpoint"])
        proxy_mps = Path(reuse["source_mps_directory"])
        source_dir = checkpoint.parent
    else:
        checkpoints = sorted(point_dir.glob("*.chk"))
        if len(checkpoints) != 1:
            raise ValueError(f"expected one checkpoint in {point_dir}, found {checkpoints}")
        checkpoint = checkpoints[0]
        proxy_mps = point_dir / "parent_mps"

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata_path = proxy_mps / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = read_json(metadata_path)
    if proxy_tag not in metadata.get("runs", {}):
        raise ValueError(f"proxy tag {proxy_tag} is absent from {proxy_mps}")

    stage_files = sorted((source_dir / "production" / "stages").glob("M*.json"))
    stage_results = [read_json(path) for path in stage_files]
    stage_results = [item for item in stage_results if "energy" in item]
    if stage_results:
        reference_result = max(
            stage_results,
            key=lambda item: int(item.get("config", {}).get("max_bond_dim", 0)),
        )
        reference_energy = float(reference_result["energy"])
        reference_bond_dim = int(reference_result["config"]["max_bond_dim"])
    else:
        summary = read_json(point_dir / "parent_reference_summary.json")
        reference_energy = float(summary["stages"][-1]["energy_Ha"])
        reference_bond_dim = int(summary["stages"][-1]["max_bond_dim"])

    system = metadata.get("system", {})
    permutation = tuple(int(value) for value in system["orbital_permutation"])
    if sorted(permutation) != list(range(13)):
        raise ValueError("parent MPS orbital permutation is invalid")
    return {
        "bond_length_angstrom": float(bond),
        "point_dir": str(point_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "proxy_mps": str(proxy_mps.resolve()),
        "proxy_tag": str(proxy_tag),
        "proxy_energy_ha": float(metadata["runs"][proxy_tag]["energy"]),
        "proxy_bond_dim": int(metadata["runs"][proxy_tag]["config"]["max_bond_dim"]),
        "reference_energy_ha": reference_energy,
        "reference_bond_dim": reference_bond_dim,
        "orbital_permutation": list(permutation),
        "orbital_symmetries_solver_order": list(system.get("orbital_symmetries", [])),
        "symmetry_mode": system.get("symmetry_mode"),
    }


def checkpoint_data(project_dir, checkpoint):
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from chemistry import fcidump_data
    from src.orbital_rotation import resolve_orbital_rotation

    data = fcidump_data(str(checkpoint))
    pairs, irreps = resolve_orbital_rotation("irrep", checkpoint, 13)
    nelec = data["NELEC"]
    if np.iterable(nelec):
        nalpha, nbeta = (int(value) for value in nelec)
    else:
        total = int(nelec)
        spin = int(data.get("MS2", 0))
        nalpha, nbeta = (total + spin) // 2, (total - spin) // 2
    if (nalpha, nbeta) != (5, 5) or len(pairs) != 28:
        raise ValueError("expected H2O/6-31G with (5,5) electrons and 28 C2v rotations")
    return {
        "nalpha": nalpha,
        "nbeta": nbeta,
        "orbital_irreps_canonical": np.asarray(irreps, dtype=int).tolist(),
        "target_irrep": int(data.get("PG_IRREP", 0)),
        "rotation_parameter_count": len(pairs),
    }


def exact_filter_selection(raw_result, checkpoint, canonical_irreps, target_rank, output_dir):
    """Replace raw GF(2) selection by independence modulo exact symmetries."""
    output_dir = Path(output_dir)
    output_path = output_dir / "exact_quotient_selection.json"
    parity_path = output_dir / "parity_matrix.txt"
    raw_data = read_json(raw_result["selection"])
    candidates = raw_data["ordered_candidates"]
    exact_rows = exact_parity_rows(canonical_irreps)
    selected = select_quotient_independent(candidates, target_rank, exact_rows, 13)
    parity = np.asarray([item["row"] for item in selected], dtype=int)
    diagnostics = selection_diagnostics(selected, exact_rows, 13)
    row_space = quotient_row_space(parity, exact_rows, 13)
    serialized = []
    for item, remainder in zip(
        selected, diagnostics["quotient_remainders_spin_orbital"]
    ):
        serialized.append(
            {
                "label": item["label"],
                "family": item["family"],
                "support": list(item["support"]),
                "score": float(item["score"]),
                "row": np.asarray(item["row"], dtype=int).tolist(),
                "quotient_remainder_spin_orbital": remainder,
            }
        )
    output = {
        "schema": "alris.h2o_631g_exact_quotient_selection",
        "version": 1,
        "checkpoint": str(Path(checkpoint).resolve()),
        "raw_score_result": str(Path(raw_result["selection"]).resolve()),
        "candidate_family": "seniority_plus_quartet",
        "target_effective_rank": int(target_rank),
        "selected": serialized,
        "parity_matrix": parity.tolist(),
        "exact_rows_spin_orbital": exact_rows.tolist(),
        "joint_row_space": [list(row) for row in row_space],
        "diagnostics": diagnostics,
    }
    np.savetxt(parity_path, parity, fmt="%d")
    atomic_json(output_path, output)
    print(
        f"[exact quotient] selected effective rank {diagnostics['effective_las_rank']} "
        f"from exact rank {diagnostics['exact_rank']}",
        flush=True,
    )
    for item in serialized:
        print(f"[exact quotient] {item['label']} score={item['score']:.12e}", flush=True)
    return {
        "selection": str(output_path),
        "parity": str(parity_path),
        "parity_matrix": parity.tolist(),
        "row_space": [list(row) for row in row_space],
        "selected": serialized,
        "diagnostics": diagnostics,
    }


def run_nc_selection(args, artifacts, rotation_solver, output_dir):
    from mps_selection import run_selection

    output_dir = Path(output_dir)
    completed = output_dir / "exact_quotient_selection.json"
    if args.resume and completed.exists():
        data = read_json(completed)
        print(f"[selection] reusing {completed}", flush=True)
        return {
            "selection": str(completed),
            "parity": str(output_dir / "parity_matrix.txt"),
            "parity_matrix": data["parity_matrix"],
            "row_space": data["joint_row_space"],
            "selected": data["selected"],
            "diagnostics": data["diagnostics"],
        }
    raw = run_selection(
        project_dir=args.project_dir,
        checkpoint=artifacts["checkpoint"],
        proxy_mps=artifacts["proxy_mps"],
        proxy_tag=artifacts["proxy_tag"],
        permutation=artifacts["orbital_permutation"],
        rotation_solver=np.asarray(rotation_solver, dtype=float),
        output_dir=output_dir / "raw_nc",
        target_rank=args.target_rank,
        workers=args.candidate_workers,
        total_threads=args.candidate_threads,
        multiply_bond_dim=args.multiply_bond_dim,
        multiply_sweeps=args.multiply_sweeps,
    )
    return exact_filter_selection(
        raw,
        artifacts["checkpoint"],
        artifacts["orbital_irreps_canonical"],
        args.target_rank,
        output_dir,
    )


def run_switching_optimization(args, artifacts, parity_canonical, x0, output_dir):
    """Use determinant-extracted sector screening, not recursive MPS splitting."""
    from src.dmrg_decoupled_energy import (
        make_context,
        optimize_with_dmrg_sector_switching,
        rotated_solver,
        screen_sector_labels,
    )
    from src.dmrg_solver import Block2DMRGSolver, DMRGConfig, solve_or_load_ground_state
    from src.orbital_rotation import irrep_pairs, params_to_U
    from common import map_rows_to_solver_order

    output_dir = Path(output_dir)
    output_path = output_dir / "optimized.json"
    if args.resume and output_path.exists():
        print(f"[optimization] reusing {output_path}", flush=True)
        return read_json(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Block2DMRGSolver.load(artifacts["proxy_mps"], n_threads=args.dmrg_threads)
    pairs = irrep_pairs(base.orbital_symmetries)
    x0 = np.asarray(x0, dtype=float)
    parity_solver = map_rows_to_solver_order(
        parity_canonical, artifacts["orbital_permutation"]
    )

    if np.max(np.abs(x0)) < 1.0e-14:
        screening_solver = base
        screening_tag = artifacts["proxy_tag"]
    else:
        screening_store = output_dir / "rotated_screening_mps"
        screening_solver = rotated_solver(
            base, x0, pairs, screening_store, args.dmrg_threads
        )
        screening_tag = "SCREEN_" + hashlib.sha256(x0.tobytes()).hexdigest()[:12]
        screening_result = solve_or_load_ground_state(
            screening_solver,
            DMRGConfig(
                max_bond_dim=args.sector_bond_dim,
                n_sweeps=args.sector_sweeps,
                energy_tol=args.sector_energy_tolerance,
                davidson_threshold=args.sector_davidson_threshold,
                mps_tag=screening_tag,
                iprint=1,
            ),
            reuse=True,
        )
        screening_tag = screening_result.mps_tag
    labels, weights = screen_sector_labels(
        screening_solver,
        parity_solver,
        reference_tag=screening_tag,
        minimum=args.screen_minimum,
        maximum=args.screen_maximum,
    )
    print(f"[screening] labels={[''.join(map(str, value)) for value in labels]}", flush=True)

    context = make_context(
        base,
        parity_solver,
        pairs,
        output_dir / "optimizer_mps",
        output_dir / "objective_cache.json",
        bond_dim=args.sector_bond_dim,
        sweeps=args.sector_sweeps,
        penalty=args.sector_penalty,
        n_threads=args.dmrg_threads,
        energy_tol=args.sector_energy_tolerance,
        davidson_threshold=args.sector_davidson_threshold,
        dmrg_iprint=1,
        cleanup_mps=True,
    )
    restart_path = output_dir / "optimizer_restart.json"
    initial_label = None
    if args.resume and restart_path.exists():
        state = read_json(restart_path)
        if state.get("rotation") is not None:
            x0 = np.asarray(state["rotation"], dtype=float)
        if state.get("sector") is not None:
            initial_label = tuple(int(value) for value in state["sector"])
    started = time.perf_counter()
    result, history, initial_scan = optimize_with_dmrg_sector_switching(
        context,
        labels,
        x0,
        maxiter=args.optimizer_maxiter,
        max_switches=args.sector_switch_maxiter,
        state_path=restart_path,
        initial_label=initial_label,
        use_analytic_gradient=True,
    )
    rotation = params_to_U(result.x, base.n_sites, pairs)
    output = {
        "schema": "alris.h2o_631g_switching_sector_optimization",
        "version": 1,
        "rotation_parameters_solver_order": np.asarray(result.x).tolist(),
        "rotation_matrix_solver_order": rotation.tolist(),
        "parity_matrix_canonical": np.asarray(parity_canonical, dtype=int).tolist(),
        "parity_matrix_solver_order": parity_solver.tolist(),
        "selected_sector": list(result.sector_label),
        "screened_sector_labels": [list(label) for label in labels],
        "sector_weights_from_proxy_determinants": weights,
        "cost_before_ha": float(initial_scan[0][0]),
        "cost_after_ha": float(result.fun),
        "switching_history": history,
        "optimizer": {
            "gradient": "analytic_rdm",
            "iterations": int(getattr(result, "nit", 0)),
            "objective_evaluations": int(getattr(result, "nfev", 0)),
            "gradient_evaluations": int(getattr(result, "njev", 0)),
            "success": bool(result.success),
            "message": str(result.message),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    atomic_json(output_path, output)
    np.savetxt(output_dir / "rotation_parameters.txt", np.asarray(result.x))
    return output


def optimize_fixed(args, artifacts, fixed_selection, output_dir):
    parity = np.asarray(fixed_selection["parity_matrix"], dtype=int)
    x0 = np.zeros(28)
    optimized = run_switching_optimization(args, artifacts, parity, x0, output_dir)
    return {
        "mode": "equilibrium_selected_fixed_splusq",
        "selection": fixed_selection,
        "optimization": optimized,
        "macrocycles": [],
    }


def optimize_reselected(args, artifacts, output_dir):
    from src.orbital_rotation import params_to_U
    from src.dmrg_solver import Block2DMRGSolver
    from src.orbital_rotation import irrep_pairs

    base = Block2DMRGSolver.load(artifacts["proxy_mps"], n_threads=args.dmrg_threads)
    pairs = irrep_pairs(base.orbital_symmetries)
    x = np.zeros(len(pairs))
    rotation = params_to_U(x, base.n_sites, pairs)
    selection = run_nc_selection(args, artifacts, rotation, Path(output_dir) / "selection_0")
    macrocycles = []
    previous_energy = None
    final_optimization = None
    final_selection = None

    for cycle in range(1, args.max_macrocycles + 1):
        cycle_dir = Path(output_dir) / f"macrocycle_{cycle}"
        parity = np.asarray(selection["parity_matrix"], dtype=int)
        optimized = run_switching_optimization(
            args, artifacts, parity, x, cycle_dir / "optimization"
        )
        x = np.asarray(optimized["rotation_parameters_solver_order"], dtype=float)
        rotation = params_to_U(x, base.n_sites, pairs)
        post = run_nc_selection(args, artifacts, rotation, cycle_dir / "reselection")
        row_stable = selection["row_space"] == post["row_space"]
        energy = float(optimized["cost_after_ha"])
        energy_change_mha = (
            None if previous_energy is None else abs(energy - previous_energy) * 1000.0
        )
        energy_stable = (
            energy_change_mha is not None
            and energy_change_mha <= args.macrocycle_energy_tolerance_mha
        )
        record = {
            "cycle": cycle,
            "input_selection": selection["selection"],
            "optimization": str((cycle_dir / "optimization" / "optimized.json").resolve()),
            "output_selection": post["selection"],
            "input_generators": [item["label"] for item in selection["selected"]],
            "output_generators": [item["label"] for item in post["selected"]],
            "row_space_stable": row_stable,
            "energy_change_mha": energy_change_mha,
            "energy_stable": energy_stable,
        }
        macrocycles.append(record)
        atomic_json(Path(output_dir) / "macrocycles.json", macrocycles)
        final_optimization = optimized
        final_selection = selection
        if row_stable and energy_stable:
            break
        previous_energy = energy
        selection = post
    return {
        "mode": "geometry_nc_reselected_splusq",
        "selection": final_selection,
        "optimization": final_optimization,
        "macrocycles": macrocycles,
    }


def final_sector_dmrg(args, artifacts, branch, output_dir):
    """Converge only the low-M winning sector and record its exact support."""
    from src.dmrg_solver import Block2DMRGSolver, DMRGConfig, rotate_integrals
    from src.orbital_rotation import irrep_pairs
    from common import map_rows_to_solver_order

    output_dir = Path(output_dir)
    result_path = output_dir / "final_decoupled.json"
    if args.resume and result_path.exists():
        print(f"[final sector] reusing {result_path}", flush=True)
        return read_json(result_path)
    base = Block2DMRGSolver.load(artifacts["proxy_mps"], n_threads=args.dmrg_threads)
    pairs = irrep_pairs(base.orbital_symmetries)
    x = np.asarray(
        branch["optimization"]["rotation_parameters_solver_order"], dtype=float
    )
    from src.orbital_rotation import params_to_U

    rotation = params_to_U(x, base.n_sites, pairs)
    h1e, g2e = rotate_integrals(base.h1e, base.g2e, rotation)
    solver = Block2DMRGSolver(
        h1e=h1e,
        g2e=g2e,
        ecore=base.ecore,
        n_elec=base.n_elec,
        spin=base.spin,
        store_dir=output_dir / "mps",
        n_threads=args.dmrg_threads,
        save_integrals=True,
        orbital_permutation=base.orbital_permutation,
        orbital_symmetries=base.orbital_symmetries,
        target_irrep=base.target_irrep,
        symmetry_mode=base.symmetry_mode,
        n_mkl_threads=1,
        stack_mem_bytes=base.stack_mem_bytes,
    )
    parity_canonical = np.asarray(branch["selection"]["parity_matrix"], dtype=int)
    parity_solver = map_rows_to_solver_order(
        parity_canonical, artifacts["orbital_permutation"]
    )
    label = tuple(int(value) for value in branch["optimization"]["selected_sector"])
    stages = []
    previous_tag = None
    for index, bond_dim in enumerate(args.final_bond_dims):
        tag = f"FINAL_M{bond_dim}"
        if tag in solver.stored_tags():
            energy = float(solver.energy_expectation(solver.get_mps(tag)))
            metadata = solver.read_metadata(solver.store_dir)["runs"][tag]
            elapsed = float(metadata.get("elapsed_seconds", 0.0))
            expectations = solver.symmetry_expectations(parity_solver, solver.get_mps(tag))
            print(f"[final sector] reusing {tag}: E={energy:.12f} Ha", flush=True)
        else:
            result = solver.sector_ground_state(
                parity_solver,
                label,
                penalty=args.sector_penalty,
                config=DMRGConfig(
                    max_bond_dim=bond_dim,
                    n_sweeps=args.final_sweeps,
                    energy_tol=args.sector_energy_tolerance,
                    davidson_threshold=args.sector_davidson_threshold,
                    mps_tag=tag,
                    iprint=1,
                ),
                mps_tag=tag,
                initial_mps_tag=previous_tag,
            )
            energy = float(result.energy)
            elapsed = float(result.elapsed_seconds)
            expectations = np.asarray(result.symmetry_expectations)
        stages.append(
            {
                "tag": tag,
                "bond_dim": int(bond_dim),
                "energy_ha": energy,
                "elapsed_seconds": elapsed,
                "symmetry_expectations": np.asarray(expectations).tolist(),
            }
        )
        previous_tag = tag
        if index >= 1:
            difference = abs(stages[-1]["energy_ha"] - stages[-2]["energy_ha"]) * 1000.0
            print(f"[final sector] M difference={difference:.6f} mHa", flush=True)
            if difference <= args.final_m_tolerance_mha:
                break

    audit = exact_sector_dimension_audit(
        artifacts["nalpha"],
        artifacts["nbeta"],
        artifacts["orbital_irreps_canonical"],
        artifacts["target_irrep"],
        parity_canonical,
        label,
    )
    final_energy = float(stages[-1]["energy_ha"])
    output = {
        "selected_sector": list(label),
        "stages": stages,
        "final_energy_ha": final_energy,
        "reference_energy_ha": float(artifacts["reference_energy_ha"]),
        "error_mha": 1000.0 * (final_energy - float(artifacts["reference_energy_ha"])),
        "dimension_audit": audit,
    }
    atomic_json(result_path, output)
    return output


def run_casscf(args, artifacts, output_dir):
    if args.skip_casscf:
        return None
    script = args.casscf_script or (
        args.project_dir / "experiments" / "h2o_631g_casscf" / "run_casscf_ladder.py"
    )
    if not Path(script).is_file():
        raise FileNotFoundError(
            f"CASSCF script is missing: {script}; pass --skip-casscf only for LAS-only tests"
        )
    command = [
        sys.executable,
        "-u",
        str(script),
        "--checkpoint",
        artifacts["checkpoint"],
        "--output-dir",
        str(output_dir),
        "--spaces",
        args.casscf_spaces,
        "--reference-energy",
        str(artifacts["reference_energy_ha"]),
        "--threads",
        str(args.casscf_threads),
        "--resume",
    ]
    subprocess.run(command, cwd=args.project_dir, check=True)
    return read_json(Path(output_dir) / "summary.json")


def prepare_fixed(args):
    equilibrium_index = min(
        range(len(args.bond_lengths)),
        key=lambda index: abs(args.bond_lengths[index] - 0.958),
    )
    bond = args.bond_lengths[equilibrium_index]
    if abs(bond - 0.958) > 1.0e-8:
        raise ValueError("the bond grid must contain the 0.958 A equilibrium point")
    artifacts = resolve_point(args.parent_root, bond, args.proxy_tag)
    artifacts.update(checkpoint_data(args.project_dir, artifacts["checkpoint"]))
    rotation = np.eye(13)
    print_stage("Prepare the equilibrium-selected fixed rank-7 S+Q set")
    selection = run_nc_selection(
        args, artifacts, rotation, args.output_dir / "fixed_selection"
    )
    atomic_json(
        args.output_dir / "fixed_selection" / "fixed_selection.json",
        {
            **selection,
            "definition": (
                "Seven S+Q generators selected by MPS-native NC at rOH=0.958 A, "
                "independent modulo exact number and C2v parity symmetries."
            ),
        },
    )


def load_fixed(args):
    path = args.output_dir / "fixed_selection" / "fixed_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"fixed selection is not prepared: {path}")
    return read_json(path)


def run_point(args, bond):
    point_output = args.output_dir / "points" / geometry_key(bond)
    final_path = point_output / "result.json"
    if args.resume and final_path.exists():
        print(f"[point] reusing completed {final_path}", flush=True)
        return
    point_output.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_point(args.parent_root, bond, args.proxy_tag)
    artifacts.update(checkpoint_data(args.project_dir, artifacts["checkpoint"]))
    atomic_json(point_output / "artifacts.json", artifacts)
    fixed_selection = load_fixed(args)

    print_stage(f"H2O/6-31G rOH={bond:.4f} A: fixed rank-7 S+Q optimization")
    fixed = optimize_fixed(args, artifacts, fixed_selection, point_output / "fixed")
    fixed_final = final_sector_dmrg(args, artifacts, fixed, point_output / "fixed" / "final")

    print_stage(f"H2O/6-31G rOH={bond:.4f} A: NC selection and reselection")
    adaptive = optimize_reselected(args, artifacts, point_output / "reselected")
    adaptive_final = final_sector_dmrg(
        args, artifacts, adaptive, point_output / "reselected" / "final"
    )

    print_stage(f"H2O/6-31G rOH={bond:.4f} A: CASSCF ladder")
    casscf = run_casscf(args, artifacts, point_output / "casscf")
    result = {
        "schema": "alris.h2o_631g_fixed_vs_reselected_curve_point",
        "version": 1,
        "completed_at": now(),
        "artifacts": artifacts,
        "fixed": {
            "generators": [item["label"] for item in fixed["selection"]["selected"]],
            "optimization": fixed["optimization"],
            "final": fixed_final,
        },
        "reselected": {
            "generators": [item["label"] for item in adaptive["selection"]["selected"]],
            "macrocycles": adaptive["macrocycles"],
            "optimization": adaptive["optimization"],
            "final": adaptive_final,
        },
        "casscf": casscf,
    }
    atomic_json(final_path, result)
    print(f"[point] completed {final_path}", flush=True)


def smallest_chemical_casscf(summary):
    if summary is None:
        return None
    eligible = [
        item for item in summary.get("results", [])
        if item.get("casscf_converged") and item.get("casscf_error_mHa") is not None
        and abs(float(item["casscf_error_mHa"])) <= 1.6
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: int(item["determinant_dimension"]))


def aggregate(args):
    rows = []
    missing = []
    for bond in args.bond_lengths:
        result_path = args.output_dir / "points" / geometry_key(bond) / "result.json"
        if not result_path.exists():
            missing.append(float(bond))
            continue
        result = read_json(result_path)
        fixed = result["fixed"]["final"]
        adaptive = result["reselected"]["final"]
        casscf = smallest_chemical_casscf(result.get("casscf"))
        rows.append(
            {
                "r_oh_angstrom": float(bond),
                "reference_energy_ha": result["artifacts"]["reference_energy_ha"],
                "fixed_energy_ha": fixed["final_energy_ha"],
                "fixed_error_mha": fixed["error_mha"],
                "fixed_selected_dimension": fixed["dimension_audit"]["selected_determinant_dimension"],
                "fixed_dmax": fixed["dimension_audit"]["dmax_determinant"],
                "reselected_energy_ha": adaptive["final_energy_ha"],
                "reselected_error_mha": adaptive["error_mha"],
                "reselected_selected_dimension": adaptive["dimension_audit"]["selected_determinant_dimension"],
                "reselected_dmax": adaptive["dimension_audit"]["dmax_determinant"],
                "reselected_generators": ";".join(result["reselected"]["generators"]),
                "macrocycles": len(result["reselected"]["macrocycles"]),
                "casscf_space": None if casscf is None else casscf["space"],
                "casscf_energy_ha": None if casscf is None else casscf["casscf_energy_Ha"],
                "casscf_error_mha": None if casscf is None else casscf["casscf_error_mHa"],
                "casscf_dimension": None if casscf is None else casscf["determinant_dimension"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "curve_data.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "system": "H2O",
        "basis": "6-31G",
        "angle_hoh_degrees": 104.5,
        "completed_points": len(rows),
        "missing_bond_lengths_angstrom": missing,
        "curve_csv": str(csv_path.resolve()),
        "comparison": "fixed equilibrium-selected rank-7 S+Q versus geometry-reselected rank-7 S+Q versus CASSCF ladder",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    if rows:
        write_plots(args.output_dir, rows)
        write_markdown(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if missing:
        raise RuntimeError(f"missing curve points: {missing}")


def write_plots(output_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bonds = np.asarray([row["r_oh_angstrom"] for row in rows])
    fixed_error = np.maximum(
        np.abs([row["fixed_error_mha"] for row in rows]), 1.0e-6
    )
    adaptive_error = np.maximum(
        np.abs([row["reselected_error_mha"] for row in rows]), 1.0e-6
    )
    casscf_error = np.asarray(
        [np.nan if row["casscf_error_mha"] is None else abs(row["casscf_error_mha"]) for row in rows]
    )

    fig, axis = plt.subplots(figsize=(8.4, 5.3))
    axis.plot(bonds, fixed_error, "s-", lw=2.0, label="Fixed equilibrium-selected rank-7 S+Q")
    axis.plot(bonds, adaptive_error, "^-", lw=2.0, label="NC-reselected rank-7 S+Q")
    if np.any(np.isfinite(casscf_error)):
        axis.plot(bonds, casscf_error, "o-", lw=2.0, label="Smallest chemically accurate CASSCF")
    axis.axhline(1.6, color="#a23b32", ls="--", lw=1.4, label="Chemical accuracy")
    axis.set_yscale("log")
    axis.set_xlabel("O-H bond length (Angstrom)")
    axis.set_ylabel("Absolute energy error (mHa)")
    axis.set_title("H2O/6-31G decoupled LAS and CASSCF")
    axis.grid(True, which="both", alpha=0.23)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "error_curves.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.3))
    axis.plot(
        bonds,
        [row["fixed_selected_dimension"] for row in rows],
        "s-",
        lw=2.0,
        label="Fixed selected sector",
    )
    axis.plot(
        bonds,
        [row["fixed_dmax"] for row in rows],
        "s--",
        lw=1.4,
        alpha=0.75,
        label="Fixed $D_{max}$",
    )
    axis.plot(
        bonds,
        [row["reselected_selected_dimension"] for row in rows],
        "^-",
        lw=2.0,
        label="Reselected selected sector",
    )
    axis.plot(
        bonds,
        [row["reselected_dmax"] for row in rows],
        "^--",
        lw=1.4,
        alpha=0.75,
        label="Reselected $D_{max}$",
    )
    casscf_dimension = np.asarray(
        [np.nan if row["casscf_dimension"] is None else row["casscf_dimension"] for row in rows]
    )
    if np.any(np.isfinite(casscf_dimension)):
        axis.plot(bonds, casscf_dimension, "o-", lw=2.0, label="CASSCF CI dimension")
    axis.set_yscale("log")
    axis.set_xlabel("O-H bond length (Angstrom)")
    axis.set_ylabel("Variational-space dimension")
    axis.set_title("Exact-symmetry-resolved selected-sector dimensions")
    axis.grid(True, which="both", alpha=0.23)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "dimension_curves.png", dpi=220)
    plt.close(fig)


def write_markdown(output_dir, rows, summary):
    lines = [
        "# H2O/6-31G fixed versus reselected decoupled LAS",
        "",
        "The fixed calculation uses the effective-rank-7 S+Q row space selected at "
        "the equilibrium geometry. The reselected calculation scores the same S+Q "
        "candidate pool at every geometry and after each orbital-optimization macrocycle.",
        "",
        "All orbital rotations preserve C2v irreps. Candidate independence and reported "
        "dimensions are evaluated modulo exact particle-number and point-group parities.",
        "",
        "| rOH (A) | Fixed error (mHa) | Fixed selected D | Fixed Dmax | Reselected error (mHa) | Reselected selected D | Reselected Dmax | CASSCF space | CASSCF D |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        casscf_dimension = (
            "" if row["casscf_dimension"] is None
            else f"{int(row['casscf_dimension']):,}"
        )
        lines.append(
            f"| {row['r_oh_angstrom']:.4f} | {row['fixed_error_mha']:.6f} | "
            f"{row['fixed_selected_dimension']:,} | {row['fixed_dmax']:,} | "
            f"{row['reselected_error_mha']:.6f} | "
            f"{row['reselected_selected_dimension']:,} | {row['reselected_dmax']:,} | "
            f"{row['casscf_space'] or 'none'} | "
            f"{casscf_dimension} |"
        )
    lines.extend(
        [
            "",
            f"Completed points: {summary['completed_points']}",
            "",
            "The selected-sector dimension is not the DMRG bond dimension. It is the "
            "exact determinant count of that LAS sector after fixing the physical C2v "
            "irrep. DMRG is only the numerical solver used to avoid constructing that basis.",
        ]
    )
    Path(output_dir, "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args):
    args.project_dir = args.project_dir.resolve()
    args.parent_root = args.parent_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not (args.project_dir / "src" / "dmrg_solver.py").exists():
        raise FileNotFoundError(args.project_dir)
    if not args.parent_root.is_dir():
        raise FileNotFoundError(args.parent_root)
    if args.target_rank != 7:
        raise ValueError("this matched experiment is intentionally fixed at effective rank 7")
    if len(args.final_bond_dims) < 2:
        raise ValueError("at least two final bond dimensions are required")
    if tuple(sorted(set(args.final_bond_dims))) != args.final_bond_dims:
        raise ValueError("--final-bond-dims must be strictly increasing")


def main():
    args = parse_args()
    validate_args(args)
    if str(args.project_dir) not in sys.path:
        sys.path.insert(0, str(args.project_dir))
    if args.aggregate_only:
        aggregate(args)
        return
    if args.prepare_fixed_only:
        prepare_fixed(args)
        return
    if args.task_index is None:
        raise ValueError("--task-index is required for a curve point")
    if not 0 <= args.task_index < len(args.bond_lengths):
        raise IndexError("task index is outside the requested bond grid")
    run_point(args, args.bond_lengths[args.task_index])


if __name__ == "__main__":
    main()
