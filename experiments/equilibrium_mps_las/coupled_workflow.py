"""High-fidelity anchor DMRG and residual-adaptive MPS coupled evaluation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from common import atomic_json, label_text, load_json
from linear_algebra import (
    canonical_generalized_eigh,
    choose_residual_labels,
    variational_curve_is_monotone,
)
from mps_krylov import (
    balanced_linear_combination,
    build_coupled_matrices,
    build_transition_signatures,
    decoupled_mpo,
    extend_sector_chain,
    extend_sector_chains_isolated,
    load_or_build_optimized_solver,
    make_residual,
    mps_tag_exists,
    projector_beam_split,
    validate_fitted_ritz,
)


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def run_anchor_schedule(
    project_dir,
    proxy_mps,
    rotation_solver,
    parity_solver,
    anchor_label,
    output_dir,
    threads,
    bond_dimensions,
    extra_bond_dimension,
    convergence_tolerance_mha,
    sweeps,
    penalty,
    energy_tolerance,
    davidson_threshold,
    twosite_to_onesite,
    stack_mem_gb,
    resume,
) -> dict:
    """Warm-start staged sector DMRG and add M=750 only when needed."""
    add_project_path(project_dir)
    from src.dmrg_solver import DMRGConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_dir = output_dir / "anchor_mps"
    solver = load_or_build_optimized_solver(
        project_dir=project_dir,
        proxy_mps=proxy_mps,
        rotation_solver=rotation_solver,
        store_dir=store_dir,
        threads=threads,
        stack_mem_gb=stack_mem_gb,
        symmetry_mode="sz",
    )
    parity_solver = np.atleast_2d(np.asarray(parity_solver, dtype=int))
    anchor_label = tuple(int(bit) for bit in anchor_label)
    stages = []
    previous_tag = None
    requested = [int(value) for value in bond_dimensions]
    for index, bond_dim in enumerate(requested):
        tag = f"ANCHOR_M{bond_dim}"
        result_path = output_dir / f"{tag}.json"
        if resume and result_path.exists() and mps_tag_exists(store_dir, tag):
            result = load_json(result_path)
            print(
                f"[anchor] reusing M={bond_dim}: "
                f"E={result['energy']:.12f} Ha",
                flush=True,
            )
        else:
            print(
                f"[anchor] stage {index + 1}/{len(requested)}: "
                f"M={bond_dim}, warm_start={previous_tag}",
                flush=True,
            )
            started = time.perf_counter()
            solved = solver.sector_ground_state(
                parity_solver,
                anchor_label,
                penalty=float(penalty),
                config=DMRGConfig(
                    max_bond_dim=bond_dim,
                    n_sweeps=int(sweeps),
                    energy_tol=float(energy_tolerance),
                    davidson_threshold=float(davidson_threshold),
                    mps_tag=tag,
                    twosite_to_onesite=twosite_to_onesite,
                    iprint=1,
                ),
                mps_tag=tag,
                initial_mps_tag=previous_tag,
                verify_tol=1.0e-3,
            )
            result = {
                "bond_dim": bond_dim,
                "tag": tag,
                "energy": float(solved.energy),
                "elapsed_seconds": time.perf_counter() - started,
                "symmetry_expectations": list(
                    solved.symmetry_expectations or []
                ),
                "sweep_history": list(solved.sweep_history),
            }
            atomic_json(result_path, result)
        stages.append(result)
        previous_tag = tag

    by_bond = {int(item["bond_dim"]): item for item in stages}
    difference_mha = abs(
        float(by_bond[500]["energy"]) - float(by_bond[350]["energy"])
    ) * 1000.0
    if difference_mha > float(convergence_tolerance_mha):
        bond_dim = int(extra_bond_dimension)
        tag = f"ANCHOR_M{bond_dim}"
        result_path = output_dir / f"{tag}.json"
        if resume and result_path.exists() and mps_tag_exists(store_dir, tag):
            result = load_json(result_path)
        else:
            print(
                f"[anchor] M500-M350={difference_mha:.6f} mHa exceeds "
                f"{convergence_tolerance_mha:.6f}; running M={bond_dim}",
                flush=True,
            )
            started = time.perf_counter()
            solved = solver.sector_ground_state(
                parity_solver,
                anchor_label,
                penalty=float(penalty),
                config=DMRGConfig(
                    max_bond_dim=bond_dim,
                    n_sweeps=int(sweeps),
                    energy_tol=float(energy_tolerance),
                    davidson_threshold=float(davidson_threshold),
                    mps_tag=tag,
                    twosite_to_onesite=twosite_to_onesite,
                    iprint=1,
                ),
                mps_tag=tag,
                initial_mps_tag=previous_tag,
                verify_tol=1.0e-3,
            )
            result = {
                "bond_dim": bond_dim,
                "tag": tag,
                "energy": float(solved.energy),
                "elapsed_seconds": time.perf_counter() - started,
                "symmetry_expectations": list(
                    solved.symmetry_expectations or []
                ),
                "sweep_history": list(solved.sweep_history),
            }
            atomic_json(result_path, result)
        stages.append(result)
        previous_tag = tag
    final = stages[-1]
    final_difference_mha = abs(
        float(stages[-1]["energy"]) - float(stages[-2]["energy"])
    ) * 1000.0
    output = {
        "schema": "quasisymmetry.equilibrium_anchor_schedule",
        "version": 1,
        "store_dir": str(store_dir),
        "anchor_label": list(anchor_label),
        "stages": stages,
        "M500_minus_M350_mha": difference_mha,
        "M500_M350_converged": difference_mha
        <= float(convergence_tolerance_mha),
        "final_stage_difference_mha": final_difference_mha,
        "anchor_converged": final_difference_mha
        <= float(convergence_tolerance_mha),
        "final_tag": final["tag"],
        "final_bond_dim": int(final["bond_dim"]),
        "decoupled_energy": float(final["energy"]),
        "bare_energy_reported": True,
    }
    atomic_json(output_dir / "anchor_summary.json", output)
    print(
        f"[anchor] final tag={final['tag']}, "
        f"E_dec={final['energy']:.12f} Ha",
        flush=True,
    )
    return {**output, "solver": solver}


def external_projector_split(
    solver,
    source_tag,
    parity_solver,
    anchor_label,
    work_dir,
    prefix,
    initial_width,
    minimum_capture,
    bond_dim,
    extra_bond_dim,
    sweeps,
    tolerance,
    fit_loss_tolerance,
    project_dir,
    workers,
    total_threads,
    exclude_anchor=True,
) -> dict:
    """Increase beam width or fitting bond dimension until leakage is captured."""
    maximum_width = 2 ** len(np.atleast_2d(parity_solver))
    width = min(int(initial_width), maximum_width)
    current_bond = int(bond_dim)
    while True:
        attempt_prefix = f"{prefix}_B{width}_M{current_bond}"
        split = projector_beam_split(
            solver=solver,
            source_tag=source_tag,
            parity_matrix=parity_solver,
            work_dir=work_dir,
            prefix=attempt_prefix,
            beam_width=width,
            bond_dim=current_bond,
            sweeps=sweeps,
            tolerance=tolerance,
            project_dir=project_dir,
            worker_root=Path(work_dir) / "projector_workers",
            workers=workers,
            total_threads=total_threads,
        )
        external = list(split["branches"])
        if exclude_anchor:
            external = [
                item
                for item in external
                if tuple(item["label"]) != tuple(anchor_label)
            ]
        external_weight = sum(float(item["norm2"]) for item in external)
        source_norm2 = max(float(split["source_norm2"]), 1.0e-30)
        capture = external_weight / source_norm2
        if (
            float(split["compression_loss"]) > float(fit_loss_tolerance)
            and current_bond < int(extra_bond_dim)
        ):
            print(
                f"[projector] fit loss {split['compression_loss']:.3e} exceeds "
                f"{fit_loss_tolerance:.3e}; retrying at M={extra_bond_dim}",
                flush=True,
            )
            current_bond = int(extra_bond_dim)
            continue
        if capture >= float(minimum_capture) or width >= maximum_width:
            return {
                **split,
                "external_branches": external,
                "external_weight": external_weight,
                "external_capture": capture,
                "effective_beam_width": width,
                "effective_bond_dim": current_bond,
                "anchor_excluded": bool(exclude_anchor),
            }
        new_width = min(maximum_width, width * 2)
        print(
            f"[projector] external capture {capture:.6%} below "
            f"{minimum_capture:.6%}; beam {width} -> {new_width}",
            flush=True,
        )
        width = new_width


def basis_depths(basis) -> dict:
    """Count retained MPS directions in every parity sector."""
    counts = {}
    for item in basis:
        label = tuple(item["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def add_projected_branches(
    solver,
    h_dec_mpo,
    branches,
    selected_labels,
    basis,
    additions,
    cycle,
    work_dir,
    bond_dim,
    sweeps,
    tolerance,
    k_cap,
    parity_solver,
    project_dir,
    workers,
    total_threads,
) -> list[dict]:
    """Orthogonalize selected projected residual branches and extend Krylov chains."""
    by_label = {
        tuple(item["label"]): item for item in branches
    }
    created = []
    requests = []
    reserved = 0
    for index, label in enumerate(selected_labels, start=1):
        if len(basis) + reserved >= int(k_cap):
            break
        branch = by_label.get(tuple(label))
        if branch is None:
            continue
        room = int(k_cap) - len(basis) - reserved
        requested = min(int(additions), room)
        print(
            f"[Krylov] cycle={cycle} sector={label_text(label)} "
            f"{index}/{len(selected_labels)} additions={requested}",
            flush=True,
        )
        requests.append(
            {
                "seed_tag": branch["tag"],
                "label": tuple(label),
                "existing_basis": basis,
                "additions": requested,
                "cycle": cycle,
                "work_dir": work_dir,
                "bond_dim": bond_dim,
                "sweeps": sweeps,
                "tolerance": tolerance,
            }
        )
        reserved += requested
    if int(workers) > 1 and len(requests) > 1:
        batches = extend_sector_chains_isolated(
            solver=solver,
            parity_matrix=parity_solver,
            requests=requests,
            project_dir=project_dir,
            worker_root=Path(work_dir) / f"krylov_workers_cycle_{cycle}",
            workers=workers,
            total_threads=total_threads,
        )
        for batch in batches:
            created.extend(batch)
        return created
    for request in requests:
        created.extend(
            extend_sector_chain(
                solver=solver,
                h_dec_mpo=h_dec_mpo,
                seed_tag=request["seed_tag"],
                label=request["label"],
                existing_basis=basis + created,
                additions=request["additions"],
                cycle=request["cycle"],
                work_dir=request["work_dir"],
                bond_dim=request["bond_dim"],
                sweeps=request["sweeps"],
                tolerance=request["tolerance"],
            )
        )
    return created


def sector_cleanliness(solver, parity_solver, basis) -> dict:
    """Measure parity expectations for one representative of every sector."""
    representatives = {}
    for item in basis:
        representatives.setdefault(tuple(item["label"]), item["tag"])
    output = {}
    for index, (label, tag) in enumerate(sorted(representatives.items()), start=1):
        print(
            f"[cleanliness] sector {index}/{len(representatives)} "
            f"{label_text(label)}",
            flush=True,
        )
        state = solver.get_mps(tag)
        expectations = solver.symmetry_expectations(parity_solver, ket=state)
        targets = np.asarray([(-1.0) ** bit for bit in label])
        output[label_text(label)] = {
            "tag": tag,
            "expectations": expectations.tolist(),
            "targets": targets.tolist(),
            "max_error": float(np.max(np.abs(expectations - targets))),
        }
    return output


def run_residual_adaptive_coupling(
    solver,
    parity_solver,
    anchor_tag,
    anchor_label,
    decoupled_energy,
    reference_energy,
    output_dir,
    initial_beam_width,
    minimum_capture,
    fit_bond_dim,
    fit_extra_bond_dim,
    fit_sweeps,
    fit_tolerance,
    fit_norm_loss_tolerance,
    fit_energy_tolerance_mha,
    initial_krylov_depth,
    residual_sectors_per_cycle,
    max_cycles,
    standard_k_cap,
    extended_k_cap,
    extended_krylov_depth,
    overlap_cutoff,
    chemical_accuracy_mha,
    energy_change_tolerance_mha,
    resume,
    project_dir,
    sector_workers,
    total_threads,
) -> dict:
    """Build coupling-seeded MPS Krylov spaces and enrich from Ritz residuals."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_solver = np.atleast_2d(np.asarray(parity_solver, dtype=int))
    anchor_label = tuple(int(bit) for bit in anchor_label)
    full_mpo = solver.hamiltonian_mpo()
    h_dec_mpo = decoupled_mpo(solver, parity_solver)
    signatures = build_transition_signatures(solver, parity_solver)
    atomic_json(
        output_dir / "transition_signatures.json",
        {
            "count": len(signatures),
            "signatures": [list(value) for value in sorted(signatures)],
        },
    )
    anchor_item = {
        "tag": anchor_tag,
        "label": list(anchor_label),
        "kind": "anchor",
        "cycle": -1,
        "depth": 0,
    }
    basis_path = output_dir / "basis_manifest.json"
    saved_basis = load_json(basis_path, {}).get("basis", []) if resume else []
    basis = saved_basis if saved_basis else [anchor_item]
    if basis[0]["tag"] != anchor_tag:
        raise ValueError("saved coupled basis does not begin with the current anchor")

    initial_residual = make_residual(
        solver,
        full_mpo,
        anchor_tag,
        decoupled_energy,
        "ANCHOR",
        output_dir,
        fit_bond_dim,
        fit_sweeps,
        fit_tolerance,
    )
    initial_split = external_projector_split(
        solver=solver,
        source_tag=initial_residual["tag"],
        parity_solver=parity_solver,
        anchor_label=anchor_label,
        work_dir=output_dir,
        prefix="ANCHOR_LEAKAGE",
        initial_width=initial_beam_width,
        minimum_capture=minimum_capture,
        bond_dim=fit_bond_dim,
        extra_bond_dim=fit_extra_bond_dim,
        sweeps=fit_sweeps,
        tolerance=fit_tolerance,
        fit_loss_tolerance=fit_norm_loss_tolerance,
        project_dir=project_dir,
        workers=sector_workers,
        total_threads=total_threads,
    )
    if len(basis) == 1:
        initial_labels = [
            tuple(item["label"])
            for item in sorted(
                initial_split["external_branches"],
                key=lambda item: -float(item["norm2"]),
            )
        ]
        basis.extend(
            add_projected_branches(
                solver=solver,
                h_dec_mpo=h_dec_mpo,
                branches=initial_split["external_branches"],
                selected_labels=initial_labels,
                basis=basis,
                additions=initial_krylov_depth,
                cycle=0,
                work_dir=output_dir,
                bond_dim=initial_split["effective_bond_dim"],
                sweeps=fit_sweeps,
                tolerance=fit_tolerance,
                k_cap=standard_k_cap,
                parity_solver=parity_solver,
                project_dir=project_dir,
                workers=sector_workers,
                total_threads=total_threads,
            )
        )
        atomic_json(basis_path, {"basis": basis})

    curve_path = output_dir / "coupled_curve.json"
    curve = load_json(curve_path, {}).get("cycles", []) if resume else []
    previous_energy = curve[-1]["energy"] if curve else None
    start_cycle = (curve[-1]["cycle"] + 1) if curve else 0
    final_solution = None
    final_matrix = None
    final_overlap = None

    for cycle in range(start_cycle, int(max_cycles) + 1):
        print(
            f"\n[coupled cycle {cycle}] K={len(basis)}, "
            f"sectors={len(basis_depths(basis))}",
            flush=True,
        )
        matrix_path = output_dir / "coupled_matrices.npz"
        hamiltonian, overlap, matrix_diagnostics = build_coupled_matrices(
            solver=solver,
            basis=basis,
            full_hamiltonian_mpo=full_mpo,
            transition_signatures=signatures,
            matrix_path=matrix_path,
            progress_path=output_dir / "matrix_progress.json",
            resume=resume,
        )
        solution = canonical_generalized_eigh(
            hamiltonian,
            overlap,
            overlap_cutoff=overlap_cutoff,
        )
        energy = float(solution["energies"][0])
        coefficients = np.asarray(solution["coefficients"][:, 0])
        ritz_tag = f"RITZ_C{cycle}"
        balanced_linear_combination(
            solver,
            [item["tag"] for item in basis],
            coefficients,
            ritz_tag,
            output_dir,
            fit_bond_dim,
            fit_sweeps,
            fit_tolerance,
        )
        fit_validation = validate_fitted_ritz(
            solver, ritz_tag, energy, full_mpo
        )
        effective_fit_bond = int(fit_bond_dim)
        if (
            fit_validation["energy_difference_mha"]
            > float(fit_energy_tolerance_mha)
            and int(fit_extra_bond_dim) > int(fit_bond_dim)
        ):
            print(
                f"[Ritz fit] energy mismatch "
                f"{fit_validation['energy_difference_mha']:.6f} mHa; "
                f"retrying at M={fit_extra_bond_dim}",
                flush=True,
            )
            ritz_tag = f"RITZ_C{cycle}_M{fit_extra_bond_dim}"
            balanced_linear_combination(
                solver,
                [item["tag"] for item in basis],
                coefficients,
                ritz_tag,
                output_dir,
                fit_extra_bond_dim,
                fit_sweeps,
                fit_tolerance,
            )
            fit_validation = validate_fitted_ritz(
                solver, ritz_tag, energy, full_mpo
            )
            effective_fit_bond = int(fit_extra_bond_dim)
        residual = make_residual(
            solver,
            full_mpo,
            ritz_tag,
            fit_validation["energy"],
            f"CYCLE{cycle}",
            output_dir,
            effective_fit_bond,
            fit_sweeps,
            fit_tolerance,
        )
        error_mha = (energy - float(reference_energy)) * 1000.0
        energy_change_mha = (
            None
            if previous_energy is None
            else abs(energy - float(previous_energy)) * 1000.0
        )
        record = {
            "cycle": cycle,
            "K": len(basis),
            "sector_count": len(basis_depths(basis)),
            "energy": energy,
            "error_mha": error_mha,
            "energy_change_mha": energy_change_mha,
            "residual_norm": residual["norm"],
            "ritz_fit": fit_validation,
            "retained_overlap_rank": solution["retained_overlap_rank"],
            "removed_overlap_directions": solution[
                "removed_overlap_directions"
            ],
            "overlap_condition_number": solution["condition_number"],
            "generalized_residual_norm": solution[
                "generalized_residual_norms"
            ][0],
            "matrix_diagnostics": matrix_diagnostics,
        }
        curve = [item for item in curve if int(item["cycle"]) != cycle]
        curve.append(record)
        curve.sort(key=lambda item: int(item["cycle"]))
        atomic_json(curve_path, {"cycles": curve})
        atomic_json(
            output_dir / f"cycle_{cycle}.json",
            {
                **record,
                "basis": basis,
                "ground_coefficients": [
                    [float(value.real), float(value.imag)]
                    for value in coefficients
                ],
                "overlap_eigenvalues": np.asarray(
                    solution["overlap_eigenvalues"], dtype=float
                ).tolist(),
            },
        )
        print(
            f"[coupled cycle {cycle}] E={energy:.12f} Ha, "
            f"error={error_mha:.6f} mHa, "
            f"dE={energy_change_mha} mHa, "
            f"||r||={residual['norm']:.6e}",
            flush=True,
        )
        final_solution = solution
        final_matrix = hamiltonian
        final_overlap = overlap
        converged = (
            abs(error_mha) <= float(chemical_accuracy_mha)
            and energy_change_mha is not None
            and energy_change_mha <= float(energy_change_tolerance_mha)
        )
        if converged:
            print("[coupled] chemical-accuracy stop criteria met", flush=True)
            break
        if cycle >= int(max_cycles):
            break

        extended = len(basis) >= int(standard_k_cap)
        k_cap = int(extended_k_cap if extended else standard_k_cap)
        additions = int(extended_krylov_depth if extended else 2)
        if len(basis) >= k_cap:
            print(f"[coupled] K cap {k_cap} exhausted", flush=True)
            break
        split = external_projector_split(
            solver=solver,
            source_tag=residual["tag"],
            parity_solver=parity_solver,
            anchor_label=anchor_label,
            work_dir=output_dir,
            prefix=f"RESIDUAL_C{cycle}",
            initial_width=initial_beam_width,
            minimum_capture=minimum_capture,
            bond_dim=effective_fit_bond,
            extra_bond_dim=fit_extra_bond_dim,
            sweeps=fit_sweeps,
            tolerance=fit_tolerance,
            fit_loss_tolerance=fit_norm_loss_tolerance,
            project_dir=project_dir,
            workers=sector_workers,
            total_threads=total_threads,
            exclude_anchor=True,
        )
        weighted = sorted(
            [
                (tuple(item["label"]), float(item["norm2"]))
                for item in split["external_branches"]
            ],
            key=lambda item: -item[1],
        )
        selected_labels = choose_residual_labels(
            weighted,
            basis_depths(basis),
            residual_sectors_per_cycle,
            anchor_label,
        )
        new_items = add_projected_branches(
            solver=solver,
            h_dec_mpo=h_dec_mpo,
            branches=split["external_branches"],
            selected_labels=selected_labels,
            basis=basis,
            additions=additions,
            cycle=cycle + 1,
            work_dir=output_dir,
            bond_dim=split["effective_bond_dim"],
            sweeps=fit_sweeps,
            tolerance=fit_tolerance,
            k_cap=k_cap,
            parity_solver=parity_solver,
            project_dir=project_dir,
            workers=sector_workers,
            total_threads=total_threads,
            exclude_anchor=False,
        )
        if not new_items:
            print("[coupled] residual produced no independent directions", flush=True)
            break
        basis.extend(new_items)
        atomic_json(basis_path, {"basis": basis})
        previous_energy = energy

    if final_solution is None or final_matrix is None or final_overlap is None:
        raise RuntimeError("coupled-space solver produced no cycle")
    cleanliness = sector_cleanliness(solver, parity_solver, basis)
    energies = [float(item["energy"]) for item in curve]
    final_record = curve[-1]
    output = {
        "schema": "quasisymmetry.equilibrium_mps_coupled",
        "version": 1,
        "method": "coupling_seeded_mps_krylov_with_residual_enrichment",
        "reference_energy": float(reference_energy),
        "decoupled_energy": float(decoupled_energy),
        "coupled_energy": float(final_record["energy"]),
        "coupled_error_mha": float(final_record["error_mha"]),
        "chemical_accuracy": abs(float(final_record["error_mha"]))
        <= float(chemical_accuracy_mha),
        "stop_change_satisfied": final_record["energy_change_mha"] is not None
        and float(final_record["energy_change_mha"])
        <= float(energy_change_tolerance_mha),
        "K": int(final_record["K"]),
        "sector_count": int(final_record["sector_count"]),
        "cycles": curve,
        "basis": basis,
        "initial_leakage": {
            "residual_norm": initial_residual["norm"],
            "external_capture": initial_split["external_capture"],
            "effective_beam_width": initial_split["effective_beam_width"],
            "effective_bond_dim": initial_split["effective_bond_dim"],
            "compression_loss": initial_split["compression_loss"],
            "discarded_beam_weight": initial_split["discarded_beam_weight"],
        },
        "variational_monotonicity": variational_curve_is_monotone(energies),
        "hamiltonian_hermiticity_error": float(
            np.max(
                np.abs(final_matrix - final_matrix.conj().T), initial=0.0
            )
        ),
        "overlap_hermiticity_error": float(
            np.max(
                np.abs(final_overlap - final_overlap.conj().T), initial=0.0
            )
        ),
        "overlap_condition_number": final_solution["condition_number"],
        "removed_overlap_directions": final_solution[
            "removed_overlap_directions"
        ],
        "sector_cleanliness": cleanliness,
    }
    atomic_json(output_dir / "coupled_summary.json", output)
    return output
