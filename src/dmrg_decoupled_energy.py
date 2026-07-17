"""Switching-sector decoupled-energy optimization with Block2 DMRG.

Only a screened list of sector labels is solved.  At a fixed label ``s`` the
objective is

    E_s(U) = min eig(P_s H(U) P_s).

After optimizing ``E_s``, the screened labels are rescanned and the optimizer
switches to a lower sector when necessary.  The complete set of ``2**M``
labels is never enumerated.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize

from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    rotate_integrals,
    restore_g2e,
)
from src.orbital_rotation import params_to_U, rotation_and_derivatives


def rotation_key(x):
    """Stable key for one exact floating-point rotation vector."""
    values = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def sector_key(label):
    """Compact text representation of a binary sector label."""
    return "".join(str(int(value)) for value in label)


def objective_key(x, label):
    """Cache key for one rotation and sector label."""
    return f"{rotation_key(x)}:{sector_key(label)}"


def load_json(path, default):
    """Load JSON when it exists, otherwise return ``default``."""
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    """Atomically save a JSON file so interrupted jobs leave valid state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)


def remove_obsolete_mps(context, keep_tags):
    """Remove optimizer MPS files that are no longer useful as warm starts.

    Block2 stores an MPS tag in each related scratch-file name.  Objective
    energies stay in the JSON cache, so only the newest state for each
    screened sector must remain on disk.
    """
    if not context.get("cleanup_mps", True):
        return

    keep_tags = {str(tag) for tag in keep_tags if tag}
    cache = context["cache"]
    known_tags = {
        str(item.get("mps_tag"))
        for item in cache["evaluations"].values()
        if item.get("mps_tag")
    }
    obsolete = known_tags - keep_tags
    store_dir = Path(context["store_dir"])
    for tag in obsolete:
        for path in store_dir.rglob("*"):
            if path.is_file() and tag in path.name:
                path.unlink(missing_ok=True)

    metadata_path = store_dir / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path, {"system": {}, "runs": {}})
        runs = metadata.get("runs", {})
        metadata["runs"] = {
            tag: value for tag, value in runs.items() if tag in keep_tags
        }
        save_json(metadata_path, metadata)


def add_neighbour_labels(labels, n_bits, minimum):
    """Add one-bit neighbours without enumerating every binary label."""
    ordered = [tuple(int(value) for value in label) for label in labels]
    seen = set(ordered)
    seeds = list(ordered)
    if not seeds:
        seeds = [tuple(0 for _ in range(int(n_bits)))]

    for seed in seeds:
        for bit in range(int(n_bits)):
            candidate = list(seed)
            candidate[bit] = 1 - candidate[bit]
            candidate = tuple(candidate)
            if candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
            if len(ordered) >= int(minimum):
                return ordered
    return ordered


def screen_sector_labels(
    solver,
    parity_matrix,
    reference_tag="GS",
    minimum=8,
    maximum=16,
    cutoff=1.0e-6,
):
    """Choose high-weight MPS sectors and a few local neighbours."""
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    ket = solver.get_mps(reference_tag)
    weighted = solver.dominant_sector_labels(
        parity_matrix,
        ket=ket,
        cutoff=float(cutoff),
        max_sectors=int(maximum),
    )
    labels = [tuple(label) for label, _ in weighted]
    labels = add_neighbour_labels(labels, parity_matrix.shape[0], minimum)
    labels = labels[: int(maximum)]
    weight_map = {sector_key(label): float(weight) for label, weight in weighted}
    return labels, weight_map


def rotated_solver(base_solver, x, pairs, store_dir, n_threads):
    """Build a DMRG solver for the integrals rotated by ``x``."""
    rotation = params_to_U(np.asarray(x, dtype=float), base_solver.n_sites, pairs)
    h1e, g2e = rotate_integrals(base_solver.h1e, base_solver.g2e, rotation)
    return Block2DMRGSolver(
        h1e=h1e,
        g2e=g2e,
        ecore=base_solver.ecore,
        n_elec=base_solver.n_elec,
        spin=base_solver.spin,
        store_dir=store_dir,
        n_threads=int(n_threads),
        save_integrals=False,
    )


def integral_derivatives(h1e, g2e, rotation, rotation_derivatives):
    """Differentiate the one- and two-electron integrals for every angle."""
    g2e = restore_g2e(g2e, h1e.shape[0])
    derivatives = []
    for derivative in rotation_derivatives:
        dh1e = (
            derivative @ h1e @ rotation.T
            + rotation @ h1e @ derivative.T
        )
        dg2e = (
            np.einsum(
                "pa,qb,rc,sd,abcd->pqrs",
                derivative, rotation, rotation, rotation, g2e,
                optimize=True,
            )
            + np.einsum(
                "pa,qb,rc,sd,abcd->pqrs",
                rotation, derivative, rotation, rotation, g2e,
                optimize=True,
            )
            + np.einsum(
                "pa,qb,rc,sd,abcd->pqrs",
                rotation, rotation, derivative, rotation, g2e,
                optimize=True,
            )
            + np.einsum(
                "pa,qb,rc,sd,abcd->pqrs",
                rotation, rotation, rotation, derivative, g2e,
                optimize=True,
            )
        )
        derivatives.append((dh1e, dg2e))
    return derivatives


def contract_integral_derivatives(integral_derivative_list, rdm1, rdm2):
    """Contract integral derivatives with Block2's conventional SZ RDMs."""
    rdm1 = np.asarray(rdm1)
    rdm2 = np.asarray(rdm2)
    if rdm1.shape[0] != 2 or rdm2.shape[0] != 3:
        raise ValueError("expected SZ RDM components (alpha,beta) and (aa,ab,bb)")

    spin_summed_rdm1 = rdm1[0] + rdm1[1]
    spin_summed_rdm2 = 0.5 * rdm2[0] + rdm2[1] + 0.5 * rdm2[2]
    gradient = []
    for dh1e, dg2e in integral_derivative_list:
        one_body = np.einsum("pq,pq->", dh1e, spin_summed_rdm1)
        two_body = np.einsum("pqrs,prsq->", dg2e, spin_summed_rdm2)
        gradient.append(float(np.real(one_body + two_body)))
    return np.asarray(gradient)


def energy_from_rdms(h1e, g2e, ecore, rdm1, rdm2):
    """Evaluate the molecular energy using Block2's conventional SZ RDMs."""
    g2e = restore_g2e(g2e, h1e.shape[0])
    spin_summed_rdm1 = np.asarray(rdm1)[0] + np.asarray(rdm1)[1]
    spin_summed_rdm2 = (
        0.5 * np.asarray(rdm2)[0]
        + np.asarray(rdm2)[1]
        + 0.5 * np.asarray(rdm2)[2]
    )
    one_body = np.einsum("pq,pq->", h1e, spin_summed_rdm1)
    two_body = np.einsum("pqrs,prsq->", g2e, spin_summed_rdm2)
    return float(np.real(float(ecore) + one_body + two_body))


def evaluate_fixed_sector(context, x, label):
    """Evaluate one rotated sector energy, using disk cache and MPS warm start."""
    x = np.asarray(x, dtype=float)
    label = tuple(int(value) for value in label)
    key = objective_key(x, label)
    cache = context["cache"]
    if key in cache["evaluations"]:
        return float(cache["evaluations"][key]["energy"])

    solver = rotated_solver(
        context["base_solver"],
        x,
        context["pairs"],
        context["store_dir"],
        context["n_threads"],
    )
    label_text = sector_key(label)
    tag = f"OPT_{label_text}_{rotation_key(x)}"
    initial_tag = cache["last_tag_by_sector"].get(label_text)
    config = DMRGConfig(
        max_bond_dim=int(context["bond_dim"]),
        n_sweeps=int(context["sweeps"]),
        energy_tol=float(context["energy_tol"]),
        davidson_threshold=float(context["davidson_threshold"]),
        mps_tag=tag,
    )

    start = time.perf_counter()
    result = solver.sector_ground_state(
        context["parity_matrix"],
        label,
        penalty=float(context["penalty"]),
        config=config,
        mps_tag=tag,
        initial_mps_tag=initial_tag,
    )
    elapsed = time.perf_counter() - start

    cache["evaluations"][key] = {
        "energy": float(result.energy),
        "elapsed_seconds": float(elapsed),
        "label": list(label),
        "rotation": x.tolist(),
        "mps_tag": tag,
        "warm_start_tag": initial_tag,
        "symmetry_expectations": list(result.symmetry_expectations or []),
    }
    cache["last_tag_by_sector"][label_text] = tag
    save_json(context["cache_path"], cache)
    return float(result.energy)


def evaluate_fixed_sector_gradient(context, x, label):
    """Analytic selected-sector energy gradient from one MPS solve and its RDMs."""
    x = np.asarray(x, dtype=float)
    label = tuple(int(value) for value in label)
    key = objective_key(x, label)
    energy = evaluate_fixed_sector(context, x, label)
    record = context["cache"]["evaluations"][key]
    if record.get("gradient") is not None:
        return np.asarray(record["gradient"], dtype=float)

    solver = rotated_solver(
        context["base_solver"],
        x,
        context["pairs"],
        context["store_dir"],
        context["n_threads"],
    )
    tag = record["mps_tag"]
    if tag not in solver.stored_tags():
        # An old objective value can outlive its MPS after scratch cleanup.
        # Recompute that point because the gradient needs the wavefunction.
        del context["cache"]["evaluations"][key]
        save_json(context["cache_path"], context["cache"])
        energy = evaluate_fixed_sector(context, x, label)
        record = context["cache"]["evaluations"][key]
        solver = rotated_solver(
            context["base_solver"],
            x,
            context["pairs"],
            context["store_dir"],
            context["n_threads"],
        )
        tag = record["mps_tag"]

    started = time.perf_counter()
    ket = solver.get_mps(tag)
    rdm1, rdm2 = solver.spin_resolved_rdms(ket)
    rotation, rotation_derivatives = rotation_and_derivatives(
        x,
        context["base_solver"].n_sites,
        context["pairs"],
    )
    derivative_list = integral_derivatives(
        context["base_solver"].h1e,
        context["base_solver"].g2e,
        rotation,
        rotation_derivatives,
    )
    gradient = contract_integral_derivatives(derivative_list, rdm1, rdm2)
    rdm_energy = energy_from_rdms(
        solver.h1e,
        solver.g2e,
        solver.ecore,
        rdm1,
        rdm2,
    )
    elapsed = time.perf_counter() - started

    record["gradient"] = gradient.tolist()
    record["gradient_elapsed_seconds"] = float(elapsed)
    record["rdm_energy"] = float(rdm_energy)
    record["rdm_energy_difference"] = float(rdm_energy - energy)
    save_json(context["cache_path"], context["cache"])
    print(
        f"[sector gradient] label={sector_key(label)} "
        f"norm={np.linalg.norm(gradient):.6e} time={elapsed:.3f}s "
        f"rdm_check={rdm_energy - energy:+.3e} Ha",
        flush=True,
    )
    return gradient


def scan_sector_energies(context, x, labels):
    """Evaluate and sort the screened sector labels at one rotation."""
    results = []
    for index, label in enumerate(labels, start=1):
        print(
            f"[sector scan] {index}/{len(labels)} label={sector_key(label)}",
            flush=True,
        )
        energy = evaluate_fixed_sector(context, x, label)
        results.append((float(energy), tuple(label)))
    results.sort(key=lambda item: item[0])
    return results


def save_optimizer_state(path, x, label, history, status, started=None, cache_path=None):
    """Save enough state to restart from the latest accepted rotation."""
    save_json(
        path,
        {
            "status": str(status),
            "rotation": np.asarray(x, dtype=float).tolist(),
            "sector": list(label),
            "history": history,
            "elapsed_seconds": (
                None if started is None else float(time.perf_counter() - started)
            ),
            "cache_path": None if cache_path is None else str(cache_path),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


def make_context(
    base_solver,
    parity_matrix,
    pairs,
    store_dir,
    cache_path,
    bond_dim=100,
    sweeps=6,
    penalty=30.0,
    n_threads=4,
    energy_tol=1.0e-6,
    davidson_threshold=1.0e-8,
    cleanup_mps=True,
):
    """Create the dictionary used by the DMRG sector objective."""
    cache = load_json(
        cache_path,
        {"schema": "quasisymmetry.dmrg_objective_cache", "version": 1,
         "evaluations": {}, "last_tag_by_sector": {}},
    )
    return {
        "base_solver": base_solver,
        "parity_matrix": np.atleast_2d(np.asarray(parity_matrix, dtype=int)),
        "pairs": pairs,
        "store_dir": str(store_dir),
        "cache_path": str(cache_path),
        "cache": cache,
        "bond_dim": int(bond_dim),
        "sweeps": int(sweeps),
        "penalty": float(penalty),
        "n_threads": int(n_threads),
        "energy_tol": float(energy_tol),
        "davidson_threshold": float(davidson_threshold),
        "cleanup_mps": bool(cleanup_mps),
    }


def optimize_with_dmrg_sector_switching(
    context,
    labels,
    x0,
    maxiter=8,
    max_switches=2,
    state_path=None,
    callback=None,
    initial_label=None,
    use_analytic_gradient=True,
):
    """Optimize, rescan, and switch among a screened set of DMRG sectors."""
    labels = [tuple(int(value) for value in label) for label in labels]
    x = np.asarray(x0, dtype=float)
    started = time.perf_counter()
    if initial_label is None:
        initial_scan = scan_sector_energies(context, x, labels)
        current_energy, current_label = initial_scan[0]
    else:
        current_label = tuple(int(value) for value in initial_label)
        if current_label not in labels:
            labels.insert(0, current_label)
        current_energy = evaluate_fixed_sector(context, x, current_label)
        initial_scan = [(float(current_energy), current_label)]
    history = []
    result = None

    for switch in range(int(max_switches) + 1):
        print(
            f"[switching] stage={switch} sector={sector_key(current_label)} "
            f"energy={current_energy:.12f}",
            flush=True,
        )

        def objective(values):
            return evaluate_fixed_sector(context, values, current_label)

        def gradient(values):
            return evaluate_fixed_sector_gradient(context, values, current_label)

        def save_progress(values):
            if state_path is not None:
                save_optimizer_state(
                    state_path,
                    values,
                    current_label,
                    history,
                    "optimizing",
                    started=started,
                    cache_path=context["cache_path"],
                )
            keep_tags = list(context["cache"]["last_tag_by_sector"].values())
            accepted_key = objective_key(values, current_label)
            accepted = context["cache"]["evaluations"].get(accepted_key, {})
            if accepted.get("mps_tag"):
                keep_tags.append(accepted["mps_tag"])
            remove_obsolete_mps(context, keep_tags)
            if callback is not None:
                callback(values)

        result = scipy.optimize.minimize(
            objective,
            x,
            method="L-BFGS-B",
            jac=gradient if use_analytic_gradient else None,
            options={"maxiter": int(maxiter)},
            callback=save_progress,
        )
        x = np.asarray(result.x, dtype=float)
        new_scan = scan_sector_energies(context, x, labels)
        new_energy, new_label = new_scan[0]
        history.append(
            {
                "switch": switch,
                "start_sector": list(current_label),
                "start_energy": float(current_energy),
                "optimized_energy": float(result.fun),
                "best_sector_after_rescan": list(new_label),
                "best_energy_after_rescan": float(new_energy),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_iterations": int(getattr(result, "nit", 0)),
                "objective_evaluations": int(getattr(result, "nfev", 0)),
                "gradient_evaluations": int(getattr(result, "njev", 0)),
                "gradient_mode": (
                    "analytic_rdm" if use_analytic_gradient else "finite_difference"
                ),
            }
        )
        if state_path is not None:
            save_optimizer_state(
                state_path,
                x,
                new_label,
                history,
                "rescanned",
                started=started,
                cache_path=context["cache_path"],
            )

        if new_label == current_label:
            result.fun = float(new_energy)
            break
        current_label = new_label
        current_energy = new_energy
        result.fun = float(new_energy)

    if result is None:
        raise RuntimeError("switching-sector optimizer did not run")
    if state_path is not None:
        save_optimizer_state(
            state_path,
            x,
            current_label,
            history,
            "complete",
            started=started,
            cache_path=context["cache_path"],
        )
    result.sector_label = tuple(current_label)
    result.screened_labels = [tuple(label) for label in labels]
    result.elapsed_seconds = float(time.perf_counter() - started)
    return result, history, initial_scan
