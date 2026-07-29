"""Shared configuration, provenance, and restart helpers for LAS experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


SYSTEMS = {
    "h2o": {
        "display_name": "H2O",
        "basis": "6-31g",
        "point_group": "C2v",
        "geometry": {
            "r_oh_angstrom": 0.958,
            "hoh_angle_degrees": 104.5,
        },
        "norb": 13,
        "nelec": 10,
        "spin": 0,
        "rotation_parameters": 28,
        "candidate_workers": 4,
        "certified_reference_energy": -76.1208994639,
        "determinant_coupled_energy": -76.120688301322,
        "determinant_coupled_k": 873,
        "determinant_sector_count": 48,
    },
    "n2": {
        "display_name": "N2",
        "basis": "6-31g",
        "point_group": "D2h",
        "geometry": {"r_nn_angstrom": 1.0977},
        "norb": 18,
        "nelec": 14,
        "spin": 0,
        "rotation_parameters": 24,
        "candidate_workers": 8,
        "certified_reference_energy": -109.1049311347,
    },
}


DEFAULTS = {
    "target_rank": 7,
    "max_macrocycles": 3,
    "macrocycle_energy_tolerance_mha": 0.1,
    "min_dominant_sectors": 8,
    "max_dominant_sectors": 16,
    "sector_bond_dim": 100,
    "sector_sweeps": 6,
    "sector_penalty": 30.0,
    "optimizer_maxiter": 8,
    "sector_switch_maxiter": 2,
    "anchor_bond_dims": [100, 200, 350, 500],
    "anchor_extra_bond_dim": 750,
    "anchor_convergence_mha": 0.2,
    "anchor_sweeps": 20,
    "projector_beam_width": 32,
    "leakage_capture": 0.999,
    "fit_bond_dim": 500,
    "fit_extra_bond_dim": 750,
    "fit_sweeps": 8,
    "fit_tolerance": 1.0e-10,
    "fit_norm_loss_tolerance": 1.0e-6,
    "projector_absolute_weight_cutoff": 1.0e-10,
    "projector_relative_weight_cutoff": 1.0e-10,
    "projector_fit_loss_multiplier": 1.0,
    "fit_energy_tolerance_mha": 0.1,
    "initial_krylov_depth": 2,
    "residual_sectors_per_cycle": 16,
    "max_enrichment_cycles": 10,
    "standard_k_cap": 384,
    "extended_k_cap": 768,
    "extended_krylov_depth": 4,
    "overlap_cutoff": 1.0e-10,
    "chemical_accuracy_mha": 1.6,
    "energy_change_tolerance_mha": 0.1,
}


def now() -> str:
    """Return a compact local timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def label_text(label) -> str:
    """Return a compact binary sector label."""
    return "".join(str(int(bit)) for bit in label)


def parse_label(text: str) -> tuple[int, ...]:
    """Parse a compact binary sector label."""
    if any(char not in "01" for char in text):
        raise ValueError(f"invalid binary sector label: {text!r}")
    return tuple(int(char) for char in text)


def atomic_json(path, data) -> None:
    """Atomically write JSON so walltime interruption leaves valid state."""
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
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path, default=None):
    """Load JSON or return a caller-supplied default."""
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path, chunk_size=1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path, include_contents=False) -> dict:
    """Fingerprint file names, sizes, and optionally contents in a directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"MPS directory does not exist: {root}")
    entries = []
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = str(item.relative_to(root))
        size = int(item.stat().st_size)
        digest.update(relative.encode())
        digest.update(str(size).encode())
        record = {"path": relative, "size": size}
        if include_contents:
            record["sha256"] = sha256_file(item)
            digest.update(record["sha256"].encode())
        entries.append(record)
    return {
        "path": str(root.resolve()),
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "fingerprint": digest.hexdigest(),
    }


def parse_reference_energy(path) -> float:
    """Read an energy from a structured or text DMRG result."""
    path = Path(path)
    if path.suffix == ".json":
        data = load_json(path)
        for key in (
            "energy",
            "energy_Ha",
            "E_DMRG",
            "reference_energy",
            "certified_reference_energy",
        ):
            if key in data:
                return float(data[key])
        stages = data.get("stages", [])
        if stages:
            return float(stages[-1]["energy_Ha"])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.replace("=", " ").split()
            if line.startswith("E_DMRG ") and len(fields) >= 2:
                return float(fields[1])
            if "energy" in line.lower():
                for field in reversed(fields):
                    try:
                        return float(field)
                    except ValueError:
                        continue
    raise ValueError(f"no reference energy found in {path}")


def project_git_commit(project_dir) -> str:
    """Return the shared-project revision without changing repository state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(project_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def process_rss_mib() -> float:
    """Current-process maximum RSS in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def child_rss_mib() -> float:
    """Maximum RSS among completed child processes in MiB."""
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def checkpoint_metadata(checkpoint, system_name, project_dir) -> dict:
    """Validate molecular dimensions and irrep-restricted rotation packing."""
    project_dir = Path(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from chemistry import fcidump_data
    from src.orbital_rotation import n_params, resolve_orbital_rotation

    expected = SYSTEMS[system_name]
    data = fcidump_data(str(checkpoint))
    norb = int(np.asarray(data["H1"]).shape[0])
    raw_nelec = data["NELEC"]
    if np.iterable(raw_nelec):
        n_alpha, n_beta = (int(value) for value in raw_nelec)
        nelec = n_alpha + n_beta
    else:
        nelec = int(raw_nelec)
        n_alpha = (nelec + int(data.get("MS2", 0))) // 2
        n_beta = nelec - n_alpha
    pairs, irreps = resolve_orbital_rotation("irrep", checkpoint, norb)
    metadata = {
        "path": str(Path(checkpoint).resolve()),
        "sha256": sha256_file(checkpoint),
        "norb": norb,
        "nelec": nelec,
        "spin": n_alpha - n_beta,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "parent_qubits": 2 * norb,
        "fixed_spin_dimension": math.comb(norb, n_alpha)
        * math.comb(norb, n_beta),
        "rotation_parameters": n_params(norb, pairs),
        "orbital_irreps": np.asarray(irreps, dtype=int).tolist(),
        "point_group": data.get("POINT_GROUP"),
        "target_irrep": int(data.get("PG_IRREP", 0)),
    }
    checks = {
        "norb": expected["norb"],
        "nelec": expected["nelec"],
        "spin": expected["spin"],
        "rotation_parameters": expected["rotation_parameters"],
    }
    for key, value in checks.items():
        if metadata[key] != value:
            raise ValueError(
                f"{system_name} checkpoint {key}={metadata[key]}, expected {value}"
            )
    point_group = str(metadata["point_group"] or "").lower()
    if point_group != expected["point_group"].lower():
        raise ValueError(
            f"{system_name} checkpoint point group {metadata['point_group']!r}, "
            f"expected {expected['point_group']!r}"
        )
    return metadata


def choose_proxy_tag(metadata, requested=None) -> str:
    """Choose an existing M=100 or M=200 tag from saved MPS metadata."""
    runs = metadata.get("runs", {})
    if requested:
        if requested not in runs:
            raise ValueError(f"proxy MPS tag {requested!r} is not stored")
        return str(requested)
    for tag in ("M200", "M100", "GS"):
        if tag in runs:
            return tag
    candidates = sorted(
        runs,
        key=lambda tag: int(runs[tag]["config"].get("max_bond_dim", 10**9)),
    )
    if not candidates:
        raise ValueError("proxy MPS metadata contains no stored runs")
    return str(candidates[0])


def validate_artifacts(
    system_name,
    checkpoint,
    proxy_mps,
    reference_result,
    project_dir,
    proxy_tag=None,
    reference_tolerance=5.0e-7,
    allowed_proxy_bond_dims=(100, 200),
) -> dict:
    """Validate all reusable inputs before running expensive stages."""
    expected = SYSTEMS[system_name]
    checkpoint_info = checkpoint_metadata(checkpoint, system_name, project_dir)
    metadata_path = Path(proxy_mps) / "metadata.json"
    integrals_path = Path(proxy_mps) / "integrals.npz"
    if not metadata_path.exists() or not integrals_path.exists():
        raise FileNotFoundError(
            "proxy MPS must contain metadata.json and integrals.npz"
        )
    metadata = load_json(metadata_path)
    system = metadata.get("system", {})
    proxy_checks = {
        "n_sites": expected["norb"],
        "n_elec": expected["nelec"],
        "spin": expected["spin"],
    }
    for key, value in proxy_checks.items():
        if int(system.get(key, -1)) != int(value):
            raise ValueError(
                f"proxy MPS {key}={system.get(key)!r}, expected {value}"
            )
    permutation = tuple(
        int(value)
        for value in system.get(
            "orbital_permutation", range(expected["norb"])
        )
    )
    if sorted(permutation) != list(range(expected["norb"])):
        raise ValueError("proxy orbital_permutation is not a valid permutation")
    stored_irreps = tuple(int(value) for value in system.get("orbital_symmetries", []))
    canonical_irreps = tuple(checkpoint_info["orbital_irreps"])
    expected_irreps = tuple(canonical_irreps[index] for index in permutation)
    if stored_irreps and stored_irreps != expected_irreps:
        raise ValueError(
            "proxy orbital symmetry labels do not match checkpoint labels "
            "after applying orbital_permutation"
        )
    tag = choose_proxy_tag(metadata, proxy_tag)
    tag_info = Path(proxy_mps) / f"{tag}-mps_info.bin"
    if not tag_info.exists():
        raise FileNotFoundError(f"proxy MPS info file is missing: {tag_info}")
    selected_bond_dim = int(
        metadata["runs"][tag]["config"].get("max_bond_dim", 0)
    )
    if (
        allowed_proxy_bond_dims is not None
        and selected_bond_dim not in set(int(value) for value in allowed_proxy_bond_dims)
    ):
        allowed = ", ".join(str(value) for value in allowed_proxy_bond_dims)
        raise ValueError(
            f"proxy MPS tag {tag!r} has bond dimension {selected_bond_dim}; "
            f"expected one of {allowed}"
        )
    reference_energy = parse_reference_energy(reference_result)
    if abs(reference_energy - expected["certified_reference_energy"]) > float(
        reference_tolerance
    ):
        raise ValueError(
            f"reference result {reference_energy:.12f} Ha differs from certified "
            f"{expected['certified_reference_energy']:.12f} Ha"
        )
    return {
        "system": system_name,
        "checkpoint": checkpoint_info,
        "proxy_mps": {
            **directory_fingerprint(proxy_mps),
            "metadata_sha256": sha256_file(metadata_path),
            "integrals_sha256": sha256_file(integrals_path),
            "integral_fingerprint": system.get("fingerprint"),
            "orbital_permutation": list(permutation),
            "orbital_symmetries": list(stored_irreps),
            "symmetry_mode": system.get("symmetry_mode"),
            "selected_tag": tag,
            "selected_tag_energy": float(metadata["runs"][tag]["energy"]),
            "selected_tag_bond_dim": selected_bond_dim,
        },
        "reference_result": {
            "path": str(Path(reference_result).resolve()),
            "sha256": sha256_file(reference_result),
            "energy": reference_energy,
        },
        "validated_at": now(),
    }


def map_rows_to_solver_order(rows, permutation) -> np.ndarray:
    """Map canonical spatial or spin-resolved parity rows to solver order."""
    rows = np.atleast_2d(np.asarray(rows, dtype=int))
    permutation = np.asarray(permutation, dtype=int)
    norb = len(permutation)
    if rows.shape[1] == norb:
        return rows[:, permutation]
    if rows.shape[1] == 2 * norb:
        output = np.zeros_like(rows)
        for new_index, old_index in enumerate(permutation):
            output[:, 2 * new_index] = rows[:, 2 * old_index]
            output[:, 2 * new_index + 1] = rows[:, 2 * old_index + 1]
        return output
    raise ValueError("parity rows must have norb or 2*norb columns")


def map_rotation_to_canonical(rotation_solver, permutation) -> np.ndarray:
    """Map a rotation matrix from solver orbital order to canonical order."""
    rotation_solver = np.asarray(rotation_solver, dtype=float)
    permutation = np.asarray(permutation, dtype=int)
    output = np.zeros_like(rotation_solver)
    output[np.ix_(permutation, permutation)] = rotation_solver
    return output


def copy_mps_tag(source_dir, target_dir, tag) -> int:
    """Copy one Block2 MPS tag and the files needed to reload it."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    exact = {f"{tag}-mps_info.bin"}
    prefixes = (f"F.MPS.{tag}.", f"F.MPS.INFO.{tag}.")
    count = 0
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if source.name in exact or source.name.startswith(prefixes):
            shutil.copy2(source, target_dir / source.name)
            count += 1
    if not (target_dir / f"{tag}-mps_info.bin").exists():
        raise FileNotFoundError(f"MPS tag {tag!r} was not found in {source_dir}")
    return count


def copy_solver_store(source_dir, target_dir, tags) -> dict:
    """Create an isolated Block2 worker store containing selected MPS tags."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "integrals.npz"):
        shutil.copy2(source_dir / name, target_dir / name)
    copied = {}
    for tag in sorted(set(tags)):
        copied[tag] = copy_mps_tag(source_dir, target_dir, tag)
    return copied


def cap_worker_store_resources(
    store_dir,
    n_threads,
    maximum_stack_bytes=4 * 1024**3,
) -> None:
    """Bound resources recorded in an isolated Block2 worker store."""
    metadata_path = Path(store_dir) / "metadata.json"
    metadata = load_json(metadata_path)
    system = metadata.setdefault("system", {})
    stored_stack = int(system.get("stack_mem_bytes") or maximum_stack_bytes)
    system["stack_mem_bytes"] = min(stored_stack, int(maximum_stack_bytes))
    system["n_threads"] = int(n_threads)
    atomic_json(metadata_path, metadata)


def stage_complete(state, name, outputs) -> bool:
    """Return whether a restartable stage and all outputs are complete."""
    entry = state.get("stages", {}).get(name, {})
    return entry.get("status") == "complete" and all(
        Path(path).exists() for path in outputs
    )


def run_subprocess_stage(
    state_path,
    state,
    name,
    command,
    outputs,
    resume,
    cwd,
) -> None:
    """Run a streamed subprocess stage with durable metadata."""
    outputs = [str(Path(path)) for path in outputs]
    if resume and stage_complete(state, name, outputs):
        print(f"[{name}] already complete; reusing outputs", flush=True)
        return
    run_dir = Path(state["run_dir"])
    log_path = run_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "status": "running",
        "command": [str(value) for value in command],
        "outputs": outputs,
        "log": str(log_path),
        "started": now(),
    }
    state.setdefault("stages", {})[name] = entry
    atomic_json(state_path, state)
    print(f"\n=== {name} ===", flush=True)
    print("command:", " ".join(str(value) for value in command), flush=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=Path(cwd),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("subprocess output stream was not created")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    entry.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "max_child_rss_mib": child_rss_mib(),
            "return_code": return_code,
            "finished": now(),
        }
    )
    if return_code != 0:
        entry["status"] = "failed"
        atomic_json(state_path, state)
        raise RuntimeError(f"stage {name} failed; see {log_path}")
    missing = [path for path in outputs if not Path(path).exists()]
    if missing:
        entry["status"] = "failed"
        entry["missing_outputs"] = missing
        atomic_json(state_path, state)
        raise RuntimeError(f"stage {name} did not create {missing}")
    entry["status"] = "complete"
    atomic_json(state_path, state)
