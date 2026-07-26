"""Build final tapered Pauli LCUs after the MPS coupled space is fixed."""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import atomic_json, label_text


def add_project_path(project_dir) -> None:
    """Make the shared project importable without modifying it."""
    project_dir = str(Path(project_dir).resolve())
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def save_operator(path, operator) -> dict:
    """Save one Pauli LCU as compressed JSON and return compact diagnostics."""
    from src.clifford_sectors import qubit_operator_to_data

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = qubit_operator_to_data(operator)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    coefficients = [complex(value) for value in operator.terms.values()]
    return {
        "path": str(path),
        "pauli_count": len(coefficients),
        "lcu_one_norm": float(sum(abs(value) for value in coefficients)),
        "maximum_coefficient": float(
            max((abs(value) for value in coefficients), default=0.0)
        ),
        "compressed_bytes": int(path.stat().st_size),
    }


def physical_to_clifford_label(label, offset) -> tuple[int, ...]:
    """Convert unsigned physical parity bits to signed Clifford-frame bits."""
    return tuple(int(bit) ^ int(shift) for bit, shift in zip(label, offset))


def represented_sector_pairs(basis, hamiltonian, tolerance=1.0e-12):
    """Return unique sector pairs with a represented nonzero coupled block."""
    labels = sorted({tuple(int(bit) for bit in item["label"]) for item in basis})
    positions = {
        label: [
            index
            for index, item in enumerate(basis)
            if tuple(int(bit) for bit in item["label"]) == label
        ]
        for label in labels
    }
    pairs = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            block = np.asarray(hamiltonian)[
                np.ix_(positions[left], positions[right])
            ]
            maximum = float(np.max(np.abs(block), initial=0.0))
            if maximum > float(tolerance):
                pairs.append((left, right, maximum))
    return labels, pairs


def build_final_clifford_outputs(
    project_dir,
    checkpoint,
    rotation_canonical,
    parity_canonical,
    symmetry_manifest,
    coupled_summary,
    coupled_matrix_path,
    output_dir,
    pair_tolerance=1.0e-12,
) -> dict:
    """Construct only LCUs represented by the final selected MPS basis."""
    add_project_path(project_dir)
    from chemistry import load_moldata
    from src.clifford_sectors import (
        apply_clifford_to_basis_bits,
        build_clifford_frame,
        load_symmetry_manifest,
        molecular_hamiltonian_to_jw,
        tapered_operator,
    )
    from src.selected_sector_lanczos import spin_orbital_parity_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    moldata = load_moldata(str(checkpoint))
    rotation_canonical = np.asarray(rotation_canonical, dtype=float)
    if rotation_canonical.shape != (moldata.norb, moldata.norb):
        raise ValueError("canonical rotation has the wrong shape")
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation_canonical)

    manifest = load_symmetry_manifest(symmetry_manifest)
    manifest_rows = spin_orbital_parity_matrix(
        manifest["parity_matrix"], moldata.norb
    )
    parity_rows = spin_orbital_parity_matrix(
        np.atleast_2d(np.asarray(parity_canonical, dtype=int)),
        moldata.norb,
    )
    if not np.array_equal(manifest_rows, parity_rows):
        raise ValueError(
            "symmetry manifest and final canonical parity matrix contain "
            "different ordered rows"
        )

    print("[Clifford] converting optimized molecular Hamiltonian to JW LCU", flush=True)
    jw_hamiltonian = molecular_hamiltonian_to_jw(
        rotated_hamiltonian, moldata.nelec
    )
    print(
        f"[Clifford] parent JW terms={len(jw_hamiltonian.terms):,}; "
        "constructing one binary Clifford frame",
        flush=True,
    )
    frame = build_clifford_frame(
        jw_hamiltonian,
        manifest["symmetries"],
        2 * moldata.norb,
    )
    zero_bits = [0] * (2 * moldata.norb)
    offset = tuple(
        int(bit)
        for bit in apply_clifford_to_basis_bits(
            zero_bits, frame["clifford"]
        )[: frame["n_symmetries"]]
    )

    coupled = json.loads(Path(coupled_summary).read_text(encoding="utf-8"))
    basis = list(coupled["basis"])
    saved = np.load(coupled_matrix_path)
    hamiltonian = np.asarray(saved["hamiltonian"], dtype=np.complex128)
    if hamiltonian.shape[0] < len(basis):
        raise ValueError("saved coupled matrix is smaller than the final MPS basis")
    hamiltonian = hamiltonian[: len(basis), : len(basis)]
    labels, pairs = represented_sector_pairs(
        basis, hamiltonian, tolerance=pair_tolerance
    )

    diagonal = []
    for number, label in enumerate(labels, start=1):
        clifford_label = physical_to_clifford_label(label, offset)
        print(
            f"[Clifford diagonal] {number}/{len(labels)} "
            f"physical={label_text(label)} frame={label_text(clifford_label)}",
            flush=True,
        )
        operator = tapered_operator(frame, clifford_label, clifford_label)
        diagnostics = save_operator(
            output_dir / "diagonal" / f"sector_{label_text(label)}.json.gz",
            operator,
        )
        diagonal.append(
            {
                "physical_label": list(label),
                "clifford_label": list(clifford_label),
                **diagnostics,
            }
        )

    inter_sector = []
    for number, (bra, ket, maximum) in enumerate(pairs, start=1):
        clifford_bra = physical_to_clifford_label(bra, offset)
        clifford_ket = physical_to_clifford_label(ket, offset)
        print(
            f"[Clifford pair] {number}/{len(pairs)} "
            f"<{label_text(bra)}|H|{label_text(ket)}> "
            f"max coupled element={maximum:.6e}",
            flush=True,
        )
        operator = tapered_operator(frame, clifford_bra, clifford_ket)
        diagnostics = save_operator(
            output_dir
            / "inter_sector"
            / f"bra_{label_text(bra)}__ket_{label_text(ket)}.json.gz",
            operator,
        )
        inter_sector.append(
            {
                "physical_bra_label": list(bra),
                "physical_ket_label": list(ket),
                "clifford_bra_label": list(clifford_bra),
                "clifford_ket_label": list(clifford_ket),
                "maximum_coupled_matrix_element": maximum,
                "adjoint_supplies_reverse_pair": True,
                **diagnostics,
            }
        )

    output = {
        "schema": "quasisymmetry.equilibrium_final_clifford_lcus",
        "version": 1,
        "parent_qubits": int(frame["n_qubits"]),
        "selected_generators": int(frame["n_symmetries"]),
        "tapered_qubits": int(frame["n_residual_qubits"]),
        "physical_to_clifford_label_offset": list(offset),
        "final_coupled_K": len(basis),
        "retained_sector_count": len(labels),
        "represented_inter_sector_pair_count": len(pairs),
        "pair_tolerance": float(pair_tolerance),
        "diagonal": diagonal,
        "inter_sector": inter_sector,
        "sparse_matrix_conversion": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "clifford_lcu_manifest.json", output)
    return output
