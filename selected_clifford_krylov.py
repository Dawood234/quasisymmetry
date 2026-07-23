#!/usr/bin/env python3
"""Evaluate selected LAS sectors with coupling-seeded Krylov spaces.

The optimized anchor sector and leakage-ranked external sectors are reused from
the selected Clifford workflow.  Instead of solving many lowest eigenstates in
each external sector, this script starts from the part of ``H|phi_anchor>`` in
that sector and repeatedly applies its diagonal sector Hamiltonian.  The final
coupled basis is therefore designed around the states that actually communicate
with the anchor.
"""

import argparse
import json
import time
from pathlib import Path

import ffsim
import numpy as np

from chemistry import CHEMICAL_PRECISION, load_moldata
from selected_clifford_lanczos import (
    atomic_json,
    load_or_build_sector,
    parse_dmrg_energy,
    save_diagonal_lcu,
    selected_labels,
    stage,
    timestamp,
    validate_anchor_energy,
)
from src.clifford_sectors import (
    build_clifford_frame,
    load_symmetry_manifest,
    molecular_hamiltonian_to_jw,
    z_symmetries_from_parity_matrix,
)
from src.orbital_rotation import rotation_from_oo_data
from src.selected_sector_lanczos import (
    coupled_krylov_matrix,
    coupling_seeded_krylov_basis,
    krylov_depth_curve,
    label_text,
    sector_leakage_weights,
    selected_sector_supports,
    spin_orbital_parity_matrix,
)


def parse_depths(text):
    """Parse a comma-separated sequence of positive Krylov depths."""
    depths = sorted(set(int(value) for value in str(text).split(",")))
    if not depths or depths[0] < 1:
        raise ValueError("Krylov depths must be positive integers")
    return depths


def load_or_build_krylov(path, support, coupling_seed, full_operator, args):
    """Reload one sector Krylov basis or build and checkpoint it."""
    path = Path(path)
    requested_depth = max(args.krylov_depths)
    support_addresses = np.asarray(support["full_addresses"], dtype=np.int64)
    coupling_seed = np.asarray(coupling_seed, dtype=np.complex128)

    if args.resume and path.exists():
        saved = np.load(path, allow_pickle=False)
        saved_support = np.asarray(saved["full_addresses"], dtype=np.int64)
        saved_seed = np.asarray(saved["coupling_seed"], dtype=np.complex128)
        saved_tolerance = float(saved["tolerance"].item())
        saved_basis = np.asarray(saved["basis"], dtype=np.complex128)
        saved_breakdown = bool(saved["breakdown"].item())
        if (
            np.array_equal(saved_support, support_addresses)
            and np.allclose(saved_seed, coupling_seed, atol=1.0e-13, rtol=1.0e-13)
            and saved_tolerance <= args.krylov_tolerance
            and (saved_basis.shape[1] >= requested_depth or saved_breakdown)
        ):
            print(
                f"[Krylov {label_text(support['label'])}] reusing depth "
                f"{saved_basis.shape[1]} from {path}",
                flush=True,
            )
            depth = min(requested_depth, saved_basis.shape[1])
            return {
                **support,
                "basis": saved_basis[:, :depth],
                "projected_hamiltonian": np.asarray(
                    saved["projected_hamiltonian"]
                )[:depth, :depth],
                "seed_norm": float(saved["seed_norm"].item()),
                "depth": depth,
                "matvec_count": int(saved["matvec_count"].item()),
                "matvec_seconds": float(saved["matvec_seconds"].item()),
                "elapsed_seconds": float(saved["elapsed_seconds"].item()),
                "breakdown": saved_breakdown,
                "reused": True,
            }

    text_label = label_text(support["label"])
    print(
        f"[Krylov {text_label}] START depth={requested_depth}, "
        f"dimension={support['dimension']:,}",
        flush=True,
    )
    result = coupling_seeded_krylov_basis(
        full_operator,
        full_operator.shape[0],
        support_addresses,
        coupling_seed,
        requested_depth,
        tolerance=args.krylov_tolerance,
        print_every=args.print_every_krylov,
    )
    result = {**support, **result, "reused": False}
    np.savez_compressed(
        path,
        label=np.asarray(support["label"], dtype=np.uint8),
        full_addresses=support_addresses,
        coupling_seed=coupling_seed,
        basis=result["basis"],
        projected_hamiltonian=result["projected_hamiltonian"],
        seed_norm=np.asarray(result["seed_norm"]),
        depth=np.asarray(result["depth"]),
        matvec_count=np.asarray(result["matvec_count"]),
        matvec_seconds=np.asarray(result["matvec_seconds"]),
        elapsed_seconds=np.asarray(result["elapsed_seconds"]),
        breakdown=np.asarray(result["breakdown"]),
        tolerance=np.asarray(args.krylov_tolerance),
    )
    print(
        f"[Krylov {text_label}] DONE in {result['elapsed_seconds']:.2f} s; "
        f"depth={result['depth']}; H actions={result['matvec_count']}; "
        f"seed norm={result['seed_norm']:.8e}",
        flush=True,
    )
    print(f"[Krylov {text_label}] checkpoint: {path}", flush=True)
    return result


def parse_args():
    """Command-line controls for coupling-seeded final evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimized_json")
    parser.add_argument("--reference_result", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--outname", required=True)
    parser.add_argument("--anchor_checkpoint_dir", default=None)
    parser.add_argument("--max_sectors", type=int, default=16)
    parser.add_argument(
        "--krylov_depths",
        default="1,2,4,8,12,16,24",
        help="comma-separated nested Krylov depths to evaluate",
    )
    parser.add_argument("--krylov_tolerance", type=float, default=1.0e-12)
    parser.add_argument("--print_every_krylov", type=int, default=5)
    parser.add_argument(
        "--sector_selection",
        choices=("leakage", "mps"),
        default="leakage",
    )
    parser.add_argument("--lanczos_tolerance", type=float, default=1.0e-9)
    parser.add_argument("--lanczos_maxiter", type=int, default=None)
    parser.add_argument("--print_every_matvec", type=int, default=25)
    parser.add_argument("--roots_per_sector", type=int, default=1)
    parser.add_argument("--anchor_tolerance", type=float, default=1.0e-2)
    parser.add_argument("--chemical_accuracy", type=float, default=CHEMICAL_PRECISION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_lcu_files", action="store_true")
    args = parser.parse_args()
    args.krylov_depths = parse_depths(args.krylov_depths)
    return args


def main():
    """Run the restartable coupling-seeded Clifford/Krylov workflow."""
    args = parse_args()
    total_start = time.perf_counter()
    work_dir = Path(args.work_dir).resolve()
    krylov_dir = work_dir / "sectors"
    lcu_dir = work_dir / "diagonal_lcus"
    krylov_dir.mkdir(parents=True, exist_ok=True)
    lcu_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work_dir / "progress.json"

    stage("1/7 Load optimized frame and parent DMRG reference")
    input_data = json.loads(Path(args.optimized_json).read_text(encoding="utf-8"))
    reference_energy = parse_dmrg_energy(args.reference_result)
    moldata = load_moldata(input_data["molpath"])
    rotation = rotation_from_oo_data(input_data, moldata.norb)
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    full_operator = ffsim.linear_operator(
        rotated_hamiltonian, norb=moldata.norb, nelec=moldata.nelec
    )
    full_dimension = int(full_operator.shape[0])
    parity_matrix = np.atleast_2d(np.loadtxt(input_data["parity"], dtype=int))
    mps_labels = selected_labels(input_data, args.max_sectors)
    anchor = input_data.get("selected_sector")
    anchor = mps_labels[0] if anchor is None else tuple(int(bit) for bit in anchor)
    print("system orbitals:", moldata.norb, flush=True)
    print("fixed-spin electrons:", moldata.nelec, flush=True)
    print("fixed-spin CI dimension:", f"{full_dimension:,}", flush=True)
    print("parent DMRG reference energy:", f"{reference_energy:.12f} Ha", flush=True)
    print("optimized anchor label:", label_text(anchor), flush=True)
    print("requested Krylov depths:", args.krylov_depths, flush=True)

    stage("2/7 Build JW Pauli LCU and one Clifford frame")
    clifford_start = time.perf_counter()
    jw_hamiltonian = molecular_hamiltonian_to_jw(
        rotated_hamiltonian, moldata.nelec
    )
    manifest_path = input_data.get("symmetry_manifest")
    if manifest_path:
        manifest = load_symmetry_manifest(manifest_path)
        symmetries = manifest["symmetries"]
        manifest_rows = spin_orbital_parity_matrix(
            manifest["parity_matrix"], moldata.norb
        )
        parity_rows = spin_orbital_parity_matrix(parity_matrix, moldata.norb)
        if not np.array_equal(manifest_rows, parity_rows):
            raise ValueError("symmetry manifest and parity file contain different rows")
    else:
        symmetries = z_symmetries_from_parity_matrix(parity_matrix, moldata.norb)
    frame = build_clifford_frame(
        jw_hamiltonian, symmetries, 2 * moldata.norb
    )
    clifford_seconds = time.perf_counter() - clifford_start
    print("parent JW Pauli terms:", len(jw_hamiltonian.terms), flush=True)
    print("selected independent generators:", frame["n_symmetries"], flush=True)
    print(
        "qubits: parent -> residual =",
        frame["n_qubits"],
        "->",
        frame["n_residual_qubits"],
        flush=True,
    )
    print("Clifford/JW construction time:", f"{clifford_seconds:.2f} s", flush=True)

    stage("3/7 Solve the anchor and select sectors from its leakage")
    anchor_support = selected_sector_supports(
        parity_matrix,
        [anchor],
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )[anchor]
    checkpoint_dir = (
        Path(args.anchor_checkpoint_dir)
        if args.anchor_checkpoint_dir
        else work_dir / "anchor"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    anchor_checkpoint = checkpoint_dir / f"sector_{label_text(anchor)}.npz"
    anchor_result = load_or_build_sector(
        anchor_checkpoint,
        anchor_support,
        full_operator,
        full_dimension,
        args,
        root_count=1,
    )
    validate_anchor_energy(
        input_data,
        anchor,
        anchor_result["energies"][0],
        args.anchor_tolerance,
    )
    leakage_rank, leakage_norm_sq, anchor_residual = sector_leakage_weights(
        full_operator,
        full_dimension,
        anchor_support["full_addresses"],
        anchor_result["vectors"][:, 0],
        anchor_result["energies"][0],
        parity_matrix,
        moldata.norb,
        moldata.nelec,
    )
    leakage_map = {
        label: weight for label, weight in leakage_rank if label != anchor
    }
    external_norm_sq = sum(leakage_map.values())
    if args.sector_selection == "leakage":
        labels = [anchor] + list(leakage_map)[: args.max_sectors - 1]
        for label in mps_labels:
            if label not in labels and len(labels) < args.max_sectors:
                labels.append(label)
    else:
        labels = list(mps_labels)
        if anchor in labels:
            labels.remove(anchor)
        labels = [anchor] + labels[: args.max_sectors - 1]
    selected_leakage = sum(leakage_map.get(label, 0.0) for label in labels)
    selected_fraction = (
        0.0 if external_norm_sq == 0.0 else selected_leakage / external_norm_sq
    )
    print(
        f"anchor external leakage norm: {np.sqrt(external_norm_sq):.8e} Ha",
        flush=True,
    )
    print(
        f"selected leakage fraction: {selected_fraction:.6%}",
        flush=True,
    )
    print(
        f"final {args.sector_selection}-selected labels:",
        [label_text(label) for label in labels],
        flush=True,
    )
    atomic_json(
        work_dir / "leakage_selection.json",
        {
            "anchor": list(anchor),
            "external_leakage_norm_squared": float(external_norm_sq),
            "total_residual_norm_squared": float(leakage_norm_sq),
            "selected_leakage_fraction": float(selected_fraction),
            "ranking": [
                {"label": list(label), "weight": float(weight)}
                for label, weight in leakage_map.items()
            ],
            "selected_labels": [list(label) for label in labels],
        },
    )

    stage("4/7 Generate selected supports and optional diagonal LCUs")
    support_start = time.perf_counter()
    supports = selected_sector_supports(
        parity_matrix,
        labels,
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )
    support_seconds = time.perf_counter() - support_start
    selected_dimension = sum(item["dimension"] for item in supports.values())
    print("selected determinant count:", f"{selected_dimension:,}", flush=True)
    print(
        "selected fraction of fixed-spin space:",
        f"{selected_dimension / full_dimension:.6%}",
        flush=True,
    )

    lcu_metadata = {}
    if args.skip_lcu_files:
        print("LCU serialization skipped by --skip_lcu_files", flush=True)
    else:
        for number, label in enumerate(labels, start=1):
            text_label = label_text(label)
            path = lcu_dir / f"sector_{text_label}.json.gz"
            print(f"[LCU] {number}/{len(labels)} {text_label}", flush=True)
            pauli_count, one_norm = save_diagonal_lcu(
                path,
                frame,
                label,
                supports[label]["clifford_label"],
            )
            lcu_metadata[text_label] = {
                "path": str(path),
                "pauli_count": int(pauli_count),
                "lcu_one_norm": float(one_norm),
            }

    stage("5/7 Build coupling-seeded sector Krylov bases")
    sector_bases = {}
    for number, label in enumerate(labels, start=1):
        if label == anchor:
            continue
        text_label = label_text(label)
        support = supports[label]
        coupling_seed = anchor_residual[support["full_addresses"]]
        print(
            f"[Krylov progress] {number}/{len(labels)} sector {text_label}",
            flush=True,
        )
        sector_bases[label] = load_or_build_krylov(
            krylov_dir / f"sector_{text_label}.npz",
            support,
            coupling_seed,
            full_operator,
            args,
        )
        atomic_json(
            progress_path,
            {
                "stage": "sector_krylov",
                "completed_labels": [
                    label_text(done) for done in sector_bases
                ],
                "completed_count": len(sector_bases),
                "total_count": len(labels) - 1,
                "updated": timestamp(),
            },
        )

    decoupled_energy = float(anchor_result["energies"][0])
    print("decoupled energy:", f"{decoupled_energy:.12f} Ha", flush=True)
    print(
        "decoupled error:",
        f"{1000.0 * (decoupled_energy - reference_energy):.6f} mHa",
        flush=True,
    )

    stage("6/7 Build coupled Krylov Hamiltonian and depth curve")
    coupled_path = work_dir / "coupled_krylov_hamiltonian.npy"
    candidate_path = work_dir / "candidate_metadata.json"
    expected_candidates = [
        {"label": list(anchor), "depth": 0, "kind": "anchor"}
    ] + [
        {
            "label": list(label),
            "depth": depth,
            "kind": "krylov",
        }
        for label in sorted(sector_bases)
        for depth in range(1, sector_bases[label]["depth"] + 1)
    ]
    reuse_coupled = False
    if args.resume and coupled_path.exists() and candidate_path.exists():
        saved_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        reuse_coupled = saved_candidates.get("candidates") == expected_candidates
    if reuse_coupled:
        coupled = np.load(coupled_path)
        candidates = saved_candidates["candidates"]
        coupled_seconds = float(saved_candidates.get("elapsed_seconds", 0.0))
        print("reusing coupled Krylov Hamiltonian:", coupled_path, flush=True)
    else:
        coupled, raw_candidates, coupled_seconds = coupled_krylov_matrix(
            full_operator,
            full_dimension,
            supports[anchor],
            anchor_result["vectors"][:, 0],
            sector_bases,
        )
        candidates = [
            {
                "label": list(item["label"]),
                "depth": int(item["depth"]),
                "kind": item["kind"],
            }
            for item in raw_candidates
        ]
        np.save(coupled_path, coupled)
        atomic_json(
            candidate_path,
            {"candidates": candidates, "elapsed_seconds": coupled_seconds},
        )

    hermiticity_error = float(np.max(np.abs(coupled - coupled.conj().T)))
    curve = krylov_depth_curve(
        coupled,
        candidates,
        args.krylov_depths,
        reference_energy,
        args.chemical_accuracy,
    )
    for row in curve:
        print(
            f"[depth {row['depth']:2d}] dimension={row['dimension']:4d} "
            f"energy={row['energy']:.12f} Ha "
            f"error={row['error_mHa']:.6f} mHa",
            flush=True,
        )
    final = curve[-1]
    first_converged = next((row for row in curve if row["converged"]), None)
    print("coupled Hermiticity error:", f"{hermiticity_error:.3e}", flush=True)
    print("coupled construction time:", f"{coupled_seconds:.2f} s", flush=True)
    print("coupled convergence reached:", first_converged is not None, flush=True)

    stage("7/7 Save final metrics and restart metadata")
    sector_metadata = {}
    for label, result in sector_bases.items():
        text_label = label_text(label)
        sector_metadata[text_label] = {
            "dimension": int(result["dimension"]),
            "krylov_depth": int(result["depth"]),
            "seed_norm": float(result["seed_norm"]),
            "anchor_leakage_weight": float(leakage_map.get(label, 0.0)),
            "matvec_count": int(result["matvec_count"]),
            "matvec_seconds": float(result["matvec_seconds"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "breakdown": bool(result["breakdown"]),
            "reused": bool(result["reused"]),
        }
        if text_label in lcu_metadata:
            sector_metadata[text_label]["diagonal_lcu"] = lcu_metadata[text_label]

    output = {
        "schema": "quasisymmetry.selected_clifford_krylov",
        "version": 1,
        "method": "selected Clifford sectors with coupling-seeded Krylov bases",
        "full_fixed_spin_matrix_constructed": False,
        "all_sector_labels_enumerated": False,
        "optimized_json": str(Path(args.optimized_json).resolve()),
        "reference_method": "DMRG",
        "E_reference": float(reference_energy),
        "E_decoupled": decoupled_energy,
        "E_coupled": float(final["energy"]),
        "dE": float(decoupled_energy - reference_energy),
        "K": (
            int(first_converged["dimension"])
            if first_converged is not None
            else int(final["dimension"])
        ),
        "K_pt": None,
        "converged": first_converged is not None,
        "coupled_curve": curve,
        "sector_label_count": len(labels),
        "candidate_state_count": int(final["dimension"]),
        "sector_labels": [list(label) for label in labels],
        "sector_selection_method": args.sector_selection,
        "anchor_sector": list(anchor),
        "anchor_external_leakage_norm_squared": float(external_norm_sq),
        "selected_leakage_fraction": float(selected_fraction),
        "sector_krylov_bases": candidates,
        "sector_results": sector_metadata,
        "n_parent_qubits": int(frame["n_qubits"]),
        "n_tapered_qubits": int(frame["n_residual_qubits"]),
        "qubit_reduction": int(frame["n_symmetries"]),
        "full_fixed_spin_dimension": full_dimension,
        "selected_determinant_count": int(selected_dimension),
        "parent_jw_pauli_count": len(jw_hamiltonian.terms),
        "parent_jw_lcu_one_norm": float(
            sum(abs(complex(value)) for value in jw_hamiltonian.terms.values())
        ),
        "clifford_pauli_count": len(frame["hamiltonian"].terms),
        "clifford_lcu_one_norm": float(
            sum(abs(complex(value)) for value in frame["hamiltonian"].terms.values())
        ),
        "coupled_hermiticity_error": hermiticity_error,
        "timings": {
            "clifford_and_jw_seconds": clifford_seconds,
            "selected_support_seconds": support_seconds,
            "coupled_matrix_seconds": coupled_seconds,
            "total_seconds": float(time.perf_counter() - total_start),
        },
        "args": {
            **vars(args),
            "krylov_depths": list(args.krylov_depths),
        },
    }
    atomic_json(args.outname, output)
    atomic_json(
        progress_path,
        {"stage": "complete", "output": str(args.outname), "updated": timestamp()},
    )
    print("final metrics:", Path(args.outname).resolve(), flush=True)
    print("work directory:", work_dir, flush=True)
    print("total elapsed:", f"{output['timings']['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
