#!/usr/bin/env python3
"""Run a restartable H2O/6-31G CASCI/CASSCF active-space ladder.

The active spaces are nested windows of the canonical RHF orbitals.  Orbital
indices are zero-based and refer to the ordering stored in the input checkpoint.
Each active space has an independent PySCF checkpoint and result JSON.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from math import comb
from pathlib import Path


REFERENCE_ENERGY = -76.1208994639

SPACE_CONFIGS = {
    "2e2o": {"nelecas": 2, "ncas": 2, "active_orbitals": [4, 5]},
    "4e4o": {"nelecas": 4, "ncas": 4, "active_orbitals": [3, 4, 5, 6]},
    "6e6o": {"nelecas": 6, "ncas": 6, "active_orbitals": [2, 3, 4, 5, 6, 7]},
    "8e8o": {
        "nelecas": 8,
        "ncas": 8,
        "active_orbitals": list(range(1, 9)),
    },
    "8e10o": {
        "nelecas": 8,
        "ncas": 10,
        "active_orbitals": list(range(1, 11)),
    },
    "8e12o": {
        "nelecas": 8,
        "ncas": 12,
        "active_orbitals": list(range(1, 13)),
    },
    # Full-space validation is intentionally excluded from the local default.
    "10e13o": {
        "nelecas": 10,
        "ncas": 13,
        "active_orbitals": list(range(13)),
    },
}

DEFAULT_SPACES = "2e2o,4e4o,6e6o,8e8o,8e10o,8e12o"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run nested H2O/6-31G CASCI and CASSCF calculations from the "
            "existing C2v RHF checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--spaces",
        default=DEFAULT_SPACES,
        help=(
            "comma-separated spaces chosen from "
            + ",".join(SPACE_CONFIGS)
            + f" (default: {DEFAULT_SPACES})"
        ),
    )
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=REFERENCE_ENERGY,
        help="total reference energy in Ha",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=12000)
    parser.add_argument("--casscf-max-macrocycles", type=int, default=100)
    parser.add_argument("--casscf-conv-tol", type=float, default=1.0e-9)
    parser.add_argument("--casscf-conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument(
        "--casci-only",
        action="store_true",
        help="run only the fixed-RHF-orbital CASCI stages",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed JSON stages and partial CASSCF checkpoints",
    )
    return parser.parse_args()


def configure_threads(count):
    value = str(count)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    # Avoid a second BLAS thread pool underneath OpenMP.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_json(path):
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def requested_spaces(text):
    names = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in SPACE_CONFIGS]
    if unknown:
        raise ValueError(
            f"unknown active spaces {unknown}; choose from {list(SPACE_CONFIGS)}"
        )
    if not names:
        raise ValueError("at least one active space is required")
    return names


def determinant_dimension(ncas, active_nelec):
    nalpha = active_nelec // 2
    nbeta = active_nelec - nalpha
    return comb(ncas, nalpha) * comb(ncas, nbeta)


def validate_input(mol, mf, scf_data):
    nmo = int(mf.mo_coeff.shape[1])
    if nmo != 13:
        raise ValueError(f"expected 13 H2O/6-31G orbitals, found {nmo}")
    if tuple(mol.nelec) != (5, 5):
        raise ValueError(f"expected H2O electron counts (5, 5), found {mol.nelec}")
    if mol.groupname.upper() != "C2V":
        raise ValueError(f"expected C2v checkpoint, found {mol.groupname}")
    required_scf_fields = {"e_tot", "mo_coeff", "mo_energy", "mo_occ"}
    missing = required_scf_fields.difference(scf_data)
    if missing:
        raise ValueError(f"RHF checkpoint is missing fields: {sorted(missing)}")


def configure_multiconfigurational_solver(solver, wfnsym, args):
    solver.max_memory = args.max_memory_mb
    solver.fix_spin_(shift=1.0, ss=0.0)
    solver.fcisolver.wfnsym = wfnsym
    solver.fcisolver.conv_tol = 1.0e-10
    solver.fcisolver.max_cycle = 100


def load_partial_casscf_orbitals(checkpoint, expected_shape):
    from pyscf import lib

    if not checkpoint.exists():
        return None
    try:
        orbitals = lib.chkfile.load(str(checkpoint), "mcscf/mo_coeff")
    except (KeyError, OSError):
        return None
    if orbitals is None or tuple(orbitals.shape) != tuple(expected_shape):
        return None
    return orbitals


def spin_data(solver):
    ss, multiplicity = solver.fcisolver.spin_square(
        solver.ci,
        solver.ncas,
        solver.nelecas,
    )
    return float(ss), float(multiplicity)


def active_natural_occupations(solver):
    import numpy as np

    one_rdm = solver.fcisolver.make_rdm1(
        solver.ci,
        solver.ncas,
        solver.nelecas,
    )
    return [float(value) for value in np.linalg.eigvalsh(one_rdm)[::-1]]


def run_space(name, mol, mf, orbital_irreps, args):
    from pyscf import mcscf

    config = SPACE_CONFIGS[name]
    active_nelec = int(config["nelecas"])
    ncas = int(config["ncas"])
    active_orbitals = list(config["active_orbitals"])
    ncore = (mol.nelectron - active_nelec) // 2
    active_pair = (active_nelec // 2, active_nelec // 2)
    dimension = determinant_dimension(ncas, active_nelec)

    space_dir = args.output_dir / name
    result_path = space_dir / "result.json"
    casscf_checkpoint = space_dir / "casscf.chk"
    space_dir.mkdir(parents=True, exist_ok=True)

    result = load_json(result_path) if args.resume else {}
    result.update(
        {
            "space": name,
            "ncas": ncas,
            "nelecas": list(active_pair),
            "ncore": ncore,
            "nvirtual": int(mf.mo_coeff.shape[1] - ncore - ncas),
            "active_orbitals_zero_based": active_orbitals,
            "active_orbital_irreps": [orbital_irreps[i] for i in active_orbitals],
            "determinant_dimension": dimension,
            "reference_energy_Ha": args.reference_energy,
            "checkpoint": str(args.checkpoint.resolve()),
            "casscf_checkpoint": str(casscf_checkpoint.resolve()),
        }
    )

    print("\n" + "=" * 78, flush=True)
    print(
        f"[{timestamp()}] {name}: CAS({active_nelec}e,{ncas}o), "
        f"ncore={ncore}, dimension={dimension:,}",
        flush=True,
    )
    print(f"active orbitals: {active_orbitals}", flush=True)
    print(f"active irreps:   {result['active_orbital_irreps']}", flush=True)

    casci_complete = result.get("casci", {}).get("status") == "complete"
    if args.resume and casci_complete:
        print("[CASCI] already complete; reusing result", flush=True)
    else:
        print("[CASCI] starting fixed-RHF-orbital calculation", flush=True)
        casci = mcscf.CASCI(
            mf,
            ncas=ncas,
            nelecas=active_pair,
            ncore=ncore,
        )
        configure_multiconfigurational_solver(casci, "A1", args)
        initial_mo = mcscf.sort_mo(
            casci,
            mf.mo_coeff,
            active_orbitals,
            base=0,
        )
        start = time.perf_counter()
        energy = float(casci.kernel(initial_mo)[0])
        elapsed = time.perf_counter() - start
        ss, multiplicity = spin_data(casci)
        result["casci"] = {
            "status": "complete",
            "completed_at": timestamp(),
            "energy_Ha": energy,
            "error_Ha": energy - args.reference_energy,
            "error_mHa": 1000.0 * (energy - args.reference_energy),
            "wall_time_s": elapsed,
            "spin_square": ss,
            "multiplicity": multiplicity,
        }
        atomic_json_dump(result_path, result)
        print(
            f"[CASCI] E={energy:.12f} Ha, "
            f"error={result['casci']['error_mHa']:.6f} mHa, "
            f"time={elapsed:.2f} s",
            flush=True,
        )

    if args.casci_only:
        return result

    casscf_complete = result.get("casscf", {}).get("status") == "complete"
    if args.resume and casscf_complete:
        print("[CASSCF] already complete; reusing result", flush=True)
        return result

    print("[CASSCF] starting orbital optimization", flush=True)
    casscf = mcscf.CASSCF(
        mf,
        ncas=ncas,
        nelecas=active_pair,
        ncore=ncore,
    )
    configure_multiconfigurational_solver(casscf, "A1", args)
    casscf.conv_tol = args.casscf_conv_tol
    casscf.conv_tol_grad = args.casscf_conv_tol_grad
    casscf.max_cycle_macro = args.casscf_max_macrocycles
    casscf.chkfile = str(casscf_checkpoint)

    initial_mo = None
    if args.resume:
        initial_mo = load_partial_casscf_orbitals(
            casscf_checkpoint,
            mf.mo_coeff.shape,
        )
        if initial_mo is not None:
            print("[CASSCF] restarting from saved MCSCF orbitals", flush=True)
    if initial_mo is None:
        initial_mo = mcscf.sort_mo(
            casscf,
            mf.mo_coeff,
            active_orbitals,
            base=0,
        )

    start = time.perf_counter()
    energy = float(casscf.kernel(initial_mo)[0])
    elapsed = time.perf_counter() - start
    ss, multiplicity = spin_data(casscf)
    result["casscf"] = {
        "status": "complete" if casscf.converged else "not_converged",
        "completed_at": timestamp(),
        "converged": bool(casscf.converged),
        "energy_Ha": energy,
        "error_Ha": energy - args.reference_energy,
        "error_mHa": 1000.0 * (energy - args.reference_energy),
        "wall_time_s": elapsed,
        "spin_square": ss,
        "multiplicity": multiplicity,
        "active_natural_occupations": active_natural_occupations(casscf),
    }
    atomic_json_dump(result_path, result)
    print(
        f"[CASSCF] E={energy:.12f} Ha, "
        f"error={result['casscf']['error_mHa']:.6f} mHa, "
        f"converged={casscf.converged}, time={elapsed:.2f} s",
        flush=True,
    )
    if not casscf.converged:
        raise RuntimeError(
            f"{name} CASSCF did not converge; resume from {casscf_checkpoint}"
        )
    return result


def write_summary(output_dir, names):
    rows = []
    for name in names:
        result_path = output_dir / name / "result.json"
        if not result_path.exists():
            continue
        result = load_json(result_path)
        row = {
            "space": name,
            "ncas": result["ncas"],
            "nelecas_alpha": result["nelecas"][0],
            "nelecas_beta": result["nelecas"][1],
            "ncore": result["ncore"],
            "nvirtual": result["nvirtual"],
            "determinant_dimension": result["determinant_dimension"],
            "casci_energy_Ha": result.get("casci", {}).get("energy_Ha"),
            "casci_error_mHa": result.get("casci", {}).get("error_mHa"),
            "casci_wall_time_s": result.get("casci", {}).get("wall_time_s"),
            "casscf_energy_Ha": result.get("casscf", {}).get("energy_Ha"),
            "casscf_error_mHa": result.get("casscf", {}).get("error_mHa"),
            "casscf_converged": result.get("casscf", {}).get("converged"),
            "casscf_wall_time_s": result.get("casscf", {}).get("wall_time_s"),
        }
        rows.append(row)

    summary = {
        "updated_at": timestamp(),
        "spaces": names,
        "results": rows,
    }
    atomic_json_dump(output_dir / "summary.json", summary)

    csv_path = output_dir / "summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = list(rows[0]) if rows else [
        "space",
        "ncas",
        "nelecas_alpha",
        "nelecas_beta",
        "ncore",
        "nvirtual",
        "determinant_dimension",
        "casci_energy_Ha",
        "casci_error_mHa",
        "casci_wall_time_s",
        "casscf_energy_Ha",
        "casscf_error_mHa",
        "casscf_converged",
        "casscf_wall_time_s",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    configure_threads(args.threads)

    import numpy
    import pyscf
    from pyscf import lib, scf, symm

    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = requested_spaces(args.spaces)

    mol = lib.chkfile.load_mol(str(args.checkpoint))
    mf = scf.RHF(mol)
    mf.update_from_chk(str(args.checkpoint))
    scf_data = lib.chkfile.load(str(args.checkpoint), "scf")
    validate_input(mol, mf, scf_data)
    # Historical PySCF checkpoints did not store ``scf/converged``.  The
    # converged energy and complete MO datasets above are the reusable artifact.
    mf.converged = True

    orbital_irrep_ids = [int(value) for value in mf.get_orbsym()]
    orbital_irreps = [
        symm.irrep_id2name(mol.groupname, irrep_id)
        for irrep_id in orbital_irrep_ids
    ]

    configuration = {
        "created_at": timestamp(),
        "python": sys.version,
        "numpy_version": numpy.__version__,
        "pyscf_version": pyscf.__version__,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "output_dir": str(args.output_dir),
        "reference_energy_Ha": args.reference_energy,
        "spaces": names,
        "threads": args.threads,
        "max_memory_mb": args.max_memory_mb,
        "casci_only": args.casci_only,
        "resume": args.resume,
        "molecule": {
            "group": mol.groupname,
            "nelec": list(mol.nelec),
            "norb": int(mf.mo_coeff.shape[1]),
            "rhf_energy_Ha": float(mf.e_tot),
            "orbital_irreps": orbital_irreps,
        },
    }
    atomic_json_dump(args.output_dir / "configuration.json", configuration)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint SHA256: {configuration['checkpoint_sha256']}")
    print(f"Output directory: {args.output_dir}")
    print(f"PySCF: {pyscf.__version__}")
    print(f"Threads: {args.threads}")
    print(f"Maximum PySCF memory: {args.max_memory_mb} MB")
    print(f"Reference energy: {args.reference_energy:.12f} Ha")
    print(f"Spaces: {names}")

    for name in names:
        try:
            run_space(name, mol, mf, orbital_irreps, args)
        finally:
            write_summary(args.output_dir, names)

    print("\nCompleted requested active-space ladder.")
    print(f"JSON summary: {args.output_dir / 'summary.json'}")
    print(f"CSV summary:  {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
