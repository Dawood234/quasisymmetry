"""Shared CLI vocabulary for optimize and metrics.

``--reference`` (optimize only)
    Which wavefunction / cost engine to use
    (``fci`` / ``hf`` → CI costs; ``dmrg`` → Block2 MPS costs).

``--orbital_rotation`` (optimize / rotate / ``--U`` tools)
    ``full`` — all ``binom(n,2)`` planes (default).
    ``irrep`` — only intra-irrep pairs; needs a symmetry-adapted
    Hamiltonian from ``make_pyscf_hamiltonian.py --point_group``.

``--backend`` (metrics only)
    Sector eigensolver: ``fci``, ``davidson``, or ``dmrg``.

Metrics K methods (``--coupled_energy_method``)
    ``perturbation``  one-shot PT ordering (no overlap reference needed)
    ``reference``     overlap ordering against a DMRG wavefunction only

``dmrg`` always means Block2. Shared flags: ``--bond_dim``,
``--wavefunction_dir``, ``--n_threads``.
"""

from __future__ import annotations

import argparse

REFERENCE_CHOICES = ("fci", "hf", "dmrg")
METRICS_BACKEND_CHOICES = ("fci", "dmrg", "davidson")
ORBITAL_ROTATION_CHOICES = ("full", "irrep")

OPTIMIZE_EPILOG = """
--reference picks both the wavefunction and the cost engine
-----------------------------------------------------------
  --reference fci     PySCF FCI CI vector + ffsim costs (default)
  --reference hf      Hartree-Fock CI vector + ffsim costs
  --reference dmrg    Block2 MPS + MPS-native NC/variance or
                      switching-sector decoupled-energy optimization

  The decoupled and fixed_sector costs currently require --reference fci or
  hf. The switching_sector cost also supports --reference dmrg.

--orbital_rotation packing
--------------------------
  --orbital_rotation full   SO(n), binom(n,2) angles (default)
  --orbital_rotation irrep  intra-irrep pairs only (needs --point_group chk)

examples
--------
  python optimize_symmetries.py mol.FCIDUMP parity.txt
  python optimize_symmetries.py mol.chk parity.txt --orbital_rotation irrep
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference hf
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference dmrg --bond_dim 250
"""

METRICS_EPILOG = """
--backend (sector eigensolver)
-----------------------------
  --backend fci         eigsh/eigh on each sector block (default)
  --backend davidson    PySCF Davidson on the same blocks
  --backend dmrg        Block2 sector-targeted DMRG

--coupled_energy_method (K selection; CI backends only)
-------------------------------------------------------
  perturbation   one-shot PT ordering (default); needs only an energy target
  reference      overlap ordering vs a DMRG wavefunction (loads Block2 GS)

  --backend dmrg always uses one-shot PT for K.
  --solver is an alias of --backend.

examples
--------
  python metrics.py oo.json
  python metrics.py oo.json --backend davidson --coupled_energy_method perturbation
  python metrics.py oo.json --coupled_energy_method reference --bond_dim 250
  python metrics.py oo.json --backend dmrg --bond_dim 250 --penalty 30
"""


def add_dmrg_common_args(parser: argparse.ArgumentParser) -> None:
    """Bond dimension / store / threads used by any ``dmrg`` path."""
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=250,
        help="Block2 DMRG bond dimension (dmrg paths / overlap-K reference)",
    )
    parser.add_argument(
        "--wavefunction_dir",
        default=None,
        help="directory for Block2 MPS files (reuse across runs)",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Block2 OpenMP thread count",
    )


def add_orbital_rotation_arg(parser: argparse.ArgumentParser) -> None:
    """``--orbital_rotation {full,irrep}`` shared by optimize entry points."""
    parser.add_argument(
        "--orbital_rotation",
        choices=ORBITAL_ROTATION_CHOICES,
        default="full",
        metavar="{full,irrep}",
        help=(
            "Orbital-rotation packing: full=SO(n) upper triangle (default); "
            "irrep=only intra-irrep pairs (needs a symmetry-adapted Hamiltonian "
            "from make_pyscf_hamiltonian.py --point_group)."
        ),
    )


def add_optimize_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` for ``optimize_symmetries.py`` (no ``--backend``)."""
    parser.add_argument(
        "--reference",
        choices=REFERENCE_CHOICES,
        default="fci",
        metavar="{fci,hf,dmrg}",
        help=(
            "REFERENCE STATE and cost engine: "
            "fci=PySCF FCI + ffsim costs (default); "
            "hf=Hartree-Fock + ffsim costs; "
            "dmrg=Block2 MPS + MPS-native NC/variance. "
            "Sector energy costs need fci or hf."
        ),
    )
    add_orbital_rotation_arg(parser)
    add_dmrg_common_args(parser)
    _attach_epilog(parser, OPTIMIZE_EPILOG)


def add_metrics_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--backend`` for ``metrics.py`` (no ``--reference``)."""
    parser.add_argument(
        "--backend",
        "--solver",
        dest="backend",
        choices=METRICS_BACKEND_CHOICES,
        default="fci",
        metavar="{fci,davidson,dmrg}",
        help=(
            "SECTOR SOLVER: fci=eigsh/eigh on each sector block (default); "
            "davidson=PySCF Davidson on the same blocks; "
            "dmrg=Block2 sector-targeted DMRG. "
            "--solver is a deprecated alias of --backend."
        ),
    )
    add_dmrg_common_args(parser)
    _attach_epilog(parser, METRICS_EPILOG)


def optimize_cost_engine(reference: str) -> str:
    """Derived cost engine label for optimize banners / JSON."""
    return "dmrg" if reference == "dmrg" else "statevector"


def print_workflow_banner(script: str, reference: str | None = None, backend: str | None = None, **extra) -> None:
    """Print a short resolved-settings banner so the run mode is obvious."""
    lines = []
    if reference is not None:
        lines.append(
            f"[workflow] reference={reference}  (wavefunction / energy used as truth)"
        )
    if script == "optimize" and reference is not None:
        engine = backend or optimize_cost_engine(reference)
        lines.append(f"[workflow] cost_engine={engine}  (from --reference)")
    elif script == "metrics" and backend is not None:
        lines.append(f"[workflow] backend={backend}  (sector solver)")
    for key, value in extra.items():
        if value is not None:
            lines.append(f"[workflow] {key}={value}")
    if lines:
        print("\n".join(lines), flush=True)


def _attach_epilog(parser: argparse.ArgumentParser, epilog: str) -> None:
    """Append recipes to the parser epilog without clobbering an existing one."""
    existing = parser.epilog or ""
    parser.epilog = (existing + "\n" + epilog).strip()
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
