"""Reusable block2 DMRG solver for the quasisymmetry pipeline.

This module wraps ``pyblock2`` in fermionic SZ mode and provides:

* ground-state DMRG directly from spatial-orbital integrals or FCIDUMP files,
* local, reloadable wavefunction storage (block2 MPS files + JSON metadata),
* diagonal and orbitally-rotated parity quasi-symmetry MPOs
  ``S̃ = U^dagger S U`` (for MPS-native orbital optimization),
* sector-targeted (penalized) DMRG for decoupled-energy diagnostics,
* conversion of the MPS to a PySCF/ffsim CI vector or a JW statevector for
  drop-in use as a reference state in the existing statevector pipeline.

The module intentionally depends only on ``numpy`` and ``block2`` so DMRG
stages can run on machines without pyscf/ffsim (e.g. directly from FCIDUMP).

Conventions
-----------
* Integrals are spatial-orbital, chemist notation ``g2e[p, q, r, s] = (pq|rs)``
  (FCIDUMP convention). Packed 4-fold or 8-fold ``g2e`` is accepted.
* Parity matrices follow the repository convention: rows are symmetries,
  columns are either ``norb`` spatial parities or ``2 * norb`` spin-resolved
  parities in interleaved order ``[a0, b0, a1, b1, ...]``.
* Sector labels are tuples of 0/1 per symmetry; label bit ``0`` means parity
  eigenvalue ``+1`` and bit ``1`` means ``-1`` (matching
  ``metrics.symmetry_sectors``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from itertools import product as iter_product
from math import comb
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except ImportError:  # pragma: no cover - exercised only without block2
    DMRGDriver = None
    SymmetryTypes = None

logger = logging.getLogger(__name__)

DEFAULT_STORE_ROOT = Path("wavefunctions")
METADATA_FILENAME = "metadata.json"
INTEGRALS_FILENAME = "integrals.npz"

#: Guard for expanding products of parity factors into explicit terms.
MAX_PARITY_TERMS = 65536

#: Guard for dense reconstructions of the wavefunction.
MAX_CI_DIMENSION = 2**26
MAX_STATEVECTOR_QUBITS = 26

#: block2 keeps a single global frame per process, so only one DMRGDriver may
#: be used at a time; solvers re-create their driver when they lose this slot.
_ACTIVE_SOLVER: "Block2DMRGSolver | None" = None


def _require_pyblock2() -> None:
    if DMRGDriver is None or SymmetryTypes is None:
        raise ImportError(
            "pyblock2 (pip package 'block2') is required for src.dmrg_solver."
        )


def restore_g2e(g2e: np.ndarray, norb: int) -> np.ndarray:
    """Return the full 4-index two-electron tensor ``(pq|rs)``.

    Accepts a full 4D tensor (returned unchanged), a 4-fold packed 2D array of
    shape ``(npair, npair)``, or an 8-fold packed 1D array of length
    ``npair * (npair + 1) / 2`` with ``npair = norb * (norb + 1) / 2``.
    """
    g2e = np.asarray(g2e)
    if g2e.ndim == 4:
        return g2e

    npair = norb * (norb + 1) // 2
    pairs = [(i, j) for i in range(norb) for j in range(i + 1)]
    full = np.zeros((norb, norb, norb, norb), dtype=g2e.dtype)

    def _fill(i: int, j: int, k: int, l: int, value: float) -> None:
        for a, b in ((i, j), (j, i)):
            for c, d in ((k, l), (l, k)):
                full[a, b, c, d] = value
                full[c, d, a, b] = value

    if g2e.ndim == 2:
        if g2e.shape != (npair, npair):
            raise ValueError(
                f"4-fold g2e must have shape {(npair, npair)}, got {g2e.shape}"
            )
        for ij, (i, j) in enumerate(pairs):
            for kl, (k, l) in enumerate(pairs):
                _fill(i, j, k, l, g2e[ij, kl])
        return full

    if g2e.ndim == 1:
        expected = npair * (npair + 1) // 2
        if g2e.shape[0] != expected:
            raise ValueError(
                f"8-fold g2e must have length {expected}, got {g2e.shape[0]}"
            )
        idx = 0
        for ij, (i, j) in enumerate(pairs):
            for kl in range(ij + 1):
                k, l = pairs[kl]
                _fill(i, j, k, l, g2e[idx])
                idx += 1
        return full

    raise ValueError(f"unsupported g2e with ndim={g2e.ndim}")


def _string_address(occupied: Sequence[int]) -> int:
    """Rank of an occupation bitstring in the PySCF ``cistring`` ordering.

    PySCF orders alpha/beta strings by ascending integer value of the
    occupation bitmask; the rank of a combination is
    ``sum_i C(p_i, i + 1)`` with occupied orbitals ``p_i`` sorted ascending.
    """
    return sum(comb(p, i + 1) for i, p in enumerate(sorted(occupied)))


def _integral_fingerprint(h1e: np.ndarray, g2e: np.ndarray, ecore: float) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(h1e).tobytes())
    digest.update(np.ascontiguousarray(g2e).tobytes())
    digest.update(np.float64(ecore).tobytes())
    return digest.hexdigest()[:16]


def rotation_from_parameters(
    x: np.ndarray,
    norb: int,
    pairs=None,
) -> np.ndarray:
    """Orbital rotation ``U = expm(A(x))`` from upper-triangle parameters.

    Delegates to :func:`src.orbital_rotation.params_to_U`. Optional ``pairs``
    restricts free angles to intra-irrep planes.
    """
    from src.orbital_rotation import params_to_U

    return params_to_U(x, norb, pairs)


def rotate_integrals(
    h1e: np.ndarray, g2e: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Integrals of ``U H U^dagger``, matching ffsim's ``rotated(U)``.

    ffsim contracts the rotation on the row index of each tensor leg
    (``h' = U h U^T`` for real rotations), so the same convention is used
    here to keep DMRG results comparable with the statevector pipeline.
    """
    norb = h1e.shape[0]
    g2e_full = restore_g2e(g2e, norb)
    h1e_rot = rotation @ h1e @ rotation.T
    g2e_rot = np.einsum(
        "pa,qb,rc,sd,abcd->pqrs",
        rotation, rotation, rotation, rotation, g2e_full,
        optimize=True,
    )
    return h1e_rot, g2e_rot


def normalize_nelec(nelec, ms2: int | None = None) -> tuple[int, int]:
    """Return ``(n_elec_total, 2 * Sz)`` from an int, a ``(na, nb)`` pair,
    or an int plus an explicit ``MS2``."""
    if np.iterable(nelec):
        n_alpha, n_beta = (int(x) for x in nelec)
        return n_alpha + n_beta, n_alpha - n_beta
    return int(nelec), int(ms2 or 0)


def find_sector_determinants(
    parity_matrix: np.ndarray,
    sector_label: Sequence[int],
    norb: int,
    n_elec: int,
    spin: int,
    maximum: int = 32,
) -> list[str]:
    """Find a small fixed-spin determinant seed in one parity sector.

    The depth-first search stops after ``maximum`` valid determinants.  It
    does not enumerate the fixed-spin determinant basis or solve other sectors.
    """
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    label = tuple(int(value) for value in sector_label)
    if len(label) != parity_matrix.shape[0]:
        raise ValueError("sector_label length must match parity rows")
    if parity_matrix.shape[1] == norb:
        alpha = parity_matrix % 2
        beta = parity_matrix % 2
    elif parity_matrix.shape[1] == 2 * norb:
        alpha = parity_matrix[:, 0::2] % 2
        beta = parity_matrix[:, 1::2] % 2
    else:
        raise ValueError("parity matrix columns must be norb or 2 * norb")

    n_alpha = (int(n_elec) + int(spin)) // 2
    n_beta = int(n_elec) - n_alpha
    zero_parity = tuple(0 for _ in label)
    stack = [(0, 0, 0, zero_parity, "")]
    local_states = (
        (1, 1, "2"),
        (1, 0, "a"),
        (0, 1, "b"),
        (0, 0, "0"),
    )
    determinants = []

    while stack and len(determinants) < int(maximum):
        orbital, used_alpha, used_beta, parity, symbols = stack.pop()
        if orbital == norb:
            if used_alpha == n_alpha and used_beta == n_beta and parity == label:
                determinants.append(symbols)
            continue

        remaining_after = norb - orbital - 1
        for add_alpha, add_beta, symbol in local_states:
            next_alpha = used_alpha + add_alpha
            next_beta = used_beta + add_beta
            if next_alpha > n_alpha or next_beta > n_beta:
                continue
            if next_alpha + remaining_after < n_alpha:
                continue
            if next_beta + remaining_after < n_beta:
                continue
            next_parity = tuple(
                int(
                    parity[k]
                    ^ (add_alpha * int(alpha[k, orbital]))
                    ^ (add_beta * int(beta[k, orbital]))
                )
                for k in range(len(label))
            )
            stack.append(
                (orbital + 1, next_alpha, next_beta, next_parity, symbols + symbol)
            )

    if not determinants:
        raise ValueError(
            f"sector {label} has no determinant with "
            f"N_alpha={n_alpha}, N_beta={n_beta}"
        )
    return determinants


def find_sector_determinant(
    parity_matrix: np.ndarray,
    sector_label: Sequence[int],
    norb: int,
    n_elec: int,
    spin: int,
) -> str:
    """Return one determinant in a requested fixed-spin parity sector."""
    return find_sector_determinants(
        parity_matrix, sector_label, norb, n_elec, spin, maximum=1
    )[0]


@dataclass(frozen=True)
class DMRGConfig:
    """Sweep schedule and runtime settings for a DMRG solve."""

    max_bond_dim: int = 250
    n_sweeps: int = 20
    energy_tol: float = 1e-8
    davidson_threshold: float = 1e-10
    mps_tag: str = "GS"
    bond_dims: tuple[int, ...] = field(default=())
    noises: tuple[float, ...] = field(default=())

    def schedule(self) -> tuple[list[int], list[float], list[float]]:
        """Return (bond_dims, noises, davidson thresholds) per sweep."""
        n = self.n_sweeps
        if self.bond_dims:
            bond_dims = list(self.bond_dims)
            bond_dims += [bond_dims[-1]] * (n - len(bond_dims))
        elif n >= 10:
            bd = self.max_bond_dim
            bond_dims = (
                [max(2, bd // 4)] * 4 + [max(2, bd // 2)] * 4 + [bd] * (n - 8)
            )
        else:
            bond_dims = [self.max_bond_dim] * n

        if self.noises:
            noises = list(self.noises)
            noises += [0.0] * (n - len(noises))
        else:
            n_noisy = max(2, min(8, n - 2))
            noises = (
                [1e-4] * (n_noisy // 2)
                + [1e-5] * (n_noisy - n_noisy // 2)
                + [0.0] * (n - n_noisy)
            )

        thrds = [self.davidson_threshold] * n
        return bond_dims[:n], noises[:n], thrds


@dataclass
class DMRGResult:
    """Outcome of a DMRG solve, with enough metadata to reload the MPS."""

    energy: float
    mps_tag: str
    store_dir: str
    config: DMRGConfig
    elapsed_seconds: float
    sector_label: tuple[int, ...] | None = None
    symmetry_expectations: tuple[float, ...] | None = None
    energies: tuple[float, ...] | None = None
    """For multi-root solves: bare energies of each extracted root."""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["config"] = asdict(self.config)
        return data


def permute_parity_matrix(
    parity_matrix: np.ndarray, permutation: Sequence[int]
) -> np.ndarray:
    """Reorder parity-matrix columns to match an orbital permutation.

    Spatial matrices (``n_sym × norb``) permute orbital columns. Spin-resolved
    matrices (``n_sym × 2 norb``) permute interleaved ``[a0,b0,...]`` pairs.
    """
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    perm = np.asarray(permutation, dtype=int)
    norb = len(perm)
    if parity_matrix.shape[1] == norb:
        return parity_matrix[:, perm]
    if parity_matrix.shape[1] == 2 * norb:
        out = np.zeros_like(parity_matrix)
        for new_p, old_p in enumerate(perm):
            out[:, 2 * new_p] = parity_matrix[:, 2 * old_p]
            out[:, 2 * new_p + 1] = parity_matrix[:, 2 * old_p + 1]
        return out
    raise ValueError(
        "parity matrix columns must be norb or 2 * norb for the permutation"
    )


def permute_integrals(
    h1e: np.ndarray, g2e: np.ndarray, permutation: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an orbital permutation to spatial integrals."""
    perm = np.asarray(permutation, dtype=int)
    norb = len(perm)
    g2e_full = restore_g2e(g2e, norb)
    h1e_p = h1e[np.ix_(perm, perm)]
    g2e_p = g2e_full[np.ix_(perm, perm, perm, perm)]
    return h1e_p, g2e_p


class Block2DMRGSolver:
    """Fermionic (SZ-mode) block2 DMRG solver with local wavefunction storage.

    All MPS tensors live inside ``store_dir`` (the block2 scratch directory),
    so a solved wavefunction can be reloaded later with
    :meth:`Block2DMRGSolver.load` without re-running DMRG.
    """

    def __init__(
        self,
        h1e: np.ndarray,
        g2e: np.ndarray,
        ecore: float,
        n_elec,
        spin: int | None = None,
        store_dir: str | Path | None = None,
        n_threads: int = 4,
        save_integrals: bool = True,
        orbital_permutation: Sequence[int] | None = None,
        reorder: str | None = None,
    ) -> None:
        _require_pyblock2()
        h1e = np.ascontiguousarray(h1e, dtype=np.float64)
        g2e = np.ascontiguousarray(g2e, dtype=np.float64)
        self.ecore = float(ecore)
        n_sites = int(h1e.shape[0])

        if reorder is not None and orbital_permutation is not None:
            raise ValueError("pass at most one of reorder= and orbital_permutation=")
        if reorder is not None:
            orbital_permutation = self.compute_orbital_ordering(
                h1e, g2e, method=reorder
            )
            h1e, g2e = permute_integrals(h1e, g2e, orbital_permutation)
        if orbital_permutation is None:
            orbital_permutation = tuple(range(n_sites))
        else:
            orbital_permutation = tuple(int(p) for p in orbital_permutation)

        self.h1e = np.ascontiguousarray(h1e, dtype=np.float64)
        self.g2e = np.ascontiguousarray(g2e, dtype=np.float64)
        self.n_sites = int(self.h1e.shape[0])
        self.n_elec, self.spin = normalize_nelec(n_elec, spin)
        self.orbital_permutation = orbital_permutation
        self.fingerprint = _integral_fingerprint(self.h1e, self.g2e, self.ecore)

        if store_dir is None:
            store_dir = DEFAULT_STORE_ROOT / f"dmrg_{self.fingerprint}"
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.n_threads = int(n_threads)
        self.driver = None
        self._hamiltonian_mpo = None
        self._activate()

        if save_integrals:
            self._save_integrals()

    @staticmethod
    def compute_orbital_ordering(
        h1e: np.ndarray, g2e: np.ndarray, method: str = "fiedler"
    ) -> tuple[int, ...]:
        """Return an orbital permutation (Fiedler or gaopt) without building a solver."""
        _require_pyblock2()
        tmp = tempfile.mkdtemp(prefix="block2_reorder_")
        try:
            driver = DMRGDriver(
                scratch=tmp, symm_type=SymmetryTypes.SZ, n_threads=1
            )
            g2e_full = restore_g2e(g2e, h1e.shape[0])
            perm = driver.orbital_reordering(
                np.asarray(h1e, dtype=np.float64),
                np.asarray(g2e_full, dtype=np.float64),
                method=method,
            )
            return tuple(int(p) for p in perm)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def remap_parity_matrix(self, parity_matrix: np.ndarray) -> np.ndarray:
        """Map a parity matrix from the original orbital order into this solver's."""
        return permute_parity_matrix(parity_matrix, self.orbital_permutation)

    def _activate(self) -> None:
        """Make this solver the owner of the block2 global frame.

        block2 keeps a single global frame (scratch space, allocators) per
        process and pyblock2 requires that only the most recently created
        ``DMRGDriver`` is used. To support several solvers in one process,
        each solver rebuilds its driver (and drops driver-bound caches such
        as the Hamiltonian MPO) whenever another solver has been active in
        the meantime. MPS handles from :meth:`get_mps` are likewise only
        valid until a different solver is activated.
        """
        global _ACTIVE_SOLVER
        if _ACTIVE_SOLVER is self and self.driver is not None:
            return
        self.driver = DMRGDriver(
            scratch=str(self.store_dir),
            symm_type=SymmetryTypes.SZ,
            n_threads=self.n_threads,
        )
        self.driver.initialize_system(
            n_sites=self.n_sites, n_elec=self.n_elec, spin=self.spin
        )
        self._hamiltonian_mpo = None
        _ACTIVE_SOLVER = self

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fcidump(
        cls,
        fcidump_path: str | Path,
        store_dir: str | Path | None = None,
        n_threads: int = 4,
        **kwargs,
    ) -> "Block2DMRGSolver":
        """Build a solver directly from an FCIDUMP file (no pyscf needed)."""
        _require_pyblock2()
        tmp_scratch = tempfile.mkdtemp(prefix="block2_fcidump_read_")
        try:
            reader = DMRGDriver(
                scratch=tmp_scratch, symm_type=SymmetryTypes.SZ, n_threads=1
            )
            reader.read_fcidump(str(fcidump_path), pg="c1", iprint=0)
            h1e = np.array(reader.h1e)
            g2e = np.array(reader.g2e)
            ecore = float(reader.ecore)
            n_elec = int(reader.n_elec)
            spin = int(reader.spin)
        finally:
            shutil.rmtree(tmp_scratch, ignore_errors=True)
        return cls(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            n_elec=n_elec,
            spin=spin,
            store_dir=store_dir,
            n_threads=n_threads,
            **kwargs,
        )

    @classmethod
    def from_dumpdata(
        cls,
        dumpdata: Mapping,
        store_dir: str | Path | None = None,
        n_threads: int = 4,
        **kwargs,
    ) -> "Block2DMRGSolver":
        """Build a solver from a ``chemistry.fcidump_data`` dictionary."""
        n_elec, spin = normalize_nelec(
            dumpdata["NELEC"], dumpdata.get("MS2", 0)
        )
        return cls(
            h1e=np.asarray(dumpdata["H1"]),
            g2e=np.asarray(dumpdata["H2"]),
            ecore=float(dumpdata["ECORE"]),
            n_elec=n_elec,
            spin=spin,
            store_dir=store_dir,
            n_threads=n_threads,
            **kwargs,
        )

    @classmethod
    def load(
        cls, store_dir: str | Path, n_threads: int | None = None
    ) -> "Block2DMRGSolver":
        """Reload a solver (system + stored integrals) from a store directory.

        MPS tags recorded in the metadata can then be fetched with
        :meth:`get_mps` without re-running DMRG.
        """
        store_dir = Path(store_dir)
        metadata = cls.read_metadata(store_dir)
        integrals_path = store_dir / INTEGRALS_FILENAME
        if not integrals_path.exists():
            raise FileNotFoundError(
                f"{integrals_path} not found; the store was created with "
                "save_integrals=False and cannot be reloaded standalone."
            )
        data = np.load(integrals_path)
        perm = None
        if "orbital_permutation" in data.files:
            perm = tuple(int(p) for p in data["orbital_permutation"])
        solver = cls(
            h1e=data["h1e"],
            g2e=data["g2e"],
            ecore=float(data["ecore"]),
            n_elec=int(metadata["system"]["n_elec"]),
            spin=int(metadata["system"]["spin"]),
            store_dir=store_dir,
            n_threads=n_threads or int(metadata["system"].get("n_threads", 4)),
            save_integrals=False,
            orbital_permutation=perm,
        )
        if solver.fingerprint != metadata["system"]["fingerprint"]:
            raise ValueError(
                f"integral fingerprint mismatch in {store_dir}; the stored "
                "MPS belongs to a different Hamiltonian."
            )
        return solver

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def metadata_path(self) -> Path:
        return self.store_dir / METADATA_FILENAME

    @staticmethod
    def read_metadata(store_dir: str | Path) -> dict:
        path = Path(store_dir) / METADATA_FILENAME
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _save_integrals(self) -> None:
        np.savez_compressed(
            self.store_dir / INTEGRALS_FILENAME,
            h1e=self.h1e,
            g2e=self.g2e,
            ecore=self.ecore,
            orbital_permutation=np.asarray(self.orbital_permutation, dtype=int),
        )

    def _record_run(self, result: DMRGResult) -> None:
        if self.metadata_path.exists():
            metadata = self.read_metadata(self.store_dir)
        else:
            metadata = {"system": {}, "runs": {}}
        metadata["system"] = {
            "n_sites": self.n_sites,
            "n_elec": self.n_elec,
            "spin": self.spin,
            "ecore": self.ecore,
            "fingerprint": self.fingerprint,
            "n_threads": self.n_threads,
            "orbital_permutation": list(self.orbital_permutation),
        }
        run = result.to_dict()
        run["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metadata.setdefault("runs", {})[result.mps_tag] = run
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

    def stored_tags(self) -> list[str]:
        """MPS tags previously solved and recorded in this store."""
        if not self.metadata_path.exists():
            return []
        return list(self.read_metadata(self.store_dir).get("runs", {}).keys())

    def get_mps(self, tag: str = "GS", nroots: int = 1):
        """Load a stored MPS (or MultiMPS when ``nroots > 1``) by tag."""
        self._activate()
        return self.driver.load_mps(tag=tag, nroots=nroots)

    def split_root(self, multi_mps, iroot: int, tag: str):
        """Extract a single-root MPS from a state-averaged MultiMPS."""
        self._activate()
        return self.driver.split_mps(multi_mps, int(iroot), tag)

    # ------------------------------------------------------------------
    # MPOs
    # ------------------------------------------------------------------

    def hamiltonian_mpo(self):
        """The (cached) optimized quantum-chemistry MPO for ``H``."""
        self._activate()
        if self._hamiltonian_mpo is None:
            self._hamiltonian_mpo = self.driver.get_qc_mpo(
                h1e=self.h1e, g2e=self.g2e, ecore=self.ecore, iprint=0
            )
        return self._hamiltonian_mpo

    @staticmethod
    def _parity_factor_options(
        orbital: int, alpha: bool, beta: bool
    ) -> list[tuple[str, list[int], float]]:
        """Expansion options for one orbital factor of a parity operator."""
        options: list[tuple[str, list[int], float]] = [("", [], 1.0)]
        if alpha:
            options.append(("cd", [orbital, orbital], -2.0))
        if beta:
            options.append(("CD", [orbital, orbital], -2.0))
        if alpha and beta:
            options.append(("cdCD", [orbital] * 4, 4.0))
        return options

    def _parity_terms(
        self, parity_row: np.ndarray
    ) -> list[tuple[str, list[int], float]]:
        """Expand one parity-matrix row into explicit second-quantized terms.

        Spatial rows (length ``norb``) give ``prod_p (1-2n_pa)(1-2n_pb)``;
        spin-resolved rows (length ``2 * norb``) use the interleaved
        ``[a0, b0, ...]`` column convention.
        """
        parity_row = np.asarray(parity_row, dtype=int)
        factors = []
        if parity_row.shape[0] == self.n_sites:
            for p in np.flatnonzero(parity_row):
                factors.append(self._parity_factor_options(int(p), True, True))
        elif parity_row.shape[0] == 2 * self.n_sites:
            for p in range(self.n_sites):
                alpha = bool(parity_row[2 * p])
                beta = bool(parity_row[2 * p + 1])
                if alpha or beta:
                    factors.append(self._parity_factor_options(p, alpha, beta))
        else:
            raise ValueError(
                "parity row length must be norb or 2 * norb, got "
                f"{parity_row.shape[0]} for norb={self.n_sites}"
            )

        n_terms = int(np.prod([len(f) for f in factors])) if factors else 1
        if n_terms > MAX_PARITY_TERMS:
            raise ValueError(
                f"parity operator expands to {n_terms} terms "
                f"(> {MAX_PARITY_TERMS}); support is too large for the "
                "term-expansion construction."
            )

        terms = []
        for combo in iter_product(*factors):
            expr = "".join(c[0] for c in combo)
            idxs = [i for c in combo for i in c[1]]
            coeff = float(np.prod([c[2] for c in combo]))
            terms.append((expr, idxs, coeff))
        return terms

    def parity_mpo(self, parity_row: np.ndarray):
        """MPO of the diagonal parity quasi-symmetry for one parity row."""
        self._activate()
        builder = self.driver.expr_builder()
        for expr, idxs, coeff in self._parity_terms(parity_row):
            if expr == "":
                builder.add_const(coeff)
            else:
                builder.add_term(expr, idxs, coeff)
        return self.driver.get_mpo(builder.finalize(), iprint=0)

    def _rotated_occupation_density(
        self, rotation: np.ndarray, orbital: int
    ) -> np.ndarray:
        """One-body matrix of ``U^dagger n_p U``: ``D_qr = U_{p q} U_{p r}``."""
        row = np.asarray(rotation, dtype=np.float64)[int(orbital)]
        return np.outer(row, row)

    def _spin_parity_factor_mpo(
        self, density: np.ndarray, alpha: bool, beta: bool
    ):
        """MPO for one rotated factor ``(1-2ñ_α)^a (1-2ñ_β)^b``.

        ``density`` is the spatial one-body matrix of ``ñ = U^dagger n_p U``.
        """
        self._activate()
        builder = self.driver.expr_builder()
        builder.add_const(1.0)
        if alpha:
            builder.add_sum_term("cd", -2.0 * density)
        if beta:
            builder.add_sum_term("CD", -2.0 * density)
        if alpha and beta:
            # a†_qα a†_sβ a_tβ a_rα with coeff 4 D[q,r] D[s,t]
            builder.add_sum_term(
                "cCDd", 4.0 * np.einsum("qr,st->qstr", density, density)
            )
        return self.driver.get_mpo(builder.finalize(), iprint=0)

    def rotated_parity_factor_mpos(
        self, parity_row: np.ndarray, rotation: np.ndarray
    ) -> list:
        """Factor MPOs of ``U^dagger S U`` for one parity-matrix row.

        Each factor is a single-orbital (or single-spin) rotated parity and
        the factors commute. Apply them sequentially with :meth:`apply_mpo`
        to act with the full ``S̃``; for a single factor the list has length 1
        and equals ``[rotated_parity_mpo(...)]``.
        """
        parity_row = np.asarray(parity_row, dtype=int)
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (self.n_sites, self.n_sites):
            raise ValueError(
                f"rotation must have shape {(self.n_sites, self.n_sites)}, "
                f"got {rotation.shape}"
            )

        factors = []
        if parity_row.shape[0] == self.n_sites:
            for p in np.flatnonzero(parity_row):
                density = self._rotated_occupation_density(rotation, int(p))
                factors.append(self._spin_parity_factor_mpo(density, True, True))
        elif parity_row.shape[0] == 2 * self.n_sites:
            for p in range(self.n_sites):
                alpha = bool(parity_row[2 * p])
                beta = bool(parity_row[2 * p + 1])
                if not (alpha or beta):
                    continue
                density = self._rotated_occupation_density(rotation, p)
                factors.append(
                    self._spin_parity_factor_mpo(density, alpha, beta)
                )
        else:
            raise ValueError(
                "parity row length must be norb or 2 * norb, got "
                f"{parity_row.shape[0]} for norb={self.n_sites}"
            )
        if not factors:
            # Empty support: S = 1
            self._activate()
            factors = [self.driver.get_identity_mpo()]
        return factors

    def rotated_parity_mpo(self, parity_row: np.ndarray, rotation: np.ndarray):
        """MPO of ``U^dagger S U`` when the row support has a single factor.

        For multi-orbital rows use :meth:`rotated_parity_factor_mpos` and
        :meth:`apply_rotated_parity` instead — forming one dense product MPO
        is unnecessary for the cost functions and scales poorly.
        """
        factors = self.rotated_parity_factor_mpos(parity_row, rotation)
        if len(factors) != 1:
            raise ValueError(
                f"rotated_parity_mpo expects a single-factor row, got "
                f"{len(factors)} factors; use apply_rotated_parity instead"
            )
        return factors[0]

    def apply_mpo(
        self,
        mpo,
        ket=None,
        tag: str = "APPLY",
        bond_dim: int | None = None,
        n_sweeps: int = 8,
        tol: float = 1e-10,
    ):
        """Return a new MPS equal to ``mpo |ket>`` (fitted, not normalized).

        Noise is kept at zero: noisy multiplies are unreliable for general
        (non-QC) MPOs in block2. The returned MPS has the physical norm
        ``||mpo|ket>||``; ``driver.multiply``'s return value is that norm.
        """
        self._activate()
        if ket is None:
            ket = self.get_mps()
        if bond_dim is None:
            try:
                bond_dim = max(int(ket.info.get_max_bond_dimension()), 2)
            except Exception:
                try:
                    bond_dim = max(int(ket.info.bond_dim), 2)
                except Exception:
                    bond_dim = 200

        bra = self.driver.get_random_mps(
            tag=tag, bond_dim=bond_dim, nroots=1
        )
        self.driver.multiply(
            bra,
            mpo,
            ket,
            n_sweeps=n_sweeps,
            tol=tol,
            bond_dims=[bond_dim] * n_sweeps,
            noises=[0.0] * n_sweeps,
            thrds=[tol] * n_sweeps,
            iprint=0,
        )
        return bra

    def apply_rotated_parity(
        self,
        parity_row: np.ndarray,
        rotation: np.ndarray,
        ket=None,
        tag: str = "SPARITY",
        bond_dim: int | None = None,
        n_sweeps: int = 8,
        tol: float = 1e-10,
    ):
        """Apply ``U^dagger S U`` to ``ket`` by sequential factor multiplies."""
        if ket is None:
            ket = self.get_mps()
        current = ket
        for i, mpo in enumerate(
            self.rotated_parity_factor_mpos(parity_row, rotation)
        ):
            current = self.apply_mpo(
                mpo,
                ket=current,
                tag=f"{tag}_f{i}",
                bond_dim=bond_dim,
                n_sweeps=n_sweeps,
                tol=tol,
            )
        return current

    def mps_overlap(self, bra, ket=None) -> complex:
        """``<bra|ket>`` via the identity MPO."""
        if ket is None:
            ket = self.get_mps()
        self._activate()
        return self.driver.expectation(bra, self.driver.get_identity_mpo(), ket)

    def mps_norm2(self, ket) -> float:
        """``||ket||^2 = <ket|ket>``."""
        return float(np.real(self.mps_overlap(ket, ket)))

    def _spin_parity_vectors(
        self, parity_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-symmetry alpha/beta support vectors, shape (n_syms, norb)."""
        parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        if parity_matrix.shape[1] == self.n_sites:
            alpha = parity_matrix.copy()
            beta = parity_matrix.copy()
        elif parity_matrix.shape[1] == 2 * self.n_sites:
            alpha = parity_matrix[:, 0::2]
            beta = parity_matrix[:, 1::2]
        else:
            raise ValueError(
                "parity matrix columns must be norb or 2 * norb, got "
                f"{parity_matrix.shape[1]} for norb={self.n_sites}"
            )
        return alpha % 2, beta % 2

    def decoupled_integrals(
        self, parity_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Integrals of the sector-block-diagonal part of the Hamiltonian.

        Every second-quantized term that flips any parity ``S_k`` (odd overlap
        between the term's spin orbitals and the symmetry support) is zeroed.
        The result ``H_dec = (h1a, h1b, gaa, gab, gbb)`` commutes exactly with
        every ``S_k`` and its restriction to a sector coincides with the
        sector Hamiltonians built by ``metrics.subspace_matrix``.
        """
        alpha, beta = self._spin_parity_vectors(parity_matrix)
        g2e_full = restore_g2e(self.g2e, self.n_sites)

        h1a, h1b = self.h1e.copy(), self.h1e.copy()
        gaa, gab, gbb = g2e_full.copy(), g2e_full.copy(), g2e_full.copy()
        for a_k, b_k in zip(alpha, beta):
            h1a[(a_k[:, None] ^ a_k[None, :]) == 1] = 0.0
            h1b[(b_k[:, None] ^ b_k[None, :]) == 1] = 0.0
            pair_a = a_k[:, None] ^ a_k[None, :]
            pair_b = b_k[:, None] ^ b_k[None, :]
            gaa[(pair_a[:, :, None, None] ^ pair_a[None, None, :, :]) == 1] = 0.0
            gab[(pair_a[:, :, None, None] ^ pair_b[None, None, :, :]) == 1] = 0.0
            gbb[(pair_b[:, :, None, None] ^ pair_b[None, None, :, :]) == 1] = 0.0
        return h1a, h1b, gaa, gab, gbb

    def _qc_expr_builder(self, integrals=None):
        """Expression builder pre-filled with an electronic Hamiltonian.

        ``integrals`` is an optional spin-resolved ``(h1a, h1b, gaa, gab,
        gbb)`` tuple in chemist notation; the stored spin-free integrals are
        used when omitted.
        """
        if integrals is None:
            g2e_full = restore_g2e(self.g2e, self.n_sites)
            h1a = h1b = self.h1e
            gaa = gab = gbb = g2e_full
        else:
            h1a, h1b, gaa, gab, gbb = integrals
        gba = gab.transpose(2, 3, 0, 1)

        builder = self.driver.expr_builder()
        builder.add_sum_term("cd", h1a)
        builder.add_sum_term("CD", h1b)
        for expr, tensor in (
            ("ccdd", gaa), ("cCDd", gab), ("CcdD", gba), ("CCDD", gbb)
        ):
            builder.add_sum_term(expr, 0.5 * tensor.transpose(0, 2, 3, 1))
        builder.add_const(self.ecore)
        return builder

    def sector_hamiltonian_mpo(
        self,
        parity_matrix: np.ndarray,
        sector_label: Sequence[int],
        penalty: float,
    ):
        """MPO of ``H_dec + penalty * sum_k (1 - sigma_k S_k) / 2``.

        ``H_dec`` is the sector-block-diagonal (decoupled) Hamiltonian from
        :meth:`decoupled_integrals`; because it commutes with every ``S_k``,
        the penalty separates sectors exactly and the ground state in the
        sector ``sigma_k = (-1)**sector_label[k]`` is obtained without
        cross-sector contamination.
        """
        parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        sector_label = tuple(int(b) for b in sector_label)
        if len(sector_label) != parity_matrix.shape[0]:
            raise ValueError("sector_label length must match parity rows")

        self._activate()
        builder = self._qc_expr_builder(self.decoupled_integrals(parity_matrix))
        for row, bit in zip(parity_matrix, sector_label):
            sigma = -1.0 if bit else 1.0
            builder.add_const(penalty / 2)
            for expr, idxs, coeff in self._parity_terms(row):
                scaled = -sigma * penalty / 2 * coeff
                if expr == "":
                    builder.add_const(scaled)
                else:
                    builder.add_term(expr, idxs, scaled)
        return self.driver.get_mpo(builder.finalize(), iprint=0)

    # ------------------------------------------------------------------
    # Solves
    # ------------------------------------------------------------------

    def _run_dmrg(
        self,
        mpo,
        config: DMRGConfig,
        tag: str,
        nroots: int = 1,
        initial_mps_tag: str | None = None,
        initial_determinants: Sequence[str] | None = None,
        projected_mps_tags: Sequence[str] | None = None,
        projection_weight: float = 10.0,
    ) -> tuple[float | list[float], float]:
        self._activate()
        bond_dims, noises, thrds = config.schedule()
        if initial_determinants is not None:
            coefficients = np.ones(len(initial_determinants), dtype=float)
            coefficients /= np.linalg.norm(coefficients)
            ket = self.driver.get_mps_from_csf_coefficients(
                list(initial_determinants),
                coefficients,
                tag,
                full_fci=True,
                iprint=0,
            )
        elif initial_mps_tag is not None and initial_mps_tag in self.stored_tags():
            initial = self.get_mps(initial_mps_tag, nroots=nroots)
            ket = initial.deep_copy(tag)
        else:
            ket = self.driver.get_random_mps(
                tag=tag, bond_dim=bond_dims[0], nroots=nroots
            )
        projected_mps = [
            self.get_mps(projected_tag)
            for projected_tag in (projected_mps_tags or [])
        ]
        start = time.perf_counter()
        energy = self.driver.dmrg(
            mpo,
            ket,
            n_sweeps=config.n_sweeps,
            tol=config.energy_tol,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            proj_mpss=projected_mps or None,
            proj_weights=(
                [float(projection_weight)] * len(projected_mps)
                if projected_mps else None
            ),
            iprint=0,
        )
        # ``deep_copy`` creates a valid warm-start MPS but does not write the
        # tag-specific info file that ``load_mps`` needs later.  Persist both
        # random and warm-started states in the same format.
        # The sweep already writes the MPS tensors.  Only the tag-specific
        # info file is missing for a state created through ``deep_copy``.
        ket.info.save_data(str(self.store_dir / f"{tag}-mps_info.bin"))
        elapsed = time.perf_counter() - start
        if nroots == 1:
            return float(energy), elapsed
        return [float(e) for e in energy], elapsed

    def run_ground_state(self, config: DMRGConfig | None = None) -> DMRGResult:
        """Solve for the ground state and persist the MPS in the local store."""
        config = config or DMRGConfig()
        energy, elapsed = self._run_dmrg(
            self.hamiltonian_mpo(), config, config.mps_tag
        )
        result = DMRGResult(
            energy=energy,
            mps_tag=config.mps_tag,
            store_dir=str(self.store_dir),
            config=config,
            elapsed_seconds=elapsed,
        )
        self._record_run(result)
        logger.info(
            "DMRG ground state: E = %.10f (tag=%s, store=%s)",
            energy, config.mps_tag, self.store_dir,
        )
        return result

    def sector_ground_state(
        self,
        parity_matrix: np.ndarray,
        sector_label: Sequence[int],
        penalty: float = 10.0,
        config: DMRGConfig | None = None,
        verify_tol: float = 1e-2,
        mps_tag: str | None = None,
        initial_mps_tag: str | None = None,
    ) -> DMRGResult:
        """Sector-restricted ground state of the decoupled Hamiltonian.

        This is the DMRG analog of diagonalizing the sector blocks built by
        ``metrics.subspace_matrix``: cross-sector couplings are removed
        exactly (see :meth:`decoupled_integrals`) and a penalty selects the
        requested sector. After the solve the parity expectations are
        measured; a warning is logged when they do not match the requested
        sector within ``verify_tol`` (e.g. because the penalty was too weak).
        """
        sector_label = tuple(int(b) for b in sector_label)
        tag = mps_tag or ("SECTOR_" + "".join(map(str, sector_label)))
        config = config or DMRGConfig(mps_tag=tag)
        if config.mps_tag != tag:
            config = DMRGConfig(**{**asdict(config), "mps_tag": tag})

        mpo = self.sector_hamiltonian_mpo(parity_matrix, sector_label, penalty)
        target_determinants = find_sector_determinants(
            parity_matrix,
            sector_label,
            self.n_sites,
            self.n_elec,
            self.spin,
            maximum=32,
        )
        warm_start_available = (
            initial_mps_tag is not None and initial_mps_tag in self.stored_tags()
        )
        energy, elapsed = self._run_dmrg(
            mpo,
            config,
            tag,
            initial_mps_tag=initial_mps_tag,
            initial_determinants=(
                None if warm_start_available else target_determinants
            ),
        )

        ket = self.get_mps(tag)
        expectations = self.symmetry_expectations(parity_matrix, ket=ket)
        targets = np.array([(-1.0) ** b for b in sector_label])
        if not np.allclose(expectations, targets, atol=verify_tol):
            logger.warning(
                "warm-started sector %s was not clean; retrying from a target "
                "determinant",
                sector_label,
            )
            retry_energy, retry_elapsed = self._run_dmrg(
                mpo,
                config,
                tag,
                initial_determinants=target_determinants,
            )
            energy = retry_energy
            elapsed += retry_elapsed
            ket = self.get_mps(tag)
            expectations = self.symmetry_expectations(parity_matrix, ket=ket)
        if not np.allclose(expectations, targets, atol=verify_tol):
            raise RuntimeError(
                f"sector {sector_label} targeting failed: <S_k>={expectations}, "
                f"targets={targets}, penalty={penalty}"
            )

        # Report the bare electronic energy in this state (penalty contribution
        # vanishes on a clean sector; using <H> keeps the diagnostic meaningful
        # even when the sector is only approximately selected).
        energy = self.energy_expectation(ket)

        result = DMRGResult(
            energy=energy,
            mps_tag=tag,
            store_dir=str(self.store_dir),
            config=config,
            elapsed_seconds=elapsed,
            sector_label=sector_label,
            symmetry_expectations=tuple(float(x) for x in expectations),
        )
        self._record_run(result)
        return result

    def sector_excited_states(
        self,
        parity_matrix: np.ndarray,
        sector_label: Sequence[int],
        nroots: int = 5,
        penalty: float = 30.0,
        config: DMRGConfig | None = None,
        verify_tol: float = 1e-2,
    ) -> list[tuple[float, str]]:
        """Lowest ``nroots`` states from sequential sector-targeted DMRG.

        Every root starts from determinants in the requested parity sector.
        Previously converged roots are projected out in subsequent solves.
        This is slower than an unrestricted random state-averaged solve, but it
        prevents the roots from silently converging to another parity sector.
        """
        if nroots < 1:
            raise ValueError("nroots must be >= 1")
        sector_label = tuple(int(b) for b in sector_label)
        base_tag = "SECTOR_" + "".join(map(str, sector_label))
        config = config or DMRGConfig(mps_tag=base_tag)
        mpo = self.sector_hamiltonian_mpo(parity_matrix, sector_label, penalty)
        target_determinants = find_sector_determinants(
            parity_matrix,
            sector_label,
            self.n_sites,
            self.n_elec,
            self.spin,
            maximum=max(32, 4 * int(nroots)),
        )
        targets = np.array([(-1.0) ** b for b in sector_label])
        results: list[tuple[float, str]] = []
        previous_tags = []
        for iroot in range(int(nroots)):
            root_tag = f"{base_tag}_r{iroot}"
            root_config = DMRGConfig(
                **{**asdict(config), "mps_tag": root_tag}
            )
            anchor_tag = (
                base_tag
                if iroot == 0 and base_tag in self.stored_tags()
                else None
            )
            _penalized_energy, elapsed = self._run_dmrg(
                mpo,
                root_config,
                root_tag,
                initial_mps_tag=anchor_tag,
                initial_determinants=(
                    None if anchor_tag is not None else target_determinants
                ),
                projected_mps_tags=previous_tags,
                projection_weight=max(10.0, float(penalty)),
            )
            ket = self.get_mps(root_tag)
            expectations = self.symmetry_expectations(parity_matrix, ket=ket)
            if not np.allclose(expectations, targets, atol=verify_tol):
                raise RuntimeError(
                    f"sector {sector_label} root {iroot} targeting failed: "
                    f"<S_k>={expectations}, targets={targets}, penalty={penalty}"
                )
            bare = self.energy_expectation(ket)
            results.append((bare, root_tag))
            previous_tags.append(root_tag)
            self._record_run(DMRGResult(
                energy=bare,
                mps_tag=root_tag,
                store_dir=str(self.store_dir),
                config=root_config,
                elapsed_seconds=elapsed,
                sector_label=sector_label,
                symmetry_expectations=tuple(float(x) for x in expectations),
            ))

        results.sort(key=lambda item: item[0])
        logger.info(
            "sector %s: %d roots, E0=%.10f",
            sector_label, len(results), results[0][0],
        )
        return results

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def expectation(self, mpo, ket=None, bra=None) -> float:
        """``<bra|MPO|ket>`` (defaults to the stored ground state)."""
        if ket is None:
            ket = self.get_mps()
        if bra is None:
            bra = ket
        self._activate()
        return float(self.driver.expectation(bra, mpo, ket))

    def energy_expectation(self, ket=None) -> float:
        """``<ket|H|ket>`` for a stored or supplied MPS."""
        return self.expectation(self.hamiltonian_mpo(), ket=ket)

    def spin_resolved_rdms(self, ket=None) -> tuple[np.ndarray, np.ndarray]:
        """Return conventional spin-resolved 1- and 2-particle RDMs.

        The returned arrays follow the Block2 SZ conventions documented by
        ``DMRGDriver.get_conventional_npdm``.  The first axes contain
        ``(alpha, beta)`` for the 1-RDM and ``(aa, ab, bb)`` for the 2-RDM.
        """
        if ket is None:
            ket = self.get_mps()
        self._activate()
        rdm1 = self.driver.get_conventional_1pdm(ket, iprint=0)
        rdm2 = self.driver.get_conventional_2pdm(ket, iprint=0)
        return np.asarray(rdm1), np.asarray(rdm2)

    def symmetry_expectations(
        self, parity_matrix: np.ndarray, ket=None
    ) -> np.ndarray:
        """``<S_k>`` for every row of a parity matrix."""
        parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        if ket is None:
            ket = self.get_mps()
        return np.array(
            [self.expectation(self.parity_mpo(row), ket=ket)
             for row in parity_matrix]
        )

    def bipartite_entanglement(self, ket=None) -> np.ndarray:
        """Von Neumann entropies (nats) across every orbital-chain cut."""
        if ket is None:
            ket = self.get_mps()
        self._activate()
        return np.asarray(self.driver.get_bipartite_entanglement(ket))

    def orbital_entropies(self, ket=None, orb_type: int = 1) -> np.ndarray:
        """1-orbital (``orb_type=1``) or 2-orbital (``orb_type=2``) entropies."""
        if ket is None:
            ket = self.get_mps()
        self._activate()
        return np.asarray(
            self.driver.get_orbital_entropies(ket, orb_type=orb_type, iprint=0)
        )

    def mutual_information(self, ket=None) -> np.ndarray:
        """Orbital mutual information ``I_ij = s_i + s_j - s_ij`` (diag zeroed)."""
        s1 = self.orbital_entropies(ket=ket, orb_type=1)
        s2 = self.orbital_entropies(ket=ket, orb_type=2)
        info = s1[:, None] + s1[None, :] - s2
        np.fill_diagonal(info, 0.0)
        return info

    # ------------------------------------------------------------------
    # Dense reconstructions (small systems / validation)
    # ------------------------------------------------------------------

    def _determinants(self, ket, cutoff: float, fci_conv: bool):
        self._activate()
        dets, coeffs = self.driver.get_csf_coefficients(
            ket, cutoff=cutoff, fci_conv=fci_conv, iprint=0
        )
        return np.asarray(dets), np.asarray(coeffs)

    def to_ci_vector(self, ket=None, cutoff: float = 1e-14) -> np.ndarray:
        """Flattened CI vector in the PySCF/ffsim convention.

        The result matches ``pyscf.fci`` (and ffsim) ordering: entry
        ``addr_alpha * dim_beta + addr_beta`` with lexicographic string
        addressing, so it is a drop-in replacement for the flattened
        ``fcivec`` returned by ``optimize_symmetries.get_fci``.
        """
        if ket is None:
            ket = self.get_mps()
        n_alpha = (self.n_elec + self.spin) // 2
        n_beta = self.n_elec - n_alpha
        dim_alpha = comb(self.n_sites, n_alpha)
        dim_beta = comb(self.n_sites, n_beta)
        if dim_alpha * dim_beta > MAX_CI_DIMENSION:
            raise ValueError(
                f"CI dimension {dim_alpha * dim_beta} exceeds "
                f"{MAX_CI_DIMENSION}; dense reconstruction refused."
            )

        dets, coeffs = self._determinants(ket, cutoff, fci_conv=True)
        vector = np.zeros(dim_alpha * dim_beta, dtype=np.complex128)
        for det, coeff in zip(dets, coeffs):
            occ_alpha = [p for p in range(self.n_sites) if det[p] in (1, 3)]
            occ_beta = [p for p in range(self.n_sites) if det[p] in (2, 3)]
            address = (
                _string_address(occ_alpha) * dim_beta
                + _string_address(occ_beta)
            )
            vector[address] = coeff
        return vector

    def to_statevector(self, ket=None, cutoff: float = 1e-14) -> np.ndarray:
        """Statevector over ``2 * norb`` qubits in the JW interleaved order.

        Spin orbitals are ordered ``[a0, b0, a1, b1, ...]`` with qubit 0 as
        the leftmost (most significant) tensor factor, matching
        ``openfermion.get_sparse_operator``.
        """
        if ket is None:
            ket = self.get_mps()
        n_qubits = 2 * self.n_sites
        if n_qubits > MAX_STATEVECTOR_QUBITS:
            raise ValueError(
                f"{n_qubits} qubits exceeds {MAX_STATEVECTOR_QUBITS}; "
                "dense reconstruction refused."
            )

        dets, coeffs = self._determinants(ket, cutoff, fci_conv=False)
        vector = np.zeros(1 << n_qubits, dtype=np.complex128)
        for det, coeff in zip(dets, coeffs):
            index = 0
            for p in range(self.n_sites):
                if det[p] in (1, 3):
                    index |= 1 << (n_qubits - 1 - 2 * p)
                if det[p] in (2, 3):
                    index |= 1 << (n_qubits - 1 - (2 * p + 1))
            vector[index] = coeff
        return vector

    # ------------------------------------------------------------------
    # Sector utilities
    # ------------------------------------------------------------------

    def dominant_sector_labels(
        self,
        parity_matrix: np.ndarray,
        ket=None,
        cutoff: float = 1e-6,
        max_sectors: int | None = None,
    ) -> list[tuple[tuple[int, ...], float]]:
        """Sector labels present in an MPS, sorted by total weight.

        Samples the determinant expansion of ``ket`` and classifies each
        determinant by the parity-matrix rows, returning
        ``[(label, weight), ...]``. This bounds which sectors are worth
        solving with :meth:`sector_ground_state` on large systems where
        sectors cannot be enumerated.
        """
        parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        if ket is None:
            ket = self.get_mps()
        dets, coeffs = self._determinants(ket, cutoff, fci_conv=False)

        weights: dict[tuple[int, ...], float] = {}
        for det, coeff in zip(dets, coeffs):
            label = tuple(
                int(self._determinant_parity_bit(det, row))
                for row in parity_matrix
            )
            weights[label] = weights.get(label, 0.0) + abs(coeff) ** 2

        ordered = sorted(weights.items(), key=lambda kv: -kv[1])
        if max_sectors is not None:
            ordered = ordered[:max_sectors]
        return ordered

    def _determinant_parity_bit(self, det: np.ndarray, row: np.ndarray) -> int:
        """Parity label bit (0 for +1, 1 for -1) of one determinant."""
        row = np.asarray(row, dtype=int)
        count = 0
        if row.shape[0] == self.n_sites:
            for p in np.flatnonzero(row):
                occ = det[p]
                count += (1 if occ in (1, 3) else 0) + (1 if occ in (2, 3) else 0)
        elif row.shape[0] == 2 * self.n_sites:
            for p in range(self.n_sites):
                if row[2 * p] and det[p] in (1, 3):
                    count += 1
                if row[2 * p + 1] and det[p] in (2, 3):
                    count += 1
        else:
            raise ValueError("parity row length must be norb or 2 * norb")
        return count % 2


def solve_or_load_ground_state(
    solver: Block2DMRGSolver,
    config: DMRGConfig | None = None,
    reuse: bool = True,
) -> DMRGResult:
    """Return the stored ground-state result if present, else solve and store."""
    config = config or DMRGConfig()
    if reuse and config.mps_tag in solver.stored_tags():
        run = solver.read_metadata(solver.store_dir)["runs"][config.mps_tag]
        logger.info(
            "reusing stored wavefunction %s (E = %.10f)",
            solver.store_dir, run["energy"],
        )
        stored_config = DMRGConfig(**{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in run["config"].items()
        })
        return DMRGResult(
            energy=float(run["energy"]),
            mps_tag=config.mps_tag,
            store_dir=str(solver.store_dir),
            config=stored_config,
            elapsed_seconds=float(run["elapsed_seconds"]),
        )
    return solver.run_ground_state(config)


def get_dmrg_reference(
    dumpdata: Mapping,
    store_dir: str | Path | None = None,
    config: DMRGConfig | None = None,
    reuse: bool = True,
    n_threads: int = 4,
) -> tuple[float, np.ndarray]:
    """DMRG drop-in replacement for ``optimize_symmetries.get_fci``.

    Solves (or reloads) the ground state for a ``chemistry.fcidump_data``
    dictionary and returns ``(energy, flattened CI vector)`` in the same
    convention as ``get_fci(dumpdata)``.
    """
    solver = Block2DMRGSolver.from_dumpdata(
        dumpdata, store_dir=store_dir, n_threads=n_threads
    )
    result = solve_or_load_ground_state(solver, config=config, reuse=reuse)
    return result.energy, solver.to_ci_vector(solver.get_mps(result.mps_tag))
