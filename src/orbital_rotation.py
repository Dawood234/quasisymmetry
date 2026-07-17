"""Orbital-rotation packing: full SO(n) or intra-irrep (point-group) pairs.

Parameters ``x`` pack into a skew generator ``A`` with free entries only on the
allowed planes ``(i, j)``, then ``U = expm(A)``. Full mode uses every upper-
triangle pair (same order as ``np.triu_indices(norb, k=1)``). Irrep mode keeps
only pairs that share an irrep label, so
``N_sym = sum_Gamma binom(|Gamma|, 2)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.linalg

PairList = list[tuple[int, int]]


def full_pairs(norb: int) -> PairList:
    """All upper-triangle orbital pairs, in ``triu_indices`` order."""
    rows, cols = np.triu_indices(norb, k=1)
    return list(zip(map(int, rows), map(int, cols)))


def irrep_pairs(irreps: Sequence[int]) -> PairList:
    """Intra-irrep pairs only (same label), still in ascending ``(i, j)`` order."""
    irreps = np.asarray(irreps)
    norb = len(irreps)
    pairs: PairList = []
    for i in range(norb):
        for j in range(i + 1, norb):
            if irreps[i] == irreps[j]:
                pairs.append((i, j))
    return pairs


def n_params(norb: int, pairs: PairList | None = None) -> int:
    """Number of free rotation angles for the given packing."""
    if pairs is None:
        return norb * (norb - 1) // 2
    return len(pairs)


def params_to_U(
    x: np.ndarray,
    norb: int,
    pairs: PairList | None = None,
) -> np.ndarray:
    """Build ``U = expm(A(x))`` from upper-triangle (or restricted) parameters."""
    x = np.asarray(x, dtype=float).ravel()
    expected = n_params(norb, pairs)
    if x.size != expected:
        raise ValueError(
            f"rotation parameter length {x.size} does not match packing "
            f"size {expected} (norb={norb}, restricted={pairs is not None})"
        )
    generator = np.zeros((norb, norb), dtype=float)
    if pairs is None:
        generator[np.triu_indices(norb, k=1)] = x
    else:
        for k, (i, j) in enumerate(pairs):
            generator[i, j] = x[k]
    generator -= generator.T
    return scipy.linalg.expm(generator)


def parameter_generator(
    x: np.ndarray,
    norb: int,
    pairs: PairList | None = None,
) -> np.ndarray:
    """Return the skew-symmetric matrix ``A(x)`` used in ``U = exp(A)``."""
    x = np.asarray(x, dtype=float).ravel()
    selected_pairs = full_pairs(norb) if pairs is None else pairs
    if x.size != len(selected_pairs):
        raise ValueError(
            f"rotation parameter length {x.size} does not match packing "
            f"size {len(selected_pairs)} (norb={norb}, "
            f"restricted={pairs is not None})"
        )
    generator = np.zeros((norb, norb), dtype=float)
    for value, (i, j) in zip(x, selected_pairs):
        generator[i, j] = value
        generator[j, i] = -value
    return generator


def rotation_and_derivatives(
    x: np.ndarray,
    norb: int,
    pairs: PairList | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``U = exp(A(x))`` and every exact derivative ``dU/dx_k``.

    The Frechet derivative accounts for the noncommutativity of finite orbital
    rotations.  Replacing it by ``E_k U`` is correct only at special points.
    """
    generator = parameter_generator(x, norb, pairs)
    selected_pairs = full_pairs(norb) if pairs is None else pairs
    rotation = scipy.linalg.expm(generator)
    derivatives = []
    for i, j in selected_pairs:
        direction = np.zeros_like(generator)
        direction[i, j] = 1.0
        direction[j, i] = -1.0
        derivative = scipy.linalg.expm_frechet(
            generator,
            direction,
            compute_expm=False,
        )
        derivatives.append(np.asarray(derivative, dtype=float))
    return rotation, derivatives


def load_orbital_irreps(molpath: str | Path) -> np.ndarray | None:
    """Load per-orbital irrep labels from a symmetry-adapted ``.chk`` or FCIDUMP.

    Returns ``None`` when no useful point-group labels are available (no
    symmetry on the molecule, or FCIDUMP ``ORBSYM`` is missing / all identical).
    """
    path = Path(molpath)
    if path.suffix == ".chk":
        return _irreps_from_chk(path)
    if path.suffix == ".FCIDUMP" or path.name.endswith("FCIDUMP"):
        return _irreps_from_fcidump(path)
    raise ValueError("molpath must be a .chk or FCIDUMP file")


def _irreps_from_chk(path: Path) -> np.ndarray | None:
    import pyscf
    from pyscf import symm

    mol = pyscf.lib.chkfile.load_mol(str(path))
    if not getattr(mol, "symmetry", False):
        return None
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(str(path))
    if hasattr(mf, "get_orbsym"):
        labels = mf.get_orbsym()
    else:
        labels = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff
        )
    return _canonicalize_irreps(labels)


def _irreps_from_fcidump(path: Path) -> np.ndarray | None:
    import pyscf

    data = pyscf.tools.fcidump.read(str(path), verbose=False)
    labels = data.get("ORBSYM")
    if labels is None:
        return None
    labels = np.asarray(labels).ravel()
    # Placeholder dumps often set every ORBSYM entry to 1.
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return _canonicalize_irreps(labels)


def _canonicalize_irreps(labels) -> np.ndarray:
    """Map string or int irrep labels to dense integer ids (order of first appearance)."""
    arr = np.asarray(labels)
    if arr.dtype.kind in "iuf":
        return arr.astype(int)
    mapping: dict = {}
    out = np.empty(arr.size, dtype=int)
    next_id = 0
    for i, lab in enumerate(arr.ravel()):
        key = str(lab)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out[i] = mapping[key]
    return out


def resolve_orbital_rotation(
    mode: str,
    molpath: str | Path,
    norb: int,
) -> tuple[PairList | None, np.ndarray | None]:
    """Resolve ``(pairs, irreps)`` for ``full`` or ``irrep`` packing.

    ``pairs is None`` means full ``SO(n)`` packing.
    """
    mode = (mode or "full").lower()
    if mode == "full":
        return None, None
    if mode != "irrep":
        raise ValueError("orbital_rotation mode must be 'full' or 'irrep'")

    irreps = load_orbital_irreps(molpath)
    if irreps is None:
        raise ValueError(
            "irrep orbital rotations require a symmetry-adapted Hamiltonian "
            "(regenerate with make_pyscf_hamiltonian.py --point_group, or an "
            "FCIDUMP with distinct ORBSYM labels)"
        )
    if len(irreps) != norb:
        raise ValueError(
            f"irrep length {len(irreps)} does not match norb={norb}"
        )
    pairs = irrep_pairs(irreps)
    if not pairs:
        raise ValueError(
            "irrep packing has no free pairs (all irrep blocks have size 1)"
        )
    return pairs, np.asarray(irreps, dtype=int)


def pairs_from_oo_data(data: dict, norb: int) -> PairList | None:
    """Rebuild packing from OO JSON fields (``orbital_rotation`` / ``irreps``)."""
    mode = str(data.get("orbital_rotation", "full")).lower()
    if mode == "full":
        return None
    if mode != "irrep":
        raise ValueError(f"unknown orbital_rotation mode in OO data: {mode!r}")
    irreps = data.get("irreps")
    if irreps is None:
        molpath = data.get("molpath")
        if molpath is None:
            raise ValueError(
                "OO JSON has orbital_rotation=irrep but neither irreps nor molpath"
            )
        pairs, _ = resolve_orbital_rotation("irrep", molpath, norb)
        return pairs
    return irrep_pairs(irreps)


def rotation_from_oo_data(data: dict, norb: int) -> np.ndarray:
    """Build ``U`` from OO JSON ``rotation`` (+ optional irrep packing)."""
    x = np.asarray(data.get("rotation", []), dtype=float)
    if x.size == 0:
        return np.eye(norb)
    pairs = pairs_from_oo_data(data, norb)
    return params_to_U(x, norb, pairs)
