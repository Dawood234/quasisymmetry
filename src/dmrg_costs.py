"""MPS-native orbital-optimization costs for the quasisymmetry pipeline.

The statevector cost in ``optimize_symmetries`` is

    NC(x) = sum_k ||[H(U), S_k] U|ψ⟩||²
          = sum_k ||[H, U^dagger S_k U] |ψ⟩||²

with ``U = expm(A(x))``. This module evaluates the right-hand side entirely
with block2 MPO/MPS operations on a **fixed** DMRG reference ``|ψ⟩``:

* DMRG is run once (or reloaded from a local wavefunction store);
* ``η = H|ψ⟩`` is cached;
* each optimizer step builds the rotated parity operators
  ``S̃_k = U^dagger S_k U`` as factor MPOs and applies them by multiply;
* the NC residual is ``||H S̃|ψ⟩ - S̃|η⟩||²`` and the variance cost is
  ``1 - |⟨ψ|S̃|ψ⟩|²``.

Output of the optimizer is still the rotation vector ``x``, so
``rotate_fcidump.py``, ``metrics.py`` and ``solve_dmrg.py --U`` stay
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    DMRGResult,
    rotation_from_parameters,
    solve_or_load_ground_state,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiplyConfig:
    """Sweep settings for MPO–MPS multiplies used inside the cost."""

    bond_dim: int | None = None  # default: inherit from the reference MPS
    n_sweeps: int = 8
    tol: float = 1e-10
    bra_bond_dim_factor: float = 1.5
    """Extra room for ``H|φ⟩`` / ``S̃|η⟩`` relative to the reference bond dim."""


class DMRGOrbitalCosts:
    """Callable NC / variance costs over orbital-rotation parameters ``x``.

    Parameters
    ----------
    solver:
        A :class:`Block2DMRGSolver` whose ground-state MPS is already solved
        (or will be loaded via ``mps_tag``).
    parity_matrix:
        Quasi-symmetry incidence matrix (``n_sym × norb`` or
        ``n_sym × 2 norb``), same convention as ``optimize_symmetries``.
    mps_tag:
        Tag of the reference MPS inside ``solver.store_dir``.
    multiply:
        Fit settings for intermediate MPO–MPS multiplies.
    """

    def __init__(
        self,
        solver: Block2DMRGSolver,
        parity_matrix: np.ndarray,
        mps_tag: str = "GS",
        multiply: MultiplyConfig | None = None,
        pairs=None,
    ) -> None:
        self.solver = solver
        self.parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        self.mps_tag = mps_tag
        self.multiply = multiply or MultiplyConfig()
        self.pairs = pairs
        self._eval_count = 0
        self._tag_serial = 0

        self.solver._activate()
        # Working copy of the reference MPS. block2 multiply may rewrite the
        # ket's on-disk tensors while sweeping; never point this at the stored
        # "GS" tag or later reloads / simultaneous multiplies will corrupt it.
        stored = self.solver.get_mps(mps_tag)
        self.ket = stored.deep_copy("COST_KET")
        self._h_mpo = self.solver.hamiltonian_mpo()
        self._eta = None  # cached H|ψ⟩ (also a working copy)
        self._ref_bond_dim = self._bond_dim_of(self.ket)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _bond_dim_of(mps) -> int:
        try:
            return max(int(mps.info.get_max_bond_dimension()), 2)
        except Exception:
            try:
                return max(int(mps.info.bond_dim), 2)
            except Exception:
                return 200

    def _next_tag(self, prefix: str) -> str:
        self._tag_serial += 1
        return f"{prefix}_{self._tag_serial}"

    def _bra_bond_dim(self, base: int | None = None) -> int:
        base = int(base or self.multiply.bond_dim or self._ref_bond_dim)
        return max(2, int(np.ceil(base * self.multiply.bra_bond_dim_factor)))

    def _apply(self, mpo, ket, prefix: str, bond_dim: int | None = None):
        # Defensive copy so the source ket's disk files are not rewritten.
        ket_tag = self._next_tag(f"{prefix}_SRC")
        ket_work = ket.deep_copy(ket_tag)
        return self.solver.apply_mpo(
            mpo,
            ket=ket_work,
            tag=self._next_tag(prefix),
            bond_dim=bond_dim or self._bra_bond_dim(),
            n_sweeps=self.multiply.n_sweeps,
            tol=self.multiply.tol,
        )

    def _apply_symmetry(self, row: np.ndarray, rotation: np.ndarray, ket, prefix: str):
        ket_work = ket.deep_copy(self._next_tag(f"{prefix}_SRC"))
        return self.solver.apply_rotated_parity(
            row,
            rotation,
            ket=ket_work,
            tag=self._next_tag(prefix),
            bond_dim=self._bra_bond_dim(),
            n_sweeps=self.multiply.n_sweeps,
            tol=self.multiply.tol,
        )

    def _ensure_eta(self) -> None:
        if self._eta is None:
            logger.info("caching H|psi> for MPS-native costs")
            self._eta = self._apply(
                self._h_mpo, self.ket, "ETA",
                bond_dim=self._bra_bond_dim(self._ref_bond_dim),
            )

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def variance(self, x: np.ndarray) -> float:
        """``sum_k (1 - |<ψ|U^dagger S_k U|ψ>|^2)``."""
        self._eval_count += 1
        rotation = rotation_from_parameters(x, self.solver.n_sites, self.pairs)
        total = 0.0
        for row in self.parity_matrix:
            phi = self._apply_symmetry(row, rotation, self.ket, "VPHI")
            # S̃ is Hermitian and involutory, so <ψ|S̃|ψ> = <ψ|φ>
            expectation = np.real(self.solver.mps_overlap(self.ket, phi))
            total += 1.0 - expectation ** 2
        return float(total)

    def commutator(self, x: np.ndarray) -> float:
        """``sum_k ||[H, U^dagger S_k U] |ψ>||^2``."""
        self._eval_count += 1
        self._ensure_eta()
        rotation = rotation_from_parameters(x, self.solver.n_sites, self.pairs)
        total = 0.0
        for row in self.parity_matrix:
            phi = self._apply_symmetry(row, rotation, self.ket, "CPHI")
            xi = self._apply_symmetry(row, rotation, self._eta, "CXI")
            chi = self._apply(self._h_mpo, phi, "CCHI")
            # ||chi - xi||^2 = <chi|chi> + <xi|xi> - 2 Re <chi|xi>
            c2 = self.solver.mps_norm2(chi)
            x2 = self.solver.mps_norm2(xi)
            cx = self.solver.mps_overlap(chi, xi)
            total += c2 + x2 - 2.0 * float(np.real(cx))
        return float(total)

    def cost_function(self, kind: str = "NC") -> Callable[[np.ndarray], float]:
        """Return a scipy-optimize-compatible objective ``f(x)``."""
        kind = kind.lower()
        if kind in ("nc", "commutator"):
            return self.commutator
        if kind == "variance":
            return self.variance
        raise ValueError("cost kind must be 'NC' or 'variance'")

    @property
    def n_evaluations(self) -> int:
        return self._eval_count


def build_dmrg_orbital_costs(
    molpath: str,
    parity_matrix: np.ndarray,
    store_dir: str | None = None,
    config: DMRGConfig | None = None,
    multiply: MultiplyConfig | None = None,
    reuse: bool = True,
    rotation: np.ndarray | None = None,
    n_threads: int = 4,
    pairs=None,
) -> tuple[DMRGOrbitalCosts, DMRGResult, Block2DMRGSolver]:
    """Solve (or reload) the reference MPS and wrap it as orbital costs.

    ``molpath`` may be an FCIDUMP (pyscf-free) or a ``.chk`` (needs pyscf).
    """
    from pathlib import Path

    path = Path(molpath)
    config = config or DMRGConfig()
    if path.suffix == ".chk":
        from chemistry import fcidump_data

        solver = Block2DMRGSolver.from_dumpdata(
            fcidump_data(str(path)),
            store_dir=store_dir,
            n_threads=n_threads,
        )
    else:
        solver = Block2DMRGSolver.from_fcidump(
            path, store_dir=store_dir, n_threads=n_threads
        )

    if rotation is not None:
        from src.dmrg_solver import rotate_integrals

        h1e, g2e = rotate_integrals(solver.h1e, solver.g2e, rotation)
        solver = Block2DMRGSolver(
            h1e=h1e,
            g2e=g2e,
            ecore=solver.ecore,
            n_elec=solver.n_elec,
            spin=solver.spin,
            store_dir=store_dir or solver.store_dir,
            n_threads=n_threads,
        )

    result = solve_or_load_ground_state(solver, config=config, reuse=reuse)
    costs = DMRGOrbitalCosts(
        solver,
        parity_matrix,
        mps_tag=result.mps_tag,
        multiply=multiply,
        pairs=pairs,
    )
    return costs, result, solver


def commutator_scores_by_row(costs, x):
    """Return one MPS-native NC score for every parity row.

    This is the per-generator form of :meth:`DMRGOrbitalCosts.commutator`.
    The expensive reference action ``H|psi>`` is cached once and reused for
    all rows, which is required when ranking a seniority/quartet pool.
    """
    costs._eval_count += 1
    costs._ensure_eta()
    rotation = rotation_from_parameters(
        np.asarray(x, dtype=float), costs.solver.n_sites, costs.pairs
    )
    scores = []

    total_rows = len(costs.parity_matrix)
    for index, row in enumerate(costs.parity_matrix, start=1):
        print(f"[NC score] candidate {index}/{total_rows}", flush=True)
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SCORE_PHI")
        xi = costs._apply_symmetry(row, rotation, costs._eta, "SCORE_XI")
        chi = costs._apply(costs._h_mpo, phi, "SCORE_CHI")
        c2 = costs.solver.mps_norm2(chi)
        x2 = costs.solver.mps_norm2(xi)
        cx = costs.solver.mps_overlap(chi, xi)
        score = c2 + x2 - 2.0 * float(np.real(cx))
        scores.append(max(0.0, float(score)))

    return np.asarray(scores, dtype=float)


def parity_expectations_by_row(costs, x, parity_matrix):
    """Return ``<psi|U^dagger S_k U|psi>`` for selected parity rows."""
    rotation = rotation_from_parameters(
        np.asarray(x, dtype=float), costs.solver.n_sites, costs.pairs
    )
    expectations = []
    rows = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    for index, row in enumerate(rows, start=1):
        print(f"[generator sign] selected {index}/{len(rows)}", flush=True)
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SIGN_PHI")
        value = costs.solver.mps_overlap(costs.ket, phi)
        expectations.append(float(np.real(value)))
    return np.asarray(expectations, dtype=float)
