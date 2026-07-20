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

import concurrent.futures
import logging
import multiprocessing
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
            eta_bond_dim = self.multiply.bond_dim or self._ref_bond_dim
            self._eta = self._apply(
                self._h_mpo, self.ket, "ETA",
                bond_dim=self._bra_bond_dim(eta_bond_dim),
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


def commutator_scores_by_row(
    costs,
    x,
    candidate_numbers=None,
    total_candidates=None,
    worker_label=None,
):
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
    if candidate_numbers is None:
        candidate_numbers = list(range(1, total_rows + 1))
    if total_candidates is None:
        total_candidates = total_rows

    for local_index, row in enumerate(costs.parity_matrix):
        candidate_number = int(candidate_numbers[local_index])
        prefix = f"[{worker_label}] " if worker_label else ""
        print(
            f"[NC score] {prefix}candidate "
            f"{candidate_number}/{int(total_candidates)}",
            flush=True,
        )
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SCORE_PHI")
        xi = costs._apply_symmetry(row, rotation, costs._eta, "SCORE_XI")
        chi = costs._apply(costs._h_mpo, phi, "SCORE_CHI")
        c2 = costs.solver.mps_norm2(chi)
        x2 = costs.solver.mps_norm2(xi)
        cx = costs.solver.mps_overlap(chi, xi)
        score = c2 + x2 - 2.0 * float(np.real(cx))
        scores.append(max(0.0, float(score)))

    return np.asarray(scores, dtype=float)


def candidate_index_chunks(n_candidates, n_workers):
    """Split candidate indices into balanced, nonempty worker chunks."""
    n_candidates = int(n_candidates)
    n_workers = max(1, min(int(n_workers), n_candidates))
    indices = np.arange(n_candidates, dtype=int)
    chunks = np.array_split(indices, n_workers)
    return [chunk.tolist() for chunk in chunks if len(chunk)]


def _score_candidate_chunk(task):
    """Load one private MPS store and score one candidate chunk."""
    worker_number = int(task["worker_number"])
    worker_threads = int(task["worker_threads"])
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)

    solver = Block2DMRGSolver.load(
        task["store_dir"],
        n_threads=worker_threads,
    )
    costs = DMRGOrbitalCosts(
        solver,
        np.asarray(task["rows"], dtype=int),
        mps_tag=task["mps_tag"],
        multiply=MultiplyConfig(
            bond_dim=task["multiply_bond_dim"],
            n_sweeps=int(task["multiply_sweeps"]),
            tol=float(task["multiply_tol"]),
            bra_bond_dim_factor=float(task["bra_bond_dim_factor"]),
        ),
        pairs=task["pairs"],
    )
    indices = [int(index) for index in task["indices"]]
    scores = commutator_scores_by_row(
        costs,
        np.asarray(task["x"], dtype=float),
        candidate_numbers=[index + 1 for index in indices],
        total_candidates=int(task["total_candidates"]),
        worker_label=f"worker {worker_number}",
    )
    return indices, scores.tolist()


def _copy_mps_store_for_tag(source_dir, target_dir, mps_tag):
    """Copy only one saved MPS and the integrals needed to reload it."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    exact_names = {
        "integrals.npz",
        "metadata.json",
        f"{mps_tag}-mps_info.bin",
    }
    mps_prefix = f"F.MPS.{mps_tag}."
    info_prefix = f"F.MPS.INFO.{mps_tag}."
    copied = 0
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if (
            source.name in exact_names
            or source.name.startswith(mps_prefix)
            or source.name.startswith(info_prefix)
        ):
            shutil.copy2(source, target_dir / source.name)
            copied += 1
    if not (target_dir / f"{mps_tag}-mps_info.bin").exists():
        raise FileNotFoundError(
            f"MPS tag {mps_tag!r} was not found in {source_dir}"
        )
    return copied


def parallel_commutator_scores_from_store(
    store_dir,
    parity_matrix,
    x,
    pairs=None,
    multiply=None,
    n_threads=1,
    n_workers=1,
    scratch_root=None,
    mps_tag="GS",
):
    """Score parity rows in separate Block2 worker processes.

    Each process receives a private copy of the parent MPS store.  Block2 uses
    files inside that store for temporary MPS tensors, so sharing one directory
    among concurrent workers can corrupt intermediate states.
    """
    rows = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    chunks = candidate_index_chunks(len(rows), n_workers)
    if len(chunks) == 1:
        solver = Block2DMRGSolver.load(store_dir, n_threads=int(n_threads))
        costs = DMRGOrbitalCosts(
            solver,
            rows,
            mps_tag=mps_tag,
            multiply=multiply,
            pairs=pairs,
        )
        return commutator_scores_by_row(costs, x)

    multiply = multiply or MultiplyConfig()
    worker_threads = max(1, int(n_threads) // len(chunks))
    scratch_parent = Path(scratch_root or Path(store_dir).parent)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    worker_root = Path(
        tempfile.mkdtemp(prefix="nc_candidate_workers_", dir=scratch_parent)
    )

    print(
        f"[NC score] parallel workers={len(chunks)}, "
        f"threads_per_worker={worker_threads}",
        flush=True,
    )
    tasks = []
    try:
        for worker_number, indices in enumerate(chunks, start=1):
            worker_store = worker_root / f"worker_{worker_number}" / "mps"
            print(
                f"[NC score] preparing worker {worker_number}/{len(chunks)} "
                f"with {len(indices)} candidates",
                flush=True,
            )
            copied = _copy_mps_store_for_tag(store_dir, worker_store, mps_tag)
            print(
                f"[NC score] worker {worker_number} copied "
                f"{copied} parent-MPS files",
                flush=True,
            )
            tasks.append(
                {
                    "worker_number": worker_number,
                    "worker_threads": worker_threads,
                    "store_dir": str(worker_store),
                    "rows": rows[indices].tolist(),
                    "indices": indices,
                    "total_candidates": len(rows),
                    "x": np.asarray(x, dtype=float).tolist(),
                    "pairs": None if pairs is None else list(pairs),
                    "mps_tag": str(mps_tag),
                    "multiply_bond_dim": multiply.bond_dim,
                    "multiply_sweeps": multiply.n_sweeps,
                    "multiply_tol": multiply.tol,
                    "bra_bond_dim_factor": multiply.bra_bond_dim_factor,
                }
            )

        scores = np.empty(len(rows), dtype=float)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(tasks),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_score_candidate_chunk, task)
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                indices, chunk_scores = future.result()
                scores[np.asarray(indices, dtype=int)] = np.asarray(
                    chunk_scores, dtype=float
                )
                print(
                    f"[NC score] completed {len(indices)} candidates; "
                    f"first candidate={indices[0] + 1}",
                    flush=True,
                )
        return scores
    finally:
        shutil.rmtree(worker_root, ignore_errors=True)


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
