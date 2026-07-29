"""Coupled-dimension selection over sector eigenstates.

Two protocols share the same nested variational search:

* ``perturbation`` — one-shot Epstein--Nesbet PT ordering relative to the
  lowest sector state, then adaptive ``K_epsilon``;
* ``reference`` — order by overlap with a reference (e.g. FCI) state, then
  the same nested search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

COUPLED_ENERGY_DEGENERACY_FLOOR = 1e-8
CHEMICAL_PRECISION = 0.0016
DEFAULT_TAU_PT = 1e-12
DEFAULT_BLOCK_SIZE = 1


def all_sector_eigenpair_candidates(
    sector_data,
) -> list[tuple[float, object, np.ndarray, int]]:
    """All block eigenpairs (energy, sector key, full-space vector, block index)."""
    candidates: list[tuple[float, object, np.ndarray, int]] = []
    for key, data in sector_data.items():
        for block_index, (energy, vector) in enumerate(
            zip(data["evals"], data["evecs_full"])
        ):
            candidates.append((float(energy), key, vector, int(block_index)))
    candidates.sort(key=lambda item: item[0])
    return candidates


@dataclass
class CoupledDimensionResult:
    """Result of nested variational coupled-dimension search."""

    e_coupled: float | None
    K: int | None
    converged: bool
    chosen_keys: list[tuple[object, int]]
    order_indices: list[int] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    K_pt: int | None = None
    K_prefix: int | None = None
    pt_weights: np.ndarray = field(default_factory=lambda: np.asarray([]))
    reference_weights: np.ndarray = field(default_factory=lambda: np.asarray([]))

    def as_curve(self) -> dict:
        """Clifford/metrics-compatible curve dict."""
        return {
            "order": [int(i) for i in self.order_indices],
            "energies": list(self.energies),
            "K": self.K,
            "converged": self.converged,
            "K_pt": self.K_pt,
            "K_prefix": self.K_prefix,
        }

    def as_tuple(
        self,
    ) -> tuple[float | None, int, bool, list[tuple[object, int]]]:
        k = 0 if self.K is None else int(self.K)
        return self.e_coupled, k, self.converged, list(self.chosen_keys)


def one_shot_pt_weight(
    coupling: complex,
    delta: float,
    *,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> float:
    """Epstein--Nesbet second-order importance ``|v|^2 / Delta``."""
    numer = float(abs(coupling) ** 2)
    if numer == 0.0:
        return 0.0
    if abs(delta) <= degeneracy_floor:
        return float("inf")
    return numer / abs(delta)


def one_shot_pt_order(
    energies: Sequence[float],
    couplings_to_ref: Sequence[complex],
    ref_index: int,
    *,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> tuple[list[int], np.ndarray]:
    """Order candidates by decreasing one-shot PT importance relative to ``ref``."""
    n = len(energies)
    if n == 0:
        return [], np.asarray([])
    if len(couplings_to_ref) != n:
        raise ValueError("couplings_to_ref must match energies length")

    e0 = float(energies[ref_index])
    weights = np.zeros(n, dtype=np.float64)
    weights[ref_index] = float("inf")
    external: list[tuple[float, int]] = []
    for index in range(n):
        if index == ref_index:
            continue
        weight = one_shot_pt_weight(
            complex(couplings_to_ref[index]),
            float(energies[index]) - e0,
            degeneracy_floor=degeneracy_floor,
        )
        weights[index] = weight
        external.append((weight, index))

    external.sort(key=lambda item: (-item[0], float(energies[item[1]]), item[1]))
    order = [ref_index] + [index for _weight, index in external]
    return order, weights


def reference_candidate_order(weights: Sequence[float]) -> list[int]:
    """Order candidates from largest to smallest reference overlap weight."""
    return list(np.argsort(np.asarray(weights, dtype=np.float64))[::-1])


def k_pt_from_ordered_weights(
    ordered_external_weights: Sequence[float],
    tau_pt: float,
) -> int:
    """``K_PT = 1 + max{r : w_{alpha_r} >= tau_PT}`` (or 1 if none pass)."""
    k_pt = 1
    for rank, weight in enumerate(ordered_external_weights, start=1):
        if weight >= tau_pt:
            k_pt = 1 + rank
        else:
            break
    return k_pt


def build_candidate_hamiltonian(
    candidates: Sequence[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    *,
    order: Sequence[int] | None = None,
) -> np.ndarray:
    """Dense Hamiltonian matrix in the (optionally reordered) candidate basis."""
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    indices = list(range(n)) if order is None else list(order)
    if sorted(indices) != list(range(n)):
        raise ValueError("order must be a permutation of all candidates")

    vecs = [candidates[i][2] for i in indices]
    h_vecs = [apply_h(vec) for vec in vecs]
    h_mat = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        for i in range(j + 1):
            value = complex(np.vdot(vecs[i], h_vecs[j]))
            h_mat[i, j] = value
            h_mat[j, i] = np.conjugate(value)
    return 0.5 * (h_mat + h_mat.conj().T)


def nested_ground_energy(h_ordered: np.ndarray, k: int) -> float:
    """Lowest eigenvalue of the leading principal ``k x k`` submatrix."""
    if k <= 0:
        raise ValueError("k must be positive")
    return float(np.linalg.eigvalsh(h_ordered[:k, :k])[0])


def find_k_epsilon(
    h_ordered: np.ndarray,
    e_ref: float,
    epsilon: float,
    k_start: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int | None, list[float], bool]:
    """Smallest nested dimension with ``E_0^(K) - e_ref <= epsilon``."""
    n = h_ordered.shape[0]
    if n == 0:
        return None, [], False
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    cache: dict[int, float] = {}

    def energy_at(k: int) -> float:
        if k not in cache:
            cache[k] = nested_ground_energy(h_ordered, k)
        return cache[k]

    def acceptable(k: int) -> bool:
        return energy_at(k) - e_ref <= epsilon

    k_start = min(max(1, k_start), n)

    if acceptable(k_start):
        hi = k_start
        lo = 0
        k = k_start - block_size
        while k >= 1:
            if acceptable(k):
                hi = k
                k -= block_size
            else:
                lo = k
                break
        if hi == 1 and acceptable(1):
            lo = 0
    else:
        lo = k_start
        hi = None
        k = k_start + block_size
        while k <= n:
            if acceptable(k):
                hi = k
                break
            lo = k
            k += block_size
        if hi is None:
            if acceptable(n):
                hi = n
            else:
                return None, [energy_at(k) for k in range(1, n + 1)], False

    left = lo + 1
    right = hi
    while left < right:
        mid = (left + right) // 2
        if acceptable(mid):
            right = mid
        else:
            left = mid + 1

    k_eps = left
    return k_eps, [energy_at(k) for k in range(1, k_eps + 1)], True


def subset_ground_energy(h_ordered: np.ndarray, indices: Sequence[int]) -> float:
    """Lowest eigenvalue of ``h_ordered`` restricted to ``indices``."""
    idx = list(indices)
    if not idx:
        raise ValueError("indices must be non-empty")
    if len(idx) == 1:
        return float(np.real(h_ordered[idx[0], idx[0]]))
    sub = h_ordered[np.ix_(idx, idx)]
    return float(np.linalg.eigvalsh(sub)[0])


def backward_prune_indices(
    h_ordered: np.ndarray,
    prefix_k: int,
    e_ref: float,
    tol: float,
) -> tuple[list[int], float, int]:
    """Greedy backward removal of individually redundant prefix states.

    For each retained index ``j`` in the leading ``prefix_k`` block, if
    ``E_{-j} - e_ref <= tol`` then ``j`` is dropped. Repeats until no state
    is individually removable. Returns kept row/column indices into
    ``h_ordered``, the pruned ground energy, and the number of states removed.
    """
    if prefix_k <= 0:
        raise ValueError("prefix_k must be positive")
    active = list(range(prefix_k))
    if prefix_k == 1:
        return active, nested_ground_energy(h_ordered, 1), 0

    n_removed = 0
    changed = True
    while changed and len(active) > 1:
        changed = False
        for drop_pos in range(len(active) - 1, -1, -1):
            keep = [active[i] for i in range(len(active)) if i != drop_pos]
            e_minus = subset_ground_energy(h_ordered, keep)
            if e_minus - e_ref <= tol:
                active.pop(drop_pos)
                n_removed += 1
                changed = True
                break

    return active, subset_ground_energy(h_ordered, active), n_removed


def coupled_dimension_from_order(
    h_coupled: np.ndarray,
    order: Sequence[int],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    k_start: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    keys: Sequence[tuple[object, int]] | None = None,
    k_pt: int | None = None,
    pt_weights: np.ndarray | None = None,
    reference_weights: np.ndarray | None = None,
    max_total_vectors: int | None = None,
    backward_prune: bool = True,
) -> CoupledDimensionResult:
    """Nested variational ``K`` along a fixed candidate ordering."""
    order = list(order)
    n = h_coupled.shape[0]
    if n == 0 or not order:
        return CoupledDimensionResult(
            e_coupled=None, K=None, converged=False, chosen_keys=[]
        )
    if max_total_vectors is not None:
        order = order[: max(1, max_total_vectors)]
    if len(order) > n:
        raise ValueError("order longer than Hamiltonian dimension")

    h_ordered = h_coupled[np.ix_(order, order)]
    n_ord = h_ordered.shape[0]
    start = n_ord if k_start is None else min(max(1, k_start), n_ord)

    key_list: list[tuple[object, int]]
    if keys is None:
        key_list = [(i, 0) for i in order]
    else:
        if len(keys) != n:
            raise ValueError("keys must match Hamiltonian dimension")
        key_list = [keys[i] for i in order]

    def chosen(k: int) -> list[tuple[object, int]]:
        return list(key_list[:k])

    if e_exact is None:
        energies = [nested_ground_energy(h_ordered, k) for k in range(1, start + 1)]
        return CoupledDimensionResult(
            e_coupled=energies[-1],
            K=start,
            converged=False,
            chosen_keys=chosen(start),
            order_indices=order,
            energies=energies,
            K_pt=k_pt,
            pt_weights=np.asarray([]) if pt_weights is None else pt_weights,
            reference_weights=(
                np.asarray([]) if reference_weights is None else reference_weights
            ),
        )

    k_eps, energies, converged = find_k_epsilon(
        h_ordered, float(e_exact), tol, start, block_size=block_size
    )
    if k_eps is None:
        return CoupledDimensionResult(
            e_coupled=energies[-1] if energies else None,
            K=n_ord,
            converged=False,
            chosen_keys=chosen(n_ord),
            order_indices=order,
            energies=energies,
            K_pt=k_pt,
            K_prefix=None,
            pt_weights=np.asarray([]) if pt_weights is None else pt_weights,
            reference_weights=(
                np.asarray([]) if reference_weights is None else reference_weights
            ),
        )

    k_prefix = k_eps
    e_coupled = energies[k_eps - 1]
    chosen_indices = list(range(k_eps))
    if backward_prune and converged and k_eps > 1:
        chosen_indices, e_coupled, _n_removed = backward_prune_indices(
            h_ordered, k_eps, float(e_exact), tol
        )

    return CoupledDimensionResult(
        e_coupled=e_coupled,
        K=len(chosen_indices),
        converged=converged,
        chosen_keys=[key_list[i] for i in chosen_indices],
        order_indices=order,
        energies=energies,
        K_pt=k_pt,
        K_prefix=k_prefix,
        pt_weights=np.asarray([]) if pt_weights is None else pt_weights,
        reference_weights=(
            np.asarray([]) if reference_weights is None else reference_weights
        ),
    )


def one_shot_from_hamiltonian(
    h_coupled: np.ndarray,
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    keys: Sequence[tuple[object, int]] | None = None,
    max_total_vectors: int | None = None,
    backward_prune: bool = True,
) -> CoupledDimensionResult:
    """One-shot PT ranking + nested variational ``K`` on a candidate Hamiltonian."""
    n = h_coupled.shape[0]
    if n == 0:
        return CoupledDimensionResult(
            e_coupled=None, K=None, converged=False, chosen_keys=[]
        )

    energies = [float(np.real(h_coupled[i, i])) for i in range(n)]
    ref_index = int(np.argmin(energies))
    couplings = [complex(h_coupled[i, ref_index]) for i in range(n)]
    order, weights = one_shot_pt_order(
        energies, couplings, ref_index, degeneracy_floor=degeneracy_floor
    )
    ordered_external_weights = [float(weights[i]) for i in order[1:]]
    k_pt = k_pt_from_ordered_weights(ordered_external_weights, tau_pt)
    return coupled_dimension_from_order(
        h_coupled,
        order,
        e_exact=e_exact,
        tol=tol,
        k_start=k_pt,
        block_size=block_size,
        keys=keys,
        k_pt=k_pt,
        pt_weights=weights,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    )


def iterative_pt_from_hamiltonian(
    h_coupled: np.ndarray,
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    batch_size: int = 1,
    keys: Sequence[tuple[object, int]] | None = None,
    max_total_vectors: int | None = None,
) -> CoupledDimensionResult:
    """Grow ``K`` by PT couplings to the current coupled ground state."""
    matrix = np.asarray(h_coupled)
    count = matrix.shape[0]
    if count == 0:
        return CoupledDimensionResult(
            e_coupled=None, K=None, converged=False, chosen_keys=[]
        )
    if matrix.shape != (count, count):
        raise ValueError("h_coupled must be square")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if keys is not None and len(keys) != count:
        raise ValueError("keys must match Hamiltonian dimension")

    limit = count if max_total_vectors is None else min(
        count, max(1, int(max_total_vectors))
    )
    anchor = int(np.argmin(np.real(np.diag(matrix))))
    active = [anchor]
    remaining = set(range(count))
    remaining.remove(anchor)
    last_weights = np.zeros(count, dtype=float)

    while True:
        active_matrix = matrix[np.ix_(active, active)]
        eigenvalues, eigenvectors = np.linalg.eigh(active_matrix)
        energy = float(np.real(eigenvalues[0]))
        coefficients = eigenvectors[:, 0]
        if e_exact is not None and energy - float(e_exact) <= float(tol):
            break
        if not remaining or len(active) >= limit:
            break

        scored = []
        last_weights[:] = 0.0
        for candidate in remaining:
            coupling = np.dot(matrix[candidate, active], coefficients)
            denominator = float(np.real(matrix[candidate, candidate])) - energy
            weight = one_shot_pt_weight(coupling, denominator)
            last_weights[candidate] = weight
            scored.append((weight, candidate))
        scored.sort(
            key=lambda item: (
                -item[0],
                float(np.real(matrix[item[1], item[1]])),
                item[1],
            )
        )

        selected = [
            candidate for weight, candidate in scored
            if weight >= float(tau_pt)
        ][: int(batch_size)]
        if not selected and scored:
            selected = [scored[0][1]]
        selected = selected[: limit - len(active)]
        if not selected:
            break
        active.extend(selected)
        remaining.difference_update(selected)

    return coupled_dimension_from_order(
        matrix,
        active,
        e_exact=e_exact,
        tol=tol,
        k_start=len(active),
        keys=keys,
        k_pt=len(active),
        pt_weights=last_weights,
    )


def reference_from_hamiltonian(
    h_coupled: np.ndarray,
    reference_weights: Sequence[float],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    block_size: int = DEFAULT_BLOCK_SIZE,
    keys: Sequence[tuple[object, int]] | None = None,
    max_total_vectors: int | None = None,
    backward_prune: bool = True,
) -> CoupledDimensionResult:
    """Reference-overlap ordering + nested variational ``K``."""
    weights = np.asarray(reference_weights, dtype=np.float64)
    if h_coupled.shape[0] != weights.size:
        raise ValueError("reference_weights must match Hamiltonian dimension")
    order = reference_candidate_order(weights)
    return coupled_dimension_from_order(
        h_coupled,
        order,
        e_exact=e_exact,
        tol=tol,
        k_start=1,
        block_size=block_size,
        keys=keys,
        reference_weights=weights,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    )


def one_shot_coupled_energy(
    candidates: list[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    max_total_vectors: int | None = None,
    backward_prune: bool = True,
) -> CoupledDimensionResult:
    """Dense-vector wrapper around :func:`one_shot_from_hamiltonian`."""
    if not candidates:
        return CoupledDimensionResult(
            e_coupled=None, K=None, converged=False, chosen_keys=[]
        )
    h_coupled = build_candidate_hamiltonian(candidates, apply_h)
    keys = [(item[1], item[3]) for item in candidates]
    return one_shot_from_hamiltonian(
        h_coupled,
        e_exact=e_exact,
        tol=tol,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=degeneracy_floor,
        keys=keys,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    )


def reference_coupled_energy(
    candidates: list[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    reference_vector: np.ndarray,
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_total_vectors: int | None = None,
    backward_prune: bool = True,
) -> CoupledDimensionResult:
    """Dense-vector wrapper: order by ``|<psi_i|Psi_ref>|^2``, then nested ``K``."""
    if not candidates:
        return CoupledDimensionResult(
            e_coupled=None, K=None, converged=False, chosen_keys=[]
        )
    weights = np.asarray(
        [float(abs(np.vdot(item[2], reference_vector)) ** 2) for item in candidates],
        dtype=np.float64,
    )
    h_coupled = build_candidate_hamiltonian(candidates, apply_h)
    keys = [(item[1], item[3]) for item in candidates]
    return reference_from_hamiltonian(
        h_coupled,
        weights,
        e_exact=e_exact,
        tol=tol,
        block_size=block_size,
        keys=keys,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    )
