"""Pure NumPy linear algebra used by the MPS coupled-space experiment."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def hermitize(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian part of a square matrix."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def canonical_generalized_eigh(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    overlap_cutoff: float = 1.0e-10,
) -> dict:
    """Solve ``H c = E S c`` after canonical overlap orthogonalization."""
    hamiltonian = hermitize(hamiltonian)
    overlap = hermitize(overlap)
    if hamiltonian.shape != overlap.shape:
        raise ValueError("Hamiltonian and overlap matrices must have equal shape")
    overlap_values, overlap_vectors = scipy.linalg.eigh(overlap)
    keep = overlap_values > float(overlap_cutoff)
    if not np.any(keep):
        raise ValueError("all coupled-space directions are linearly dependent")
    transform = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])[None, :]
    orthogonal_hamiltonian = hermitize(
        transform.conj().T @ hamiltonian @ transform
    )
    energies, orthogonal_vectors = scipy.linalg.eigh(orthogonal_hamiltonian)
    coefficients = transform @ orthogonal_vectors
    for column in range(coefficients.shape[1]):
        norm = coefficients[:, column].conj() @ overlap @ coefficients[:, column]
        coefficients[:, column] /= np.sqrt(np.real(norm))
    residuals = []
    for energy, vector in zip(energies, coefficients.T):
        residuals.append(
            float(
                np.linalg.norm(
                    hamiltonian @ vector - energy * overlap @ vector
                )
            )
        )
    return {
        "energies": np.real_if_close(energies),
        "coefficients": coefficients,
        "overlap_eigenvalues": np.real_if_close(overlap_values),
        "retained_overlap_rank": int(np.count_nonzero(keep)),
        "removed_overlap_directions": int(np.count_nonzero(~keep)),
        "condition_number": float(
            overlap_values[keep].max() / overlap_values[keep].min()
        ),
        "generalized_residual_norms": residuals,
        "hermiticity_error_h": float(
            np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0)
        ),
        "hermiticity_error_s": float(
            np.max(np.abs(overlap - overlap.conj().T), initial=0.0)
        ),
    }


def xor_label(left, right) -> tuple[int, ...]:
    """Return the GF(2) difference between two sector labels."""
    return tuple(int(a) ^ int(b) for a, b in zip(left, right))


def transition_signature(
    alpha_support: np.ndarray,
    beta_support: np.ndarray,
    alpha_indices=(),
    beta_indices=(),
) -> tuple[int, ...]:
    """Parity-label change caused by a second-quantized operator term."""
    value = np.zeros(alpha_support.shape[0], dtype=np.uint8)
    for index in alpha_indices:
        value ^= alpha_support[:, int(index)].astype(np.uint8)
    for index in beta_indices:
        value ^= beta_support[:, int(index)].astype(np.uint8)
    return tuple(int(bit) for bit in value)


def hamiltonian_transition_signatures(
    parity_matrix: np.ndarray,
    h1e: np.ndarray,
    g2e: np.ndarray,
    threshold: float = 1.0e-14,
) -> set[tuple[int, ...]]:
    """Enumerate exact sector-label changes allowed by nonzero Hamiltonian terms."""
    parity = np.atleast_2d(np.asarray(parity_matrix, dtype=np.uint8)) % 2
    norb = int(np.asarray(h1e).shape[0])
    if parity.shape[1] == norb:
        alpha = parity
        beta = parity
    elif parity.shape[1] == 2 * norb:
        alpha = parity[:, 0::2]
        beta = parity[:, 1::2]
    else:
        raise ValueError("parity matrix must have norb or 2*norb columns")

    signatures = {tuple(0 for _ in range(len(parity)))}
    for p, q in np.argwhere(np.abs(h1e) > float(threshold)):
        signatures.add(transition_signature(alpha, beta, (p, q), ()))
        signatures.add(transition_signature(alpha, beta, (), (p, q)))

    g2e = np.asarray(g2e)
    if g2e.ndim != 4:
        raise ValueError("g2e must be restored to a four-index tensor")
    for p, q, r, s in np.argwhere(np.abs(g2e) > float(threshold)):
        signatures.add(
            transition_signature(alpha, beta, (p, r, s, q), ())
        )
        signatures.add(
            transition_signature(alpha, beta, (), (p, r, s, q))
        )
        signatures.add(
            transition_signature(alpha, beta, (p, q), (r, s))
        )
        signatures.add(
            transition_signature(alpha, beta, (r, s), (p, q))
        )
    return signatures


def coupling_is_allowed(label_a, label_b, signatures) -> bool:
    """Return whether a Hamiltonian term can connect two sector labels."""
    return xor_label(label_a, label_b) in signatures


def variational_curve_is_monotone(energies, tolerance=1.0e-10) -> bool:
    """Check that enlarging a nested variational space never raises energy."""
    values = np.asarray(energies, dtype=float)
    if values.size < 2:
        return True
    return bool(np.all(np.diff(values) <= float(tolerance)))


def choose_residual_labels(
    weighted_labels,
    existing_depths,
    maximum,
    anchor_label,
    include_anchor=True,
) -> list[tuple[int, ...]]:
    """Prioritize missing sectors, then under-resolved existing sectors."""
    ordered = []
    for label, _weight in weighted_labels:
        label = tuple(int(bit) for bit in label)
        if not include_anchor and label == tuple(anchor_label):
            continue
        if label not in existing_depths:
            ordered.append(label)
    for label, _weight in weighted_labels:
        label = tuple(int(bit) for bit in label)
        if (
            (not include_anchor and label == tuple(anchor_label))
            or label in ordered
        ):
            continue
        ordered.append(label)
    return ordered[: int(maximum)]


def retain_projector_candidates(
    candidates,
    beam_width,
    protected_prefix=None,
) -> tuple[list[dict], list[dict]]:
    """Keep the strongest branches while preserving one required label prefix."""
    ordered = sorted(candidates, key=lambda item: -float(item["norm2"]))
    width = max(1, int(beam_width))
    if protected_prefix is None:
        return ordered[:width], ordered[width:]

    protected_prefix = tuple(int(bit) for bit in protected_prefix)
    protected = next(
        (
            item
            for item in ordered
            if tuple(int(bit) for bit in item["label"]) == protected_prefix
        ),
        None,
    )
    if protected is None or protected in ordered[:width]:
        return ordered[:width], ordered[width:]

    kept = [protected]
    kept.extend(item for item in ordered if item is not protected)
    kept = sorted(kept[:width], key=lambda item: -float(item["norm2"]))
    kept_ids = {id(item) for item in kept}
    pruned = [item for item in ordered if id(item) not in kept_ids]
    return kept, pruned


def select_external_projector_branches(
    branches,
    source_norm2,
    anchor_label,
    minimum_capture,
    compression_loss=0.0,
    absolute_weight_cutoff=1.0e-10,
    relative_weight_cutoff=1.0e-10,
    fit_loss_multiplier=1.0,
) -> dict:
    """Select reliable external branches using external-only normalization."""
    source_norm2 = max(0.0, float(source_norm2))
    minimum_capture = float(minimum_capture)
    if not 0.0 < minimum_capture <= 1.0:
        raise ValueError("minimum_capture must lie in (0, 1]")
    if min(
        absolute_weight_cutoff,
        relative_weight_cutoff,
        fit_loss_multiplier,
    ) < 0.0:
        raise ValueError("projector branch thresholds must be nonnegative")

    anchor_label = tuple(int(bit) for bit in anchor_label)
    normalized = []
    for item in branches:
        copied = dict(item)
        copied["label"] = [int(bit) for bit in item["label"]]
        copied["norm2"] = max(0.0, float(item["norm2"]))
        normalized.append(copied)

    anchor_branches = [
        item for item in normalized if tuple(item["label"]) == anchor_label
    ]
    anchor_weight = sum(float(item["norm2"]) for item in anchor_branches)
    external = sorted(
        [
            item
            for item in normalized
            if tuple(item["label"]) != anchor_label
        ],
        key=lambda item: -float(item["norm2"]),
    )
    external_total = max(0.0, source_norm2 - anchor_weight)
    raw_external_weight = sum(float(item["norm2"]) for item in external)
    compression_loss = max(0.0, float(compression_loss))
    noise_floor = max(
        float(absolute_weight_cutoff),
        float(relative_weight_cutoff) * external_total,
        float(fit_loss_multiplier) * compression_loss,
    )
    external_is_numerical_noise = external_total <= noise_floor
    raw_capture = (
        1.0
        if external_is_numerical_noise
        else min(1.0, raw_external_weight / external_total)
    )

    selected = []
    rejected = []
    selected_weight = 0.0
    for item in external:
        weight = float(item["norm2"])
        if weight <= noise_floor:
            rejected.append(
                {
                    **item,
                    "rejection_reason": "below_noise_floor",
                    "cumulative_external_capture": (
                        1.0
                        if external_is_numerical_noise
                        else min(1.0, selected_weight / external_total)
                    ),
                }
            )
            continue
        capture = (
            1.0
            if external_is_numerical_noise
            else selected_weight / external_total
        )
        if capture >= minimum_capture:
            rejected.append(
                {
                    **item,
                    "rejection_reason": "beyond_capture_target",
                    "cumulative_external_capture": min(1.0, capture),
                }
            )
            continue
        selected_weight += weight
        selected.append(
            {
                **item,
                "cumulative_external_capture": (
                    1.0
                    if external_is_numerical_noise
                    else min(1.0, selected_weight / external_total)
                ),
            }
        )

    selected_capture = (
        1.0
        if external_is_numerical_noise
        else min(1.0, selected_weight / external_total)
    )
    return {
        "anchor_label": list(anchor_label),
        "anchor_present": bool(anchor_branches),
        "anchor_weight": anchor_weight,
        "anchor_fraction": (
            0.0 if source_norm2 <= 1.0e-30 else anchor_weight / source_norm2
        ),
        "source_norm2": source_norm2,
        "external_total_weight": external_total,
        "external_is_numerical_noise": external_is_numerical_noise,
        "raw_external_weight": raw_external_weight,
        "raw_external_capture": raw_capture,
        "raw_target_met": raw_capture >= minimum_capture,
        "selected_external_weight": selected_weight,
        "selected_external_capture": selected_capture,
        "selection_target_met": selected_capture >= minimum_capture,
        "unresolved_external_weight": max(
            0.0, external_total - raw_external_weight
        ),
        "noise_floor": noise_floor,
        "absolute_weight_cutoff": float(absolute_weight_cutoff),
        "relative_weight_cutoff": float(relative_weight_cutoff),
        "fit_loss_multiplier": float(fit_loss_multiplier),
        "compression_loss": compression_loss,
        "minimum_capture": minimum_capture,
        "selected_branches": selected,
        "rejected_branches": rejected,
        "selected_branch_count": len(selected),
        "rejected_branch_count": len(rejected),
    }
