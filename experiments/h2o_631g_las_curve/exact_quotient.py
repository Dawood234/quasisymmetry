"""Exact-symmetry quotient selection and sector-dimension bookkeeping."""

from collections import Counter
from itertools import combinations

import numpy as np

from src.gf2_utils import gf2_rank, gf2_rref


def spatial_to_spin_rows(rows, norb):
    """Lift spatial pair-parity rows to interleaved alpha/beta rows."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.uint8)) % 2
    if rows.shape[1] != int(norb):
        raise ValueError("spatial rows must have one column per orbital")
    lifted = np.zeros((len(rows), 2 * int(norb)), dtype=np.uint8)
    lifted[:, 0::2] = rows
    lifted[:, 1::2] = rows
    return lifted


def point_group_spatial_rows(orbital_irreps):
    """Return independent Abelian point-group character rows."""
    irreps = np.asarray(orbital_irreps, dtype=int).ravel()
    if irreps.size == 0 or np.any(irreps < 0):
        raise ValueError("orbital irreps must be nonnegative integers")
    bit_count = max(1, int(np.max(irreps)).bit_length())
    rows = np.asarray(
        [[(int(irrep) >> bit) & 1 for irrep in irreps] for bit in range(bit_count)],
        dtype=np.uint8,
    )
    return rows[np.any(rows, axis=1)]


def exact_parity_rows(orbital_irreps):
    """Return exact alpha number, beta number, and point-group parity rows."""
    norb = len(orbital_irreps)
    number_rows = np.zeros((2, 2 * norb), dtype=np.uint8)
    number_rows[0, 0::2] = 1
    number_rows[1, 1::2] = 1
    point_rows = spatial_to_spin_rows(point_group_spatial_rows(orbital_irreps), norb)
    rows = np.vstack((number_rows, point_rows))
    reduced, _ = gf2_rref(rows)
    rows = reduced[np.any(reduced, axis=1)]
    return rows


def _reduced_rows(rows):
    rows = np.atleast_2d(np.asarray(rows, dtype=np.uint8)) % 2
    reduced, _ = gf2_rref(rows)
    return reduced[np.any(reduced, axis=1)]


def quotient_remainder(spin_row, exact_rows):
    """Canonical remainder of one row modulo the exact-symmetry span."""
    vector = np.asarray(spin_row, dtype=np.uint8).ravel().copy() % 2
    for row in _reduced_rows(exact_rows):
        pivot = int(np.flatnonzero(row)[-1])
        if vector[pivot]:
            vector ^= row
    return tuple(int(value) for value in vector)


def select_quotient_independent(candidates, target_rank, exact_rows, norb):
    """Greedily retain low-score named candidates independent modulo exact rows."""
    exact_rows = _reduced_rows(exact_rows)
    exact_rank = int(gf2_rank(exact_rows))
    representatives = {}
    for candidate in sorted(candidates, key=lambda item: (item["score"], item["label"])):
        spatial = np.asarray(candidate["row"], dtype=np.uint8)
        spin_row = spatial_to_spin_rows(spatial.reshape(1, -1), norb)[0]
        remainder = quotient_remainder(spin_row, exact_rows)
        if not any(remainder) or remainder in representatives:
            continue
        item = dict(candidate)
        item["spin_row"] = spin_row
        item["quotient_remainder"] = remainder
        representatives[remainder] = item

    selected = []
    selected_rows = []
    current_rank = exact_rank
    for candidate in representatives.values():
        trial_rows = selected_rows + [candidate["spin_row"]]
        trial = np.vstack((exact_rows, np.asarray(trial_rows, dtype=np.uint8)))
        trial_rank = int(gf2_rank(trial))
        if trial_rank == current_rank:
            continue
        selected.append(candidate)
        selected_rows.append(candidate["spin_row"])
        current_rank = trial_rank
        if len(selected) == int(target_rank):
            return selected
    raise ValueError(
        f"candidate pool has quotient rank {current_rank - exact_rank}, "
        f"below requested rank {target_rank}"
    )


def quotient_row_space(spatial_rows, exact_rows, norb):
    """Canonical identity of the joint exact-plus-LAS row space."""
    spin_rows = spatial_to_spin_rows(spatial_rows, norb)
    joint = _reduced_rows(np.vstack((exact_rows, spin_rows)))
    return tuple(tuple(int(value) for value in row) for row in joint)


def selection_diagnostics(selected, exact_rows, norb):
    """Serialize the effective rank and quotient representative of each row."""
    spatial_rows = np.asarray([item["row"] for item in selected], dtype=np.uint8)
    spin_rows = spatial_to_spin_rows(spatial_rows, norb)
    exact_rank = int(gf2_rank(exact_rows))
    joint_rank = int(gf2_rank(np.vstack((exact_rows, spin_rows))))
    return {
        "exact_rank": exact_rank,
        "las_row_count": len(selected),
        "joint_rank": joint_rank,
        "effective_las_rank": joint_rank - exact_rank,
        "quotient_remainders_spin_orbital": [
            list(quotient_remainder(row, exact_rows)) for row in spin_rows
        ],
    }


def _occupation_counts(norb, nelec, point_rows, las_rows):
    counts = Counter()
    for occupied in combinations(range(norb), int(nelec)):
        occupation = np.zeros(norb, dtype=np.uint8)
        occupation[list(occupied)] = 1
        point_label = tuple(int(row @ occupation) & 1 for row in point_rows)
        las_label = tuple(int(row @ occupation) & 1 for row in las_rows)
        counts[(point_label, las_label)] += 1
    return counts


def _resolved_dimensions(norb, nalpha, nbeta, orbital_irreps, target_irrep, las_rows):
    point_rows = point_group_spatial_rows(orbital_irreps)
    target_point = tuple(
        (int(target_irrep) >> bit) & 1 for bit in range(len(point_rows))
    )
    alpha_counts = _occupation_counts(norb, nalpha, point_rows, las_rows)
    beta_counts = _occupation_counts(norb, nbeta, point_rows, las_rows)
    dimensions = Counter()
    for (point_a, las_a), count_a in alpha_counts.items():
        required_point_b = tuple(a ^ target for a, target in zip(point_a, target_point))
        for (point_b, las_b), count_b in beta_counts.items():
            if point_b != required_point_b:
                continue
            label = tuple(a ^ b for a, b in zip(las_a, las_b))
            dimensions[label] += count_a * count_b
    return dimensions


def exact_sector_dimension_audit(
    nalpha,
    nbeta,
    orbital_irreps,
    target_irrep,
    las_rows,
    selected_label,
):
    """Count exact-irrep-resolved LAS determinant and singlet-CSF dimensions."""
    norb = len(orbital_irreps)
    las_rows = np.atleast_2d(np.asarray(las_rows, dtype=np.uint8)) % 2
    selected_label = tuple(int(value) for value in selected_label)
    dimensions = _resolved_dimensions(
        norb, nalpha, nbeta, orbital_irreps, target_irrep, las_rows
    )
    if selected_label not in dimensions:
        raise ValueError(f"selected LAS sector {selected_label} is empty")

    singlet_dimensions = {}
    if int(nalpha) == int(nbeta) and int(nalpha) + 1 <= norb and int(nbeta) >= 1:
        ms_one = _resolved_dimensions(
            norb, int(nalpha) + 1, int(nbeta) - 1,
            orbital_irreps, target_irrep, las_rows,
        )
        singlet_dimensions = {
            label: int(dimension - ms_one.get(label, 0))
            for label, dimension in dimensions.items()
        }

    exact_rows = exact_parity_rows(orbital_irreps)
    spin_rows = spatial_to_spin_rows(las_rows, norb)
    exact_rank = int(gf2_rank(exact_rows))
    joint_rank = int(gf2_rank(np.vstack((exact_rows, spin_rows))))
    return {
        "target_irrep": int(target_irrep),
        "target_exact_dimension": int(sum(dimensions.values())),
        "nonempty_las_sector_count": len(dimensions),
        "exact_rank": exact_rank,
        "joint_rank": joint_rank,
        "effective_las_rank": joint_rank - exact_rank,
        "selected_label": list(selected_label),
        "selected_determinant_dimension": int(dimensions[selected_label]),
        "selected_singlet_csf_dimension": (
            None if not singlet_dimensions else int(singlet_dimensions[selected_label])
        ),
        "dmax_determinant": int(max(dimensions.values())),
        "dmax_singlet_csf": (
            None if not singlet_dimensions else int(max(singlet_dimensions.values()))
        ),
        "sector_dimensions": {
            "".join(map(str, label)): int(value)
            for label, value in sorted(dimensions.items())
        },
    }
