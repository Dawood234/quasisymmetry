"""DMRG-backed selection of seniority and quartet parity generators.

The candidate pool contains local seniorities

    S_p = (-1)^(n_{p alpha} + n_{p beta})

and quartet products ``S_p S_q``.  Candidate scores come from the MPS-native
non-commutativity calculation in :mod:`src.dmrg_costs`.  This module only
orders candidates and enforces GF(2) independence; it does not optimize
orbitals or evaluate sector energies.
"""

from itertools import combinations

import numpy as np

from src.gf2_utils import gf2_rank, gf2_rref


def seniority_quartet_candidates(norb):
    """Return all local seniority rows and pairwise quartet rows."""
    candidates = []
    seniority_rows = []

    for orbital in range(int(norb)):
        row = np.zeros(int(norb), dtype=int)
        row[orbital] = 1
        seniority_rows.append(row)
        candidates.append(
            {
                "label": f"S({orbital + 1})",
                "family": "seniority",
                "support": [orbital + 1],
                "row": row,
            }
        )

    for first, second in combinations(range(int(norb)), 2):
        row = (seniority_rows[first] + seniority_rows[second]) % 2
        candidates.append(
            {
                "label": f"Q({first + 1},{second + 1})",
                "family": "quartet",
                "support": [first + 1, second + 1],
                "row": row,
            }
        )

    return candidates


def candidate_matrix(candidates):
    """Stack candidate parity rows into one binary matrix."""
    if not candidates:
        return np.zeros((0, 0), dtype=int)
    return np.asarray([item["row"] for item in candidates], dtype=int)


def assign_candidate_scores(candidates, scores):
    """Return candidate dictionaries with their NC scores attached."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(candidates),):
        raise ValueError("one score is required for every candidate")

    scored = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["row"] = np.asarray(candidate["row"], dtype=int)
        item["score"] = float(score)
        scored.append(item)
    return scored


def select_independent_candidates(candidates, target_rank):
    """Select the lowest-score candidates that increase the GF(2) rank."""
    target_rank = int(target_rank)
    if target_rank < 1:
        raise ValueError("target_rank must be positive")

    ordered = sorted(candidates, key=lambda item: (item["score"], item["label"]))
    selected = []
    rows = []
    current_rank = 0

    for candidate in ordered:
        trial_rows = rows + [np.asarray(candidate["row"], dtype=int)]
        trial_rank = int(gf2_rank(np.asarray(trial_rows, dtype=int)))
        if trial_rank == current_rank:
            continue
        selected.append(candidate)
        rows = trial_rows
        current_rank = trial_rank
        if current_rank == target_rank:
            break

    if current_rank != target_rank:
        raise ValueError(
            f"candidate pool has GF(2) rank {current_rank}, "
            f"below requested rank {target_rank}"
        )
    return selected


def selected_parity_matrix(selected):
    """Return the selected spatial-orbital parity matrix."""
    if not selected:
        return np.zeros((0, 0), dtype=int)
    return np.asarray([item["row"] for item in selected], dtype=int)


def canonical_row_space(parity_matrix):
    """Return a hashable canonical GF(2) representation of a row space."""
    matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int)) % 2
    reduced, _ = gf2_rref(matrix)
    nonzero = [tuple(int(value) for value in row) for row in reduced if np.any(row)]
    return tuple(nonzero)


def selection_to_json(selected):
    """Convert selected candidate dictionaries to JSON-compatible values."""
    output = []
    for candidate in selected:
        output.append(
            {
                "label": candidate["label"],
                "family": candidate["family"],
                "support": list(candidate["support"]),
                "score": float(candidate["score"]),
                "row": np.asarray(candidate["row"], dtype=int).tolist(),
            }
        )
    return output
