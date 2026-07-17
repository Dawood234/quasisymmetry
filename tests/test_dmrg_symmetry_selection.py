"""Fast tests for DMRG candidate ordering and GF(2) selection."""

import numpy as np

from src.dmrg_symmetry_selection import (
    assign_candidate_scores,
    candidate_matrix,
    canonical_row_space,
    select_independent_candidates,
    selected_parity_matrix,
    seniority_quartet_candidates,
)


def test_h2o_631g_candidate_count():
    candidates = seniority_quartet_candidates(13)
    assert len(candidates) == 13 + 78
    assert candidate_matrix(candidates).shape == (91, 13)


def test_greedy_selection_reaches_requested_rank():
    candidates = seniority_quartet_candidates(13)
    scores = np.arange(len(candidates), dtype=float)
    selected = select_independent_candidates(
        assign_candidate_scores(candidates, scores), 7
    )
    parity = selected_parity_matrix(selected)
    assert parity.shape == (7, 13)
    assert len(canonical_row_space(parity)) == 7


def test_canonical_row_space_ignores_generator_basis():
    first = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=int)
    second = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=int)
    assert canonical_row_space(first) == canonical_row_space(second)

