"""Pure unit tests for selection, restart, and coupled-space mathematics."""

import json

import numpy as np

from clifford_outputs import represented_sector_pairs
from common import (
    atomic_json,
    cap_worker_store_resources,
    map_rotation_to_canonical,
    map_rows_to_solver_order,
)
from linear_algebra import (
    canonical_generalized_eigh,
    choose_residual_labels,
    hamiltonian_transition_signatures,
    variational_curve_is_monotone,
)
from src.dmrg_symmetry_selection import (
    assign_candidate_scores,
    canonical_row_space,
    select_independent_candidates,
    seniority_quartet_candidates,
    selected_parity_matrix,
)
from src.dmrg_decoupled_energy import add_neighbour_labels, objective_key


def test_gf2_greedy_selection_reaches_requested_rank():
    candidates = seniority_quartet_candidates(5)
    scores = np.linspace(0.0, 1.0, len(candidates))
    selected = select_independent_candidates(
        assign_candidate_scores(candidates, scores), 4
    )
    parity = selected_parity_matrix(selected)
    assert parity.shape == (4, 5)
    assert len(canonical_row_space(parity)) == 4


def test_generalized_eigensolver_removes_dependent_direction():
    basis = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    full_h = np.diag([-2.0, 3.0])
    overlap = basis.T @ basis
    hamiltonian = basis.T @ full_h @ basis
    result = canonical_generalized_eigh(hamiltonian, overlap, 1.0e-10)
    assert result["retained_overlap_rank"] == 2
    assert result["removed_overlap_directions"] == 1
    assert np.isclose(result["energies"][0], -2.0)
    assert result["generalized_residual_norms"][0] < 1.0e-10


def test_rotation_and_parity_permutation_round_trip():
    permutation = [2, 0, 1]
    rows = np.array([[1, 0, 1], [0, 1, 1]])
    solver_rows = map_rows_to_solver_order(rows, permutation)
    assert np.array_equal(solver_rows, rows[:, permutation])
    rotation_solver = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    canonical = map_rotation_to_canonical(rotation_solver, permutation)
    assert np.array_equal(
        canonical[np.ix_(permutation, permutation)], rotation_solver
    )


def test_transition_signatures_detect_one_body_label_change():
    parity = np.array([[1, 0], [0, 1]])
    h1e = np.array([[1.0, 0.2], [0.2, 2.0]])
    g2e = np.zeros((2, 2, 2, 2))
    signatures = hamiltonian_transition_signatures(parity, h1e, g2e)
    assert (0, 0) in signatures
    assert (1, 1) in signatures


def test_represented_pairs_and_variational_curve():
    basis = [
        {"label": [0, 0]},
        {"label": [0, 0]},
        {"label": [1, 0]},
        {"label": [0, 1]},
    ]
    matrix = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
    matrix[0, 2] = matrix[2, 0] = 0.3
    labels, pairs = represented_sector_pairs(basis, matrix)
    assert labels == [(0, 0), (0, 1), (1, 0)]
    assert [(left, right) for left, right, _ in pairs] == [((0, 0), (1, 0))]
    assert variational_curve_is_monotone([-1.0, -1.1, -1.10001])
    assert not variational_curve_is_monotone([-1.0, -0.9])


def test_atomic_restart_json_is_valid(tmp_path):
    path = tmp_path / "restart.json"
    atomic_json(path, {"cycle": 3, "basis": ["A", "B"]})
    assert json.loads(path.read_text()) == {"cycle": 3, "basis": ["A", "B"]}


def test_worker_resource_cap_does_not_expand_saved_stack(tmp_path):
    metadata = {
        "system": {
            "n_threads": 32,
            "stack_mem_bytes": 8 * 1024**3,
        }
    }
    atomic_json(tmp_path / "metadata.json", metadata)
    cap_worker_store_resources(tmp_path, n_threads=4)
    updated = json.loads((tmp_path / "metadata.json").read_text())
    assert updated["system"]["n_threads"] == 4
    assert updated["system"]["stack_mem_bytes"] == 4 * 1024**3


def test_objective_cache_key_is_exact_and_sector_specific():
    rotation = np.array([0.0, 0.25, -0.125], dtype=np.float64)
    same = objective_key(rotation.copy(), (0, 1, 0))
    assert same == objective_key(rotation, (0, 1, 0))
    assert same != objective_key(rotation, (0, 1, 1))
    changed = rotation.copy()
    changed[0] = np.nextafter(0.0, 1.0)
    assert same != objective_key(changed, (0, 1, 0))


def test_sector_screening_adds_neighbours_without_full_enumeration():
    labels = add_neighbour_labels([(0, 0, 0, 0)], n_bits=4, minimum=4)
    assert len(labels) == 4
    assert labels[0] == (0, 0, 0, 0)
    assert all(sum(label) <= 1 for label in labels)


def test_residual_enrichment_can_revisit_anchor_sector():
    weighted = [((0, 0), 0.6), ((1, 0), 0.3), ((0, 1), 0.1)]
    depths = {(0, 0): 1, (1, 0): 2}
    labels = choose_residual_labels(weighted, depths, 3, (0, 0))
    assert labels == [(0, 1), (0, 0), (1, 0)]
    external_only = choose_residual_labels(
        weighted, depths, 3, (0, 0), include_anchor=False
    )
    assert external_only == [(0, 1), (1, 0)]
