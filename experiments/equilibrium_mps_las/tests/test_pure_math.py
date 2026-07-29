"""Pure unit tests for selection, restart, and coupled-space mathematics."""

import json
from types import SimpleNamespace

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
    retain_projector_candidates,
    select_external_projector_branches,
    variational_curve_is_monotone,
)
from mps_selection import (
    candidate_checkpoint_state,
    payload_fingerprint,
    read_sign_checkpoint,
    write_candidate_checkpoint,
    write_sign_checkpoint,
)
from mps_krylov import (
    build_coupled_matrices,
    matrix_metadata_path,
    operation_record_matches,
    record_fingerprint,
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


def test_candidate_checkpoints_resume_only_matching_inputs(tmp_path):
    rows = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    context = payload_fingerprint({"proxy": "M200", "rotation": "frame-a"})
    write_candidate_checkpoint(
        tmp_path,
        index=0,
        row=rows[0],
        score=0.125,
        context_fingerprint=context,
        elapsed_seconds=3.0,
    )
    write_candidate_checkpoint(
        tmp_path,
        index=2,
        row=rows[2],
        score=0.5,
        context_fingerprint=context,
        elapsed_seconds=4.0,
    )

    state = candidate_checkpoint_state(tmp_path, rows, context)
    assert state["completed_indices"] == [0, 2]
    assert state["missing_indices"] == [1]
    assert np.isclose(state["scores"][0], 0.125)
    assert np.isnan(state["scores"][1])
    assert np.isclose(state["scores"][2], 0.5)

    changed_context = payload_fingerprint(
        {"proxy": "M200", "rotation": "frame-b"}
    )
    changed = candidate_checkpoint_state(tmp_path, rows, changed_context)
    assert changed["completed_indices"] == []
    assert changed["missing_indices"] == [0, 1, 2]


def test_generator_sign_checkpoint_rejects_changed_row_or_context(tmp_path):
    row = np.array([1, 0, 1, 0])
    context = payload_fingerprint({"selection": "row-space-a"})
    write_sign_checkpoint(
        tmp_path,
        index=0,
        row=row,
        expectation=-0.875,
        context_fingerprint=context,
        elapsed_seconds=2.0,
    )
    assert np.isclose(
        read_sign_checkpoint(tmp_path, 0, row, context),
        -0.875,
    )
    assert read_sign_checkpoint(
        tmp_path,
        0,
        np.array([1, 1, 0, 0]),
        context,
    ) is None
    assert read_sign_checkpoint(
        tmp_path,
        0,
        row,
        payload_fingerprint({"selection": "row-space-b"}),
    ) is None


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


def test_external_capture_is_invariant_to_anchor_weight():
    branches = [
        {"label": [0, 0], "tag": "anchor", "norm2": 0.2},
        {"label": [1, 0], "tag": "external", "norm2": 0.7},
    ]
    first = select_external_projector_branches(
        branches,
        source_norm2=1.0,
        anchor_label=(0, 0),
        minimum_capture=0.8,
    )
    branches[0]["norm2"] = 0.05
    second = select_external_projector_branches(
        branches,
        source_norm2=0.85,
        anchor_label=(0, 0),
        minimum_capture=0.8,
    )
    assert np.isclose(first["raw_external_capture"], 0.875)
    assert np.isclose(
        first["raw_external_capture"],
        second["raw_external_capture"],
    )


def test_complete_external_distribution_meets_target_without_full_beam():
    result = select_external_projector_branches(
        [
            {"label": [0, 0], "tag": "anchor", "norm2": 0.0191542},
            {"label": [1, 0], "tag": "large", "norm2": 0.7},
            {"label": [0, 1], "tag": "medium", "norm2": 0.2808458},
        ],
        source_norm2=1.0,
        anchor_label=(0, 0),
        minimum_capture=0.999,
    )
    assert result["raw_target_met"]
    assert result["selection_target_met"]
    assert np.isclose(result["selected_external_capture"], 1.0)


def test_tiny_projector_branch_is_not_normalized_into_krylov_basis():
    result = select_external_projector_branches(
        [
            {"label": [0, 0], "tag": "anchor", "norm2": 0.1},
            {
                "label": [1, 0],
                "tag": "physical",
                "norm2": 0.899999999999,
            },
            {"label": [0, 1], "tag": "noise", "norm2": 1.0e-12},
        ],
        source_norm2=1.0,
        anchor_label=(0, 0),
        minimum_capture=0.999,
        absolute_weight_cutoff=1.0e-10,
    )
    assert [item["tag"] for item in result["selected_branches"]] == [
        "physical"
    ]
    rejected = {
        item["tag"]: item["rejection_reason"]
        for item in result["rejected_branches"]
    }
    assert rejected["noise"] == "below_noise_floor"


def test_sub_noise_external_residual_stops_without_normalization():
    result = select_external_projector_branches(
        [
            {"label": [0], "tag": "anchor", "norm2": 0.99999999999},
            {"label": [1], "tag": "noise", "norm2": 1.0e-11},
        ],
        source_norm2=1.0,
        anchor_label=(0,),
        minimum_capture=0.999,
        absolute_weight_cutoff=1.0e-10,
    )
    assert result["external_is_numerical_noise"]
    assert result["selection_target_met"]
    assert result["selected_branches"] == []


def test_beam_pruning_preserves_anchor_prefix():
    kept, pruned = retain_projector_candidates(
        [
            {"label": [0, 0], "tag": "first", "norm2": 1.0},
            {"label": [0, 1], "tag": "second", "norm2": 0.9},
            {"label": [1, 1], "tag": "anchor-path", "norm2": 0.01},
        ],
        beam_width=2,
        protected_prefix=(1, 1),
    )
    assert "anchor-path" in {item["tag"] for item in kept}
    assert "second" in {item["tag"] for item in pruned}


def test_corrected_matrix_imports_arbitrary_legacy_subset(tmp_path):
    legacy_basis = [
        {"tag": "A", "label": [0], "kind": "anchor", "cycle": -1, "depth": 0},
        {
            "tag": "B",
            "label": [1],
            "kind": "residual_seed",
            "cycle": 0,
            "depth": 0,
        },
        {
            "tag": "C",
            "label": [1],
            "kind": "sector_krylov",
            "cycle": 0,
            "depth": 1,
        },
    ]
    legacy_matrix = tmp_path / "coupled_matrices.npz"
    hamiltonian = np.arange(9, dtype=float).reshape(3, 3)
    hamiltonian = hamiltonian + hamiltonian.T
    overlap = np.eye(3)
    np.savez_compressed(
        legacy_matrix,
        hamiltonian=hamiltonian,
        overlap=overlap,
    )
    atomic_json(
        matrix_metadata_path(legacy_matrix),
        {"basis": legacy_basis, "completed_rows": 3},
    )

    target_basis = [legacy_basis[0], legacy_basis[2]]
    imported_h, imported_s, diagnostics = build_coupled_matrices(
        solver=SimpleNamespace(get_mps=lambda tag: tag),
        basis=target_basis,
        full_hamiltonian_mpo=None,
        transition_signatures=set(),
        matrix_path=tmp_path / "coupled_matrices_v2.npz",
        progress_path=tmp_path / "matrix_progress_v2.json",
        resume=True,
        reuse_matrix_paths=[legacy_matrix],
    )
    assert np.array_equal(imported_h, hamiltonian[np.ix_([0, 2], [0, 2])])
    assert np.array_equal(imported_s, overlap[np.ix_([0, 2], [0, 2])])
    assert diagnostics["contractions"] == 0


def test_fitted_operation_restart_rejects_changed_energy_coefficient():
    saved = {
        "operation": "fitted_mps_addition",
        "left_tag": "HPSI",
        "right_tag": "PSI",
        "left_coefficient": 1.0,
        "right_coefficient": 76.0,
        "output_tag": "RESIDUAL",
        "bond_dim": 500,
        "sweeps": 8,
        "tolerance": 1.0e-10,
        "elapsed_seconds": 100.0,
    }
    expected = {
        key: value for key, value in saved.items() if key != "elapsed_seconds"
    }
    assert operation_record_matches(saved, expected)
    expected["right_coefficient"] = 76.1
    assert not operation_record_matches(saved, expected)
    assert record_fingerprint({"b": 2, "a": 1}) == record_fingerprint(
        {"a": 1, "b": 2}
    )
