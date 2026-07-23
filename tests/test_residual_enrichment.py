import numpy as np

from selected_clifford_residual_enrichment import (
    choose_residual_labels,
    load_restart,
    residual_summary,
    save_restart,
)


def test_residual_label_selection_separates_existing_and_new_sectors():
    ranking = [
        ((0, 0), 0.5),
        ((1, 0), 0.3),
        ((0, 1), 0.15),
        ((1, 1), 0.05),
    ]
    current = {(0, 0), (0, 1)}

    existing, new = choose_residual_labels(ranking, current, 2, 1)
    inside, outside = residual_summary(ranking, current)

    assert existing == [(0, 0), (0, 1)]
    assert new == [(1, 0)]
    assert np.isclose(inside, 0.65)
    assert np.isclose(outside, 0.35)


def test_residual_restart_round_trip(tmp_path):
    supports = {
        (0,): np.asarray([0, 2]),
        (1,): np.asarray([1, 3]),
    }
    bases = {
        (0,): np.eye(2, dtype=np.complex128),
        (1,): np.asarray([[1.0], [0.0]], dtype=np.complex128),
    }
    candidates = [
        {
            "label": (0,),
            "kind": "anchor",
            "depth": 0,
            "sector_column": 0,
            "cycle": 0,
            "support": supports[(0,)],
            "vector": bases[(0,)][:, 0],
        },
        {
            "label": (1,),
            "kind": "residual_krylov",
            "depth": 1,
            "sector_column": 0,
            "cycle": 1,
            "support": supports[(1,)],
            "vector": bases[(1,)][:, 0],
        },
        {
            "label": (0,),
            "kind": "residual_krylov",
            "depth": 2,
            "sector_column": 1,
            "cycle": 1,
            "support": supports[(0,)],
            "vector": bases[(0,)][:, 1],
        },
    ]
    matrix = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 2.0],
        ],
        dtype=np.complex128,
    )
    history = [{"cycle": 1, "energy_after": -0.1}]

    save_restart(
        tmp_path,
        1,
        matrix,
        candidates,
        bases,
        supports,
        history,
    )
    state, saved_matrix, saved_candidates, saved_bases, saved_supports = (
        load_restart(tmp_path)
    )

    assert state["completed_cycle"] == 1
    assert state["history"] == history
    assert np.allclose(saved_matrix, matrix)
    for label in bases:
        assert np.array_equal(saved_supports[label], supports[label])
        assert np.allclose(saved_bases[label], bases[label])
    for original, saved in zip(candidates, saved_candidates):
        assert saved["label"] == original["label"]
        assert saved["kind"] == original["kind"]
        assert saved["sector_column"] == original["sector_column"]
        assert np.allclose(saved["vector"], original["vector"])
