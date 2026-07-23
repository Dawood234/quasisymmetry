import numpy as np
import scipy.sparse.linalg

from src.selected_sector_lanczos import (
    coupled_krylov_matrix,
    coupling_capture,
    coupling_seeded_krylov_basis,
    krylov_depth_curve,
    sector_leakage_weights,
)


def test_sector_leakage_weights_rank_external_parity_sector():
    matrix = np.asarray(
        [
            [0.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, 1.4],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)

    ranked, norm_squared, residual = sector_leakage_weights(
        operator,
        full_dimension=4,
        anchor_support=np.asarray([0, 3]),
        anchor_vector=np.asarray([1.0, 0.0]),
        anchor_energy=0.0,
        parity_matrix=np.asarray([[1, 0]]),
        norb=2,
        nelec=(1, 1),
    )

    assert ranked[0][0] == (1,)
    assert np.isclose(ranked[0][1], 0.05)
    assert np.isclose(norm_squared, 0.05)
    assert np.allclose(residual, [0.0, 0.2, 0.1, 0.0])


def test_coupling_capture_measures_resolved_leakage_fraction():
    result = {
        "full_addresses": np.asarray([1, 2]),
        "vectors": np.asarray([[1.0], [0.0]]),
    }
    residual = np.asarray([0.0, 0.2, 0.1, 0.0])

    capture = coupling_capture(result, residual, leakage_weight=0.05)

    assert np.isclose(capture, 0.8)


def test_coupling_seeded_krylov_spans_relevant_external_sector():
    matrix = np.asarray(
        [
            [0.0, 0.4, 0.3, 0.0],
            [0.4, 1.0, 0.2, 0.1],
            [0.3, 0.2, 1.5, 0.25],
            [0.0, 0.1, 0.25, 2.0],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)
    support = np.asarray([1, 2, 3])
    seed = matrix[support, 0]

    result = coupling_seeded_krylov_basis(
        operator,
        full_dimension=4,
        support=support,
        coupling_seed=seed,
        max_depth=3,
        tolerance=1.0e-13,
        print_every=0,
    )

    assert result["depth"] == 3
    assert np.allclose(
        result["basis"].conj().T @ result["basis"],
        np.eye(3),
        atol=1.0e-12,
    )
    assert np.allclose(
        result["basis"][:, 0],
        seed / np.linalg.norm(seed),
    )


def test_coupled_krylov_recovers_exact_toy_ground_energy():
    matrix = np.asarray(
        [
            [0.0, 0.4, 0.3, 0.0],
            [0.4, 1.0, 0.2, 0.1],
            [0.3, 0.2, 1.5, 0.25],
            [0.0, 0.1, 0.25, 2.0],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)
    external_support = np.asarray([1, 2, 3])
    result = coupling_seeded_krylov_basis(
        operator,
        full_dimension=4,
        support=external_support,
        coupling_seed=matrix[external_support, 0],
        max_depth=3,
        tolerance=1.0e-13,
        print_every=0,
    )
    result["full_addresses"] = external_support

    coupled, candidates, _seconds = coupled_krylov_matrix(
        operator,
        full_dimension=4,
        anchor_support={
            "label": (0,),
            "full_addresses": np.asarray([0]),
        },
        anchor_vector=np.asarray([1.0]),
        sector_bases={(1,): result},
    )
    exact_energy = np.linalg.eigvalsh(matrix)[0]
    curve = krylov_depth_curve(
        coupled,
        candidates,
        depths=[1, 2, 3],
        reference_energy=exact_energy,
        tolerance=1.0e-10,
    )

    assert curve[-1]["dimension"] == 4
    assert np.isclose(curve[-1]["energy"], exact_energy, atol=1.0e-12)
    assert curve[-1]["converged"]
