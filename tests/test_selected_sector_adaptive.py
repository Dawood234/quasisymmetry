import numpy as np
import scipy.sparse.linalg

from src.selected_sector_lanczos import (
    coupling_capture,
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
