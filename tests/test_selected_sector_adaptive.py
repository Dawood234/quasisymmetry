import numpy as np
import scipy.sparse.linalg

from src.selected_sector_lanczos import (
    coupled_ground_residual,
    coupled_krylov_matrix,
    coupling_capture,
    coupling_seeded_krylov_basis,
    extend_coupled_matrix,
    krylov_depth_curve,
    residual_seeded_krylov_extension,
    sector_leakage_weights,
    sector_vector_weights,
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


def test_sector_vector_weights_resolve_arbitrary_fixed_spin_vector():
    vector = np.asarray([0.1, 0.2, 0.3, 0.4])

    ranked, norm_squared = sector_vector_weights(
        vector,
        parity_matrix=np.asarray([[1, 0]]),
        norb=2,
        nelec=(1, 1),
    )

    weights = dict(ranked)
    assert np.isclose(norm_squared, np.vdot(vector, vector).real)
    assert np.isclose(sum(weights.values()), norm_squared)
    assert set(weights) == {(0,), (1,)}


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


def test_residual_extension_is_orthogonal_to_existing_sector_basis():
    matrix = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.2, 1.0, 0.3],
            [0.0, 0.3, 2.0],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)
    existing = np.asarray([[1.0], [0.0], [0.0]])
    seed = np.asarray([0.4, 0.5, 0.6])

    extension = residual_seeded_krylov_extension(
        operator,
        full_dimension=3,
        support=np.arange(3),
        residual_seed=seed,
        existing_basis=existing,
        max_vectors=2,
        tolerance=1.0e-13,
    )

    combined = np.column_stack([existing, extension])
    assert extension.shape == (3, 2)
    assert np.allclose(combined.conj().T @ combined, np.eye(3), atol=1.0e-12)


def test_incremental_coupled_matrix_matches_direct_projection():
    matrix = np.asarray(
        [
            [0.0, 0.4, 0.3],
            [0.4, 1.0, 0.2],
            [0.3, 0.2, 1.5],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)
    candidates = [
        {
            "label": (0,),
            "support": np.asarray([0]),
            "vector": np.asarray([1.0]),
            "kind": "anchor",
            "depth": 0,
            "sector_column": 0,
        }
    ]
    new_candidates = [
        {
            "label": (1,),
            "support": np.asarray([1, 2]),
            "vector": np.asarray([1.0, 0.0]),
            "kind": "residual_krylov",
            "depth": 1,
            "sector_column": 0,
        },
        {
            "label": (1,),
            "support": np.asarray([1, 2]),
            "vector": np.asarray([0.0, 1.0]),
            "kind": "residual_krylov",
            "depth": 2,
            "sector_column": 1,
        },
    ]

    extended, all_candidates, _seconds = extend_coupled_matrix(
        operator,
        full_dimension=3,
        matrix=np.asarray([[matrix[0, 0]]]),
        candidates=candidates,
        new_candidates=new_candidates,
        print_every=0,
    )
    energy, _coefficients, _state, residual = coupled_ground_residual(
        operator,
        full_dimension=3,
        matrix=extended,
        candidates=all_candidates,
    )

    assert np.allclose(extended, matrix)
    assert np.isclose(energy, np.linalg.eigvalsh(matrix)[0], atol=1.0e-12)
    assert np.linalg.norm(residual) < 1.0e-12


def test_residual_enrichment_recovers_missing_anchor_and_external_directions():
    matrix = np.asarray(
        [
            [0.0, 0.4, 0.0, 0.1],
            [0.4, 1.0, 0.3, 0.2],
            [0.0, 0.3, 1.5, 0.4],
            [0.1, 0.2, 0.4, 2.0],
        ]
    )
    operator = scipy.sparse.linalg.aslinearoperator(matrix)
    supports = {(0,): np.asarray([0, 3]), (1,): np.asarray([1, 2])}
    anchor_block = matrix[np.ix_(supports[(0,)], supports[(0,)])]
    _anchor_energies, anchor_vectors = np.linalg.eigh(anchor_block)
    anchor = anchor_vectors[:, :1]
    candidates = [
        {
            "label": (0,),
            "support": supports[(0,)],
            "vector": anchor[:, 0],
            "kind": "anchor",
            "depth": 0,
            "sector_column": 0,
        }
    ]
    basis = {(0,): anchor, (1,): np.zeros((2, 0), dtype=np.complex128)}
    coupled = np.asarray([[_anchor_energies[0]]])

    for _cycle in range(2):
        _energy, _coefficients, _state, residual = coupled_ground_residual(
            operator, 4, coupled, candidates
        )
        new_candidates = []
        for label in ((0,), (1,)):
            extension = residual_seeded_krylov_extension(
                operator,
                4,
                supports[label],
                residual[supports[label]],
                basis[label],
                max_vectors=2,
                tolerance=1.0e-13,
            )
            first_column = basis[label].shape[1]
            basis[label] = np.column_stack([basis[label], extension])
            for offset in range(extension.shape[1]):
                column = first_column + offset
                new_candidates.append(
                    {
                        "label": label,
                        "support": supports[label],
                        "vector": basis[label][:, column],
                        "kind": "residual_krylov",
                        "depth": column + 1,
                        "sector_column": column,
                    }
                )
        if not new_candidates:
            break
        coupled, candidates, _seconds = extend_coupled_matrix(
            operator,
            4,
            coupled,
            candidates,
            new_candidates,
            print_every=0,
        )

    final_energy = np.linalg.eigvalsh(coupled)[0]
    exact_energy = np.linalg.eigvalsh(matrix)[0]
    assert len(candidates) == 4
    assert np.isclose(final_energy, exact_energy, atol=1.0e-12)
