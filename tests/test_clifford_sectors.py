import numpy as np
import openfermion as of

from chemistry import fcidump_data, load_moldata
from metrics import build_coupled_block_cache, build_physical_coupled_block_cache
from optimize_symmetries import get_fci
from src.clifford_sectors import (
    apply_clifford_to_basis_bits,
    bits_to_index,
    build_clifford_frame,
    candidate_hamiltonian,
    load_symmetry_manifest,
    molecular_hamiltonian_to_jw,
    occupation_bits,
    physical_clifford_matrix,
    perturbative_coupled_energy_curve,
    physical_sector_indices,
    sector_state_candidates,
    save_symmetry_manifest,
    solve_tapered_sector,
    solve_physical_clifford_sector,
    tapered_operator,
    z_symmetries_from_parity_matrix,
)
from src.sector_utils import subspace_matrix, symmetry_sectors
from src.selected_sector_lanczos import (
    coupled_candidate_matrix,
    selected_sector_supports,
    solve_selected_sector,
)


def test_diagonal_and_off_diagonal_tapering_matches_explicit_blocks():
    hamiltonian = (
        0.7 * of.QubitOperator("Z0")
        + 0.2 * of.QubitOperator("X0 X1")
        - 0.3 * of.QubitOperator("Y0 Y1")
        + 0.4 * of.QubitOperator("Z1")
    )
    symmetry = of.QubitOperator("Z0 Z1")
    frame = build_clifford_frame(hamiltonian, [symmetry], 2)
    full_matrix = of.get_sparse_operator(frame["hamiltonian"], 2).toarray()

    for bra in ((0,), (1,)):
        for ket in ((0,), (1,)):
            tapered = tapered_operator(frame, bra, ket)
            tapered_matrix = of.get_sparse_operator(tapered, 1).toarray()
            row = [2 * bra[0], 2 * bra[0] + 1]
            column = [2 * ket[0], 2 * ket[0] + 1]
            explicit = full_matrix[np.ix_(row, column)]
            assert np.allclose(tapered_matrix, explicit, atol=1e-12)

    block_01 = of.get_sparse_operator(tapered_operator(frame, (0,), (1,)), 1).toarray()
    block_10 = of.get_sparse_operator(tapered_operator(frame, (1,), (0,)), 1).toarray()
    assert np.allclose(block_01, block_10.conj().T, atol=1e-12)


def test_physical_clifford_matrix_matches_tapered_blocks():
    hamiltonian = (
        0.7 * of.QubitOperator("Z0")
        + 0.2 * of.QubitOperator("X0 X1")
        - 0.3 * of.QubitOperator("Y0 Y1")
        + 0.4 * of.QubitOperator("Z1")
    )
    frame = build_clifford_frame(hamiltonian, [of.QubitOperator("Z0")], 2)
    physical_matrix = physical_clifford_matrix(frame, [0, 1, 2, 3])
    supports = {(0,): [0, 1], (1,): [2, 3]}
    residual_indices = {(0,): [0, 1], (1,): [0, 1]}

    for bra_label, bra_support in supports.items():
        for ket_label, ket_support in supports.items():
            direct = physical_matrix[bra_support, :][:, ket_support].toarray()
            tapered = of.get_sparse_operator(
                tapered_operator(frame, bra_label, ket_label), 1
            ).toarray()
            assert np.allclose(direct, tapered, atol=1e-12)

    sector_results = {
        label: solve_physical_clifford_sector(
            physical_matrix,
            label,
            residual_indices[label],
            supports[label],
            1,
        )
        for label in supports
    }
    candidates = sector_state_candidates(sector_results)
    cache = build_physical_coupled_block_cache(
        physical_matrix,
        sector_results,
        candidates,
    )
    coupled, _ = candidate_hamiltonian(frame, candidates, cache)
    assert np.allclose(coupled, coupled.conj().T, atol=1e-12)


def test_signed_symmetry_manifest_round_trip(tmp_path):
    path = tmp_path / "symmetries.json"
    symmetries = [
        -of.QubitOperator("Z0 Z1"),
        of.QubitOperator("Z2 Z3"),
    ]
    parity_matrix = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]])
    save_symmetry_manifest(
        path,
        symmetries,
        parity_matrix,
        metadata={"reference": "fci", "selection_score": 0.25},
    )
    loaded = load_symmetry_manifest(path)
    assert loaded["symmetries"] == symmetries
    assert np.array_equal(loaded["parity_matrix"], parity_matrix)
    assert loaded["n_qubits"] == 4
    assert loaded["metadata"]["selection_score"] == 0.25


def test_perturbative_curve_stops_at_requested_accuracy():
    hamiltonian = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 2.0],
        ]
    )
    exact_energy = float(np.linalg.eigvalsh(hamiltonian)[0])
    curve = perturbative_coupled_energy_curve(
        hamiltonian,
        exact_energy=exact_energy,
        tolerance=0.01,
    )
    assert curve["converged"]
    assert curve["K"] == 2
    assert len(curve["energies"]) == 2
    assert len(curve["order"]) >= 2
    assert curve["order"][:2] == curve["order"][: curve["K"]]


def test_cached_coupled_blocks_match_direct_construction():
    hamiltonian = (
        0.4 * of.QubitOperator("Z0")
        - 0.2 * of.QubitOperator("Z1")
        + 0.3 * of.QubitOperator("X0")
    )
    frame = build_clifford_frame(hamiltonian, [of.QubitOperator("Z0 Z1")], 2)
    physical_sectors = {(0,): [0, 1], (1,): [0, 1]}
    sector_results = {
        label: solve_tapered_sector(frame, label, indices, 1)
        for label, indices in physical_sectors.items()
    }
    candidates = sector_state_candidates(sector_results)

    direct, _ = candidate_hamiltonian(frame, candidates)
    cache = build_coupled_block_cache(frame, sector_results, candidates, parallel=False)
    cached, returned_cache = candidate_hamiltonian(frame, candidates, cache)

    assert set(cache) == {((0,), (0,)), ((0,), (1,)), ((1,), (1,))}
    assert returned_cache is cache
    assert np.allclose(cached, direct, atol=1e-12)


def test_fixed_particle_basis_mapping_matches_clifford_state_action():
    symmetries = [
        of.QubitOperator("Z0 Z1"),
        of.QubitOperator("Z2 Z3"),
    ]
    frame = build_clifford_frame(of.QubitOperator(), symmetries, 4)
    sectors = physical_sector_indices(2, (1, 1), frame["clifford"], 2)
    assert sum(len(indices) for indices in sectors.values()) == 4

    for alpha in ((0,), (1,)):
        for beta in ((0,), (1,)):
            bits = occupation_bits(alpha, beta, 2)
            expected_bits = apply_clifford_to_basis_bits(bits, frame["clifford"])
            state = np.zeros(16)
            state[bits_to_index(bits)] = 1.0
            transformed = frame["clifford"].transform_state(state)
            actual_index = int(np.argmax(np.abs(transformed)))
            assert actual_index == bits_to_index(expected_bits)

    signed_frame = build_clifford_frame(
        of.QubitOperator(),
        [-of.QubitOperator("Z0 Z1"), of.QubitOperator("Z2 Z3")],
        4,
    )
    bits = occupation_bits((0,), (1,), 2)
    expected_bits = apply_clifford_to_basis_bits(bits, signed_frame["clifford"])
    state = np.zeros(16)
    state[bits_to_index(bits)] = 1.0
    transformed = signed_frame["clifford"].transform_state(state)
    assert int(np.argmax(np.abs(transformed))) == bits_to_index(expected_bits)


def test_selected_support_generation_matches_full_sector_grouping():
    parity_matrix = np.asarray([[1, 0], [0, 1]], dtype=int)
    symmetries = z_symmetries_from_parity_matrix(parity_matrix, 2)
    frame = build_clifford_frame(of.QubitOperator(), symmetries, 4)
    full = symmetry_sectors(parity_matrix, 2, (1, 1))
    labels = sorted(full)[:3]
    selected = selected_sector_supports(
        parity_matrix,
        labels,
        2,
        (1, 1),
        frame["clifford"],
        frame["n_symmetries"],
        print_progress=False,
    )
    for label in labels:
        assert set(selected[label]["full_addresses"]) == set(full[label])
        assert selected[label]["dimension"] == len(full[label])


def test_selected_lanczos_and_coupled_matrix_match_full_operator():
    import ffsim

    path = "hamiltonians/water/H2O_OH0.9580_104.5000.FCIDUMP"
    parity_matrix = np.asarray([[1, 0, 0, 0, 0, 0, 0]], dtype=int)
    moldata = load_moldata(path)
    symmetries = z_symmetries_from_parity_matrix(parity_matrix, moldata.norb)
    frame = build_clifford_frame(of.QubitOperator(), symmetries, 2 * moldata.norb)
    labels = [(0,), (1,)]
    supports = selected_sector_supports(
        parity_matrix,
        labels,
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
        print_progress=False,
    )
    full_operator = ffsim.linear_operator(
        moldata.hamiltonian, norb=moldata.norb, nelec=moldata.nelec
    )
    results = {}
    for label in labels:
        solved = solve_selected_sector(
            full_operator,
            full_operator.shape[0],
            supports[label]["full_addresses"],
            2,
            tolerance=1e-10,
            print_every=0,
        )
        direct = np.linalg.eigvalsh(
            subspace_matrix(full_operator, supports[label]["full_addresses"])
        )[:2]
        assert np.allclose(solved["energies"], direct, atol=1e-9)
        results[label] = {**supports[label], **solved}

    coupled, candidates, _ = coupled_candidate_matrix(
        full_operator, full_operator.shape[0], results
    )
    assert coupled.shape == (4, 4)
    assert len(candidates) == 4
    assert np.allclose(coupled, coupled.conj().T, atol=1e-10)


def test_h2o_tapered_sectors_match_determinant_blocks_and_full_fci():
    path = "hamiltonians/water/H2O_OH0.9580_104.5000.FCIDUMP"
    parity_path = "hamiltonians/water/parity_water_0.9580.txt"
    parity_matrix = np.atleast_2d(np.loadtxt(parity_path, dtype=int))
    moldata = load_moldata(path)
    symmetries = z_symmetries_from_parity_matrix(parity_matrix, moldata.norb)
    jw_hamiltonian = molecular_hamiltonian_to_jw(moldata.hamiltonian, moldata.nelec)
    frame = build_clifford_frame(jw_hamiltonian, symmetries, 2 * moldata.norb)

    physical = physical_sector_indices(
        moldata.norb,
        moldata.nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )
    determinant = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)
    assert set(physical) == set(determinant)
    assert all(len(physical[label]) == len(determinant[label]) for label in physical)

    import ffsim

    h_fixed = ffsim.linear_operator(
        moldata.hamiltonian,
        norb=moldata.norb,
        nelec=moldata.nelec,
    )
    sector_results = {}
    for label in sorted(physical):
        root_count = len(physical[label])
        solved = solve_tapered_sector(
            frame,
            label,
            physical[label],
            root_count,
        )
        direct = np.linalg.eigvalsh(subspace_matrix(h_fixed, determinant[label]))
        assert np.allclose(solved["energies"], direct, atol=1e-10)
        sector_results[label] = solved

    candidates = sector_state_candidates(sector_results)
    coupled_hamiltonian, _ = candidate_hamiltonian(frame, candidates)
    coupled_energy = float(np.linalg.eigvalsh(coupled_hamiltonian)[0])
    exact_energy, _ = get_fci(fcidump_data(path))
    assert abs(coupled_energy - exact_energy) < 1e-8
    assert frame["n_qubits"] == 14
    assert frame["n_residual_qubits"] == 9
