"""Dense toy validation of projector splitting and coupling-seeded Krylov."""

import itertools

import numpy as np

from linear_algebra import canonical_generalized_eigh


def z_operator(qubit, n_qubits):
    matrices = [np.eye(2)] * n_qubits
    matrices[qubit] = np.diag([1.0, -1.0])
    output = matrices[0]
    for matrix in matrices[1:]:
        output = np.kron(output, matrix)
    return output


def projector(symmetries, label):
    dimension = symmetries[0].shape[0]
    output = np.eye(dimension, dtype=complex)
    for symmetry, bit in zip(symmetries, label):
        output = output @ (np.eye(dimension) + ((-1) ** bit) * symmetry) / 2.0
    return output


def orthonormal_append(basis, vector, tolerance=1.0e-12):
    vector = np.asarray(vector, dtype=complex).copy()
    for item in basis:
        vector -= item * np.vdot(item, vector)
    norm = np.linalg.norm(vector)
    if norm <= tolerance:
        return False
    basis.append(vector / norm)
    return True


def test_projectors_and_residual_krylov_form_nested_variational_spaces():
    n_qubits = 3
    z = [z_operator(index, n_qubits) for index in range(n_qubits)]
    symmetries = [z[0] @ z[1], z[1] @ z[2]]
    labels = list(itertools.product((0, 1), repeat=2))
    projectors = {label: projector(symmetries, label) for label in labels}
    identity = np.eye(2**n_qubits)
    assert np.allclose(sum(projectors.values()), identity)
    for left, right in itertools.combinations(labels, 2):
        assert np.allclose(projectors[left] @ projectors[right], 0.0)

    rng = np.random.default_rng(19)
    raw = rng.normal(size=(8, 8))
    hamiltonian = (raw + raw.T) / 2.0
    state = rng.normal(size=8)
    state /= np.linalg.norm(state)
    weights = {
        label: np.linalg.norm(projectors[label] @ state) ** 2 for label in labels
    }
    assert np.isclose(sum(weights.values()), 1.0)

    anchor_label = max(labels, key=weights.get)
    anchor_projector = projectors[anchor_label]
    anchor_indices = np.flatnonzero(np.diag(anchor_projector) > 0.5)
    anchor_block = hamiltonian[np.ix_(anchor_indices, anchor_indices)]
    anchor_values, anchor_vectors = np.linalg.eigh(anchor_block)
    anchor = np.zeros(8)
    anchor[anchor_indices] = anchor_vectors[:, 0]
    anchor_energy = anchor_values[0]

    basis = [anchor.astype(complex)]
    energies = [anchor_energy]
    for label in labels:
        if label == anchor_label:
            continue
        sector_h = projectors[label] @ hamiltonian @ projectors[label]
        seed = projectors[label] @ hamiltonian @ anchor
        if not orthonormal_append(basis, seed):
            continue
        orthonormal_append(basis, sector_h @ basis[-1])
        matrix = np.asarray(
            [
                [np.vdot(left, hamiltonian @ right) for right in basis]
                for left in basis
            ]
        )
        overlap = np.asarray(
            [[np.vdot(left, right) for right in basis] for left in basis]
        )
        energies.append(
            float(canonical_generalized_eigh(matrix, overlap)["energies"][0])
        )
    assert np.all(np.diff(energies) <= 1.0e-10)
    assert energies[-1] <= anchor_energy + 1.0e-10

    coefficients = np.arange(1, len(basis) + 1, dtype=float)
    direct = sum(value * vector for value, vector in zip(coefficients, basis))
    left = sum(
        value * vector
        for value, vector in zip(coefficients[: len(basis) // 2], basis)
    )
    right = sum(
        value * vector
        for value, vector in zip(coefficients[len(basis) // 2 :], basis[len(basis) // 2 :])
    )
    assert np.allclose(left + right, direct)
