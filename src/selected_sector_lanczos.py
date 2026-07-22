"""Selected Clifford-sector supports and matrix-free Lanczos utilities.

This module never builds the complete fixed-spin Hamiltonian matrix.  It
generates determinants only for requested parity labels and applies the
molecular Hamiltonian through ffsim's compiled LinearOperator.
"""

import itertools
import time

import numpy as np
import pyscf.fci.cistring
import scipy.sparse.linalg

from src.clifford_sectors import (
    apply_clifford_to_basis_bits,
    bits_to_index,
    occupation_bits,
)


def label_text(label):
    """Return a compact binary sector label."""
    return "".join(str(int(bit)) for bit in label)


def parity_spin_blocks(parity_matrix, norb):
    """Split a parity matrix into alpha and beta orbital contributions."""
    parity = np.atleast_2d(np.asarray(parity_matrix, dtype=np.uint8)) % 2
    if parity.shape[1] == norb:
        return parity, parity
    if parity.shape[1] == 2 * norb:
        return parity[:, 0::2], parity[:, 1::2]
    raise ValueError("parity matrix must have norb or 2*norb columns")


def bitstring_syndrome(bitstring, columns):
    """Evaluate GF(2) parity rows on one spatial-orbital bitstring."""
    syndrome = np.zeros(columns.shape[0], dtype=np.uint8)
    value = int(bitstring)
    orbital = 0
    while value:
        if value & 1:
            syndrome ^= columns[:, orbital]
        value >>= 1
        orbital += 1
    return tuple(int(bit) for bit in syndrome)


def occupied_orbitals(bitstring, norb):
    """Return occupied spatial-orbital indices from an integer bitstring."""
    value = int(bitstring)
    return tuple(orbital for orbital in range(norb) if (value >> orbital) & 1)


def selected_sector_supports(
    parity_matrix,
    labels,
    norb,
    nelec,
    clifford,
    n_symmetries,
    print_progress=True,
):
    """Generate fixed-spin determinants only for the requested sectors.

    Alpha and beta strings are grouped by their partial GF(2) syndromes.  A
    requested label ``s`` needs only pairs satisfying ``s_alpha XOR s_beta=s``.
    This avoids the Cartesian product of all alpha and beta strings.
    """
    n_alpha, n_beta = (int(nelec[0]), int(nelec[1]))
    alpha_rows, beta_rows = parity_spin_blocks(parity_matrix, norb)
    labels = [tuple(int(bit) for bit in label) for label in labels]

    alpha_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), n_alpha), dtype=np.int64
    )
    beta_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), n_beta), dtype=np.int64
    )
    if print_progress:
        print(
            "[support] alpha strings:", len(alpha_strings),
            "beta strings:", len(beta_strings),
            flush=True,
        )

    beta_groups = {}
    for beta_address, beta_string in enumerate(beta_strings):
        syndrome = bitstring_syndrome(beta_string, beta_rows)
        beta_groups.setdefault(syndrome, []).append((beta_address, int(beta_string)))
    if print_progress:
        print(
            "[support] beta strings grouped into",
            len(beta_groups),
            "GF(2) syndromes",
            flush=True,
        )

    entries = {label: [] for label in labels}
    zero_bits = [0] * (2 * norb)
    label_offset = tuple(
        apply_clifford_to_basis_bits(zero_bits, clifford)[:n_symmetries]
    )
    clifford_labels = {
        label: tuple(bit ^ offset for bit, offset in zip(label, label_offset))
        for label in labels
    }
    if print_progress:
        print(
            "[support] physical-to-Clifford label offset:",
            label_text(label_offset),
            flush=True,
        )
    n_beta_strings = len(beta_strings)
    for alpha_address, alpha_string in enumerate(alpha_strings):
        alpha_syndrome = bitstring_syndrome(alpha_string, alpha_rows)
        alpha_occupied = occupied_orbitals(alpha_string, norb)
        for label in labels:
            needed_beta = tuple(
                left ^ right for left, right in zip(label, alpha_syndrome)
            )
            for beta_address, beta_string in beta_groups.get(needed_beta, []):
                beta_occupied = occupied_orbitals(beta_string, norb)
                bits = occupation_bits(alpha_occupied, beta_occupied, norb)
                transformed = apply_clifford_to_basis_bits(bits, clifford)
                transformed_label = tuple(transformed[:n_symmetries])
                if transformed_label != clifford_labels[label]:
                    raise ValueError(
                        f"physical label {label} should map to Clifford label "
                        f"{clifford_labels[label]}, not {transformed_label}"
                    )
                residual_index = bits_to_index(transformed[n_symmetries:])
                full_address = alpha_address * n_beta_strings + beta_address
                entries[label].append((residual_index, full_address))

    supports = {}
    for label in labels:
        ordered = sorted(entries[label])
        residual = np.asarray([item[0] for item in ordered], dtype=np.int64)
        addresses = np.asarray([item[1] for item in ordered], dtype=np.int64)
        if len(np.unique(residual)) != len(residual):
            raise ValueError(f"duplicate residual indices in sector {label_text(label)}")
        supports[label] = {
            "label": label,
            "clifford_label": clifford_labels[label],
            "clifford_label_offset": label_offset,
            "residual_indices": residual,
            "full_addresses": addresses,
            "dimension": int(len(addresses)),
        }
        if print_progress:
            print(
                f"[support] sector {label_text(label)}: "
                f"{len(addresses):,} physical determinants",
                flush=True,
            )
    return supports


def restricted_linear_operator(full_operator, full_dimension, support, statistics):
    """Return the action ``R_s^dagger H R_s`` on one selected support."""
    support = np.asarray(support, dtype=np.int64)
    dimension = len(support)

    operator_dtype = np.dtype(full_operator.dtype)

    def matvec(vector):
        started = time.perf_counter()
        vector = np.asarray(vector)
        work_dtype = np.result_type(operator_dtype, vector.dtype)
        full_vector = np.zeros(full_dimension, dtype=work_dtype)
        full_vector[support] = vector
        result = np.asarray(full_operator @ full_vector, dtype=work_dtype)
        statistics["matvec_count"] += 1
        statistics["matvec_seconds"] += time.perf_counter() - started
        count = statistics["matvec_count"]
        interval = statistics.get("print_every", 25)
        if interval and count % interval == 0:
            print(
                f"[Lanczos] completed {count} H actions; "
                f"cumulative action time={statistics['matvec_seconds']:.1f} s",
                flush=True,
            )
        return result[support]

    return scipy.sparse.linalg.LinearOperator(
        shape=(dimension, dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=operator_dtype,
    )


def solve_selected_sector(
    full_operator,
    full_dimension,
    support,
    n_roots,
    tolerance=1e-9,
    maxiter=None,
    print_every=25,
):
    """Solve low roots of ``R_s^dagger H R_s`` without forming its matrix."""
    support = np.asarray(support, dtype=np.int64)
    dimension = len(support)
    if dimension == 0:
        raise ValueError("cannot solve an empty sector")
    root_count = min(max(1, int(n_roots)), dimension)
    statistics = {
        "matvec_count": 0,
        "matvec_seconds": 0.0,
        "print_every": int(print_every),
    }
    operator = restricted_linear_operator(
        full_operator, full_dimension, support, statistics
    )
    started = time.perf_counter()

    if dimension <= 64 or root_count >= dimension - 1:
        identity = np.eye(dimension, dtype=operator.dtype)
        matrix = np.column_stack([operator @ identity[:, i] for i in range(dimension)])
        matrix = 0.5 * (matrix + matrix.conj().T)
        energies, vectors = np.linalg.eigh(matrix)
        energies = energies[:root_count]
        vectors = vectors[:, :root_count]
        solver = "dense_from_actions"
    else:
        # A structured vector can be exactly orthogonal to roots in unresolved
        # symmetry subspaces.  A fixed random seed is reproducible and overlaps
        # every such subspace with probability one.
        initial = np.random.default_rng(7).normal(size=dimension)
        initial /= np.linalg.norm(initial)
        energies, vectors = scipy.sparse.linalg.eigsh(
            operator,
            k=root_count,
            which="SA",
            tol=float(tolerance),
            maxiter=maxiter,
            v0=initial,
        )
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
        solver = "eigsh"

    return {
        "energies": np.real_if_close(energies),
        "vectors": np.asarray(vectors, dtype=np.complex128),
        "solver": solver,
        "elapsed_seconds": float(time.perf_counter() - started),
        "matvec_count": int(statistics["matvec_count"]),
        "matvec_seconds": float(statistics["matvec_seconds"]),
    }


def coupled_candidate_matrix(full_operator, full_dimension, sector_results):
    """Build the dense Hamiltonian in the retained sector-root basis."""
    candidates = []
    for label in sorted(sector_results):
        result = sector_results[label]
        for root, energy in enumerate(result["energies"]):
            candidates.append(
                {
                    "label": label,
                    "root": int(root),
                    "energy": float(np.real(energy)),
                    "support": np.asarray(result["full_addresses"], dtype=np.int64),
                    "vector": np.asarray(result["vectors"][:, root], dtype=np.complex128),
                }
            )

    count = len(candidates)
    matrix = np.zeros((count, count), dtype=np.complex128)
    started = time.perf_counter()
    for column, ket in enumerate(candidates):
        action_start = time.perf_counter()
        full_vector = np.zeros(full_dimension, dtype=np.complex128)
        full_vector[ket["support"]] = ket["vector"]
        h_vector = np.asarray(full_operator @ full_vector, dtype=np.complex128)
        for row, bra in enumerate(candidates):
            matrix[row, column] = np.vdot(
                bra["vector"], h_vector[bra["support"]]
            )
        print(
            f"[coupled] H action {column + 1}/{count} complete in "
            f"{time.perf_counter() - action_start:.2f} s",
            flush=True,
        )

    matrix = 0.5 * (matrix + matrix.conj().T)
    for index, candidate in enumerate(candidates):
        matrix[index, index] = candidate["energy"]
    return matrix, candidates, float(time.perf_counter() - started)
