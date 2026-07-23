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


def spin_orbital_parity_matrix(parity_matrix, norb):
    """Return parity rows in interleaved alpha/beta spin-orbital form."""
    alpha_rows, beta_rows = parity_spin_blocks(parity_matrix, norb)
    expanded = np.zeros((alpha_rows.shape[0], 2 * norb), dtype=np.uint8)
    expanded[:, 0::2] = alpha_rows
    expanded[:, 1::2] = beta_rows
    return expanded


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


def sector_leakage_weights(
    full_operator,
    full_dimension,
    anchor_support,
    anchor_vector,
    anchor_energy,
    parity_matrix,
    norb,
    nelec,
):
    """Resolve ``||(I-P_anchor) H |phi_anchor>||^2`` by parity sector."""
    anchor_support = np.asarray(anchor_support, dtype=np.int64)
    full_vector = np.zeros(full_dimension, dtype=np.complex128)
    full_vector[anchor_support] = np.asarray(anchor_vector)
    residual = np.asarray(full_operator @ full_vector, dtype=np.complex128)
    residual[anchor_support] -= float(anchor_energy) * np.asarray(anchor_vector)

    n_alpha, n_beta = (int(nelec[0]), int(nelec[1]))
    alpha_rows, beta_rows = parity_spin_blocks(parity_matrix, norb)
    alpha_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), n_alpha), dtype=np.int64
    )
    beta_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), n_beta), dtype=np.int64
    )
    alpha_syndromes = np.asarray(
        [bitstring_syndrome(value, alpha_rows) for value in alpha_strings],
        dtype=np.uint8,
    )
    beta_syndromes = np.asarray(
        [bitstring_syndrome(value, beta_rows) for value in beta_strings],
        dtype=np.uint8,
    )

    addresses = np.flatnonzero(np.abs(residual) > 1.0e-14)
    alpha_addresses = addresses // len(beta_strings)
    beta_addresses = addresses % len(beta_strings)
    labels = alpha_syndromes[alpha_addresses] ^ beta_syndromes[beta_addresses]
    powers = 1 << np.arange(alpha_rows.shape[0], dtype=np.int64)
    codes = labels @ powers
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    weights = np.bincount(
        inverse,
        weights=np.abs(residual[addresses]) ** 2,
    )

    ranked = []
    for code, weight in zip(unique_codes, weights):
        if weight <= 0.0:
            continue
        label = tuple(
            int((code >> bit) & 1) for bit in range(alpha_rows.shape[0])
        )
        ranked.append((label, float(weight)))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked, float(np.sum(weights)), residual


def coupling_capture(result, h_anchor, leakage_weight):
    """Fraction of one sector's anchor coupling represented by solved roots."""
    if leakage_weight <= 0.0:
        return 1.0
    support = np.asarray(result["full_addresses"], dtype=np.int64)
    vectors = np.asarray(result["vectors"])
    couplings = vectors.conj().T @ np.asarray(h_anchor)[support]
    captured = float(np.sum(np.abs(couplings) ** 2))
    return min(1.0, captured / float(leakage_weight))


def orthogonalize_vector(vector, basis, tolerance=1.0e-12):
    """Remove components along an orthonormal basis with two stable passes."""
    vector = np.asarray(vector, dtype=np.complex128).copy()
    basis = np.asarray(basis, dtype=np.complex128)
    if basis.size:
        for _pass in range(2):
            vector -= basis @ (basis.conj().T @ vector)
    norm = float(np.linalg.norm(vector))
    if norm <= float(tolerance):
        return None, norm
    return vector / norm, norm


def coupling_seeded_krylov_basis(
    full_operator,
    full_dimension,
    support,
    coupling_seed,
    max_depth,
    tolerance=1.0e-12,
    print_every=5,
):
    """Build ``span{q, H_ss q, ...}`` from one sector's leakage vector.

    The input ``coupling_seed`` is the selected-sector part of
    ``H|phi_anchor>``.  Starting from this vector targets the states that
    actually couple to the optimized anchor, rather than the lowest-energy
    eigenstates of the external sector.
    """
    support = np.asarray(support, dtype=np.int64)
    seed = np.asarray(coupling_seed, dtype=np.complex128)
    if len(seed) != len(support):
        raise ValueError("coupling seed must match the selected-sector support")
    if int(max_depth) < 1:
        raise ValueError("max_depth must be positive")

    seed_norm = float(np.linalg.norm(seed))
    if seed_norm <= float(tolerance):
        return {
            "basis": np.zeros((len(support), 0), dtype=np.complex128),
            "projected_hamiltonian": np.zeros((0, 0), dtype=np.complex128),
            "seed_norm": seed_norm,
            "depth": 0,
            "matvec_count": 0,
            "matvec_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "breakdown": True,
        }

    statistics = {
        "matvec_count": 0,
        "matvec_seconds": 0.0,
        "print_every": 0,
    }
    operator = restricted_linear_operator(
        full_operator, full_dimension, support, statistics
    )
    started = time.perf_counter()
    basis_vectors = [seed / seed_norm]
    actions = []
    breakdown = False

    for depth in range(int(max_depth)):
        action = np.asarray(operator @ basis_vectors[depth], dtype=np.complex128)
        actions.append(action)
        if print_every and (depth + 1) % int(print_every) == 0:
            print(
                f"[Krylov] completed depth {depth + 1}/{int(max_depth)}; "
                f"H-action time={statistics['matvec_seconds']:.1f} s",
                flush=True,
            )
        if depth + 1 == int(max_depth):
            break

        current_basis = np.column_stack(basis_vectors)
        next_vector, norm = orthogonalize_vector(
            action, current_basis, tolerance=tolerance
        )
        if next_vector is None:
            print(
                f"[Krylov] invariant subspace reached at depth {depth + 1}; "
                f"residual norm={norm:.3e}",
                flush=True,
            )
            breakdown = True
            break
        basis_vectors.append(next_vector)

    basis = np.column_stack(basis_vectors)
    action_matrix = np.column_stack(actions)
    projected = basis.conj().T @ action_matrix
    projected = 0.5 * (projected + projected.conj().T)
    return {
        "basis": basis,
        "projected_hamiltonian": projected,
        "seed_norm": seed_norm,
        "depth": int(basis.shape[1]),
        "matvec_count": int(statistics["matvec_count"]),
        "matvec_seconds": float(statistics["matvec_seconds"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "breakdown": bool(breakdown),
    }


def coupled_krylov_matrix(
    full_operator,
    full_dimension,
    anchor_support,
    anchor_vector,
    sector_bases,
):
    """Build the coupled Hamiltonian in the anchor plus sector-Krylov basis."""
    candidates = [
        {
            "label": tuple(anchor_support["label"]),
            "depth": 0,
            "support": np.asarray(
                anchor_support["full_addresses"], dtype=np.int64
            ),
            "vector": np.asarray(anchor_vector, dtype=np.complex128),
            "kind": "anchor",
        }
    ]
    for label in sorted(sector_bases):
        result = sector_bases[label]
        basis = np.asarray(result["basis"], dtype=np.complex128)
        support = np.asarray(result["full_addresses"], dtype=np.int64)
        for depth in range(basis.shape[1]):
            candidates.append(
                {
                    "label": tuple(label),
                    "depth": int(depth + 1),
                    "support": support,
                    "vector": basis[:, depth],
                    "kind": "krylov",
                }
            )

    count = len(candidates)
    matrix = np.zeros((count, count), dtype=np.complex128)
    started = time.perf_counter()
    for column, ket in enumerate(candidates):
        full_vector = np.zeros(full_dimension, dtype=np.complex128)
        full_vector[ket["support"]] = ket["vector"]
        h_vector = np.asarray(full_operator @ full_vector, dtype=np.complex128)
        for row, bra in enumerate(candidates):
            matrix[row, column] = np.vdot(
                bra["vector"], h_vector[bra["support"]]
            )
        if (column + 1) % 20 == 0 or column + 1 == count:
            print(
                f"[coupled Krylov] H action {column + 1}/{count}",
                flush=True,
            )

    matrix = 0.5 * (matrix + matrix.conj().T)
    return matrix, candidates, float(time.perf_counter() - started)


def krylov_depth_curve(matrix, candidates, depths, reference_energy, tolerance):
    """Diagonalize nested spaces containing up to each depth per sector."""
    curve = []
    for depth in sorted(set(int(value) for value in depths)):
        indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["kind"] == "anchor" or candidate["depth"] <= depth
        ]
        selected = matrix[np.ix_(indices, indices)]
        energy = float(np.linalg.eigvalsh(selected)[0])
        error = energy - float(reference_energy)
        curve.append(
            {
                "depth": depth,
                "dimension": len(indices),
                "energy": energy,
                "error_Ha": error,
                "error_mHa": 1000.0 * error,
                "converged": bool(error <= float(tolerance)),
            }
        )
    return curve


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
