"""Validation of the block2 DMRG solver against exact diagonalization.

These tests only need numpy / scipy / openfermion / block2, so they run on
machines without pyscf (e.g. native Windows). The exact reference is built by
Jordan-Wigner mapping the integrals with OpenFermion and diagonalizing in the
correct particle-number sector.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

import numpy as np
import openfermion as of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    find_sector_determinant,
    restore_g2e,
    rotate_integrals,
    rotation_from_parameters,
    solve_or_load_ground_state,
)
from src.dmrg_decoupled_energy import energy_from_rdms

FCIDUMP_PATH = Path(__file__).resolve().parents[1] / "hamiltonians" / "sentest_5_d754.FCIDUMP"
ENERGY_TOL = 1e-7


def test_find_sector_determinant_satisfies_spin_and_parities():
    parity_matrix = np.asarray(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    determinant = find_sector_determinant(
        parity_matrix,
        sector_label=(0, 1),
        norb=3,
        n_elec=4,
        spin=0,
    )
    alpha = np.asarray([symbol in ("a", "2") for symbol in determinant], dtype=int)
    beta = np.asarray([symbol in ("b", "2") for symbol in determinant], dtype=int)
    occupations = np.column_stack([alpha, beta]).ravel()
    assert alpha.sum() == 2
    assert beta.sum() == 2
    np.testing.assert_array_equal((parity_matrix @ occupations) % 2, [0, 1])


def fermion_operator_from_integrals(h1e, g2e, ecore, norb):
    """H in interleaved spin-orbital order [a0, b0, a1, b1, ...]."""
    op = of.FermionOperator((), ecore)
    for p in range(norb):
        for q in range(norb):
            if abs(h1e[p, q]) > 1e-14:
                for s in (0, 1):
                    op += of.FermionOperator(
                        ((2 * p + s, 1), (2 * q + s, 0)), h1e[p, q]
                    )
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    coeff = g2e[p, q, r, s]
                    if abs(coeff) > 1e-14:
                        for s1 in (0, 1):
                            for s2 in (0, 1):
                                op += of.FermionOperator(
                                    ((2 * p + s1, 1), (2 * r + s2, 1),
                                     (2 * s + s2, 0), (2 * q + s1, 0)),
                                    0.5 * coeff,
                                )
    return op


def occupation_of_index(index, n_qubits):
    """Occupations per qubit with qubit 0 as the leftmost tensor factor."""
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


def sector_support(n_qubits, norb, n_alpha, n_beta):
    support = []
    for index in range(1 << n_qubits):
        occ = occupation_of_index(index, n_qubits)
        if (sum(occ[0::2]) == n_alpha) and (sum(occ[1::2]) == n_beta):
            support.append(index)
    return np.array(support)


def parity_eigenvalues(parity_row, indices, n_qubits, norb):
    """Diagonal parity eigenvalues (+-1) for the given basis indices."""
    parity_row = np.asarray(parity_row, dtype=int)
    if parity_row.shape[0] == norb:
        spin_orbitals = [q for p in np.flatnonzero(parity_row)
                         for q in (2 * p, 2 * p + 1)]
    else:
        spin_orbitals = list(np.flatnonzero(parity_row))
    values = np.empty(len(indices))
    for i, index in enumerate(indices):
        occ = occupation_of_index(index, n_qubits)
        values[i] = (-1.0) ** sum(occ[q] for q in spin_orbitals)
    return values


class TestBlock2DMRGSolver(unittest.TestCase):
    """End-to-end checks of energies, conventions, sectors and persistence."""

    @classmethod
    def setUpClass(cls):
        cls.store_root = Path(tempfile.mkdtemp(prefix="dmrg_solver_test_"))
        cls.solver = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH, store_dir=cls.store_root / "gs"
        )
        cls.norb = cls.solver.n_sites
        cls.n_qubits = 2 * cls.norb
        cls.n_alpha = (cls.solver.n_elec + cls.solver.spin) // 2
        cls.n_beta = cls.solver.n_elec - cls.n_alpha

        g2e_full = restore_g2e(cls.solver.g2e, cls.norb)
        hamiltonian = fermion_operator_from_integrals(
            cls.solver.h1e, g2e_full, cls.solver.ecore, cls.norb
        )
        h_sparse = of.get_sparse_operator(
            of.jordan_wigner(hamiltonian), n_qubits=cls.n_qubits
        ).tocsr()
        cls.support = sector_support(
            cls.n_qubits, cls.norb, cls.n_alpha, cls.n_beta
        )
        cls.h_sector = h_sparse[np.ix_(cls.support, cls.support)].toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(cls.h_sector)
        cls.e_exact = eigenvalues[0]
        cls.psi_exact = np.zeros(1 << cls.n_qubits, dtype=complex)
        cls.psi_exact[cls.support] = eigenvectors[:, 0]

        cls.result = cls.solver.run_ground_state(
            DMRGConfig(max_bond_dim=150, n_sweeps=16)
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.store_root, ignore_errors=True)

    def test_ground_state_energy_matches_exact(self):
        self.assertAlmostEqual(
            self.result.energy, self.e_exact, delta=ENERGY_TOL
        )
        self.assertGreater(len(self.result.sweep_history), 0)
        self.assertIn("discarded_weight", self.result.sweep_history[-1])

    def test_spin_resolved_rdms_reproduce_energy(self):
        ket = self.solver.get_mps(self.result.mps_tag)
        rdm1, rdm2 = self.solver.spin_resolved_rdms(ket)
        energy = energy_from_rdms(
            self.solver.h1e,
            self.solver.g2e,
            self.solver.ecore,
            rdm1,
            rdm2,
        )
        self.assertAlmostEqual(energy, self.result.energy, delta=1.0e-7)

    def test_statevector_matches_exact_ground_state(self):
        psi = self.solver.to_statevector()
        self.assertAlmostEqual(np.linalg.norm(psi), 1.0, places=8)
        overlap = abs(np.vdot(self.psi_exact, psi))
        self.assertAlmostEqual(overlap, 1.0, places=6)

    def test_ci_vector_matches_pyscf_convention(self):
        """CI vector must follow the pyscf addressing used by get_fci."""
        ci = self.solver.to_ci_vector()
        dim_beta = comb(self.norb, self.n_beta)
        self.assertEqual(
            ci.shape[0], comb(self.norb, self.n_alpha) * dim_beta
        )
        self.assertAlmostEqual(np.linalg.norm(ci), 1.0, places=8)

        # Rebuild the expected CI vector from the exact interleaved
        # statevector using the same blocked -> interleaved phase as
        # optimize_symmetries.expand_state.
        expected = np.zeros_like(ci)
        alpha_strings = [s for s in range(1 << self.norb)
                         if bin(s).count("1") == self.n_alpha]
        beta_strings = [s for s in range(1 << self.norb)
                        if bin(s).count("1") == self.n_beta]
        alpha_strings.sort()
        beta_strings.sort()
        for ia, sa in enumerate(alpha_strings):
            occ_a = [p for p in range(self.norb) if (sa >> p) & 1]
            for ib, sb in enumerate(beta_strings):
                occ_b = [p for p in range(self.norb) if (sb >> p) & 1]
                index = 0
                for p in occ_a:
                    index |= 1 << (self.n_qubits - 1 - 2 * p)
                for p in occ_b:
                    index |= 1 << (self.n_qubits - 1 - (2 * p + 1))
                inversions = sum(1 for pa in occ_a for pb in occ_b if pb < pa)
                phase = -1.0 if inversions % 2 else 1.0
                expected[ia * dim_beta + ib] = phase * self.psi_exact[index]

        # Align global phase on the largest amplitude.
        pivot = np.argmax(np.abs(expected))
        ci_aligned = ci * (expected[pivot] / ci[pivot])
        np.testing.assert_allclose(ci_aligned, expected, atol=1e-6)

    def test_parity_expectations_match_exact(self):
        parity_matrix = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ])
        expectations = self.solver.symmetry_expectations(parity_matrix)
        amplitudes = self.psi_exact[self.support]
        for row, value in zip(parity_matrix, expectations):
            eigenvalues = parity_eigenvalues(
                row, self.support, self.n_qubits, self.norb
            )
            exact = np.real(
                np.sum(eigenvalues * np.abs(amplitudes) ** 2)
            )
            self.assertAlmostEqual(value, exact, delta=1.0e-6)

    def test_spin_resolved_parity_row(self):
        row = np.zeros(2 * self.norb, dtype=int)
        row[0] = 1  # alpha parity of orbital 0 only
        value = self.solver.symmetry_expectations(row.reshape(1, -1))[0]
        eigenvalues = parity_eigenvalues(
            row, self.support, self.n_qubits, self.norb
        )
        exact = np.real(np.sum(
            eigenvalues * np.abs(self.psi_exact[self.support]) ** 2
        ))
        self.assertAlmostEqual(value, exact, places=6)

    def test_sector_ground_state_penalty(self):
        parity_matrix = np.array([[1, 0, 0, 0, 0]])
        for bit in (0, 1):
            eigenvalues = parity_eigenvalues(
                parity_matrix[0], self.support, self.n_qubits, self.norb
            )
            target = 1.0 if bit == 0 else -1.0
            mask = np.isclose(eigenvalues, target)
            h_sub = self.h_sector[np.ix_(mask.nonzero()[0], mask.nonzero()[0])]
            e_sector_exact = np.linalg.eigvalsh(h_sub)[0]

            result = self.solver.sector_ground_state(
                parity_matrix, (bit,), penalty=10.0,
                config=DMRGConfig(
                    max_bond_dim=150, n_sweeps=16,
                    mps_tag=f"SECTOR_{bit}",
                ),
            )
            self.assertAlmostEqual(
                result.energy, e_sector_exact, delta=1e-6,
                msg=f"sector bit={bit}",
            )
            self.assertAlmostEqual(
                result.symmetry_expectations[0], target, delta=1e-4
            )

    def test_sector_ground_state_warm_start_is_reloadable(self):
        parity_matrix = np.array([[1, 0, 0, 0, 0]])
        first = self.solver.sector_ground_state(
            parity_matrix,
            (0,),
            penalty=10.0,
            config=DMRGConfig(
                max_bond_dim=100, n_sweeps=8, mps_tag="WARM_A"
            ),
            mps_tag="WARM_A",
        )
        second = self.solver.sector_ground_state(
            parity_matrix,
            (0,),
            penalty=10.0,
            config=DMRGConfig(
                max_bond_dim=100, n_sweeps=4, mps_tag="WARM_B"
            ),
            mps_tag="WARM_B",
            initial_mps_tag=first.mps_tag,
        )
        self.assertAlmostEqual(second.energy, first.energy, delta=1e-6)
        self.assertIn("WARM_B", self.solver.stored_tags())
        self.solver.get_mps("WARM_B")

    def test_sector_ground_state_multi_symmetry_spin_resolved(self):
        """Two symmetries, one of them a single-spin (alpha) parity."""
        norb = self.norb
        parity_matrix = np.zeros((2, 2 * norb), dtype=int)
        parity_matrix[0, 0] = parity_matrix[0, 1] = 1  # full parity, orbital 0
        parity_matrix[1, 2] = 1                        # alpha parity, orbital 1
        sector_label = (0, 1)

        eigenvalue_rows = [
            parity_eigenvalues(row, self.support, self.n_qubits, self.norb)
            for row in parity_matrix
        ]
        targets = [(-1.0) ** bit for bit in sector_label]
        mask = np.logical_and(
            np.isclose(eigenvalue_rows[0], targets[0]),
            np.isclose(eigenvalue_rows[1], targets[1]),
        )
        h_sub = self.h_sector[np.ix_(mask.nonzero()[0], mask.nonzero()[0])]
        e_sector_exact = np.linalg.eigvalsh(h_sub)[0]

        result = self.solver.sector_ground_state(
            parity_matrix, sector_label, penalty=30.0,
            config=DMRGConfig(
                max_bond_dim=250, n_sweeps=30, mps_tag="SECTOR_SPIN_01",
            ),
        )
        self.assertAlmostEqual(result.energy, e_sector_exact, delta=1e-6)
        np.testing.assert_allclose(
            result.symmetry_expectations, targets, atol=1e-4
        )

    def test_dominant_sector_labels(self):
        parity_matrix = np.array([[1, 0, 0, 0, 0]])
        labels = self.solver.dominant_sector_labels(parity_matrix)
        total_weight = sum(weight for _, weight in labels)
        self.assertAlmostEqual(total_weight, 1.0, places=6)

        eigenvalues = parity_eigenvalues(
            parity_matrix[0], self.support, self.n_qubits, self.norb
        )
        amplitudes = np.abs(self.psi_exact[self.support]) ** 2
        exact_weights = {
            (0,): float(np.sum(amplitudes[np.isclose(eigenvalues, 1.0)])),
            (1,): float(np.sum(amplitudes[np.isclose(eigenvalues, -1.0)])),
        }
        for label, weight in labels:
            self.assertAlmostEqual(weight, exact_weights[label], places=5)

    def test_persistence_reload_without_resolving(self):
        reloaded = Block2DMRGSolver.load(self.solver.store_dir)
        self.assertIn("GS", reloaded.stored_tags())
        energy = reloaded.energy_expectation(reloaded.get_mps("GS"))
        self.assertAlmostEqual(energy, self.result.energy, delta=1e-8)

        result = solve_or_load_ground_state(reloaded)
        self.assertAlmostEqual(result.energy, self.result.energy, delta=1e-12)
        self.assertEqual(
            len(result.sweep_history), len(self.result.sweep_history)
        )

        ci = reloaded.to_ci_vector(reloaded.get_mps("GS"))
        self.assertAlmostEqual(np.linalg.norm(ci), 1.0, places=8)

    def test_su2_mode_matches_exact_singlet_energy(self):
        store = self.store_root / "su2"
        solver = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH,
            store_dir=store,
            symmetry_mode="su2",
            n_threads=2,
        )
        result = solver.run_ground_state(
            DMRGConfig(
                max_bond_dim=40,
                n_sweeps=4,
                bond_dims=(20, 30, 40, 40),
                noises=(1.0e-5, 1.0e-6, 0.0, 0.0),
                mps_tag="SU2",
            )
        )
        self.assertAlmostEqual(result.energy, self.e_exact, delta=1.0e-7)
        self.assertEqual(solver.symmetry_mode, "su2")
        self.assertEqual(
            Block2DMRGSolver.load(store).symmetry_mode,
            "su2",
        )

    def test_bipartite_entanglement_shape(self):
        entropies = self.solver.bipartite_entanglement()
        self.assertEqual(len(entropies), self.norb - 1)
        self.assertTrue(np.all(entropies >= -1e-12))

    def test_rotation_parameters_give_orthogonal_matrix(self):
        rng = np.random.default_rng(7)
        x = rng.normal(scale=0.1, size=comb(self.norb, 2))
        rotation = rotation_from_parameters(x, self.norb)
        np.testing.assert_allclose(
            rotation @ rotation.T, np.eye(self.norb), atol=1e-12
        )

    def test_energy_invariant_under_orbital_rotation(self):
        rng = np.random.default_rng(11)
        x = rng.normal(scale=0.2, size=comb(self.norb, 2))
        rotation = rotation_from_parameters(x, self.norb)
        h1e_rot, g2e_rot = rotate_integrals(
            self.solver.h1e, self.solver.g2e, rotation
        )
        rotated_solver = Block2DMRGSolver(
            h1e=h1e_rot, g2e=g2e_rot, ecore=self.solver.ecore,
            n_elec=self.solver.n_elec, spin=self.solver.spin,
            store_dir=self.store_root / "rotated",
        )
        result = rotated_solver.run_ground_state(
            DMRGConfig(max_bond_dim=150, n_sweeps=16)
        )
        self.assertAlmostEqual(result.energy, self.e_exact, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
