"""Sanity checks for PT / reference coupled-dimension diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coupled_energy_core import (
    all_sector_eigenpair_candidates,
    backward_prune_indices,
    build_candidate_hamiltonian,
    coupled_dimension_from_order,
    find_k_epsilon,
    iterative_pt_from_hamiltonian,
    k_pt_from_ordered_weights,
    nested_ground_energy,
    one_shot_coupled_energy,
    one_shot_from_hamiltonian,
    one_shot_pt_order,
    one_shot_pt_weight,
    reference_coupled_energy,
)
from src.energy_diagnostics import (
    coupled_energy_perturbation,
    decoupled_energy_test,
    diagonalize_sector_blocks,
    reference_coupled_energy_k,
)


def _toy_hamiltonian_and_sectors():
    dim = 6
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.05
    h[2, 2] = 0.1
    h[3, 3] = 0.11
    h[4, 4] = 0.2
    h[5, 5] = 0.3
    coupling = 0.02
    h[0, 2] = h[2, 0] = coupling

    sectors = {"A": [0, 1], "B": [2, 3], "C": [4], "D": [5]}
    sector_data = diagonalize_sector_blocks(lambda v: h @ v, sectors, dim)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sectors, sector_data, e_exact


def _toy_with_excited_cross_coupling():
    dim = 3
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.08
    h[2, 2] = 0.12
    h[0, 2] = h[2, 0] = 0.02
    h[1, 2] = h[2, 1] = 0.05

    sectors = {"A": [0, 1], "B": [2]}
    sector_data = diagonalize_sector_blocks(lambda v: h @ v, sectors, dim)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sectors, sector_data, e_exact


class CoupledEnergySanityTests(unittest.TestCase):
    def test_one_shot_matches_exact_on_toy(self):
        h, _sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        e_pt, k_pt, converged, _ = coupled_energy_perturbation(
            lambda v: h @ v, sector_data, e_exact=e_exact, tol=1e-10
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(e_pt, e_exact, places=10)
        self.assertGreaterEqual(k_pt, 2)

    def test_one_shot_finds_excited_cross_coupling(self):
        h, _sectors, sector_data, e_exact = _toy_with_excited_cross_coupling()
        e_pt, k_pt, converged, keys = coupled_energy_perturbation(
            lambda v: h @ v, sector_data, e_exact=e_exact, tol=1e-10
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(e_pt, e_exact, places=10)
        self.assertGreater(k_pt, 2)
        block_indices = {idx for _key, idx in keys}
        self.assertIn(1, block_indices)

    def test_reference_overlap_matches_exact_on_toy(self):
        h, _sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        candidates = all_sector_eigenpair_candidates(sector_data)
        # Exact ground state as reference.
        _evals, evecs = np.linalg.eigh(0.5 * (h + h.conj().T))
        psi_ref = evecs[:, 0]
        result = reference_coupled_energy(
            candidates, lambda v: h @ v, psi_ref, e_exact=e_exact, tol=1e-10
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.e_coupled, e_exact, places=10)
        self.assertGreaterEqual(result.K, 2)

        vecs = np.column_stack([c[2] for c in candidates])
        k_min, e_coupled, converged, order = reference_coupled_energy_k(
            lambda v: h @ v, vecs, psi_ref, e_exact, chemical_precision=1e-10
        )
        self.assertTrue(converged)
        self.assertEqual(k_min, result.K)
        self.assertAlmostEqual(e_coupled, e_exact, places=10)
        self.assertEqual(list(order[:k_min]), list(result.order_indices[:k_min]))

    def test_pt_ordering_ranks_coupled_state_first(self):
        h, _sectors, sector_data, _e_exact = _toy_hamiltonian_and_sectors()
        candidates = all_sector_eigenpair_candidates(sector_data)
        energies = [c[0] for c in candidates]
        ref = int(np.argmin(energies))
        h_ref = h @ candidates[ref][2]
        couplings = [complex(np.vdot(c[2], h_ref)) for c in candidates]
        order, weights = one_shot_pt_order(energies, couplings, ref)
        self.assertEqual(order[0], ref)
        first_ext = order[1]
        self.assertGreater(weights[first_ext], 0.0)
        for index in order[2:]:
            self.assertGreaterEqual(weights[first_ext], weights[index])

    def test_k_pt_threshold_and_nested_search(self):
        weights = [1.0, 0.5, 1e-14, 0.0]
        self.assertEqual(k_pt_from_ordered_weights(weights, tau_pt=0.1), 3)
        self.assertEqual(k_pt_from_ordered_weights(weights, tau_pt=2.0), 1)

        h_ord = np.asarray(
            [
                [0.0, 0.2, 0.0],
                [0.2, 1.0, 0.1],
                [0.0, 0.1, 2.0],
            ],
            dtype=np.float64,
        )
        e_ref = float(np.linalg.eigvalsh(h_ord)[0])
        k_eps, energies, converged = find_k_epsilon(
            h_ord, e_ref, epsilon=1e-10, k_start=1, block_size=1
        )
        self.assertTrue(converged)
        self.assertEqual(k_eps, 3)
        self.assertEqual(len(energies), 3)
        self.assertLessEqual(energies[1], energies[0] + 1e-12)
        self.assertLessEqual(energies[2], energies[1] + 1e-12)
        self.assertAlmostEqual(energies[-1], e_ref, places=10)

        curve = one_shot_from_hamiltonian(h_ord, e_exact=e_ref, tol=0.01).as_curve()
        self.assertTrue(curve["converged"])
        self.assertEqual(curve["K"], 2)

    def test_degenerate_coupling_gets_infinite_weight(self):
        self.assertEqual(one_shot_pt_weight(0.1, 0.0), float("inf"))
        self.assertEqual(one_shot_pt_weight(0.0, 0.0), 0.0)

    def test_one_shot_result_exposes_k_pt(self):
        h, _sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        candidates = all_sector_eigenpair_candidates(sector_data)
        result = one_shot_coupled_energy(
            candidates, lambda v: h @ v, e_exact=e_exact, tol=1e-10, tau_pt=1e-12
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.K_pt, 1)
        self.assertGreaterEqual(result.K, 2)
        self.assertAlmostEqual(result.e_coupled, e_exact, places=10)
        self.assertEqual(len(result.chosen_keys), result.K)

    def test_iterative_pt_updates_couplings_to_current_state(self):
        matrix = np.asarray(
            [
                [0.0, 0.10, 0.00],
                [0.10, 0.30, 0.08],
                [0.00, 0.08, 0.35],
            ]
        )
        exact = float(np.linalg.eigvalsh(matrix)[0])

        result = iterative_pt_from_hamiltonian(
            matrix,
            e_exact=exact,
            tol=1.0e-12,
            batch_size=1,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.K, 3)
        self.assertAlmostEqual(result.e_coupled, exact, places=12)
        self.assertEqual(result.order_indices, [0, 1, 2])

    def test_build_candidate_hamiltonian_matches_projection(self):
        h, _sectors, sector_data, _e_exact = _toy_hamiltonian_and_sectors()
        candidates = all_sector_eigenpair_candidates(sector_data)[:3]
        h_c = build_candidate_hamiltonian(candidates, lambda v: h @ v)
        vecs = np.column_stack([c[2] for c in candidates])
        h_proj = vecs.conj().T @ h @ vecs
        h_proj = 0.5 * (h_proj + h_proj.conj().T)
        self.assertTrue(np.allclose(h_c, h_proj, atol=1e-12))
        self.assertAlmostEqual(
            nested_ground_energy(h_c, 3),
            float(np.linalg.eigvalsh(h_proj)[0]),
            places=10,
        )

    def test_decoupled_energy(self):
        h, sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        sector_gs_pairs = {
            k: (d["evals"], np.column_stack(d["evecs_full"]))
            for k, d in sector_data.items()
        }
        e_dec, best_key, _dim = decoupled_energy_test(sectors, sector_gs_pairs)
        self.assertAlmostEqual(e_dec, 0.0, places=10)
        self.assertEqual(best_key, "A")
        self.assertGreater(e_dec, e_exact)

    def test_backward_prune_removes_decoupled_junk(self):
        h = np.zeros((4, 4), dtype=np.float64)
        h[0, 0] = 0.0
        h[3, 3] = 0.1
        h[0, 3] = h[3, 0] = 0.2
        h[1, 1] = 5.0
        h[2, 2] = 6.0
        e_ref = float(np.linalg.eigvalsh(h)[0])
        kept, e_pruned, n_removed = backward_prune_indices(h, 4, e_ref, 1e-10)
        self.assertEqual(n_removed, 2)
        self.assertEqual(kept, [0, 3])
        self.assertAlmostEqual(e_pruned, e_ref, places=10)

    def test_coupled_dimension_backward_prune_reduces_k(self):
        h = np.zeros((4, 4), dtype=np.float64)
        h[0, 0] = 0.0
        h[3, 3] = 0.1
        h[0, 3] = h[3, 0] = 0.2
        h[1, 1] = 5.0
        h[2, 2] = 6.0
        order = [0, 1, 2, 3]
        e_ref = float(np.linalg.eigvalsh(h)[0])
        with_prune = coupled_dimension_from_order(
            h, order, e_exact=e_ref, tol=1e-10, backward_prune=True
        )
        without_prune = coupled_dimension_from_order(
            h, order, e_exact=e_ref, tol=1e-10, backward_prune=False
        )
        self.assertTrue(with_prune.converged)
        self.assertEqual(without_prune.K, 4)
        self.assertEqual(with_prune.K_prefix, 4)
        self.assertEqual(with_prune.K, 2)
        self.assertAlmostEqual(with_prune.e_coupled, e_ref, places=10)


if __name__ == "__main__":
    unittest.main()
