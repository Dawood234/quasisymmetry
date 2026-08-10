#!/usr/bin/env python3
"""Small tests that do not require Block2 or molecular checkpoints."""

import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path[:0] = [str(HERE), str(PROJECT)]

from exact_quotient import (  # noqa: E402
    exact_parity_rows,
    exact_sector_dimension_audit,
    select_quotient_independent,
    selection_diagnostics,
)


class ExactQuotientTests(unittest.TestCase):
    def test_selection_adds_requested_effective_rank(self):
        irreps = [0, 1, 2, 3, 0]
        exact = exact_parity_rows(irreps)
        candidates = []
        for orbital in range(5):
            row = np.zeros(5, dtype=int)
            row[orbital] = 1
            candidates.append(
                {
                    "label": f"S({orbital + 1})",
                    "family": "seniority",
                    "support": [orbital + 1],
                    "row": row.tolist(),
                    "score": float(orbital),
                }
            )
        selected = select_quotient_independent(candidates, 2, exact, 5)
        diagnostics = selection_diagnostics(selected, exact, 5)
        self.assertEqual(diagnostics["effective_las_rank"], 2)

    def test_dimension_audit_matches_brute_force(self):
        norb = 5
        irreps = [0, 1, 2, 3, 0]
        rows = np.asarray([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=int)
        label = (0, 0)
        audit = exact_sector_dimension_audit(2, 2, irreps, 0, rows, label)

        brute = 0
        for alpha in combinations(range(norb), 2):
            for beta in combinations(range(norb), 2):
                occupation = np.zeros(norb, dtype=np.uint8)
                occupation[list(alpha)] ^= 1
                occupation[list(beta)] ^= 1
                irrep = 0
                for orbital in np.flatnonzero(occupation):
                    irrep ^= irreps[orbital]
                if irrep != 0:
                    continue
                sector = tuple(int(row @ occupation) & 1 for row in rows)
                if sector == label:
                    brute += 1
        self.assertEqual(audit["selected_determinant_dimension"], brute)
        self.assertEqual(
            audit["target_exact_dimension"],
            sum(audit["sector_dimensions"].values()),
        )


if __name__ == "__main__":
    unittest.main()
