"""H2O validation of the selected-sector DMRG orbital-energy gradient."""

from pathlib import Path

import numpy as np

from src.dmrg_decoupled_energy import (
    evaluate_fixed_sector,
    evaluate_fixed_sector_gradient,
    make_context,
)
from src.dmrg_solver import Block2DMRGSolver


PROJECT_DIR = Path(__file__).resolve().parents[1]
H2O_FCIDUMP = (
    PROJECT_DIR / "hamiltonians" / "water" / "H2O_OH0.9580_104.5000.FCIDUMP"
)


def test_h2o_selected_sector_gradient_matches_central_difference(tmp_path):
    # Four independent seniority/quartet rows from the equilibrium H2O test.
    parity_matrix = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    base_solver = Block2DMRGSolver.from_fcidump(
        H2O_FCIDUMP,
        store_dir=tmp_path / "base",
        n_threads=2,
    )
    context = make_context(
        base_solver,
        parity_matrix,
        pairs=[(0, 1)],
        store_dir=tmp_path / "states",
        cache_path=tmp_path / "cache.json",
        bond_dim=160,
        sweeps=12,
        penalty=30.0,
        n_threads=2,
        energy_tol=1.0e-10,
        davidson_threshold=1.0e-11,
        cleanup_mps=False,
    )

    x = np.asarray([0.02])
    label = (0, 0, 0, 0)
    analytic = evaluate_fixed_sector_gradient(context, x, label)[0]
    epsilon = 2.0e-4
    plus = evaluate_fixed_sector(context, x + epsilon, label)
    minus = evaluate_fixed_sector(context, x - epsilon, label)
    central = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, central, atol=2.0e-6, rtol=2.0e-4)
