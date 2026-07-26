"""Cluster-only checks for the Block2 APIs used by the experiment."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


pyblock2 = pytest.importorskip("pyblock2")

from src.dmrg_solver import Block2DMRGSolver


def test_solver_exposes_required_fitted_mps_operations():
    required = (
        "apply_mpo",
        "mps_overlap",
        "mps_norm2",
        "decoupled_integrals",
        "sector_ground_state",
        "symmetry_expectations",
    )
    for name in required:
        assert callable(getattr(Block2DMRGSolver, name))


def test_apply_mpo_accepts_fitting_controls():
    parameters = inspect.signature(Block2DMRGSolver.apply_mpo).parameters
    for name in ("tag", "bond_dim", "n_sweeps", "tol"):
        assert name in parameters


def test_su2_spatial_parity_uses_spin_scalar_qc_operator():
    density = np.asarray([[0.64, 0.48], [0.48, 0.36]])
    calls = []

    def get_qc_mpo(h1e, g2e, ecore, iprint):
        calls.append((h1e, g2e, ecore, iprint))
        return "su2-parity-mpo"

    solver = SimpleNamespace(
        symmetry_mode="su2",
        driver=SimpleNamespace(get_qc_mpo=get_qc_mpo),
        _activate=lambda: None,
    )
    result = Block2DMRGSolver._spin_parity_factor_mpo(
        solver, density, True, True
    )
    assert result == "su2-parity-mpo"
    h1e, g2e, ecore, iprint = calls[0]
    np.testing.assert_allclose(h1e, -2.0 * density)
    np.testing.assert_allclose(
        g2e, 4.0 * np.einsum("ij,kl->ijkl", density, density)
    )
    assert ecore == 1.0
    assert iprint == 0


def test_su2_rejects_spin_resolved_parity_factor():
    solver = SimpleNamespace(symmetry_mode="su2", _activate=lambda: None)
    with pytest.raises(ValueError, match="spatial occupation parity only"):
        Block2DMRGSolver._spin_parity_factor_mpo(
            solver, np.eye(2), True, False
        )
