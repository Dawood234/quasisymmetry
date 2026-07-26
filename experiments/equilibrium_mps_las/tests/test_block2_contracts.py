"""Cluster-only checks for the Block2 APIs used by the experiment."""

import inspect

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
