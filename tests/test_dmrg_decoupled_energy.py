"""Fast tests for screened sectors and durable DMRG objective state."""

import json
from unittest.mock import Mock

import numpy as np
import scipy.optimize

from src.dmrg_decoupled_energy import (
    integral_derivatives,
    load_json,
    objective_key,
    optimize_with_dmrg_sector_switching,
    remove_obsolete_mps,
    save_json,
    save_optimizer_state,
    screen_sector_labels,
)
from src.dmrg_solver import rotate_integrals
from src.orbital_rotation import rotation_and_derivatives


def test_screening_uses_weights_and_neighbours_without_full_enumeration():
    solver = Mock()
    solver.get_mps.return_value = object()
    solver.dominant_sector_labels.return_value = [
        ((0, 0, 0, 0, 0, 0, 0), 0.7),
        ((1, 0, 0, 0, 0, 0, 0), 0.2),
    ]
    labels, weights = screen_sector_labels(
        solver,
        np.eye(7, dtype=int),
        minimum=8,
        maximum=16,
    )
    assert len(labels) == 8
    assert len(set(labels)) == 8
    assert weights["0000000"] == 0.7
    solver.dominant_sector_labels.assert_called_once()


def test_objective_key_is_exact_and_stable():
    x = np.asarray([0.0, 0.125, -0.25])
    assert objective_key(x, (0, 1)) == objective_key(x.copy(), (0, 1))
    assert objective_key(x, (0, 1)) != objective_key(x + 1.0e-12, (0, 1))


def test_restart_and_atomic_json_round_trip(tmp_path):
    path = tmp_path / "restart.json"
    save_optimizer_state(
        path,
        np.asarray([0.1, 0.2]),
        (1, 0),
        [{"switch": 0}],
        "optimizing",
        cache_path=tmp_path / "cache.json",
    )
    data = load_json(path, {})
    assert data["rotation"] == [0.1, 0.2]
    assert data["sector"] == [1, 0]
    assert data["status"] == "optimizing"


def test_obsolete_optimizer_mps_files_are_pruned(tmp_path):
    old_tag = "OPT_OLD"
    keep_tag = "OPT_KEEP"
    (tmp_path / f"MPS_INFO.{old_tag}").write_text("old")
    (tmp_path / f"MPS_INFO.{keep_tag}").write_text("keep")
    save_json(
        tmp_path / "metadata.json",
        {"system": {}, "runs": {old_tag: {}, keep_tag: {}}},
    )
    context = {
        "cleanup_mps": True,
        "store_dir": str(tmp_path),
        "cache": {
            "evaluations": {
                "old": {"mps_tag": old_tag},
                "keep": {"mps_tag": keep_tag},
            }
        },
    }
    remove_obsolete_mps(context, [keep_tag])
    assert not (tmp_path / f"MPS_INFO.{old_tag}").exists()
    assert (tmp_path / f"MPS_INFO.{keep_tag}").exists()
    with (tmp_path / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert list(metadata["runs"]) == [keep_tag]


def test_rotated_integral_derivatives_match_central_difference():
    rng = np.random.default_rng(8)
    norb = 4
    h1e = rng.normal(size=(norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    g2e = rng.normal(size=(norb, norb, norb, norb))
    pairs = [(0, 1), (2, 3)]
    x = np.asarray([0.12, -0.07])
    rotation, rotation_derivative_list = rotation_and_derivatives(x, norb, pairs)
    derivatives = integral_derivatives(
        h1e, g2e, rotation, rotation_derivative_list
    )

    epsilon = 1.0e-6
    for parameter, (dh1e, dg2e) in enumerate(derivatives):
        step = np.zeros_like(x)
        step[parameter] = epsilon
        plus_rotation, _ = rotation_and_derivatives(x + step, norb, pairs)
        minus_rotation, _ = rotation_and_derivatives(x - step, norb, pairs)
        plus_h1e, plus_g2e = rotate_integrals(h1e, g2e, plus_rotation)
        minus_h1e, minus_g2e = rotate_integrals(h1e, g2e, minus_rotation)
        np.testing.assert_allclose(
            dh1e,
            (plus_h1e - minus_h1e) / (2.0 * epsilon),
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            dg2e,
            (plus_g2e - minus_g2e) / (2.0 * epsilon),
            atol=2.0e-8,
        )


def test_switching_optimizer_passes_analytic_jacobian(monkeypatch):
    calls = []

    def fake_energy(context, values, label):
        return float(np.dot(values, values))

    def fake_gradient(context, values, label):
        return 2.0 * np.asarray(values)

    def fake_minimize(fun, x0, method, jac, options, callback):
        calls.append(jac)
        value = fun(np.asarray(x0))
        return scipy.optimize.OptimizeResult(
            x=np.asarray(x0),
            fun=value,
            success=True,
            message="test",
            nit=0,
            nfev=1,
            njev=1,
        )

    monkeypatch.setattr(
        "src.dmrg_decoupled_energy.evaluate_fixed_sector", fake_energy
    )
    monkeypatch.setattr(
        "src.dmrg_decoupled_energy.evaluate_fixed_sector_gradient", fake_gradient
    )
    monkeypatch.setattr(
        "src.dmrg_decoupled_energy.scipy.optimize.minimize", fake_minimize
    )

    optimize_with_dmrg_sector_switching(
        {}, [(0,)], np.asarray([0.1]), maxiter=1, max_switches=0,
        initial_label=(0,), use_analytic_gradient=True,
    )
    assert callable(calls[-1])

    optimize_with_dmrg_sector_switching(
        {}, [(0,)], np.asarray([0.1]), maxiter=1, max_switches=0,
        initial_label=(0,), use_analytic_gradient=False,
    )
    assert calls[-1] is None
