"""Tests for the staged H2O/6-31G parent-reference driver."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_h2o_631g_parent_reference import (
    reverse_extrapolation,
    solve_command,
    stage_noises,
)


def test_stage_noise_finishes_with_clean_sweeps():
    assert stage_noises(0, 6) == [1.0e-4, 1.0e-5, 0.0, 0.0, 0.0, 0.0]
    assert stage_noises(2, 4) == [1.0e-5, 1.0e-6, 0.0, 0.0]
    assert stage_noises(1, 2) == [0.0, 0.0]


def test_reverse_extrapolation_recovers_zero_weight_intercept():
    intercept = -76.123456
    slope = 3.25
    history = []
    for sweep, (bond_dim, weight) in enumerate(
        [(500, 1.0e-5), (500, 8.0e-6), (350, 3.0e-5), (250, 8.0e-5)]
    ):
        history.append(
            {
                "sweep": sweep,
                "bond_dim": bond_dim,
                "discarded_weight": weight,
                "energies": [intercept + slope * weight],
            }
        )

    result = reverse_extrapolation({"sweep_history": history})

    assert result["available"]
    assert np.isclose(result["intercept_Ha"], intercept, atol=1.0e-12)
    assert len(result["points"]) == 3


def test_solve_command_avoids_restart_copy_when_store_is_persistent(tmp_path):
    args = SimpleNamespace(
        python="python3",
        davidson_threshold=1.0e-10,
        symmetry_mode="su2",
        n_threads=8,
        n_mkl_threads=1,
        stack_mem_gb=2.0,
        ordering="fiedler",
    )
    store = tmp_path / "mps"
    command = solve_command(
        args,
        Path("input.chk"),
        store,
        store,
        Path("result.txt"),
        Path("result.json"),
        "M100",
        [100, 100],
        [1.0e-5, 0.0],
    )

    assert "--restart_dir" not in command
    assert command[command.index("--symmetry_mode") + 1] == "su2"
    assert command[command.index("--reorder") + 1] == "fiedler"
