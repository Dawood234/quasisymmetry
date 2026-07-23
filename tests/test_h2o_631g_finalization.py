import json

import pytest

from run_h2o_631g_las import final_oo_json
from selected_clifford_lanczos import validate_anchor_energy


def test_final_oo_uses_the_optimization_generator_basis(tmp_path):
    optimize = tmp_path / "optimized.json"
    optimize.write_text(
        json.dumps(
            {
                "parity": "old_parity.txt",
                "symmetry_manifest": "old_manifest.json",
                "screened_sector_labels": [[0, 1]],
            }
        ),
        encoding="utf-8",
    )
    final = tmp_path / "final.json"
    parity = tmp_path / "optimization_parity.txt"
    manifest = tmp_path / "optimization_manifest.json"

    final_oo_json(final, optimize, "molecule.chk", parity, manifest)

    saved = json.loads(final.read_text(encoding="utf-8"))
    assert saved["parity"] == str(parity)
    assert saved["symmetry_manifest"] == str(manifest)
    assert saved["screened_sector_labels"] == [[0, 1]]


def test_anchor_energy_check_rejects_inconsistent_sector():
    data = {"selected_sector": [0, 1], "cost_after": -10.0}

    with pytest.raises(RuntimeError, match="generator basis"):
        validate_anchor_energy(data, (0, 1), -9.9, tolerance=0.01)


def test_anchor_energy_check_accepts_matching_sector():
    data = {"selected_sector": [0, 1], "cost_after": -10.0}

    validate_anchor_energy(data, (0, 1), -9.999, tolerance=0.01)
