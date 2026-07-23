import json

import pytest

from run_krylov_sector_breadth import (
    final_curve_row,
    parse_counts,
    result_row,
    write_summary,
)


def test_parse_counts_sorts_and_removes_duplicates():
    assert parse_counts("32,24,32") == [24, 32]


def test_parse_counts_rejects_anchor_only_count():
    with pytest.raises(ValueError):
        parse_counts("1")


def test_final_curve_row_uses_largest_depth():
    data = {"coupled_curve": [{"depth": 8}, {"depth": 2}, {"depth": 24}]}
    assert final_curve_row(data)["depth"] == 24


def test_result_and_summary_preserve_sector_comparison(tmp_path):
    metrics = {
        "sector_label_count": 24,
        "selected_leakage_fraction": 0.99,
        "coupled_curve": [
            {
                "depth": 24,
                "dimension": 553,
                "energy": -1.2,
                "error_mHa": 1.5,
                "converged": True,
            }
        ],
        "timings": {"total_seconds": 10.0},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    row = result_row(path)
    write_summary(tmp_path, [row])

    assert row["sector_count"] == 24
    assert row["dimension"] == 553
    assert "99.000000%" in (tmp_path / "krylov_sector_breadth.md").read_text()
