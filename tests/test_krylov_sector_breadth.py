import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from run_krylov_sector_breadth import (
    final_curve_row,
    parse_counts,
    result_row,
    seed_sector_checkpoints,
    write_summary,
)
from selected_clifford_lanczos import atomic_json


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


def test_atomic_json_allows_concurrent_writers(tmp_path):
    path = tmp_path / "progress.json"

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda value: atomic_json(path, {"value": value}), range(40)))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["value"] in range(40)
    assert list(tmp_path.glob("*.tmp")) == []


def test_seed_sector_checkpoints_reuses_bases_without_overwriting(tmp_path):
    source = tmp_path / "final_selected_clifford_krylov" / "sectors"
    source.mkdir(parents=True)
    (source / "sector_0000001.npz").write_bytes(b"source")

    work_dir = tmp_path / "final_selected_clifford_krylov_s32"
    destination = work_dir / "sectors" / "sector_0000001.npz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    seed_sector_checkpoints(tmp_path, work_dir)

    assert destination.read_bytes() == b"existing"
