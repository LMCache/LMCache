# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks/cache_policy/compare_admission_vs_lfu.py."""

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from benchmarks.cache_policy.compare_admission_vs_lfu import (
    _holm_correct,
    _load_rows,
    compare,
)

_MIB = 2**20
_CAPACITIES = [50 * _MIB, 100 * _MIB, 200 * _MIB]
_SEEDS = [0, 1, 2, 3, 4]


def _row(policy: str, workload: str, capacity: int, seed: int, hit_rate: float) -> dict:
    return {
        "policy_name": policy,
        "workload_name": workload,
        "cache_capacity_bytes": capacity,
        "seed": seed,
        "token_hit_rate": hit_rate,
    }


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "raw.json"
    with open(path, "w") as f:
        json.dump(rows, f)
    return path


def _paired_rows(
    candidate_advantage: float = 0.02,
    workload: str = "mixed_zipfian",
) -> list[dict]:
    """Rows for ADMISSION_LRU/LFU at all three expected capacities, where
    ADMISSION_LRU's hit rate is ``candidate_advantage`` above LFU's for
    every seed (a clean, unambiguous "candidate wins" fixture)."""
    rows = []
    for capacity in _CAPACITIES:
        for seed in _SEEDS:
            base = 0.5 + 0.01 * seed
            rows.append(_row("LFU", workload, capacity, seed, base))
            rows.append(
                _row("ADMISSION_LRU", workload, capacity, seed, base + candidate_advantage)
            )
    return rows


def test_load_rows_pairs_by_shared_seed(tmp_path):
    path = _write_rows(tmp_path, _paired_rows())
    by_capacity = _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")
    assert set(by_capacity.keys()) == set(_CAPACITIES)
    for capacity in _CAPACITIES:
        assert set(by_capacity[capacity]["ADMISSION_LRU"].keys()) == set(_SEEDS)
        assert set(by_capacity[capacity]["LFU"].keys()) == set(_SEEDS)


def test_load_rows_rejects_missing_seed_for_one_policy(tmp_path):
    rows = _paired_rows()
    # Drop LFU's seed=4 row at the 50 MiB capacity -- ADMISSION_LRU still
    # has it, so the seed sets no longer match.
    rows = [
        r
        for r in rows
        if not (
            r["policy_name"] == "LFU"
            and r["cache_capacity_bytes"] == 50 * _MIB
            and r["seed"] == 4
        )
    ]
    path = _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="Seed sets differ"):
        _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")


def test_load_rows_rejects_duplicate_row(tmp_path):
    rows = _paired_rows()
    rows.append(_row("LFU", "mixed_zipfian", 50 * _MIB, 0, 0.5))
    path = _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="Duplicate row"):
        _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")


def test_load_rows_rejects_hit_rate_out_of_range(tmp_path):
    rows = _paired_rows()
    rows.append(_row("LFU", "mixed_zipfian", 50 * _MIB, 99, 1.5))
    path = _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")


def test_load_rows_rejects_too_few_seeds(tmp_path):
    rows = [
        r
        for r in _paired_rows()
        if r["cache_capacity_bytes"] != 50 * _MIB or r["seed"] == 0
    ]
    path = _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="At least 2 seeds"):
        _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")


def test_load_rows_rejects_missing_expected_capacity(tmp_path):
    rows = [r for r in _paired_rows() if r["cache_capacity_bytes"] != 200 * _MIB]
    path = _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="Expected cache capacities"):
        _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")


def test_load_rows_ignores_other_workloads(tmp_path):
    rows = _paired_rows()
    rows += _paired_rows(candidate_advantage=-0.9, workload="repetitive_short")
    path = _write_rows(tmp_path, rows)
    by_capacity = _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")
    for capacity in _CAPACITIES:
        for hit_rate in by_capacity[capacity]["ADMISSION_LRU"].values():
            assert 0.0 <= hit_rate <= 1.0


def test_compare_diff_direction_is_candidate_minus_baseline(tmp_path):
    path = _write_rows(tmp_path, _paired_rows(candidate_advantage=0.02))
    by_capacity = _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")
    comparisons = compare(by_capacity, "ADMISSION_LRU", "LFU", bootstrap_seed=0)
    for row in comparisons:
        assert row["hit_rate_diff_mean"] == pytest.approx(0.02)
        assert row["wins"] == len(_SEEDS)
        assert row["ties"] == 0
        assert row["losses"] == 0


def test_compare_ties_are_not_counted_as_wins(tmp_path):
    path = _write_rows(tmp_path, _paired_rows(candidate_advantage=0.0))
    by_capacity = _load_rows(path, "mixed_zipfian", "ADMISSION_LRU", "LFU")
    comparisons = compare(by_capacity, "ADMISSION_LRU", "LFU", bootstrap_seed=0)
    for row in comparisons:
        assert row["wins"] == 0
        assert row["losses"] == 0
        assert row["ties"] == len(_SEEDS)


def test_compare_row_order_independent_of_input_order(tmp_path):
    rows = _paired_rows(candidate_advantage=0.02)
    forward_path = _write_rows(tmp_path, rows)
    shuffled_path = tmp_path / "shuffled.json"
    with open(shuffled_path, "w") as f:
        json.dump(list(reversed(rows)), f)

    forward = compare(
        _load_rows(forward_path, "mixed_zipfian", "ADMISSION_LRU", "LFU"),
        "ADMISSION_LRU",
        "LFU",
        bootstrap_seed=0,
    )
    shuffled = compare(
        _load_rows(shuffled_path, "mixed_zipfian", "ADMISSION_LRU", "LFU"),
        "ADMISSION_LRU",
        "LFU",
        bootstrap_seed=0,
    )
    assert forward == shuffled


def test_holm_correct_matches_hand_computed_thresholds():
    # Three p-values, ascending: 0.001, 0.02, 0.18.
    # Thresholds in rank order: 0.05/3, 0.05/2, 0.05/1.
    p_values = {50: 0.001, 100: 0.02, 200: 0.18}
    result = _holm_correct(p_values)

    assert result[50]["holm_rank"] == 1
    assert result[50]["holm_threshold"] == pytest.approx(0.05 / 3)
    assert result[50]["holm_reject_at_p05"] is True

    assert result[100]["holm_rank"] == 2
    assert result[100]["holm_threshold"] == pytest.approx(0.05 / 2)
    assert result[100]["holm_reject_at_p05"] is True

    assert result[200]["holm_rank"] == 3
    assert result[200]["holm_threshold"] == pytest.approx(0.05 / 1)
    assert result[200]["holm_reject_at_p05"] is False


def test_holm_correct_stops_rejecting_after_first_failure():
    # First comparison fails Holm at rank 1 -- nothing downstream may be
    # rejected even if its raw p-value would individually pass alpha=0.05.
    p_values = {50: 0.04, 100: 0.03, 200: 0.02}
    result = _holm_correct(p_values)
    # Ascending order: 200 (0.02) rank 1, 100 (0.03) rank 2, 50 (0.04) rank 3.
    assert result[200]["holm_reject_at_p05"] is False
    assert result[100]["holm_reject_at_p05"] is False
    assert result[50]["holm_reject_at_p05"] is False
