# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`ReplayStatsCollector`."""

# Standard
import json
import threading

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace._stats import (
    OpStats,
    ReplayStatsCollector,
    _percentile,
)


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_p0_returns_min(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_returns_max(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_p50_on_100_values(self):
        vals = [float(i) for i in range(1, 101)]
        # nearest-rank ceil((50/100)*100)=50 → rank 50 → index 49 → value 50
        assert _percentile(vals, 50) == 50.0

    def test_whole_rank_does_not_advance_to_the_next_sample(self):
        vals = [float(i) for i in range(1, 101)]
        # (99/100)*100 lands exactly on rank 99, the case where treating the
        # rank as a 0-based index reports the single worst sample as p99.
        assert _percentile(vals, 99) == 99.0

    def test_fractional_rank_rounds_up(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        # (50/100)*7 = 3.5 → rank 4 → index 3
        assert _percentile(vals, 50) == 40.0

    def test_whole_rank_survives_inexact_division(self):
        vals = [float(i) for i in range(1, 101)]
        # 7/100.0 rounds up, so computing the ratio before multiplying gives
        # 7.000000000000001 and picks rank 8.
        assert _percentile(vals, 7) == 7.0


class TestReplayStatsCollector:
    def test_record_and_summary(self):
        s = ReplayStatsCollector()
        for latency_ms in (1.0, 2.0, 3.0, 4.0, 100.0):
            s.record("op.foo", latency_ms / 1000.0)
        summary = s.summary()
        assert "op.foo" in summary
        stats = summary["op.foo"]
        assert isinstance(stats, OpStats)
        assert stats.count == 5
        assert stats.error_count == 0
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(100.0)
        # 22 = mean of {1,2,3,4,100}
        assert stats.mean_ms == pytest.approx(22.0)

    def test_p90_of_ten_samples_is_the_ninth(self):
        s = ReplayStatsCollector()
        for latency_ms in range(1, 11):
            s.record("op.foo", latency_ms / 1000.0)
        stats = s.summary()["op.foo"]
        assert stats.p90_ms == pytest.approx(9.0)
        assert stats.max_ms == pytest.approx(10.0)

    def test_records_failed_separately(self):
        s = ReplayStatsCollector()
        s.record("op.foo", 0.001, failed=False)
        s.record("op.foo", 0.001, failed=True)
        summary = s.summary()
        assert summary["op.foo"].count == 1
        assert summary["op.foo"].error_count == 1

    def test_error_only_bucket(self):
        """A qualname that only ever failed still appears with zero
        latency stats."""
        s = ReplayStatsCollector()
        s.record("op.failing", 0.0, failed=True)
        summary = s.summary()
        assert summary["op.failing"].count == 0
        assert summary["op.failing"].error_count == 1
        assert summary["op.failing"].mean_ms == 0.0

    def test_duration(self):
        s = ReplayStatsCollector()
        assert s.total_duration_s() == 0.0
        s.mark_start(100.0)
        s.mark_end(105.5)
        assert s.total_duration_s() == pytest.approx(5.5)

    def test_export_csv(self, tmp_path):
        s = ReplayStatsCollector()
        s.record("op.a", 0.001)
        s.record("op.a", 0.002)
        s.record("op.b", 0.003)
        path = str(tmp_path / "out.csv")
        s.export_csv(path)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0].split(",") == [
            "qualname",
            "count",
            "errors",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
        ]
        # two op rows sorted alphabetically
        assert lines[1].startswith("op.a,")
        assert lines[2].startswith("op.b,")

    def test_export_json(self, tmp_path):
        s = ReplayStatsCollector()
        s.mark_start(0.0)
        s.record("op.a", 0.001)
        s.mark_end(1.0)
        path = str(tmp_path / "out.json")
        s.export_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["duration_s"] == pytest.approx(1.0)
        assert "op.a" in data["ops"]
        assert data["ops"]["op.a"]["count"] == 1

    def test_thread_safety(self):
        s = ReplayStatsCollector()

        def worker():
            for _ in range(100):
                s.record("op.x", 0.0001)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert s.summary()["op.x"].count == 400
