# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult

_MB = 1024 * 1024


def _result(durations: list[float], timed_out: int = 0) -> BenchResult:
    return BenchResult(
        operation="store",
        in_flight=1,
        num_keys=64,
        data_size_bytes=_MB,
        round_durations=durations,
        success_counts=[64] * (len(durations) + timed_out),
        timed_out_rounds=timed_out,
    )


def test_timed_out_round_excluded_from_duration_and_throughput() -> None:
    r = _result([0.10, 0.11, 0.09, 0.12], timed_out=1)

    assert len(r.per_round_throughput_mbps) == 4
    assert r.min_throughput_mbps > 0.0
    assert r.max_duration == 0.12
    assert r.std_duration > 0.0


def test_timeout_does_not_change_key_accounting() -> None:
    r = _result([0.10, 0.11, 0.09, 0.12], timed_out=1)

    assert r.attempted_rounds == 5
    assert r.total_keys == 320
    assert r.actual_hit_rate == 1.0


def test_all_rounds_timed_out_reports_zeros() -> None:
    r = _result([], timed_out=3)

    assert r.attempted_rounds == 3
    assert r.avg_duration == 0.0
    assert r.avg_throughput_mbps == 0.0
    assert r.p99_duration == 0.0
