# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for dropping warmup attempts from a benchmark result.

``warmup`` counts attempts, while ``round_durations`` only has an entry for
attempts that completed.  Slicing the two by the same offset therefore dropped
a measurement duration whenever a warmup attempt timed out, and discarding the
timeout counter undercounted the measurement attempts.
"""

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.command import _strip_warmup
from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult


def _result(
    durations: list[float],
    flags: list[bool],
    successes: list[int] | None = None,
    expected_hit_count: int = 0,
) -> BenchResult:
    """Build a lookup result; ``flags`` has one entry per attempted round."""
    return BenchResult(
        operation="Lookup",
        in_flight=1,
        num_keys=1,
        data_size_bytes=0,
        round_durations=durations,
        success_counts=([1] * len(flags) if successes is None else successes),
        timed_out_flags=flags,
        expected_hit_count=expected_hit_count,
    )


def test_timed_out_warmup_does_not_drop_a_measurement_duration() -> None:
    """A warmup timeout adds no duration, so no measurement duration is lost."""
    raw = _result(durations=[0.10, 0.20], flags=[True, False, False])

    result = _strip_warmup(raw, warmup=1)

    assert result.attempted_rounds == 2
    assert result.round_durations == [0.10, 0.20]
    assert result.total_success == 2
    assert result.total_keys == 2
    assert result.actual_hit_rate == 1.0
    assert result.timed_out_rounds == 0


def test_completed_warmup_drops_one_duration() -> None:
    """A completed warmup attempt owns exactly one duration, which is dropped."""
    raw = _result(durations=[0.10, 0.20, 0.30], flags=[False, False, False])

    result = _strip_warmup(raw, warmup=1)

    assert result.round_durations == [0.20, 0.30]
    assert result.attempted_rounds == 2


def test_measurement_timeout_is_preserved() -> None:
    """Timeouts that happen after warmup stay counted as measurement attempts."""
    raw = _result(
        durations=[0.10, 0.30],
        flags=[False, True, False],
        successes=[1, 0, 1],
    )

    result = _strip_warmup(raw, warmup=1)

    assert result.timed_out_rounds == 1
    assert result.attempted_rounds == 2
    assert result.total_keys == 2
    assert result.total_success == 1
    assert result.round_durations == [0.30]


def test_mixed_warmup_drops_only_completed_durations() -> None:
    """Two warmup attempts, one of which timed out, own only one duration."""
    raw = _result(
        durations=[0.10, 0.20, 0.30],
        flags=[True, False, False, False],
    )

    result = _strip_warmup(raw, warmup=2)

    assert result.attempted_rounds == 2
    assert result.round_durations == [0.20, 0.30]


def test_warmup_larger_than_attempts_keeps_nothing() -> None:
    """Over-specified warmup must not produce negative slices."""
    raw = _result(durations=[0.10], flags=[False, True])

    result = _strip_warmup(raw, warmup=5)

    assert result.attempted_rounds == 0
    assert result.round_durations == []
    assert result.success_counts == []
    assert result.timed_out_rounds == 0
    assert result.total_keys == 0


def test_expected_hit_count_scales_with_kept_attempts() -> None:
    """The expected hit count is prorated over the attempts that remain."""
    raw = _result(
        durations=[0.10, 0.20, 0.30],
        flags=[False, False, False, False],
        expected_hit_count=4,
    )

    result = _strip_warmup(raw, warmup=1)

    assert result.expected_hit_count == 3
