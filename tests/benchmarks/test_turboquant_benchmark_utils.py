# SPDX-License-Identifier: Apache-2.0
"""Tests for the TurboQuant benchmark reporting helpers."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import math

# Third Party
import pytest

_UTILS_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "serde"
    / "turboquant"
    / "benchmark_utils.py"
)
_SPEC = spec_from_file_location("turboquant_benchmark_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [(0.0, 1.0), (50.0, 2.5), (95.0, 3.85), (100.0, 4.0)],
)
def test_percentile_uses_linear_interpolation(
    percentile: float, expected: float
) -> None:
    assert _UTILS.percentile([4.0, 1.0, 3.0, 2.0], percentile) == pytest.approx(
        expected
    )


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _UTILS.percentile([], 50.0)

    with pytest.raises(ValueError, match="between 0 and 100"):
        _UTILS.percentile([1.0], 101.0)


def test_summarize_timings_reports_latency_and_raw_throughput() -> None:
    summary = _UTILS.summarize_timings(
        timings_ms=[4.0, 1.0, 3.0, 2.0],
        raw_bytes=1024**3,
    )

    assert summary.mean_ms == pytest.approx(2.5)
    assert summary.p50_ms == pytest.approx(2.5)
    assert summary.p95_ms == pytest.approx(3.85)
    assert summary.raw_gib_per_s == pytest.approx(400.0)


@pytest.mark.parametrize(
    ("timings_ms", "raw_bytes", "match"),
    [
        ([], 1024, "at least one"),
        ([0.0], 1024, "positive"),
        ([math.nan], 1024, "finite"),
        ([1.0], 0, "raw_bytes"),
    ],
)
def test_summarize_timings_rejects_invalid_measurements(
    timings_ms: list[float], raw_bytes: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _UTILS.summarize_timings(timings_ms, raw_bytes)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_positive_int_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _UTILS.positive_int(value)


def test_non_negative_int_accepts_zero_and_rejects_negative() -> None:
    assert _UTILS.non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        _UTILS.non_negative_int("-1")
