# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers for the TurboQuant microbenchmark."""

# Standard
from typing import NamedTuple
import argparse
import math


class TimingSummary(NamedTuple):
    """Aggregate latency and throughput for one benchmark operation."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    raw_gib_per_s: float


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]

    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_timings(timings_ms: list[float], raw_bytes: int) -> TimingSummary:
    """Summarize positive latency samples and raw-data throughput."""
    if not timings_ms:
        raise ValueError("timings_ms must contain at least one sample")
    if raw_bytes <= 0:
        raise ValueError("raw_bytes must be positive")
    if any(not math.isfinite(value) for value in timings_ms):
        raise ValueError("timing samples must be finite")
    if any(value <= 0.0 for value in timings_ms):
        raise ValueError("timing samples must be positive")

    mean_ms = math.fsum(timings_ms) / len(timings_ms)
    raw_gib = raw_bytes / 1024**3
    return TimingSummary(
        mean_ms=mean_ms,
        p50_ms=percentile(timings_ms, 50.0),
        p95_ms=percentile(timings_ms, 95.0),
        raw_gib_per_s=raw_gib / (mean_ms / 1000.0),
    )


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {parsed}"
        )
    return parsed
