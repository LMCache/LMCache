# SPDX-License-Identifier: Apache-2.0
"""A distribution a component keeps in its ``stats()``.

OpenTelemetry has no instrument read at scrape time for a histogram, so a
component counts observations into fixed buckets itself, and
``observability`` exports the counts as Prometheus histogram series:
``_bucket`` (cumulative, labelled ``le``), ``_sum`` and ``_count``.
``histogram_quantile`` works on them as on a native histogram.
"""

# Standard
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

# Bucket bounds shared by histograms of the same kind of value.
CHUNK_COUNT_BOUNDS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DURATION_SECONDS_BOUNDS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
)


@dataclass
class Histogram:
    """Observations counted into buckets with fixed upper bounds.

    Not thread-safe: the owning component guards it with its own lock.

    Attributes:
        bounds: Bucket upper bounds, ascending. An observation goes in the
            first bucket whose bound is at least its value.
        bucket_counts: Observations per bucket, one more than ``bounds``:
            the last counts observations above every bound.
        sum: Total of every observed value.
        count: Number of observations.
    """

    bounds: tuple[float, ...]
    bucket_counts: list[int]
    sum: float = 0.0
    count: int = 0

    @classmethod
    def with_bounds(cls, bounds: Sequence[float]) -> "Histogram":
        """Return an empty histogram with the given bucket upper bounds.

        Args:
            bounds: Bucket upper bounds, ascending.

        Raises:
            ValueError: If ``bounds`` is empty or not strictly ascending.
        """
        if not bounds or any(a >= b for a, b in pairwise(bounds)):
            raise ValueError(f"bounds must be non-empty and ascending: {bounds}")
        return cls(bounds=tuple(bounds), bucket_counts=[0] * (len(bounds) + 1))

    def observe(self, value: float) -> None:
        """Count one observation of ``value``."""
        self.bucket_counts[bisect_left(self.bounds, value)] += 1
        self.sum += value
        self.count += 1

    def copy(self) -> "Histogram":
        """Return an independent snapshot of this histogram."""
        return Histogram(
            bounds=self.bounds,
            bucket_counts=list(self.bucket_counts),
            sum=self.sum,
            count=self.count,
        )

    def cumulative_buckets(self) -> list[tuple[str, int]]:
        """Return ``(le, observations at or below le)`` per bucket.

        Returns:
            One pair per bound, then ``("+Inf", count)``: the Prometheus
            ``_bucket`` series.
        """
        buckets: list[tuple[str, int]] = []
        observations_at_or_below = 0
        # bucket_counts has one more entry, above every bound: not zipped.
        for bound, bucket_count in zip(self.bounds, self.bucket_counts, strict=False):
            observations_at_or_below += bucket_count
            buckets.append((f"{bound:g}", observations_at_or_below))
        buckets.append(("+Inf", self.count))
        return buckets
