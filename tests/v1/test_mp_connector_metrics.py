# SPDX-License-Identifier: Apache-2.0
"""Tests for LMCache MP connector metrics."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector metrics require vLLM")

# Third Party
from prometheus_client import Counter, Gauge, Histogram  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
    LMCacheMPPromMetrics,
)


class _BoundHistogram:
    def __init__(self) -> None:
        self.observations: list[float] = []

    def observe(self, value: float) -> None:
        self.observations.append(value)


class _Histogram:
    instance: "_Histogram"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.children: dict[tuple[object, ...], _BoundHistogram] = {}
        _Histogram.instance = self

    def labels(self, *values: object) -> _BoundHistogram:
        return self.children.setdefault(values, _BoundHistogram())


def test_lookup_stats_snapshot_and_aggregation() -> None:
    stats = LMCacheMPConnectorStats()
    stats.record_lookup(0.025)

    snapshot = stats.clone_and_reset()
    aggregate = LMCacheMPConnectorStats()
    aggregate.aggregate(snapshot)

    assert stats.is_empty()
    assert aggregate.reduce() == {
        "LMCache MP lookup count": 1,
        "LMCache MP lookup avg latency (ms)": 25.0,
    }


def test_prometheus_observes_lookup_duration() -> None:
    metric_types: dict[Any, Any] = {
        Counter: _Histogram,
        Gauge: _Histogram,
        Histogram: _Histogram,
    }
    metrics = LMCacheMPPromMetrics(
        SimpleNamespace(kv_transfer_config=object()),
        metric_types,
        ["model_name", "engine"],
        {0: ["model", "0"]},
    )

    metrics.observe({"lookup_duration_seconds": [0.01, 0.02]})

    assert _Histogram.instance.kwargs["name"] == (
        "vllm:lmcache_mp_lookup_duration_seconds"
    )
    assert _Histogram.instance.children[("model", "0")].observations == [0.01, 0.02]
