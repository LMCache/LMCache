# SPDX-License-Identifier: Apache-2.0
"""vLLM metrics for the LMCache MP connector."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

_LOOKUP_DURATION = "lookup_duration_seconds"


@dataclass
class LMCacheMPConnectorStats(KVConnectorStats):
    """Lookup latency observations collected by the scheduler connector."""

    def __post_init__(self) -> None:
        self.data.setdefault(_LOOKUP_DURATION, [])

    def record_lookup(self, duration_seconds: float) -> None:
        self.data[_LOOKUP_DURATION].append(duration_seconds)

    def clone_and_reset(self) -> "LMCacheMPConnectorStats":
        snapshot = LMCacheMPConnectorStats(
            data={_LOOKUP_DURATION: self.data[_LOOKUP_DURATION]}
        )
        self.reset()
        return snapshot

    def reset(self) -> None:
        self.data: dict[str, Any] = {_LOOKUP_DURATION: []}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not isinstance(other, LMCacheMPConnectorStats):
            raise TypeError(f"Cannot aggregate {type(other)}")
        self.data[_LOOKUP_DURATION].extend(other.data[_LOOKUP_DURATION])
        return self

    def reduce(self) -> dict[str, int | float]:
        durations = self.data[_LOOKUP_DURATION]
        if not durations:
            return {}
        return {
            "LMCache MP lookup count": len(durations),
            "LMCache MP lookup avg latency (ms)": round(
                sum(durations) / len(durations) * 1000, 3
            ),
        }

    def is_empty(self) -> bool:
        return not self.data[_LOOKUP_DURATION]


class LMCacheMPPromMetrics(KVConnectorPromMetrics):
    """Expose LMCache MP lookup latency through vLLM Prometheus metrics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        histogram = self._histogram_cls(
            name="vllm:lmcache_mp_lookup_duration_seconds",
            documentation="LMCache MP lookup latency observed by vLLM.",
            labelnames=labelnames,
        )
        self._lookup_duration = {
            engine_idx: histogram.labels(*labelvalues)
            for engine_idx, labelvalues in per_engine_labelvalues.items()
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for duration in transfer_stats_data.get(_LOOKUP_DURATION, []):
            self._lookup_duration[engine_idx].observe(duration)
