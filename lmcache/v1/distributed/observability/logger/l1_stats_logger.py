# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from typing import Any, Deque, Dict, List, Optional
import threading
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.distributed.observability.stats.l1_stats import L1Stats

_stats_lock = threading.Lock()


def stats_safe(func):
    def wrapper(self, *args, **kwargs):
        with _stats_lock:
            return func(self, *args, **kwargs)

    return wrapper


# Latency histogram buckets in seconds, covering sub-millisecond to 10 s range
_LATENCY_BUCKETS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
]


class L1ManagerStatsLogger(L1ManagerListener, PrometheusLogger):
    def __init__(
        self,
        labels: Optional[Dict[str, str]] = None,
        config: Optional[Any] = None,
    ):
        if labels is None:
            labels = {}
        PrometheusLogger.__init__(self, labels=labels, config=config)

        self.stats: L1Stats = L1Stats()
        labelnames: List[str] = list(labels.keys())

        # Prometheus L1-level counters
        self._l1_read_keys_counter = self._create_counter(
            "lmcache_mp:l1_read_keys",
            "Total number of keys reserved for read on L1",
            labelnames,
        )
        self._l1_write_keys_counter = self._create_counter(
            "lmcache_mp:l1_write_keys",
            "Total number of keys reserved for write on L1",
            labelnames,
        )
        self._l1_evicted_keys_counter = self._create_counter(
            "lmcache_mp:l1_evicted_keys",
            "Total number of keys evicted from L1 by the manager",
            labelnames,
        )

        # Prometheus L1-level histograms
        self._l1_read_latency_histogram = self._create_histogram(
            "lmcache_mp:l1_read_latency",
            "L1 cache read latency in seconds",
            labelnames,
            buckets=_LATENCY_BUCKETS,
        )
        self._l1_write_latency_histogram = self._create_histogram(
            "lmcache_mp:l1_write_latency",
            "L1 cache write latency in seconds",
            labelnames,
            buckets=_LATENCY_BUCKETS,
        )

        # Per-batch start timestamps for L1 latency tracking (FIFO, O(1) per call)
        # This is safe only because L1Manager serializes all callbacks under its lock
        self._l1_read_start_times: Deque[float] = deque()
        self._l1_write_start_times: Deque[float] = deque()

    @stats_safe
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        self._l1_read_start_times.append(time.perf_counter())
        self.stats.interval_l1_read_keys += len(keys)

    @stats_safe
    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        if self._l1_read_start_times:
            self.stats.l1_read_latency.append(
                time.perf_counter() - self._l1_read_start_times.popleft()
            )

    @stats_safe
    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        self._l1_write_start_times.append(time.perf_counter())
        self.stats.interval_l1_write_keys += len(keys)

    @stats_safe
    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        if self._l1_write_start_times:
            self.stats.l1_write_latency.append(
                time.perf_counter() - self._l1_write_start_times.popleft()
            )

    @stats_safe
    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        self.stats.interval_l1_evicted_keys += len(keys)

    def log_prometheus(self) -> None:
        """Log accumulated stats to Prometheus and reset internal counters."""
        with _stats_lock:
            stats = self.stats
            self.stats = L1Stats()

        # L1 counters
        self._log_counter(self._l1_read_keys_counter, stats.interval_l1_read_keys)
        self._log_counter(self._l1_write_keys_counter, stats.interval_l1_write_keys)
        self._log_counter(self._l1_evicted_keys_counter, stats.interval_l1_evicted_keys)

        # L1 histograms
        self._log_histogram(self._l1_read_latency_histogram, stats.l1_read_latency)
        self._log_histogram(self._l1_write_latency_histogram, stats.l1_write_latency)
