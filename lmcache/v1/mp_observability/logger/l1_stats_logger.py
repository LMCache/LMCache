# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.mp_observability.stats.l1_stats import L1Stats

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
        self._l1_read_keys_counter = self.create_counter(
            "lmcache_mp:l1_read_keys",
            "Total number of keys finished for read on L1",
            labelnames,
        )
        self._l1_write_keys_counter = self.create_counter(
            "lmcache_mp:l1_write_keys",
            "Total number of keys finished for write on L1",
            labelnames,
        )
        self._l1_evicted_keys_counter = self.create_counter(
            "lmcache_mp:l1_evicted_keys",
            "Total number of keys evicted from L1 by the manager",
            labelnames,
        )

    @stats_safe
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        # No ops. Record read count once it is actually finished.
        pass

    @stats_safe
    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        self.stats.interval_l1_read_keys += len(keys)

    @stats_safe
    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        # No ops. Record write counts once it is actually finished.
        pass

    @stats_safe
    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        self.stats.interval_l1_write_keys += len(keys)

    @stats_safe
    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        self.stats.interval_l1_evicted_keys += len(keys)

    @stats_safe
    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        self.stats.interval_l1_write_keys += len(keys)

    def log_prometheus(self) -> None:
        """Log accumulated stats to Prometheus and reset internal counters."""
        with _stats_lock:
            stats = self.stats
            self.stats = L1Stats()

        # L1 counters
        self.log_counter(self._l1_read_keys_counter, stats.interval_l1_read_keys)
        self.log_counter(self._l1_write_keys_counter, stats.interval_l1_write_keys)
        self.log_counter(self._l1_evicted_keys_counter, stats.interval_l1_evicted_keys)
