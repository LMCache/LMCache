# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import StorageManagerListener
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.mp_observability.stats.storage_manager_stats import (
    StorageManagerStats,
)

_stats_lock = threading.Lock()


def stats_safe(func):
    def wrapper(self, *args, **kwargs):
        with _stats_lock:
            return func(self, *args, **kwargs)

    return wrapper


class StorageManagerStatsLogger(StorageManagerListener, PrometheusLogger):
    def __init__(
        self,
        labels: Optional[Dict[str, str]] = None,
        config: Optional[Any] = None,
    ):
        if labels is None:
            labels = {}
        PrometheusLogger.__init__(self, labels=labels, config=config)

        self.stats: StorageManagerStats = StorageManagerStats()
        labelnames: List[str] = list(labels.keys())

        # Prometheus StorageManager-level counters
        self._sm_read_requests_counter = self.create_counter(
            "lmcache_mp:sm_read_requests",
            "Total number of StorageManager read (prefetch) requests",
            labelnames,
        )
        self._sm_read_succeed_keys_counter = self.create_counter(
            "lmcache_mp:sm_read_succeed_keys",
            "Total number of keys that were succeed in reading from LMCache",
            labelnames,
        )
        self._sm_read_failed_keys_counter = self.create_counter(
            "lmcache_mp:sm_read_failed_keys",
            "Total number of keys that were cache failed in reading LMCache",
            labelnames,
        )
        self._sm_write_requests_counter = self.create_counter(
            "lmcache_mp:sm_write_requests",
            "Total number of StorageManager write (reserve) requests",
            labelnames,
        )
        self._sm_write_succeed_keys_counter = self.create_counter(
            "lmcache_mp:sm_write_succeed_keys",
            "Total number of keys successfully allocated for write in SM",
            labelnames,
        )
        self._sm_write_failed_keys_counter = self.create_counter(
            "lmcache_mp:sm_write_failed_keys",
            "Total number of keys that failed allocation for write in SM",
            labelnames,
        )

    @stats_safe
    def on_sm_read_prefetched(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        self.stats.interval_sm_read_requests += 1
        self.stats.interval_sm_read_succeed_keys += len(succeeded_keys)
        self.stats.interval_sm_read_failed_keys += len(failed_keys)

    @stats_safe
    def on_sm_read_prefetched_finished(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        pass

    @stats_safe
    def on_sm_reserved_write(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        self.stats.interval_sm_write_requests += 1
        self.stats.interval_sm_write_succeed_keys += len(succeeded_keys)
        self.stats.interval_sm_write_failed_keys += len(failed_keys)

    @stats_safe
    def on_sm_write_finished(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        pass

    def log_prometheus(self) -> None:
        """Log accumulated stats to Prometheus and reset internal counters."""
        with _stats_lock:
            stats = self.stats
            self.stats = StorageManagerStats()

        self.log_counter(
            self._sm_read_requests_counter, stats.interval_sm_read_requests
        )
        self.log_counter(
            self._sm_read_succeed_keys_counter, stats.interval_sm_read_succeed_keys
        )
        self.log_counter(
            self._sm_read_failed_keys_counter, stats.interval_sm_read_failed_keys
        )
        self.log_counter(
            self._sm_write_requests_counter, stats.interval_sm_write_requests
        )
        self.log_counter(
            self._sm_write_succeed_keys_counter, stats.interval_sm_write_succeed_keys
        )
        self.log_counter(
            self._sm_write_failed_keys_counter, stats.interval_sm_write_failed_keys
        )
