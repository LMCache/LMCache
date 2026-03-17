# SPDX-License-Identifier: Apache-2.0

"""StorageManager metrics subscriber — OTel counters for SM events."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class SMMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for StorageManager operations.

    Metric parity with the old ``StorageManagerStatsLogger``:
    - ``lmcache_mp.sm_read_requests``     — SM read (prefetch) requests
    - ``lmcache_mp.sm_read_succeed_keys`` — keys that were cache hits
    - ``lmcache_mp.sm_read_failed_keys``  — keys that were cache misses
    - ``lmcache_mp.sm_write_requests``     — SM write (reserve) requests
    - ``lmcache_mp.sm_write_succeed_keys`` — keys successfully allocated
    - ``lmcache_mp.sm_write_failed_keys``  — keys that failed allocation
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.sm")
        self._read_requests = meter.create_counter(
            "lmcache_mp.sm_read_requests",
            description="Total StorageManager read (prefetch) requests",
        )
        self._read_succeed = meter.create_counter(
            "lmcache_mp.sm_read_succeed_keys",
            description="Total keys that were cache hits in SM read",
        )
        self._read_failed = meter.create_counter(
            "lmcache_mp.sm_read_failed_keys",
            description="Total keys that were cache misses in SM read",
        )
        self._write_requests = meter.create_counter(
            "lmcache_mp.sm_write_requests",
            description="Total StorageManager write (reserve) requests",
        )
        self._write_succeed = meter.create_counter(
            "lmcache_mp.sm_write_succeed_keys",
            description="Total keys successfully allocated for write in SM",
        )
        self._write_failed = meter.create_counter(
            "lmcache_mp.sm_write_failed_keys",
            description="Total keys that failed allocation for write in SM",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.SM_READ_PREFETCHED: self._on_read_prefetched,
            EventType.SM_WRITE_RESERVED: self._on_write_reserved,
        }

    def _on_read_prefetched(self, event: Event) -> None:
        self._read_requests.add(1)
        self._read_succeed.add(len(event.metadata["succeeded_keys"]))
        self._read_failed.add(len(event.metadata["failed_keys"]))

    def _on_write_reserved(self, event: Event) -> None:
        self._write_requests.add(1)
        self._write_succeed.add(len(event.metadata["succeeded_keys"]))
        self._write_failed.add(len(event.metadata["failed_keys"]))
