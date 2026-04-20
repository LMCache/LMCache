# SPDX-License-Identifier: Apache-2.0

"""StorageManager metrics subscriber — OTel counters for SM events."""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class SMMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for StorageManager operations.

    Metrics are tagged with ``cache_salt`` so operators can attribute SM
    traffic to individual tenants. Request counters are incremented once
    per distinct ``cache_salt`` touched by the call.

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
        succeeded = event.metadata.get("succeeded_keys", [])
        failed = event.metadata.get("failed_keys", [])
        self._emit_request(self._read_requests, succeeded, failed)
        self._emit_by_salt(self._read_succeed, succeeded)
        self._emit_by_salt(self._read_failed, failed)

    def _on_write_reserved(self, event: Event) -> None:
        succeeded = event.metadata.get("succeeded_keys", [])
        failed = event.metadata.get("failed_keys", [])
        self._emit_request(self._write_requests, succeeded, failed)
        self._emit_by_salt(self._write_succeed, succeeded)
        self._emit_by_salt(self._write_failed, failed)

    @staticmethod
    def _emit_by_salt(counter: metrics.Counter, keys: list) -> None:
        """Group ``keys`` by ``cache_salt`` and emit one ``add`` per group."""
        if not keys:
            return
        groups = Counter(getattr(k, "cache_salt", "") for k in keys)
        for salt, count in groups.items():
            counter.add(count, {"cache_salt": salt})

    @staticmethod
    def _emit_request(counter: metrics.Counter, *key_lists: list) -> None:
        """Emit one request increment per distinct ``cache_salt`` touched.

        When no keys are present (should not happen for a real SM call),
        record the request against an empty-salt tenant so the series
        still reflects that the call occurred.
        """
        salts: set[str] = set()
        for keys in key_lists:
            for k in keys:
                salts.add(getattr(k, "cache_salt", ""))
        if not salts:
            counter.add(1, {"cache_salt": ""})
            return
        for salt in salts:
            counter.add(1, {"cache_salt": salt})
