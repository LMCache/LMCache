# SPDX-License-Identifier: Apache-2.0

"""L2 storage metrics subscriber — OTel counters for L2 store/prefetch events."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L2MetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L2 store and prefetch operations.

    Metrics:
    - ``lmcache_mp.l2_store_tasks``         — store tasks submitted to L2
    - ``lmcache_mp.l2_store_keys``          — keys submitted for L2 store
    - ``lmcache_mp.l2_store_completed``     — store tasks completed
    - ``lmcache_mp.l2_store_succeeded_keys`` — keys successfully stored to L2
    - ``lmcache_mp.l2_store_failed_keys``   — keys that failed to store to L2
    - ``lmcache_mp.l2_prefetch_lookups``    — prefetch lookup requests
    - ``lmcache_mp.l2_prefetch_lookup_keys`` — keys submitted for lookup
    - ``lmcache_mp.l2_prefetch_hit_keys``   — prefix keys found in L2
    - ``lmcache_mp.l2_prefetch_load_tasks`` — load tasks submitted
    - ``lmcache_mp.l2_prefetch_load_keys``  — keys submitted for load
    - ``lmcache_mp.l2_prefetch_loaded_keys`` — keys successfully loaded from L2
    - ``lmcache_mp.l2_prefetch_failed_keys`` — keys that failed to load
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.l2")

        # Store counters
        self._store_tasks = meter.create_counter(
            "lmcache_mp.l2_store_tasks",
            description="Total L2 store tasks submitted",
        )
        self._store_keys = meter.create_counter(
            "lmcache_mp.l2_store_keys",
            description="Total keys submitted for L2 store",
        )
        self._store_completed = meter.create_counter(
            "lmcache_mp.l2_store_completed",
            description="Total L2 store tasks completed",
        )
        self._store_succeeded_keys = meter.create_counter(
            "lmcache_mp.l2_store_succeeded_keys",
            description="Total keys successfully stored to L2",
        )
        self._store_failed_keys = meter.create_counter(
            "lmcache_mp.l2_store_failed_keys",
            description="Total keys that failed to store to L2",
        )

        # Prefetch counters
        self._prefetch_lookups = meter.create_counter(
            "lmcache_mp.l2_prefetch_lookups",
            description="Total L2 prefetch lookup requests",
        )
        self._prefetch_lookup_keys = meter.create_counter(
            "lmcache_mp.l2_prefetch_lookup_keys",
            description="Total keys submitted for L2 prefetch lookup",
        )
        self._prefetch_hit_keys = meter.create_counter(
            "lmcache_mp.l2_prefetch_hit_keys",
            description="Total prefix keys found in L2 lookup",
        )
        self._prefetch_load_tasks = meter.create_counter(
            "lmcache_mp.l2_prefetch_load_tasks",
            description="Total L2 prefetch load tasks submitted",
        )
        self._prefetch_load_keys = meter.create_counter(
            "lmcache_mp.l2_prefetch_load_keys",
            description="Total keys submitted for L2 load",
        )
        self._prefetch_loaded_keys = meter.create_counter(
            "lmcache_mp.l2_prefetch_loaded_keys",
            description="Total keys successfully loaded from L2",
        )
        self._prefetch_failed_keys = meter.create_counter(
            "lmcache_mp.l2_prefetch_failed_keys",
            description="Total keys that failed to load from L2",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED: self._on_lookup_submitted,
            EventType.L2_PREFETCH_LOOKUP_COMPLETED: self._on_lookup_completed,
            EventType.L2_PREFETCH_LOAD_SUBMITTED: self._on_load_submitted,
            EventType.L2_PREFETCH_LOAD_COMPLETED: self._on_load_completed,
        }

    def _on_store_submitted(self, event: Event) -> None:
        self._store_tasks.add(1)
        self._store_keys.add(event.metadata["key_count"])

    def _on_store_completed(self, event: Event) -> None:
        self._store_completed.add(1)
        self._store_succeeded_keys.add(event.metadata["succeeded_count"])
        self._store_failed_keys.add(event.metadata["failed_count"])

    def _on_lookup_submitted(self, event: Event) -> None:
        self._prefetch_lookups.add(1)
        self._prefetch_lookup_keys.add(event.metadata["key_count"])

    def _on_lookup_completed(self, event: Event) -> None:
        self._prefetch_hit_keys.add(event.metadata["prefix_hit_count"])

    def _on_load_submitted(self, event: Event) -> None:
        self._prefetch_load_tasks.add(event.metadata["adapter_count"])
        self._prefetch_load_keys.add(event.metadata["key_count"])

    def _on_load_completed(self, event: Event) -> None:
        self._prefetch_loaded_keys.add(event.metadata["loaded_count"])
        self._prefetch_failed_keys.add(event.metadata["failed_count"])
