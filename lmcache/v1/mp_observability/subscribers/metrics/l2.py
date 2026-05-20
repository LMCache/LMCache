# SPDX-License-Identifier: Apache-2.0

"""L2 storage metrics subscriber — OTel counters for L2 store/prefetch events."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


def _l2_name_attrs(event: Event) -> dict[str, Any]:
    """Build ``{"l2_name": ...}`` if the event carries an ``l2_name``
    metadata key, else ``{}``.  Keeps the counter dimensionless when
    the emission site hasn't been updated to carry the label yet."""
    l2_name = event.metadata.get("l2_name")
    if l2_name is None:
        return {}
    return {"l2_name": str(l2_name)}


class L2MetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L2 store and prefetch operations.

    Metrics:
    - ``lmcache_mp.l2_store_submitted`` — store requests submitted to L2
    - ``lmcache_mp.l2_store_submitted_objects`` — chunks submitted for L2 store
    - ``lmcache_mp.l2_store_completed`` — store requests completed (attr: ``l2_name``)
    - ``lmcache_mp.l2_store_completed_objects`` — chunks successfully stored to L2
    - ``lmcache_mp.l2_load_completed`` — per-adapter load tasks completed
      (attr: ``l2_name``)
    - ``lmcache_mp.l2_prefetch_lookup`` — prefetch lookup requests
    - ``lmcache_mp.l2_prefetch_lookup_objects`` — chunks submitted for lookup
    - ``lmcache_mp.l2_prefetch_hit`` — prefix chunks found in L2
    - ``lmcache_mp.l2_prefetch_load_submitted`` — load tasks submitted
    - ``lmcache_mp.l2_prefetch_load_submitted_objects`` — chunks submitted for load
    - ``lmcache_mp.l2_prefetch_load_completed`` — chunks successfully loaded from L2

    The ``l2_name``-labeled counters (``l2_store_completed``, ``l2_load_completed``)
    let dashboards compute per-backend IOPS via
    ``rate(<counter>_total{l2_name="..."}[1m])``.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.l2")

        # Store counters
        self._store_submitted = meter.create_counter(
            "lmcache_mp.l2_store_submitted",
            description="Total L2 store requests submitted",
            unit="requests",
        )
        self._store_submitted_objects = meter.create_counter(
            "lmcache_mp.l2_store_submitted_objects",
            description="Total chunks submitted for L2 store",
            unit="chunks",
        )
        self._store_completed = meter.create_counter(
            "lmcache_mp.l2_store_completed",
            description="Total L2 store requests completed",
            unit="requests",
        )
        self._store_completed_objects = meter.create_counter(
            "lmcache_mp.l2_store_completed_objects",
            description="Total chunks successfully stored to L2",
            unit="chunks",
        )

        # Per-adapter load task counter (for IOPS via rate()).
        # Labeled by ``l2_name`` so dashboards can slice per backend.
        self._load_completed = meter.create_counter(
            "lmcache_mp.l2_load_completed",
            description="Total L2 load tasks completed (per-adapter)",
            unit="requests",
        )

        # Prefetch lookup counters
        self._prefetch_lookup_submitted = meter.create_counter(
            "lmcache_mp.l2_prefetch_lookup",
            description="Total L2 prefetch lookup requests submitted",
            unit="requests",
        )
        self._prefetch_lookup_submitted_objects = meter.create_counter(
            "lmcache_mp.l2_prefetch_lookup_objects",
            description="Total chunks submitted for L2 prefetch lookup",
            unit="chunks",
        )
        self._prefetch_lookup_hit = meter.create_counter(
            "lmcache_mp.l2_prefetch_hit",
            description="Total prefix chunks found in L2 lookup",
            unit="chunks",
        )

        # Prefetch load counters
        self._prefetch_load_submitted = meter.create_counter(
            "lmcache_mp.l2_prefetch_load_submitted",
            description="Total L2 prefetch load requests submitted (per-adapter)",
            unit="requests",
        )
        self._prefetch_load_submitted_objects = meter.create_counter(
            "lmcache_mp.l2_prefetch_load_submitted_objects",
            description="Total chunks submitted for L2 load",
            unit="chunks",
        )
        self._prefetch_load_completed = meter.create_counter(
            "lmcache_mp.l2_prefetch_load_completed",
            description="Total chunks successfully loaded from L2",
            unit="chunks",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_LOAD_TASK_COMPLETED: self._on_load_task_completed,
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED: self._on_lookup_submitted,
            EventType.L2_PREFETCH_LOOKUP_COMPLETED: self._on_lookup_completed,
            EventType.L2_PREFETCH_LOAD_SUBMITTED: self._on_load_submitted,
            EventType.L2_PREFETCH_LOAD_COMPLETED: self._on_load_completed,
        }

    def _on_store_submitted(self, event: Event) -> None:
        self._store_submitted.add(1)
        self._store_submitted_objects.add(event.metadata["key_count"])

    def _on_store_completed(self, event: Event) -> None:
        attrs = _l2_name_attrs(event)
        self._store_completed.add(1, attributes=attrs)
        self._store_completed_objects.add(event.metadata["succeeded_count"])

    def _on_load_task_completed(self, event: Event) -> None:
        self._load_completed.add(1, attributes=_l2_name_attrs(event))

    def _on_lookup_submitted(self, event: Event) -> None:
        self._prefetch_lookup_submitted.add(1)
        self._prefetch_lookup_submitted_objects.add(event.metadata["key_count"])

    def _on_lookup_completed(self, event: Event) -> None:
        self._prefetch_lookup_hit.add(event.metadata["prefix_hit_count"])

    def _on_load_submitted(self, event: Event) -> None:
        self._prefetch_load_submitted.add(event.metadata["adapter_count"])
        self._prefetch_load_submitted_objects.add(event.metadata["key_count"])

    def _on_load_completed(self, event: Event) -> None:
        self._prefetch_load_completed.add(event.metadata["loaded_count"])
