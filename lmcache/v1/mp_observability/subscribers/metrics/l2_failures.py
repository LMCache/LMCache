# SPDX-License-Identifier: Apache-2.0

"""L2 failure metrics subscriber — OTel counters for L2 failures.

This covers the health-monitoring surface for L2:

- ``lmcache_mp.l2_prefetch_failure`` (see LM-291) — count of keys that failed to
  load from L2 to L1. Tagged by ``reason``:
    * ``l1_oom``       — L1 had no room to receive the prefetched object.
    * ``l1_contended`` — the key appeared in L1 (write-locked by a concurrent
      request) between the prefetch's L1 claim and its buffer reservation, so
      the load was skipped.
    * ``not_found``    — L2 reported the key present during lookup but the
      load returned no data (adapter-level inconsistency, e.g. concurrent
      delete).
- ``lmcache_mp.l2_store_failure`` — count of chunks whose L1->L2 store task
  failed, so the data never reached L2. Tagged by ``l2_name`` (which backend
  is dropping writes, when several are configured) and ``model_name``.

  Nothing else counts a failed store. ``L2_STORE_COMPLETED`` is published for
  both outcomes, so ``l2_store_completed`` counts *finished* tasks, and
  ``l2_store_completed_objects`` is driven by ``key_count_per_salt``, which only
  the success branch populates.  ``l2_usage_bytes`` measures occupancy rather
  than outcome: it reads 0 for a cold cache as well as a broken one, and it does
  not fall — it can even rise — when a warm cache starts rejecting writes.

The ``serde_failure`` reason is intentionally omitted until the serde PR
lands; once it does, it becomes an additive third value of the same tag
with no breaking change to dashboards.

All emissions carry ``model_name``, derived from the ``ObjectKey``\ s the
event covers.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L2FailureMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L2 prefetch and store failures."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.health")
        self._prefetch_counter = meter.create_counter(
            "lmcache_mp.l2_prefetch_failure",
            description=(
                "Count of keys that were expected in L2 but failed to load "
                "into L1. Tagged by ``reason`` = l1_oom | l1_contended | "
                "not_found and ``model_name``."
            ),
            unit="chunks",
        )
        self._store_counter = meter.create_counter(
            "lmcache_mp.l2_store_failure",
            description=(
                "Count of chunks whose L1->L2 store task failed and which are "
                "therefore absent from L2. Tagged by ``l2_name`` and "
                "``model_name``."
            ),
            unit="chunks",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_PREFETCH_FAILED: self._on_prefetch_failed,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
        }

    def _on_prefetch_failed(self, event: Event) -> None:
        reason: str = event.metadata["reason"]
        keys: list[ObjectKey] = event.metadata["keys"]
        for model_name, count in Counter(k.model_name for k in keys).items():
            self._prefetch_counter.add(
                count,
                {"reason": reason, "model_name": model_name},
            )

    def _on_store_completed(self, event: Event) -> None:
        """Count the chunks of a store task that did not reach L2.

        ``L2_STORE_COMPLETED`` is published once per finished store task
        regardless of its outcome, so a failure is identified here by a
        non-zero ``failed_count`` rather than by the event type.
        """
        failed_count = int(event.metadata.get("failed_count", 0))
        if failed_count <= 0:
            return
        per_model: dict[str, int] = event.metadata.get("failed_count_per_model") or {
            # Emission site predates ``failed_count_per_model``; keep the
            # failure countable without the per-model breakdown.
            "": failed_count
        }
        l2_name = event.metadata.get("l2_name")
        base_attrs = {} if l2_name is None else {"l2_name": str(l2_name)}
        for model_name, count in per_model.items():
            attrs = dict(base_attrs)
            if model_name:
                attrs["model_name"] = model_name
            self._store_counter.add(count, attrs)
