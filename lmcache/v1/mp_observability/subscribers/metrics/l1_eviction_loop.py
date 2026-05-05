# SPDX-License-Identifier: Apache-2.0

"""L1 eviction-loop metrics subscriber.

Exposes counters that distinguish "the eviction loop is alive" from
"eviction actually fired" — a subtle distinction that matters when
debugging benchmarks that complete faster than the 1Hz polling rate.

Use case: a workload that writes many chunks in a 50 ms burst will
finish before the eviction loop's next ``time.sleep(1)`` returns, so
no eviction fires during the run even when the pool exceeds the
watermark.  Counters here let dashboards show that immediately, rather
than requiring code inspection or grep on debug logs.

Metrics:
  - ``lmcache_mp.l1_eviction_loop_ticks``      — every loop iteration
  - ``lmcache_mp.l1_eviction_loop_triggered``  — only iterations where
    ``usage >= watermark`` and the policy ran
  - ``lmcache_mp.l1_usage_ratio``              — histogram of
    ``used_bytes / total_bytes`` sampled at every tick

Source event: :data:`EventType.L1_EVICTION_LOOP_TICK`, published by
:class:`L1EvictionController` once per ``eviction_loop`` cycle.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L1EvictionLoopSubscriber(EventSubscriber):
    """Maintains OTel instruments for the L1 eviction loop's lifecycle."""

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.l1")
        self._ticks = meter.create_counter(
            "lmcache_mp.l1_eviction_loop_ticks",
            description=(
                "Number of L1 eviction-loop iterations.  Increments once "
                "per loop cycle (default ~1Hz), regardless of whether "
                "eviction actually ran.  Lets dashboards distinguish a "
                "stalled loop from one that's running but staying below "
                "the watermark."
            ),
        )
        self._triggered = meter.create_counter(
            "lmcache_mp.l1_eviction_loop_triggered",
            description=(
                "Iterations of the L1 eviction loop where ``usage >= "
                "watermark`` and the policy ran.  Compare to "
                "``l1_eviction_loop_ticks`` to see what fraction of "
                "loop iterations actually triggered eviction."
            ),
        )
        self._usage_hist = meter.create_histogram(
            "lmcache_mp.l1_usage_ratio",
            description=(
                "Distribution of L1 ``used_bytes / total_bytes`` sampled "
                "once per eviction-loop tick.  Useful for spotting "
                "fast-fill / slow-drain patterns versus steady-state "
                "operation."
            ),
            unit="ratio",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.L1_EVICTION_LOOP_TICK: self._on_tick}

    def _on_tick(self, event: Event) -> None:
        usage = float(event.metadata.get("usage", 0.0))
        triggered = bool(event.metadata.get("triggered", False))
        self._ticks.add(1)
        if triggered:
            self._triggered.add(1)
        self._usage_hist.record(usage)
