# SPDX-License-Identifier: Apache-2.0

"""Lookup metrics subscriber — OTel counters for L1+L2 token-level hit rate.

Exposes two counters driven by the ``MP_LOOKUP_PREFETCH_END`` event.  Their
ratio is the fraction of tokens requested by a lookup that were served from
the L1 or L2 caches (L0/GPU prefix cache is vLLM-owned and not observable
here):

    rate(lmcache_mp_lookup_hit_tokens_total[5m])
    / rate(lmcache_mp_lookup_requested_tokens_total[5m])

See ``docs/design/v1/mp_observability/L1_L2_HIT_RATE_PLAN.md`` for the full
rationale behind co-locating numerator and denominator on a single event.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class LookupMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for L1+L2 token-level cache hit rate.

    Metrics:
    - ``lmcache_mp.lookup_requested_tokens`` — tokens submitted for lookup
      (denominator).  Counts only the chunk-aligned portion; sub-chunk
      trailing tokens are excluded because they cannot hit by design.
    - ``lmcache_mp.lookup_hit_tokens`` — tokens found in L1+L2 during the
      lookup (numerator).  Counts the contiguous prefix hit only.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.lookup")

        self._requested_tokens = meter.create_counter(
            "lmcache_mp.lookup_requested_tokens",
            description=(
                "Total tokens submitted for lookup (denominator of the "
                "L1+L2 token-level hit rate). Only chunk-aligned tokens "
                "are counted."
            ),
            unit="tokens",
        )
        self._hit_tokens = meter.create_counter(
            "lmcache_mp.lookup_hit_tokens",
            description=(
                "Total tokens found in L1+L2 during lookup (numerator of "
                "the L1+L2 token-level hit rate). Counts the contiguous "
                "prefix hit only."
            ),
            unit="tokens",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_LOOKUP_PREFETCH_END: self._on_lookup_prefetch_end,
        }

    def _on_lookup_prefetch_end(self, event: Event) -> None:
        self._requested_tokens.add(event.metadata["requested_tokens"])
        self._hit_tokens.add(event.metadata["hit_tokens"])
