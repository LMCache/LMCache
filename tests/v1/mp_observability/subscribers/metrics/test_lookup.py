# SPDX-License-Identifier: Apache-2.0

"""Tests for LookupMetricsSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel counter values
and assert exact counts after publishing known events through the EventBus.

OTel only allows one MeterProvider per process, so we use a module-scoped
provider and assert on counter **deltas** between before/after snapshots.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.lookup import (
    LookupMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_counters() -> dict[str, int]:
    """Snapshot all counter values from the module-level reader."""
    data = _reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue  # skip histogram data points
                    result[metric.name] = int(dp.value)
    return result


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute the difference between two counter snapshots."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = LookupMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = _read_counters()

    def get_delta() -> dict[str, int]:
        return _counter_delta(before, _read_counters())

    return get_delta


# ---------------------------------------------------------------------------
# Hit-rate counters
# ---------------------------------------------------------------------------


class TestLookupHitRateCounters:
    def test_full_hit(self, bus, subscriber, snapshot):
        """All requested tokens are found in L1+L2."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-1",
                metadata={
                    "found_count": 4,
                    "requested_tokens": 1024,
                    "hit_tokens": 1024,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.lookup_requested_tokens"] == 1024
        assert delta["lmcache_mp.lookup_hit_tokens"] == 1024

    def test_partial_hit(self, bus, subscriber, snapshot):
        """A prefix of the requested tokens is served from cache."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-2",
                metadata={
                    "found_count": 2,
                    "requested_tokens": 1024,
                    "hit_tokens": 512,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.lookup_requested_tokens"] == 1024
        assert delta["lmcache_mp.lookup_hit_tokens"] == 512

    def test_full_miss(self, bus, subscriber, snapshot):
        """Nothing is cached; both counters still receive the denominator."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={
                    "found_count": 0,
                    "requested_tokens": 1024,
                    "hit_tokens": 0,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.lookup_requested_tokens"] == 1024
        assert delta.get("lmcache_mp.lookup_hit_tokens", 0) == 0

    def test_early_exit_contributes_zero(self, bus, subscriber, snapshot):
        """Early-exit lookups (no layout_desc or empty chunk_hashes) emit
        `MP_LOOKUP_PREFETCH_END` with both token fields set to 0, so they
        must not move either counter — preserves the ratio."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-4",
                metadata={
                    "found_count": 0,
                    "requested_tokens": 0,
                    "hit_tokens": 0,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta.get("lmcache_mp.lookup_requested_tokens", 0) == 0
        assert delta.get("lmcache_mp.lookup_hit_tokens", 0) == 0

    def test_multiple_lookups_accumulate(self, bus, subscriber, snapshot):
        """Counters should accumulate across multiple completed lookups."""
        bus.start()
        # 3 full-hit lookups @ 512 tokens each
        for i in range(3):
            bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP_PREFETCH_END,
                    session_id=f"hit-{i}",
                    metadata={
                        "found_count": 2,
                        "requested_tokens": 512,
                        "hit_tokens": 512,
                    },
                )
            )
        # 2 partial-hit lookups: 1024 requested, 256 hit
        for i in range(2):
            bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP_PREFETCH_END,
                    session_id=f"partial-{i}",
                    metadata={
                        "found_count": 1,
                        "requested_tokens": 1024,
                        "hit_tokens": 256,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        # 3*512 + 2*1024 = 1536 + 2048 = 3584
        assert delta["lmcache_mp.lookup_requested_tokens"] == 3584
        # 3*512 + 2*256 = 1536 + 512 = 2048
        assert delta["lmcache_mp.lookup_hit_tokens"] == 2048


# ---------------------------------------------------------------------------
# Subscription wiring
# ---------------------------------------------------------------------------


class TestLookupMetricsSubscriptions:
    def test_subscribes_only_to_prefetch_end(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_LOOKUP_PREFETCH_END in subs
        # Must not subscribe to START — the denominator must not be attributed
        # to lookups that will never fire END (abandoned, early-exit).
        assert EventType.MP_LOOKUP_PREFETCH_START not in subs
        assert len(subs) == 1
