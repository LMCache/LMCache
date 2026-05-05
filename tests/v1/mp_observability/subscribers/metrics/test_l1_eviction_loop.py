# SPDX-License-Identifier: Apache-2.0

"""Tests for L1EvictionLoopSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel counter and
histogram values and verify per-tick attribution.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1_eviction_loop import (
    L1EvictionLoopSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15


def _read_counters() -> dict[str, int]:
    data = _reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                total = 0
                any_value = False
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    total += int(dp.value)
                    any_value = True
                if any_value:
                    result[metric.name] = result.get(metric.name, 0) + total
    return result


def _histogram_count(name: str) -> int:
    data = _reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for dp in metric.data.data_points:
                    if hasattr(dp, "count"):
                        total += int(dp.count)
    return total


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L1EvictionLoopSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before_counters = _read_counters()
    before_hist = _histogram_count("lmcache_mp.l1_usage_ratio")

    def get_delta() -> tuple[dict[str, int], int]:
        after_counters = _read_counters()
        after_hist = _histogram_count("lmcache_mp.l1_usage_ratio")
        return _delta(before_counters, after_counters), after_hist - before_hist

    return get_delta


def _tick(triggered: bool, usage: float = 0.5) -> Event:
    return Event(
        event_type=EventType.L1_EVICTION_LOOP_TICK,
        metadata={"usage": usage, "watermark": 0.8, "triggered": triggered},
    )


class TestL1EvictionLoopSubscriber:
    def test_subscribes_only_to_eviction_loop_tick(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_EVICTION_LOOP_TICK in subs
        assert len(subs) == 1

    def test_below_watermark_increments_only_ticks(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(5):
            bus.publish(_tick(triggered=False, usage=0.4))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        d_counters, d_hist = snapshot()
        assert d_counters["lmcache_mp.l1_eviction_loop_ticks"] == 5
        assert d_counters.get("lmcache_mp.l1_eviction_loop_triggered", 0) == 0
        assert d_hist == 5  # all ticks recorded into the usage histogram

    def test_triggered_increments_both_counters(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(3):
            bus.publish(_tick(triggered=True, usage=0.95))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        d_counters, d_hist = snapshot()
        assert d_counters["lmcache_mp.l1_eviction_loop_ticks"] == 3
        assert d_counters["lmcache_mp.l1_eviction_loop_triggered"] == 3
        assert d_hist == 3

    def test_mixed_ticks(self, bus, subscriber, snapshot):
        """Ratio of triggered to ticks reflects how often eviction fired."""
        bus.start()
        # 4 below-watermark, 6 triggered
        for _ in range(4):
            bus.publish(_tick(triggered=False, usage=0.5))
        for _ in range(6):
            bus.publish(_tick(triggered=True, usage=0.9))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        d_counters, d_hist = snapshot()
        assert d_counters["lmcache_mp.l1_eviction_loop_ticks"] == 10
        assert d_counters["lmcache_mp.l1_eviction_loop_triggered"] == 6
        assert d_hist == 10

    def test_missing_metadata_uses_safe_defaults(self, bus, subscriber, snapshot):
        """A tick event with empty metadata still increments ticks and the
        usage histogram (with default 0.0), without crashing."""
        bus.start()
        bus.publish(Event(event_type=EventType.L1_EVICTION_LOOP_TICK, metadata={}))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        d_counters, d_hist = snapshot()
        assert d_counters["lmcache_mp.l1_eviction_loop_ticks"] == 1
        assert d_counters.get("lmcache_mp.l1_eviction_loop_triggered", 0) == 0
        assert d_hist == 1
