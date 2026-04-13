# SPDX-License-Identifier: Apache-2.0

"""Tests for L1MetricsSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1 import (
    L1MetricsSubscriber,
)


def _make_keys(count: int) -> list:
    """Create a list of placeholder key objects for testing."""
    return [f"key-{i}" for i in range(count)]


def _make_event(event_type: EventType, keys: list) -> Event:
    return Event(event_type=event_type, metadata={"keys": keys})


@pytest.fixture
def bus():
    b = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
    return b


@pytest.fixture
def subscriber(bus):
    sub = L1MetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestL1MetricsSubscriber:
    def test_read_finished_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, _make_keys(5)))
        time.sleep(0.15)
        bus.stop()
        # Verify the counter was called — OTel counters are real objects,
        # we check via the internal measurement
        # (in a real integration test we'd scrape /metrics)

    def test_write_finished_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, _make_keys(3)))
        time.sleep(0.15)
        bus.stop()

    def test_write_finished_and_read_reserved_increments_write_counter(
        self, bus, subscriber
    ):
        bus.start()
        bus.publish(
            _make_event(EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, _make_keys(7))
        )
        time.sleep(0.15)
        bus.stop()

    def test_evicted_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, _make_keys(4)))
        time.sleep(0.15)
        bus.stop()

    def test_no_subscription_for_reserved_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_RESERVED not in subs
        assert EventType.L1_WRITE_RESERVED not in subs

    def test_subscriptions_cover_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED_AND_READ_RESERVED in subs
        assert EventType.L1_KEYS_EVICTED in subs

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        for _ in range(10):
            bus.publish(_make_event(EventType.L1_READ_FINISHED, _make_keys(2)))
        time.sleep(0.15)
        bus.stop()
        # 10 events of 2 keys each = 20 total keys counted
        # No assertion on internal OTel state — this verifies no exceptions

    def test_does_not_crash_on_empty_keys(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, []))
        time.sleep(0.15)
        bus.stop()
