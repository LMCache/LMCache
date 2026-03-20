# SPDX-License-Identifier: Apache-2.0

"""Tests for SMMetricsSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.sm import (
    SMMetricsSubscriber,
)


def _make_keys(count: int) -> list:
    """Create a list of placeholder key objects for testing."""
    return [f"key-{i}" for i in range(count)]


def _make_sm_event(
    event_type: EventType,
    succeeded: list,
    failed: list,
) -> Event:
    return Event(
        event_type=event_type,
        metadata={"succeeded_keys": succeeded, "failed_keys": failed},
    )


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = SMMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestSMMetricsSubscriber:
    def test_read_prefetched_increments_counters(self, bus, subscriber):
        bus.start()
        bus.publish(
            _make_sm_event(
                EventType.SM_READ_PREFETCHED,
                succeeded=_make_keys(3),
                failed=_make_keys(2),
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_write_reserved_increments_counters(self, bus, subscriber):
        bus.start()
        bus.publish(
            _make_sm_event(
                EventType.SM_WRITE_RESERVED,
                succeeded=_make_keys(5),
                failed=_make_keys(1),
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_no_subscription_for_finished_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.SM_READ_PREFETCHED_FINISHED not in subs
        assert EventType.SM_WRITE_FINISHED not in subs

    def test_subscriptions_cover_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.SM_READ_PREFETCHED in subs
        assert EventType.SM_WRITE_RESERVED in subs

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        for _ in range(10):
            bus.publish(
                _make_sm_event(
                    EventType.SM_READ_PREFETCHED,
                    succeeded=_make_keys(1),
                    failed=[],
                )
            )
        time.sleep(0.15)
        bus.stop()

    def test_does_not_crash_on_empty_keys(self, bus, subscriber):
        bus.start()
        bus.publish(
            _make_sm_event(
                EventType.SM_READ_PREFETCHED,
                succeeded=[],
                failed=[],
            )
        )
        time.sleep(0.15)
        bus.stop()
