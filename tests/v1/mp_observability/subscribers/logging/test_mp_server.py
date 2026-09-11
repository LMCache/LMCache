# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerLoggingSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.mp_server import (
    MPServerLoggingSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = MPServerLoggingSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestMPServerLoggingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    def test_store_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0", "stored_count": 5},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1", "retrieved_count": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-3",
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={"found_count": 10},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_multiple_events_no_crash(self, bus, subscriber):
        bus.start()
        for i in range(10):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0"},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()


class TestLookupHitRateSummary:
    """MP_LOOKUP_PREFETCH_END feeds a throttled token-weighted summary that
    resets at each line (rate over the lookups since the previous line)."""

    def _end(self, requested, hit):
        return Event(
            event_type=EventType.MP_LOOKUP_PREFETCH_END,
            session_id="req-1",
            metadata={"requested_tokens": requested, "hit_tokens": hit},
        )

    def test_first_lookup_logs_and_resets(self, subscriber, caplog):
        with caplog.at_level("INFO"):
            subscriber._on_lookup_prefetch_end(self._end(1024, 768))
        assert "lookup hit rate (over 1 lookup(s)): prefix=75.0%" in caplog.text
        assert subscriber._requested == 0

    def test_throttled_then_token_weighted_aggregate(self, subscriber, caplog):
        subscriber._on_lookup_prefetch_end(self._end(1024, 768))  # logs + resets
        caplog.clear()
        with caplog.at_level("INFO"):
            subscriber._on_lookup_prefetch_end(self._end(100, 0))
            subscriber._on_lookup_prefetch_end(self._end(100, 100))
            assert "hit rate" not in caplog.text  # within the interval
            subscriber._next_log = 0.0
            subscriber._on_lookup_prefetch_end(self._end(200, 100))
        assert "lookup hit rate (over 3 lookup(s)): prefix=50.0%" in caplog.text

    def test_early_exit_is_not_counted(self, subscriber, caplog):
        with caplog.at_level("INFO"):
            subscriber._on_lookup_prefetch_end(self._end(0, 0))
        assert "hit rate" not in caplog.text
        assert subscriber._lookups == 0
