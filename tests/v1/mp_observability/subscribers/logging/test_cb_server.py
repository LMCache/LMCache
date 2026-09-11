# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendLoggingSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.cb_server import (
    BlendLoggingSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = BlendLoggingSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestBlendLoggingSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    def test_no_subscription_for_lifecycle_sentinels(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START not in subs
        assert EventType.CB_REQUEST_END not in subs
        assert EventType.CB_RETRIEVE_SUBMITTED not in subs

    def test_lookup_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id="req-1",
                metadata={"num_tokens": 128},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-1",
                metadata={
                    "num_tokens": 128,
                    "fingerprint_hits": 4,
                    "storage_hits": 3,
                    "stale_chunks": 1,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-3",
                metadata={"instance_id": 1, "num_chunks": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-3",
                metadata={"instance_id": 1, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_fingerprints_registered_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                session_id="req-5",
                metadata={"num_chunks": 8, "num_tokens": 256},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_chunks_evicted_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_CHUNKS_EVICTED,
                session_id="req-5",
                metadata={"num_chunks": 2},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_multiple_events_no_crash(self, bus, subscriber):
        bus.start()
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id=f"req-{i}",
                    metadata={"num_tokens": 100},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"req-{i}",
                    metadata={
                        "num_tokens": 100,
                        "fingerprint_hits": 2,
                        "storage_hits": 2,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(0.15)
        bus.stop()


class TestLookupHitRateSummary:
    """CB_LOOKUP_END feeds a throttled prefix + non-prefix summary."""

    def _end(self, requested, prefix, seg_tail, non_prefix):
        return Event(
            event_type=EventType.CB_LOOKUP_END,
            session_id="req-1",
            metadata={
                "requested_tokens": requested,
                "prefix_hit_tokens": prefix,
                "segmented_prefix_hit_tokens": seg_tail,
                "non_prefix_hit_tokens": non_prefix,
            },
        )

    def test_both_planes_with_segmented_tail_as_prefix(self, subscriber, caplog):
        with caplog.at_level("INFO"):
            subscriber._on_lookup_end(self._end(1024, 512, 256, 256))
        assert (
            "lookup hit rate (over 1 lookup(s)): prefix=75.0% non_prefix=25.0%"
            in caplog.text
        )

    def test_miss_counts_zero(self, subscriber, caplog):
        with caplog.at_level("INFO"):
            subscriber._on_lookup_end(self._end(512, 0, 0, 0))
        assert (
            "lookup hit rate (over 1 lookup(s)): prefix=0.0% non_prefix=0.0%"
            in caplog.text
        )
