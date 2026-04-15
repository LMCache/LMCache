# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerTracingSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing import (
    MPServerTracingSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = MPServerTracingSubscriber()
    bus.register_subscriber(sub)
    return sub


def _drain(bus: EventBus) -> None:
    """Start and stop the bus to flush all queued events."""
    bus.start()
    time.sleep(0.15)
    bus.stop()


class TestMPServerTracingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_REQUEST_START in subs
        assert EventType.MP_STORE_SUBMITTED in subs
        assert EventType.MP_RETRIEVE_SUBMITTED in subs
        assert EventType.MP_SESSION_END in subs
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    # ------------------------------------------------------------------
    # Root span creation
    # ------------------------------------------------------------------

    def test_root_span_created_on_request_start(self, bus, subscriber):
        bus.start()
        bus.publish(Event(event_type=EventType.MP_REQUEST_START, session_id="req-root"))
        time.sleep(0.15)
        assert "req-root" in subscriber._root_spans
        bus.stop()

    def test_no_root_span_before_any_event(self, subscriber):
        assert len(subscriber._root_spans) == 0

    # ------------------------------------------------------------------
    # Session-end closes root immediately when no stores in flight
    # ------------------------------------------------------------------

    def test_session_end_closes_root_immediately_when_no_store(self, bus, subscriber):
        bus.start()
        now = time.time()
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id="req-lookup-only",
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-lookup-only",
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-lookup-only",
                timestamp=now + 0.010,
                metadata={"found_count": 4},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id="req-lookup-only",
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert "req-lookup-only" not in subscriber._root_spans
        assert "req-lookup-only" not in subscriber._pending_store_count
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_finishes(self, bus, subscriber):
        bus.start()
        now = time.time()
        sid = "req-deferred"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        # SESSION_END arrives before STORE_END
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (store in flight)
        assert sid in subscriber._root_spans
        assert sid in subscriber._deferred_session_end_ts

        # Now GPU store completes
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"stored_count": 2, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert sid not in subscriber._root_spans
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Multiple stores: root stays open until all complete
    # ------------------------------------------------------------------

    def test_multiple_stores_all_must_finish(self, bus, subscriber):
        bus.start()
        now = time.time()
        sid = "req-multi-store"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # count=2 — still open
        assert sid in subscriber._root_spans

        # First store ends — count=1, still open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        assert sid in subscriber._root_spans

        # Second store ends — count=0, closes now
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"stored_count": 1, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert sid not in subscriber._root_spans
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Lazy root creation on store-only path (no lookup)
    # ------------------------------------------------------------------

    def test_lazy_root_on_store_only_path(self, bus, subscriber):
        bus.start()
        now = time.time()
        sid = "req-store-only"

        # No MP_REQUEST_START — root created lazily on MP_STORE_START
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"device": "cuda:0"},
            )
        )
        time.sleep(0.15)

        assert sid in subscriber._root_spans

        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.020,
                metadata={"stored_count": 3, "device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=sid,
                timestamp=now + 0.025,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert sid not in subscriber._root_spans

    # ------------------------------------------------------------------
    # Retrieve deferral: SESSION_END races GPU retrieve
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_retrieve_finishes(self, bus, subscriber):
        bus.start()
        now = time.time()
        sid = "req-deferred-retrieve"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        # SESSION_END arrives before RETRIEVE_END (the race condition)
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (retrieve in flight)
        assert sid in subscriber._root_spans
        assert sid in subscriber._deferred_session_end_ts

        # Now GPU retrieve completes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"retrieved_count": 4, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert sid not in subscriber._root_spans
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_retrieve_count

    def test_session_end_deferred_until_both_store_and_retrieve_finish(
        self, bus, subscriber
    ):
        """Root span stays open until both a store and a retrieve finish."""
        bus.start()
        now = time.time()
        sid = "req-store-and-retrieve"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Both in flight — root still open
        assert sid in subscriber._root_spans

        # Store finishes first — retrieve still pending, root stays open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        assert sid in subscriber._root_spans

        # Retrieve finishes — now both counters are zero → root closes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"retrieved_count": 2, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert sid not in subscriber._root_spans
        assert sid not in subscriber._deferred_session_end_ts

    # ------------------------------------------------------------------
    # Existing lifecycle tests (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_store_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": "cuda:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": "cuda:0", "stored_count": 5},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_retrieve_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-2",
                metadata={"device": "cuda:1"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": "cuda:1", "retrieved_count": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_lookup_prefetch_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-3",
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={"found_count": 10},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_end_does_not_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="orphan",
                metadata={"stored_count": 1, "device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_start_cleaned_on_shutdown(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="leaked",
                metadata={"device": "cuda:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()
        subscriber.shutdown()
        assert len(subscriber._pending) == 0

    def test_multiple_concurrent_sessions(self, bus, subscriber):
        bus.start()
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=f"req-{i}",
                    metadata={"device": "cuda:0"},
                )
            )
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": "cuda:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0
