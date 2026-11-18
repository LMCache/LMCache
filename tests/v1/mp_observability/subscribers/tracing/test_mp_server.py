# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerTracingSubscriber."""

# Standard
from unittest.mock import patch
import time

# Third Party
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing import (
    MPServerTracingSubscriber,
)
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import (
    SpanRegistry,
)
import lmcache.v1.mp_observability.subscribers.tracing.mp_server as mp_server_module


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def registry():
    return SpanRegistry()


@pytest.fixture
def subscriber(registry, bus):
    sub = MPServerTracingSubscriber(registry)
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def exporter():
    """Real OTel provider with in-memory exporter; patches the module tracer.

    Span attributes are only recorded by a real ``TracerProvider``.
    ``_HAS_OTEL`` is forced True so the handlers don't early-return.
    """
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    real_tracer = provider.get_tracer("lmcache_mp.server")
    with (
        patch.object(mp_server_module, "_tracer", real_tracer),
        patch.object(mp_server_module, "_HAS_OTEL", True),
    ):
        yield exp
    exp.shutdown()


def _root_span(exporter: InMemorySpanExporter, sid: str):
    """Return the finished root span for *sid*, or None."""
    for span in exporter.get_finished_spans():
        if span.name == "request" and span.attributes.get("session_id") == sid:
            return span
    return None


class TestMPServerTracingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_REQUEST_START in subs
        assert EventType.MP_STORE_SUBMITTED in subs
        assert EventType.MP_RETRIEVE_SUBMITTED in subs
        assert EventType.MP_REQUEST_END in subs
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    # ------------------------------------------------------------------
    # Root span creation
    # ------------------------------------------------------------------

    def test_root_span_created_on_request_start(self, bus, registry, subscriber):
        bus.start()
        bus.publish(Event(event_type=EventType.MP_REQUEST_START, session_id="req-root"))
        time.sleep(0.15)
        assert registry.get("req-root", "request") is not None
        bus.stop()

    def test_no_root_span_before_any_event(self, registry):
        assert registry.get("any-session", "request") is None

    # ------------------------------------------------------------------
    # Session-end closes root immediately when no stores in flight
    # ------------------------------------------------------------------

    def test_session_end_closes_root_immediately_when_no_store(
        self, bus, registry, subscriber
    ):
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
                event_type=EventType.MP_REQUEST_END,
                session_id="req-lookup-only",
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get("req-lookup-only", "request") is None
        assert "req-lookup-only" not in subscriber._pending_store_count
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_finishes(self, bus, registry, subscriber):
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
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (store in flight)
        assert registry.get(sid, "request") is not None
        assert sid in subscriber._deferred_session_ends

        # Now GPU store completes
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"stored_count": 2, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_ends
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Multiple stores: root stays open until all complete
    # ------------------------------------------------------------------

    def test_multiple_stores_all_must_finish(self, bus, registry, subscriber):
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
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # count=2 — still open
        assert registry.get(sid, "request") is not None

        # First store ends — count=1, still open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "request") is not None

        # Second store ends — count=0, closes now
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Lazy root creation on store-only path (no lookup)
    # ------------------------------------------------------------------

    def test_lazy_root_on_store_only_path(self, bus, registry, subscriber):
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
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "request") is not None

        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.020,
                metadata={"stored_count": 3, "device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.025,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None

    # ------------------------------------------------------------------
    # Retrieve deferral: SESSION_END races GPU retrieve
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_retrieve_finishes(
        self, bus, registry, subscriber
    ):
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
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (retrieve in flight)
        assert registry.get(sid, "request") is not None
        assert sid in subscriber._deferred_session_ends

        # Now GPU retrieve completes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"retrieved_count": 4, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_ends
        assert sid not in subscriber._pending_retrieve_count

    def test_session_end_deferred_until_both_store_and_retrieve_finish(
        self, bus, registry, subscriber
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
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Both in flight — root still open
        assert registry.get(sid, "request") is not None

        # Store finishes first — retrieve still pending, root stays open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "request") is not None

        # Retrieve finishes — now both counters are zero → root closes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"retrieved_count": 2, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_ends

    # ------------------------------------------------------------------
    # Child spans registered in shared registry for sub-span parenting
    # ------------------------------------------------------------------

    def test_retrieve_span_registered_in_registry_while_open(
        self, bus, registry, subscriber
    ):
        """An open mp.retrieve span is accessible in the registry under
        ``span_name="retrieve"`` so that a future sub-span subscriber can
        look it up as a parent."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-reg",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get("req-reg", "retrieve") is not None
        bus.stop()

    def test_retrieve_span_deregistered_after_close(self, bus, registry, subscriber):
        """Registry entry is cleaned up when mp.retrieve ends so stale
        contexts do not leak into subsequent requests."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-reg2",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-reg2",
                metadata={"device": f"{torch_device_type}:0", "retrieved_count": 1},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert registry.get("req-reg2", "retrieve") is None

    # ------------------------------------------------------------------
    # Existing lifecycle tests (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_store_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0", "stored_count": 5},
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
                metadata={"device": f"{torch_device_type}:1"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1", "retrieved_count": 3},
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
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
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
                metadata={"device": f"{torch_device_type}:0"},
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
                    metadata={"device": f"{torch_device_type}:0"},
                )
            )
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0


class TestHitRateAttributes:
    """Verify hit_tokens / requested_tokens / hit_rate are set on the root span.

    Uses the module-level ``exporter`` fixture so span attributes are actually
    recorded by a real TracerProvider.
    """

    def test_hit_rate_attrs_set_on_root_span(self, exporter):
        """LP_END with hit_tokens=512, requested_tokens=1024 → hit_rate=0.5."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-normal"

        bus.publish(
            Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "hit_tokens": 512,
                    "requested_tokens": 1024,
                    "found_count": 2,
                    "l1_hit_tokens": 512,
                    "l2_hit_tokens": 0,
                    "early_exit_reason": "",
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = _root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 512
        assert root.attributes["requested_tokens"] == 1024
        assert abs(root.attributes["hit_rate"] - 0.5) < 1e-9

    def test_hit_rate_zero_when_requested_tokens_is_zero(self, exporter):
        """LP_END with requested_tokens=0 yields hit_rate=0.0 without error."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-zero"

        bus.publish(
            Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "hit_tokens": 0,
                    "requested_tokens": 0,
                    "found_count": 0,
                    "l1_hit_tokens": 0,
                    "l2_hit_tokens": 0,
                    "early_exit_reason": "",
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = _root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 0
        assert root.attributes["requested_tokens"] == 0
        assert root.attributes["hit_rate"] == 0.0
        # Emitters predating the request-level fields: rates read 0.0.
        assert root.attributes["prompt_tokens"] == 0
        assert root.attributes["engine_hit_rate"] == 0.0
        assert root.attributes["request_hit_rate"] == 0.0

    def test_request_level_attrs_set_on_root_span(self, exporter):
        """LP_END with prompt_tokens=2000, engine_hit_tokens=500 and
        request_hit_tokens=1500 → engine_hit_rate=0.25, request_hit_rate=0.75."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-request-level"

        bus.publish(
            Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "hit_tokens": 1024,
                    "requested_tokens": 1536,
                    "found_count": 6,
                    "apc_covered_tokens": 512,
                    "l1_hit_tokens": 1024,
                    "l2_hit_tokens": 0,
                    "prompt_tokens": 2000,
                    "engine_hit_tokens": 500,
                    "request_hit_tokens": 1500,
                    "early_exit_reason": "",
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = _root_span(exporter, sid)
        assert root is not None
        assert root.attributes["prompt_tokens"] == 2000
        assert root.attributes["engine_hit_tokens"] == 500
        assert root.attributes["request_hit_tokens"] == 1500
        assert abs(root.attributes["engine_hit_rate"] - 0.25) < 1e-9
        assert abs(root.attributes["request_hit_rate"] - 0.75) < 1e-9

    def test_no_hit_rate_attrs_on_store_only_path(self, exporter):
        """Store-only request (no LP_END) leaves root span without hit rate attrs."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-store-only"

        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED, session_id=sid, timestamp=now
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.020,
                metadata={"stored_count": 3, "device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.025,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = _root_span(exporter, sid)
        assert root is not None
        assert "hit_tokens" not in root.attributes
        assert "hit_rate" not in root.attributes


class TestTierAttributionAttributes:
    """Verify the per-tier hit split and early-exit reason on the root span."""

    def _root_span_for(self, exporter, sid: str, end_metadata: dict):
        """Run a REQUEST/LP_START/LP_END/REQUEST_END cycle for ``sid``.

        Args:
            exporter: In-memory span exporter from the ``exporter`` fixture.
            sid: Session ID correlating the four events.
            end_metadata: Metadata for the ``MP_LOOKUP_PREFETCH_END`` event.

        Returns:
            The finished root span for ``sid``, or None.
        """
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(MPServerTracingSubscriber(SpanRegistry()))
        bus.start()
        now = time.time()
        for event_type, offset, metadata in (
            (EventType.MP_REQUEST_START, 0.0, {}),
            (EventType.MP_LOOKUP_PREFETCH_START, 0.001, {}),
            (EventType.MP_LOOKUP_PREFETCH_END, 0.010, end_metadata),
            (EventType.MP_REQUEST_END, 0.020, {}),
        ):
            bus.publish(
                Event(
                    event_type=event_type,
                    session_id=sid,
                    timestamp=now + offset,
                    metadata=metadata,
                )
            )
        time.sleep(0.15)
        bus.stop()
        return _root_span(exporter, sid)

    def test_tier_attributes_on_normal_path(self, exporter):
        """768 hit tokens split 512/256 of 1024 requested."""
        root = self._root_span_for(
            exporter,
            "req-tier-normal",
            {
                "found_count": 3,
                "requested_tokens": 1024,
                "hit_tokens": 768,
                "l1_hit_tokens": 512,
                "l2_hit_tokens": 256,
                "l1_hit_keys": 20,
                "l2_hit_keys": 4,
                "early_exit_reason": "",
            },
        )

        assert root is not None
        assert root.attributes["l1_hit_tokens"] == 512
        assert root.attributes["l2_hit_tokens"] == 256
        assert root.attributes["l1_hit_rate"] == pytest.approx(0.5)
        assert root.attributes["l2_hit_rate"] == pytest.approx(0.25)
        assert root.attributes["l1_hit_keys"] == 20
        assert root.attributes["l2_hit_keys"] == 4
        assert root.attributes["early_exit_reason"] == ""
        # Typed, unlike the generic child-span loop that stringifies metadata.
        assert isinstance(root.attributes["l1_hit_tokens"], int)
        assert isinstance(root.attributes["l1_hit_rate"], float)
        assert isinstance(root.attributes["l1_hit_keys"], int)

    @pytest.mark.parametrize("reason", ["no_gpu_context", "empty_chunk_hashes"])
    def test_early_exit_reason_and_zero_rates(self, exporter, reason):
        """An early exit names itself; a zero denominator yields 0.0 rates."""
        root = self._root_span_for(
            exporter,
            f"req-early-{reason}",
            {
                "found_count": 0,
                "requested_tokens": 0,
                "hit_tokens": 0,
                "l1_hit_tokens": 0,
                "l2_hit_tokens": 0,
                "early_exit_reason": reason,
            },
        )

        assert root is not None
        assert root.attributes["early_exit_reason"] == reason
        assert root.attributes["l1_hit_rate"] == 0.0
        assert root.attributes["l2_hit_rate"] == 0.0


class TestRequestTraceSpan:
    """With the token trace enabled, every ``MP_LOOKUP`` becomes a
    zero-duration ``mp.request.trace`` span under the root request span."""

    _TOKEN_IDS = (5, 6, 7, 8)

    def test_not_subscribed_by_default(self, registry):
        subs = MPServerTracingSubscriber(registry).get_subscriptions()
        assert EventType.MP_LOOKUP not in subs

    def test_subscribed_when_token_trace_enabled(self, registry):
        subs = MPServerTracingSubscriber(registry, token_trace_enabled=True)
        assert EventType.MP_LOOKUP in subs.get_subscriptions()

    @staticmethod
    def _drive(
        exporter,
        sid: str,
        token_ids: tuple[int, ...],
        *,
        registry: SpanRegistry | None = None,
        plain_request: bool = True,
        lookups: int = 1,
    ) -> dict:
        """Publish *lookups* lookups for *sid* and return finished spans by name.

        Args:
            registry: Shared registry; pass one to pre-open a root span.
            plain_request: Bracket the lookups with ``MP_REQUEST_START`` and
                ``MP_REQUEST_END`` as the plain lookup path does.
            lookups: Number of ``MP_LOOKUP`` events to publish.
        """
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = registry if registry is not None else SpanRegistry()
        bus.register_subscriber(
            MPServerTracingSubscriber(registry, token_trace_enabled=True)
        )
        bus.start()
        try:
            if plain_request:
                bus.publish(
                    Event(event_type=EventType.MP_REQUEST_START, session_id=sid)
                )
            for _ in range(lookups):
                bus.publish(
                    Event(
                        event_type=EventType.MP_LOOKUP,
                        session_id=sid,
                        metadata={
                            "request_id": sid,
                            "chunk_hashes": [b"h"],
                            "model_name": "m",
                            "chunk_size": 256,
                            "seq_len": len(token_ids),
                            "dtypes": [],
                            "shapes": [],
                            "token_ids": token_ids,
                        },
                    )
                )
            if plain_request:
                bus.publish(Event(event_type=EventType.MP_REQUEST_END, session_id=sid))
            time.sleep(0.2)
        finally:
            bus.stop()
        return {s.name: s for s in exporter.get_finished_spans()}

    def test_blend_lookup_nests_under_cb_root_without_creating_request(self, exporter):
        """A blend request's root is ``cb.request``; no ``request`` root may
        be opened for it, since nothing on the blend path would close it."""
        sid = "req-blend"
        registry = SpanRegistry()
        cb_root = mp_server_module._tracer.start_span("cb.request")
        registry.open(sid, "cb.request", cb_root, trace.set_span_in_context(cb_root))

        spans = self._drive(
            exporter, sid, self._TOKEN_IDS, registry=registry, plain_request=False
        )

        assert registry.get(sid, "request") is None
        assert spans["mp.request.trace"].parent.span_id == cb_root.context.span_id
        cb_root.end()

    def test_lookup_without_any_root_is_parentless(self, exporter):
        sid = "req-orphan"
        registry = SpanRegistry()

        spans = self._drive(
            exporter, sid, self._TOKEN_IDS, registry=registry, plain_request=False
        )

        assert registry.get(sid, "request") is None
        assert spans["mp.request.trace"].parent is None

    def test_repeat_lookup_emits_one_span_per_lookup(self, exporter):
        """The subscriber keeps no per-request state; a re-looked-up request
        produces a span per lookup, all carrying the same session id."""
        sid = "req-repeat"
        self._drive(exporter, sid, self._TOKEN_IDS, lookups=2)
        traces = [
            s for s in exporter.get_finished_spans() if s.name == "mp.request.trace"
        ]
        assert len(traces) == 2
        assert {s.attributes["session_id"] for s in traces} == {sid}

    def test_span_carries_prompt_under_root(self, exporter):
        sid = "req-trace"
        spans = self._drive(exporter, sid, self._TOKEN_IDS)
        trace_span = spans["mp.request.trace"]
        root = spans["request"]
        assert trace_span.parent.span_id == root.context.span_id
        assert trace_span.attributes["session_id"] == sid
        assert trace_span.attributes["lmcache.model"] == "m"
        assert trace_span.attributes["lmcache.input_ids"] == self._TOKEN_IDS
        assert trace_span.start_time == trace_span.end_time
        assert root.start_time <= trace_span.start_time
