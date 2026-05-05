# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendTracingSubscriber."""

# Standard
from unittest.mock import patch
import time

# Third Party
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing.cb_server import (
    BlendTracingSubscriber,
)
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import (
    SpanRegistry,
)
import lmcache.v1.mp_observability.subscribers.tracing.cb_server as cb_server_module


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def registry():
    return SpanRegistry()


@pytest.fixture
def subscriber(registry, bus):
    sub = BlendTracingSubscriber(registry)
    bus.register_subscriber(sub)
    return sub


class TestBlendTracingSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START in subs
        assert EventType.CB_REQUEST_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_SUBMITTED in subs
        assert EventType.CB_RETRIEVE_SUBMITTED in subs
        assert EventType.CB_STORE_FINAL_SUBMITTED in subs
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_START in subs
        assert EventType.CB_STORE_PRE_COMPUTED_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_STORE_FINAL_START in subs
        assert EventType.CB_STORE_FINAL_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    # ------------------------------------------------------------------
    # Root span creation
    # ------------------------------------------------------------------

    def test_root_span_created_on_request_start(self, bus, registry, subscriber):
        bus.start()
        bus.publish(Event(event_type=EventType.CB_REQUEST_START, session_id="req-root"))
        time.sleep(0.15)
        assert registry.get("req-root", "cb.request") is not None
        bus.stop()

    def test_no_root_span_before_any_event(self, registry):
        assert registry.get("any-session", "cb.request") is None

    # ------------------------------------------------------------------
    # Session end closes root immediately when no GPU ops in flight
    # ------------------------------------------------------------------

    def test_session_end_closes_root_immediately_when_no_gpu_ops(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-lookup-only"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 64,
                    "fingerprint_hits": 2,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._pending_gpu_ops
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store_pre_computed
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_pre_computed_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-store-pre"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 0},
            )
        )
        # SESSION_END arrives before GPU store completes
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open
        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        # GPU store completes
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU retrieve
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_retrieve_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-retrieve"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 1},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 1, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"instance_id": 1, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store_final
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_final_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-store-final"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 2},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 2, "num_tokens": 256},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"instance_id": 2, "stored_chunks": 16, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Multiple GPU ops: all must finish before root closes
    # ------------------------------------------------------------------

    def test_multiple_gpu_ops_all_must_finish(self, bus, registry, subscriber):
        bus.start()
        now = time.time()
        sid = "req-multi-gpu-ops"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
                metadata={"instance_id": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Both in flight — root still open
        assert registry.get(sid, "cb.request") is not None

        # Store finishes first — retrieve still pending, root stays open
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "cb.request") is not None

        # Retrieve finishes — all done, root closes
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"instance_id": 0, "num_chunks": 4},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"instance_id": 0, "num_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts

    # ------------------------------------------------------------------
    # Child span lifecycles
    # ------------------------------------------------------------------

    def test_lookup_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id="req-lookup",
                metadata={"num_tokens": 128},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-lookup",
                metadata={
                    "num_tokens": 128,
                    "fingerprint_hits": 3,
                    "storage_hits": 2,
                    "stale_chunks": 1,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_store_pre_computed_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="req-sp",
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="req-sp",
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_retrieve_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-ret",
                metadata={"instance_id": 1, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-ret",
                metadata={"instance_id": 1, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_store_final_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id="req-sf",
                metadata={"instance_id": 2, "num_tokens": 512},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id="req-sf",
                metadata={"instance_id": 2, "stored_chunks": 32, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Point events
    # ------------------------------------------------------------------

    def test_fingerprints_registered_no_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                session_id="req-fp",
                metadata={"num_chunks": 8, "num_tokens": 256},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_chunks_evicted_no_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_CHUNKS_EVICTED,
                session_id="req-ev",
                metadata={"num_chunks": 2},
            )
        )
        time.sleep(0.15)
        bus.stop()

    # ------------------------------------------------------------------
    # Error resilience
    # ------------------------------------------------------------------

    def test_unmatched_end_does_not_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="orphan",
                metadata={"stored_chunks": 2, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_start_cleaned_on_shutdown(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="leaked",
                metadata={"instance_id": 0, "num_tokens": 64},
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
                    event_type=EventType.CB_LOOKUP_START,
                    session_id=f"req-{i}",
                    metadata={"num_tokens": 100},
                )
            )
        for i in range(5):
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
        assert len(subscriber._pending) == 0


class TestCBHitRateAttributes:
    """Verify hit_tokens / requested_tokens / hit_rate are set on cb.request root span.

    Uses a real OTel TracerProvider backed by InMemorySpanExporter so that span
    attributes are actually recorded.  The module-level ``_tracer`` is patched
    for the duration of each test; ``_HAS_OTEL`` is forced True.
    """

    @pytest.fixture
    def exporter(self):
        """Real OTel provider with in-memory exporter; patches module tracer."""
        exp = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        real_tracer = provider.get_tracer("lmcache_mp.blend")
        with (
            patch.object(cb_server_module, "_tracer", real_tracer),
            patch.object(cb_server_module, "_HAS_OTEL", True),
        ):
            yield exp
        exp.shutdown()

    def _root_span(self, exporter: InMemorySpanExporter, sid: str):
        """Return the finished cb.request root span for *sid*, or None."""
        for span in exporter.get_finished_spans():
            if span.name == "cb.request" and span.attributes.get("session_id") == sid:
                return span
        return None

    def test_hit_rate_attrs_set_on_root_span(self, exporter):
        """CB_LOOKUP_END with hit_tokens=512, requested_tokens=1024 → hit_rate=0.5."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-normal"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 1024,
                    "fingerprint_hits": 4,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 512,
                    "requested_tokens": 1024,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 512
        assert root.attributes["requested_tokens"] == 1024
        assert abs(root.attributes["hit_rate"] - 0.5) < 1e-9

    def test_hit_rate_zero_when_requested_tokens_is_zero(self, exporter):
        """CB_LOOKUP_END with requested_tokens=0 yields hit_rate=0.0 without error."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-zero"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 0,
                    "requested_tokens": 0,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 0
        assert root.attributes["requested_tokens"] == 0
        assert root.attributes["hit_rate"] == 0.0

    def test_total_miss_hit_rate(self, exporter):
        """CB_LOOKUP_END with storage_hits=0 but requested_tokens>0 → hit_rate=0.0."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-miss"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 1024,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 0,
                    "requested_tokens": 1024,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 0
        assert root.attributes["requested_tokens"] == 1024
        assert root.attributes["hit_rate"] == 0.0
