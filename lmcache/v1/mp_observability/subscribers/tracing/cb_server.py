# SPDX-License-Identifier: Apache-2.0

"""OTel tracing subscriber for Cache Blending (CB) operations.

Creates a root ``"cb.request"`` span per session wrapping all CB child spans.
Opens at ``CB_REQUEST_START``; closes at ``CB_REQUEST_END``, deferred until
any in-flight GPU store/retrieve callbacks complete.

Accepts an optional :class:`~lmcache.v1.mp_observability.subscribers.tracing\
.span_registry.SpanRegistry` so the ``"cb.request"`` span is automatically
nested under the MP server ``"request"`` root when both subscribers share the
same registry.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.blend")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class BlendTracingSubscriber(EventSubscriber):
    """Creates OTel spans from CB (Cache Blending) START/END event pairs.

    Each session gets one root ``"cb.request"`` span that nests all child
    spans (``cb.lookup``, ``cb.store_pre_computed``, ``cb.retrieve``,
    ``cb.store_final``).  The root is opened at ``CB_REQUEST_START`` and
    closed at ``CB_REQUEST_END``, with deferral if GPU ops are still in
    flight.

    When a shared :class:`SpanRegistry` is provided, ``"cb.request"`` is
    nested under the MP server ``"request"`` span for the same session.
    """

    # Maps each START event to its span name (used for creation and registry key).
    _SPAN_DEFS: dict[EventType, str] = {
        EventType.CB_STORE_PRE_COMPUTED_START: "cb.store_pre_computed",
        EventType.CB_LOOKUP_START: "cb.lookup",
        EventType.CB_RETRIEVE_START: "cb.retrieve",
        EventType.CB_STORE_FINAL_START: "cb.store_final",
    }

    _END_TO_START: dict[EventType, EventType] = {
        EventType.CB_STORE_PRE_COMPUTED_END: EventType.CB_STORE_PRE_COMPUTED_START,
        EventType.CB_LOOKUP_END: EventType.CB_LOOKUP_START,
        EventType.CB_RETRIEVE_END: EventType.CB_RETRIEVE_START,
        EventType.CB_STORE_FINAL_END: EventType.CB_STORE_FINAL_START,
    }

    # END events that correspond to a SUBMITTED sentinel (decrement ops counter)
    _GPU_OP_END_EVENTS: frozenset[EventType] = frozenset(
        {
            EventType.CB_STORE_PRE_COMPUTED_END,
            EventType.CB_RETRIEVE_END,
            EventType.CB_STORE_FINAL_END,
        }
    )

    def __init__(self, registry: SpanRegistry | None = None) -> None:
        self._registry = registry if registry is not None else SpanRegistry()

        # session_id -> (span, start_event_type) for pending child spans
        self._pending: dict[str, tuple[Any, EventType]] = {}

        # session_id -> number of in-flight GPU ops (SUBMITTED without matching END)
        self._pending_gpu_ops: dict[str, int] = {}

        # session_id -> REQUEST_END timestamp saved when GPU ops are in flight
        self._deferred_session_end_ts: dict[str, float] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the event-to-callback mapping for this subscriber."""
        return {
            # Root span lifecycle
            EventType.CB_REQUEST_START: self._on_request_start,
            EventType.CB_STORE_PRE_COMPUTED_SUBMITTED: self._on_submitted,
            EventType.CB_RETRIEVE_SUBMITTED: self._on_submitted,
            EventType.CB_STORE_FINAL_SUBMITTED: self._on_submitted,
            EventType.CB_REQUEST_END: self._on_session_end,
            # Child spans
            EventType.CB_STORE_PRE_COMPUTED_START: self._on_start,
            EventType.CB_STORE_PRE_COMPUTED_END: self._on_end,
            EventType.CB_LOOKUP_START: self._on_start,
            EventType.CB_LOOKUP_END: self._on_end,
            EventType.CB_RETRIEVE_START: self._on_start,
            EventType.CB_RETRIEVE_END: self._on_end,
            EventType.CB_STORE_FINAL_START: self._on_start,
            EventType.CB_STORE_FINAL_END: self._on_end,
            # Point events
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_point,
            EventType.CB_CHUNKS_EVICTED: self._on_point,
        }

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """End all leaked spans on bus shutdown."""
        for key, (span, _) in self._pending.items():
            try:
                span.end()
            except Exception:
                pass
        self._pending.clear()

        sessions = (
            set(self._pending_gpu_ops)
            | set(self._deferred_session_end_ts)
            | self._registry.all_session_ids()
        )
        for sid in sessions:
            self._registry.clear_session(sid)
        self._pending_gpu_ops.clear()
        self._deferred_session_end_ts.clear()

    # ------------------------------------------------------------------
    # Root span handlers
    # ------------------------------------------------------------------

    def _on_request_start(self, event: Event) -> None:
        """Create the ``"cb.request"`` root span, nested under MP's root if present.

        Args:
            event: ``CB_REQUEST_START`` event with ``session_id`` set.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        # Nest under the MP server's "request" span when running alongside it.
        mp_root_ctx = self._registry.get_context(sid, "request")
        root_span = _tracer.start_span(
            "cb.request",
            context=mp_root_ctx,
            start_time=int(event.timestamp * 1e9),
        )
        root_span.set_attribute("session_id", sid)
        self._registry.open(
            sid, "cb.request", root_span, trace.set_span_in_context(root_span)
        )

    def _on_submitted(self, event: Event) -> None:
        """Increment the in-flight GPU-ops counter for the session.

        Args:
            event: One of ``CB_STORE_PRE_COMPUTED_SUBMITTED``,
                ``CB_RETRIEVE_SUBMITTED``, or ``CB_STORE_FINAL_SUBMITTED``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        self._pending_gpu_ops[sid] = self._pending_gpu_ops.get(sid, 0) + 1

    def _on_session_end(self, event: Event) -> None:
        """Close the root span, or defer if GPU ops are still in flight.

        Args:
            event: ``CB_REQUEST_END`` event carrying the logical end timestamp.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        if self._pending_gpu_ops.get(sid, 0) == 0:
            self._close_request_span(sid, event.timestamp)
        else:
            self._deferred_session_end_ts[sid] = event.timestamp

    # ------------------------------------------------------------------
    # Child span handlers
    # ------------------------------------------------------------------

    def _on_start(self, event: Event) -> None:
        """Create a child span nested under the ``"cb.request"`` root span.

        Args:
            event: One of the CB ``*_START`` event types.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        span_name = self._SPAN_DEFS[event.event_type]

        parent_ctx = self._registry.get_context(sid, "cb.request")
        span = _tracer.start_span(
            span_name,
            context=parent_ctx,
            start_time=int(event.timestamp * 1e9),
        )
        span.set_attribute("session_id", sid)
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))

        key = f"{sid}:{event.event_type.value}"
        self._pending[key] = (span, event.event_type)

        logical = self._SPAN_DEFS.get(event.event_type)
        if logical:
            self._registry.open(sid, logical, span, trace.set_span_in_context(span))

    def _on_end(self, event: Event) -> None:
        """Close a pending child span and handle GPU-ops counter deferral.

        For GPU-backed END events (store_pre_computed, retrieve, store_final),
        decrements the in-flight counter; if it reaches zero and a deferred
        session-end timestamp exists, closes the root span.

        Args:
            event: One of the CB ``*_END`` event types.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        start_type = self._END_TO_START[event.event_type]
        key = f"{sid}:{start_type.value}"
        entry = self._pending.pop(key, None)
        if entry is None:
            logger.debug(
                "No pending CB span for %s session=%s",
                event.event_type.value,
                sid,
            )
        else:
            span, _ = entry
            for k, v in event.metadata.items():
                span.set_attribute(k, str(v))
            span.end(end_time=int(event.timestamp * 1e9))

        logical = self._SPAN_DEFS.get(start_type)
        if logical:
            self._registry.pop(sid, logical)

        if event.event_type in self._GPU_OP_END_EVENTS:
            if (count := self._pending_gpu_ops.get(sid, 0)) > 0:
                if count == 1:
                    self._pending_gpu_ops.pop(sid)
                else:
                    self._pending_gpu_ops[sid] = count - 1
            if (
                sid in self._deferred_session_end_ts
                and self._pending_gpu_ops.get(sid, 0) == 0
            ):
                deferred_ts = self._deferred_session_end_ts.pop(sid)
                self._close_request_span(sid, deferred_ts)

    def _on_point(self, event: Event) -> None:
        """Emit an instant span for point events (no paired END).

        Args:
            event: ``CB_FINGERPRINTS_REGISTERED`` or ``CB_CHUNKS_EVICTED``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        ts_ns = int(event.timestamp * 1e9)
        parent_ctx = self._registry.get_context(sid, "cb.request")
        span = _tracer.start_span(
            event.event_type.value,
            context=parent_ctx,
            start_time=ts_ns,
        )
        span.set_attribute("session_id", sid)
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.end(end_time=ts_ns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_request_span(self, session_id: str, end_ts: float) -> None:
        """End the ``"cb.request"`` root span and clean up per-session state.

        Args:
            session_id: The request session identifier.
            end_ts: Wall-clock timestamp to stamp as the span end time.
        """
        entry = self._registry.pop(session_id, "cb.request")
        if entry is not None:
            root_span, _ = entry
            try:
                root_span.end(end_time=int(end_ts * 1e9))
            except Exception:
                pass
        self._pending_gpu_ops.pop(session_id, None)
        self._deferred_session_end_ts.pop(session_id, None)
