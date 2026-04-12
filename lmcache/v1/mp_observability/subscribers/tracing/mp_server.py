# SPDX-License-Identifier: Apache-2.0

"""MP Server span subscriber — OTel spans for store/retrieve/lookup operations.

Creates spans from START/END event pairs using explicit ``start_span()`` /
``span.end()`` with caller-provided timestamps.  Pending spans are stashed
in a ``dict[str, Span]`` keyed by ``session_id``.

Supports parent-child span relationships: child spans (e.g.
``mp.store.reserve_write``) are linked to their parent span
(e.g. ``mp.store``) via ``session_id`` correlation.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventCallback,
    EventSubscriber,
)

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.server")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class MPServerTracingSubscriber(EventSubscriber):
    """Creates OTel spans from MP server START/END event pairs.

    Parent spans (``mp.store``, ``mp.retrieve``, ``mp.lookup_prefetch``)
    are created from top-level START/END events.  Child spans
    (``mp.store.reserve_write``, ``mp.store.gpu_copy``, etc.) are
    automatically parented to the active parent span for the same
    ``session_id``.
    """

    # -- Parent span definitions --
    _SPAN_DEFS: dict[EventType, str] = {
        EventType.MP_STORE_START: "mp.store",
        EventType.MP_RETRIEVE_START: "mp.retrieve",
        EventType.MP_LOOKUP_PREFETCH_START: "mp.lookup_prefetch",
    }

    _END_TO_START: dict[EventType, EventType] = {
        EventType.MP_STORE_END: EventType.MP_STORE_START,
        EventType.MP_RETRIEVE_END: EventType.MP_RETRIEVE_START,
        EventType.MP_LOOKUP_PREFETCH_END: (EventType.MP_LOOKUP_PREFETCH_START),
    }

    # -- Child span definitions --
    # Maps child START event -> (span name, parent START event)
    _CHILD_SPAN_DEFS: dict[EventType, tuple[str, EventType]] = {
        EventType.MP_STORE_RESERVE_WRITE_START: (
            "mp.store.reserve_write",
            EventType.MP_STORE_START,
        ),
        EventType.MP_STORE_GPU_COPY_START: (
            "mp.store.gpu_copy",
            EventType.MP_STORE_START,
        ),
        EventType.MP_RETRIEVE_READ_PREFETCHED_START: (
            "mp.retrieve.read_prefetched",
            EventType.MP_RETRIEVE_START,
        ),
        EventType.MP_RETRIEVE_GPU_COPY_START: (
            "mp.retrieve.gpu_copy",
            EventType.MP_RETRIEVE_START,
        ),
    }

    _CHILD_END_TO_START: dict[EventType, EventType] = {
        EventType.MP_STORE_RESERVE_WRITE_END: (EventType.MP_STORE_RESERVE_WRITE_START),
        EventType.MP_STORE_GPU_COPY_END: (EventType.MP_STORE_GPU_COPY_START),
        EventType.MP_RETRIEVE_READ_PREFETCHED_END: (
            EventType.MP_RETRIEVE_READ_PREFETCHED_START
        ),
        EventType.MP_RETRIEVE_GPU_COPY_END: (EventType.MP_RETRIEVE_GPU_COPY_START),
    }

    def __init__(self) -> None:
        # session_id:event_type -> (span, start_event_type)
        self._pending: dict[str, tuple[Any, EventType]] = {}
        # session_id:parent_start_type -> OTel Context (for child spans)
        self._parent_ctx: dict[str, Any] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        subs: dict[EventType, EventCallback] = {}
        # Parent span events
        for start_type in self._SPAN_DEFS:
            subs[start_type] = self._on_start
        for end_type in self._END_TO_START:
            subs[end_type] = self._on_end
        # Child span events
        for start_type in self._CHILD_SPAN_DEFS:
            subs[start_type] = self._on_child_start
        for end_type in self._CHILD_END_TO_START:
            subs[end_type] = self._on_child_end
        return subs

    # -- Parent span handlers --

    def _on_start(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        span_name = self._SPAN_DEFS[event.event_type]
        span = _tracer.start_span(
            span_name,
            start_time=int(event.timestamp * 1e9),
        )
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.set_attribute("session_id", event.session_id)

        key = self._make_key(event.session_id, event.event_type)
        self._pending[key] = (span, event.event_type)

        # Stash OTel context so child spans can be parented
        ctx = trace.set_span_in_context(span)
        ctx_key = self._make_key(event.session_id, event.event_type)
        self._parent_ctx[ctx_key] = ctx

    def _on_end(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        start_type = self._END_TO_START[event.event_type]
        key = self._make_key(event.session_id, start_type)
        entry = self._pending.pop(key, None)
        if entry is None:
            logger.debug(
                "No pending span for %s session=%s",
                event.event_type.value,
                event.session_id,
            )
            return

        span, _ = entry
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.end(end_time=int(event.timestamp * 1e9))

        # Clean up parent context
        self._parent_ctx.pop(key, None)

    # -- Child span handlers --

    def _on_child_start(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        span_name, parent_start_type = self._CHILD_SPAN_DEFS[event.event_type]

        # Look up parent context
        parent_key = self._make_key(event.session_id, parent_start_type)
        parent_ctx = self._parent_ctx.get(parent_key)

        span = _tracer.start_span(
            span_name,
            context=parent_ctx,
            start_time=int(event.timestamp * 1e9),
        )
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.set_attribute("session_id", event.session_id)

        key = self._make_key(event.session_id, event.event_type)
        self._pending[key] = (span, event.event_type)

    def _on_child_end(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        start_type = self._CHILD_END_TO_START[event.event_type]
        key = self._make_key(event.session_id, start_type)
        entry = self._pending.pop(key, None)
        if entry is None:
            logger.debug(
                "No pending child span for %s session=%s",
                event.event_type.value,
                event.session_id,
            )
            return

        span, _ = entry
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.end(end_time=int(event.timestamp * 1e9))

    # -- Helpers --

    @staticmethod
    def _make_key(session_id: str, event_type: EventType) -> str:
        return f"{session_id}:{event_type.value}"

    def shutdown(self) -> None:
        # End any leaked spans
        for key, (span, _) in self._pending.items():
            try:
                span.end()
            except Exception:
                pass
        self._pending.clear()
        self._parent_ctx.clear()
