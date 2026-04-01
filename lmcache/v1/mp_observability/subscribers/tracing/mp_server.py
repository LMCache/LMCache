# SPDX-License-Identifier: Apache-2.0

"""MP Server span subscriber — OTel spans for store/retrieve/lookup operations.

Creates spans from START/END event pairs using explicit ``start_span()`` /
``span.end()`` with caller-provided timestamps.  Pending spans are stashed
in a ``dict[str, Span]`` keyed by ``session_id``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.server")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class MPServerTracingSubscriber(EventSubscriber):
    """Creates OTel spans from MP server START/END event pairs."""

    # Maps START event types to (span name, END event type)
    _SPAN_DEFS: dict[EventType, str] = {
        EventType.MP_STORE_START: "mp.store",
        EventType.MP_RETRIEVE_START: "mp.retrieve",
        EventType.MP_LOOKUP_PREFETCH_START: "mp.lookup_prefetch",
    }

    _END_TO_START: dict[EventType, EventType] = {
        EventType.MP_STORE_END: EventType.MP_STORE_START,
        EventType.MP_RETRIEVE_END: EventType.MP_RETRIEVE_START,
        EventType.MP_LOOKUP_PREFETCH_END: EventType.MP_LOOKUP_PREFETCH_START,
    }

    def __init__(self) -> None:
        # session_id -> (span, start_event_type) for pending spans
        self._pending: dict[str, tuple[Any, EventType]] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_start,
            EventType.MP_STORE_END: self._on_end,
            EventType.MP_RETRIEVE_START: self._on_start,
            EventType.MP_RETRIEVE_END: self._on_end,
            EventType.MP_LOOKUP_PREFETCH_START: self._on_start,
            EventType.MP_LOOKUP_PREFETCH_END: self._on_end,
        }

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

        # Key: (session_id, start_event_type)
        key = f"{event.session_id}:{event.event_type.value}"
        self._pending[key] = (span, event.event_type)

    def _on_end(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        start_type = self._END_TO_START[event.event_type]
        key = f"{event.session_id}:{start_type.value}"
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

    def shutdown(self) -> None:
        # End any leaked spans
        for key, (span, _) in self._pending.items():
            try:
                span.end()
            except Exception:
                pass
        self._pending.clear()
