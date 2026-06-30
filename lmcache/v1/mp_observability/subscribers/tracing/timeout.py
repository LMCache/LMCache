# SPDX-License-Identifier: Apache-2.0

"""OTel tracing subscriber for timeout errors.

Records each :class:`~lmcache.v1.mp_observability.errors.LMCacheTimeoutError`
as a zero-duration OTel span named ``"timeout"``.  The span carries the OTel
exception semantic-convention attributes (``exception.type`` /
``exception.message`` / ``exception.stacktrace``) as an ``"exception"`` span
event and is marked with ERROR status — the same shape OTel's
``Span.record_exception`` produces, but driven from the EventBus drain thread
where the original exception object is no longer available.

When the event carries a ``session_id`` that matches an open ``"request"`` span
in the shared :class:`~lmcache.v1.mp_observability.subscribers.tracing\
.span_registry.SpanRegistry`, the timeout span is nested under that request;
otherwise it is emitted as a standalone root span.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _tracer = trace.get_tracer("lmcache_mp.timeout")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class TimeoutTracingSubscriber(EventSubscriber):
    """Creates an OTel span recording each timeout error.

    Args:
        registry: Shared span registry used to look up the open ``"request"``
            span to nest the timeout span under.  When ``None`` a private
            registry is created, so the subscriber is usable standalone (the
            timeout span then has no parent to find and is emitted as a root
            span).
    """

    def __init__(self, registry: SpanRegistry | None = None) -> None:
        self._registry = registry if registry is not None else SpanRegistry()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TIMEOUT_RAISED: self._on_timeout}

    def _on_timeout(self, event: Event) -> None:
        """Emit a zero-duration ``"timeout"`` span recording the exception.

        Args:
            event: ``TIMEOUT_RAISED`` event with ``message``,
                ``exception_type``, and ``stacktrace`` in its metadata.
        """
        if not _HAS_OTEL:
            return
        parent_ctx = (
            self._registry.get_context(event.session_id, "request")
            if event.session_id
            else None
        )
        start_ns = int(event.timestamp * 1e9)
        message = str(event.metadata.get("message", ""))
        exception_type = str(event.metadata.get("exception_type", "TimeoutError"))

        span = _tracer.start_span("timeout", context=parent_ctx, start_time=start_ns)
        span.set_attribute("session_id", event.session_id)
        span.add_event(
            "exception",
            attributes={
                "exception.type": exception_type,
                "exception.message": message,
                "exception.stacktrace": str(event.metadata.get("stacktrace", "")),
            },
            timestamp=start_ns,
        )
        span.set_status(Status(StatusCode.ERROR, message))
        span.end(end_time=start_ns)
