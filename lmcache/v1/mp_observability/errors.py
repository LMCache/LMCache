# SPDX-License-Identifier: Apache-2.0

"""Observability-aware timeout error for the LMCache runtime.

This module defines :class:`LMCacheTimeoutError`, a drop-in replacement for the
built-in :class:`TimeoutError` that reports itself to the MP observability
EventBus on construction.  Code that raises a timeout should raise this class
instead of the built-in so the timeout is surfaced as an
:attr:`~lmcache.v1.mp_observability.event.EventType.TIMEOUT_RAISED` event (and
from there into OTel metrics / logs / traces via the timeout subscribers).

The ``ban-raw-timeout-error`` pre-commit hook enforces this convention across
``lmcache/`` by rejecting any bare ``TimeoutError`` raise.
"""

# Future
from __future__ import annotations

# Standard
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)

logger = init_logger(__name__)


class LMCacheTimeoutError(TimeoutError):
    """A timeout error that reports itself to MP observability on construction.

    Subclasses the built-in :class:`TimeoutError`, so existing
    ``except TimeoutError`` (and ``except asyncio.TimeoutError``, which is the
    same type on Python 3.11+) handlers continue to catch it unchanged.

    On construction it publishes a
    :attr:`~lmcache.v1.mp_observability.event.EventType.TIMEOUT_RAISED` event to
    the global EventBus carrying the message and the construction stack trace.
    The publish is skipped entirely when observability is disabled — which is
    the default outside the MP server process — so the class behaves exactly
    like the built-in ``TimeoutError`` (minus one boolean check) on the non-MP
    path and never adds an OTel dependency at runtime there.

    Publishing never raises: any failure is swallowed and logged at debug level
    so that observability can never break error handling.

    Args:
        message: Human-readable description of what timed out.  Becomes both
            the exception string and the ``message`` field of the emitted
            event.
        session_id: Request/session id used to correlate the timeout event with
            the originating request's other observability events (e.g. to nest
            the timeout span under that request's root span).  Empty string
            means "no correlation id available".
    """

    def __init__(self, message: str, *, session_id: str = "") -> None:
        super().__init__(message)
        if not is_observability_enabled():
            return
        # Capture the stack at the raise site (drop this __init__ frame) so the
        # event records where the timeout originated, per the OTel exception
        # semantic conventions (``exception.stacktrace``).
        stacktrace = "".join(traceback.format_stack()[:-1])
        self._publish_timeout_event(message, stacktrace, session_id)

    def _publish_timeout_event(
        self, message: str, stacktrace: str, session_id: str
    ) -> None:
        """Publish a ``TIMEOUT_RAISED`` event; never raises.

        Args:
            message: The timeout message to record.
            stacktrace: The pre-formatted construction stack trace.
            session_id: Correlating session id, or empty string.
        """
        try:
            get_event_bus().publish(
                Event(
                    event_type=EventType.TIMEOUT_RAISED,
                    metadata={
                        "message": message,
                        "exception_type": type(self).__name__,
                        "stacktrace": stacktrace,
                    },
                    session_id=session_id,
                )
            )
        except Exception:
            logger.debug("Failed to publish TIMEOUT_RAISED event", exc_info=True)
