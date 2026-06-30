# SPDX-License-Identifier: Apache-2.0

"""Timeout logging subscriber — warning logs for timeout errors.

Logs each :class:`~lmcache.v1.mp_observability.errors.LMCacheTimeoutError`
with its message and construction stack trace.  Logs are emitted via Python's
standard logging module; when OpenTelemetry is installed, ``init_logger``
attaches an OTel ``LoggingHandler`` so records are forwarded to OTel when a
``LoggerProvider`` is configured at startup.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class TimeoutLoggingSubscriber(EventSubscriber):
    """Logs timeout errors at warning level with their stack trace."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TIMEOUT_RAISED: self._on_timeout}

    def _on_timeout(self, event: Event) -> None:
        logger.warning(
            "Timeout raised (%s): %s\n%s",
            event.metadata["exception_type"],
            event.metadata["message"],
            event.metadata["stacktrace"],
        )
