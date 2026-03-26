# SPDX-License-Identifier: Apache-2.0

"""MP Server logging subscriber — debug logs for store/retrieve/lookup events.

Logs are emitted via Python's standard logging module.  When OpenTelemetry
is installed, ``init_logger`` automatically attaches an OTel
``LoggingHandler`` so records are forwarded to OTel when a
``LoggerProvider`` is configured at startup.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class MPServerLoggingSubscriber(EventSubscriber):
    """Logs MP server store/retrieve/lookup events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: self._on_retrieve_start,
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
            EventType.MP_LOOKUP_PREFETCH_START: self._on_lookup_prefetch_start,
            EventType.MP_LOOKUP_PREFETCH_END: self._on_lookup_prefetch_end,
        }

    def _on_store_start(self, event: Event) -> None:
        logger.debug(
            "MP store start: session=%s device=%s",
            event.session_id,
            event.metadata.get("device"),
        )

    def _on_store_end(self, event: Event) -> None:
        logger.debug(
            "MP store end: session=%s device=%s stored_count=%s",
            event.session_id,
            event.metadata.get("device"),
            event.metadata.get("stored_count"),
        )

    def _on_retrieve_start(self, event: Event) -> None:
        logger.debug(
            "MP retrieve start: session=%s device=%s",
            event.session_id,
            event.metadata.get("device"),
        )

    def _on_retrieve_end(self, event: Event) -> None:
        logger.debug(
            "MP retrieve end: session=%s device=%s retrieved_count=%s",
            event.session_id,
            event.metadata.get("device"),
            event.metadata.get("retrieved_count"),
        )

    def _on_lookup_prefetch_start(self, event: Event) -> None:
        logger.debug(
            "MP lookup/prefetch start: session=%s",
            event.session_id,
        )

    def _on_lookup_prefetch_end(self, event: Event) -> None:
        logger.debug(
            "MP lookup/prefetch end: session=%s found_count=%s",
            event.session_id,
            event.metadata.get("found_count"),
        )
