# SPDX-License-Identifier: Apache-2.0

"""MP Server logging subscriber — debug logs for store/retrieve/lookup events.

Logs are emitted via Python's standard logging module.  When OpenTelemetry
is installed, ``init_logger`` automatically attaches an OTel
``LoggingHandler`` so records are forwarded to OTel when a
``LoggerProvider`` is configured at startup.
"""

# Future
from __future__ import annotations

# Standard
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class MPServerLoggingSubscriber(EventSubscriber):
    """Logs MP server store/retrieve/lookup events at debug level."""

    #: Minimum seconds between two hit-rate summary lines.
    HIT_RATE_LOG_INTERVAL_S = 30.0

    def __init__(self) -> None:
        # Token-weighted hit-rate accumulators, reset at each summary line.
        self._requested = 0
        self._hit = 0
        self._lookups = 0
        self._next_log = 0.0

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: self._on_retrieve_start,
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
            EventType.MP_LOOKUP_PREFETCH_START: self._on_lookup_prefetch_start,
            EventType.MP_LOOKUP_PREFETCH_END: self._on_lookup_prefetch_end,
            EventType.MP_VLLM_BLOCK_ALLOCATION: self._on_block_allocation,
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
        requested = int(event.metadata.get("requested_tokens") or 0)
        if requested <= 0:
            return  # early-exit lookup: nothing was actually looked up
        self._requested += requested
        self._hit += int(event.metadata.get("hit_tokens") or 0)
        self._lookups += 1
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self.HIT_RATE_LOG_INTERVAL_S
        logger.info(
            "lookup hit rate (over %d lookup(s)): prefix=%.1f%%",
            self._lookups,
            100.0 * self._hit / self._requested,
        )
        self._requested = self._hit = self._lookups = 0

    def _on_block_allocation(self, event: Event) -> None:
        records = event.metadata.get("records", [])
        for rec in records:
            logger.debug(
                "vLLM block allocation: req_id=%s "
                "new_blocks=%d new_tokens=%d "
                "block_ids=%s",
                rec.req_id,
                len(rec.new_block_ids),
                len(rec.new_token_ids),
                rec.new_block_ids[:10],
            )
