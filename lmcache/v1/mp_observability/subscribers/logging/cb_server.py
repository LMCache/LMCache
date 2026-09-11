# SPDX-License-Identifier: Apache-2.0

"""Blend logging subscriber — debug logs for cache blending events."""

# Future
from __future__ import annotations

# Standard
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class BlendLoggingSubscriber(EventSubscriber):
    """Logs cache blending (CB) events at debug level."""

    #: Minimum seconds between two hit-rate summary lines.
    HIT_RATE_LOG_INTERVAL_S = 30.0

    def __init__(self) -> None:
        # Token-weighted hit-rate accumulators, reset at each summary line.
        self._requested = 0
        self._prefix_hit = 0
        self._non_prefix_hit = 0
        self._lookups = 0
        self._next_log = 0.0

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the mapping of event types to handler callbacks."""
        return {
            EventType.CB_LOOKUP_START: self._on_lookup_start,
            EventType.CB_LOOKUP_END: self._on_lookup_end,
            EventType.CB_RETRIEVE_START: self._on_retrieve_start,
            EventType.CB_RETRIEVE_END: self._on_retrieve_end,
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_fingerprints_registered,
            EventType.CB_CHUNKS_EVICTED: self._on_chunks_evicted,
        }

    def _on_lookup_start(self, event: Event) -> None:
        logger.debug(
            "CB lookup start: session=%s num_tokens=%s",
            event.session_id,
            event.metadata.get("num_tokens"),
        )

    def _on_lookup_end(self, event: Event) -> None:
        logger.debug(
            "CB lookup end: session=%s num_tokens=%s"
            " fingerprint_hits=%s storage_hits=%s stale_chunks=%s no_gpu_context=%s",
            event.session_id,
            event.metadata.get("num_tokens"),
            event.metadata.get("fingerprint_hits"),
            event.metadata.get("storage_hits"),
            event.metadata.get("stale_chunks"),
            event.metadata.get("no_gpu_context"),
        )
        md = event.metadata
        requested = int(md.get("requested_tokens") or 0)
        if requested <= 0:
            return
        self._requested += requested
        # Segmented-tail chunks are pure loads, so they count as prefix.
        self._prefix_hit += int(md.get("prefix_hit_tokens") or 0) + int(
            md.get("segmented_prefix_hit_tokens") or 0
        )
        self._non_prefix_hit += int(md.get("non_prefix_hit_tokens") or 0)
        self._lookups += 1
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self.HIT_RATE_LOG_INTERVAL_S
        logger.info(
            "lookup hit rate (over %d lookup(s)): prefix=%.1f%% non_prefix=%.1f%%",
            self._lookups,
            100.0 * self._prefix_hit / self._requested,
            100.0 * self._non_prefix_hit / self._requested,
        )
        self._requested = self._prefix_hit = self._non_prefix_hit = 0
        self._lookups = 0

    def _on_retrieve_start(self, event: Event) -> None:
        logger.debug(
            "CB retrieve start: session=%s instance_id=%s num_chunks=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_chunks"),
        )

    def _on_retrieve_end(self, event: Event) -> None:
        logger.debug(
            "CB retrieve end: session=%s instance_id=%s num_chunks=%s success=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_chunks"),
            event.metadata.get("success"),
        )

    def _on_fingerprints_registered(self, event: Event) -> None:
        logger.debug(
            "CB fingerprint table: +%s chunks (%s tokens)",
            event.metadata.get("num_chunks"),
            event.metadata.get("num_tokens"),
        )

    def _on_chunks_evicted(self, event: Event) -> None:
        logger.debug(
            "CB fingerprint table: evicted %s stale chunks",
            event.metadata.get("num_chunks"),
        )
