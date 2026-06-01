# SPDX-License-Identifier: Apache-2.0

"""L2 storage logging subscriber — debug logs for L2 store/prefetch events."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class L2LoggingSubscriber(EventSubscriber):
    """Logs L2 store and prefetch events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED: self._on_lookup_submitted,
            EventType.L2_PREFETCH_LOOKUP_COMPLETED: self._on_lookup_completed,
            EventType.L2_PREFETCH_LOAD_SUBMITTED: self._on_load_submitted,
            EventType.L2_PREFETCH_LOAD_COMPLETED: self._on_load_completed,
        }

    def _on_store_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 store submitted: %d keys to adapter %d",
            len(event.metadata.get("keys", [])),
            event.metadata["adapter_index"],
        )

    def _on_store_completed(self, event: Event) -> None:
        logger.debug(
            "L2 store completed: adapter %d, %d succeeded, %d failed",
            event.metadata["adapter_index"],
            len(event.metadata.get("succeeded_keys", [])),
            len(event.metadata.get("failed_keys", [])),
        )

    def _on_lookup_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch lookup submitted: request %d, %d keys to %d adapters",
            event.metadata["request_id"],
            len(event.metadata.get("keys", [])),
            event.metadata["adapter_count"],
        )

    def _on_lookup_completed(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch lookup completed: request %d, %d prefix hits",
            event.metadata["request_id"],
            event.metadata["prefix_hit_count"],
        )

    def _on_load_submitted(self, event: Event) -> None:
        logger.debug(
            "L2 prefetch load submitted: request %d, %d keys to %d adapters",
            event.metadata["request_id"],
            len(event.metadata.get("keys", [])),
            event.metadata["adapter_count"],
        )

    def _on_load_completed(self, event: Event) -> None:
        loaded = event.metadata.get("loaded_keys", [])
        failed = event.metadata.get("failed_keys", [])
        logger.debug(
            "L2 prefetch load completed: request %d, %d loaded, %d failed",
            event.metadata["request_id"],
            len(loaded),
            len(failed),
        )
