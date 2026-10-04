# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-operation transfer logging for DAX-Coordinated L1."""

# Future
from __future__ import annotations

# Standard
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)
_PENDING_MAX_AGE_SECONDS = 60.0
_CorrelationKey = tuple[str, str, str]
_TRANSFER_EVENTS = {
    EventType.MP_STORE_START: ("PUT", None),
    EventType.MP_STORE_END: ("PUT", "stored_count"),
    EventType.MP_RETRIEVE_START: ("GET", None),
    EventType.MP_RETRIEVE_END: ("GET", "retrieved_count"),
}


class DevDaxTransferLoggingSubscriber(EventSubscriber):
    """Log every completed Device-DAX L0/L1 transfer at INFO level.

    The MP transfer module already publishes START and END events as ordered
    CUDA-stream callbacks. END carries the actual bytes reserved for PUT or
    returned for GET, so this subscriber does not reconstruct model geometry.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str, _CorrelationKey], float] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Handle transfer pairs and expire unmatched starts on eviction ticks."""
        return {
            **{kind: self._on_transfer for kind in _TRANSFER_EVENTS},
            EventType.L1_EVICTION_LOOP_TICK: self._prune_pending,
        }

    @staticmethod
    def _correlation_key(event: Event) -> _CorrelationKey | None:
        device = event.metadata.get("device")
        if not event.session_id or device is None:
            return None
        engine_id = event.metadata.get("engine_id", "")
        return event.session_id, str(device), str(engine_id)

    def _on_transfer(self, event: Event) -> None:
        direction, count_field = _TRANSFER_EVENTS[event.event_type]
        key = self._correlation_key(event)
        if count_field is None:
            if key is not None:
                self._pending[direction, key] = event.timestamp
            return
        if key is None:
            logger.warning(
                "Device-DAX transfer completion lacks correlation metadata: "
                "direction=%s request_id=%s",
                direction,
                event.session_id,
            )
            return

        started_at = self._pending.pop((direction, key), None)
        if started_at is None or event.timestamp <= started_at:
            logger.warning(
                "Device-DAX transfer completion lacks a valid START event: "
                "direction=%s request_id=%s device=%s engine_id=%s",
                direction,
                key[0],
                key[1],
                key[2],
            )
            return

        elapsed_seconds = event.timestamp - started_at
        total_bytes = int(event.metadata.get("total_bytes", 0))
        bandwidth_gb_s = total_bytes / elapsed_seconds / 1e9
        logger.info(
            "Device-DAX transfer completed: backend=dax_coordinated_l1 "
            "direction=%s request_id=%s device=%s engine_id=%s "
            "objects=%d tokens=%d payload_bytes=%d latency_ms=%.3f "
            "bandwidth_GB_s=%.3f",
            direction,
            key[0],
            key[1],
            key[2],
            int(event.metadata.get(count_field, 0)),
            int(event.metadata.get("num_tokens", 0)),
            total_bytes,
            elapsed_seconds * 1000.0,
            bandwidth_gb_s,
        )

    def _prune_pending(self, event: Event) -> None:
        deadline = time.time() - _PENDING_MAX_AGE_SECONDS
        stale = [
            key for key, timestamp in self._pending.items() if timestamp < deadline
        ]
        for key in stale:
            del self._pending[key]
