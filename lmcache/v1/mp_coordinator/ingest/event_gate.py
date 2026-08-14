# SPDX-License-Identifier: Apache-2.0
"""Admission stage of the coordinator's cache-event ingest layer.

Every cache event the coordinator acts on enters through
:meth:`EventGate.ingest`, which owns the per-emitter stream cursor and
decides what reaches the consumers holding the state.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster

logger = init_logger(__name__)


class IngestResult(str, Enum):
    """Outcome of offering one batch to the gate; only ``ADMITTED``
    reaches the consumers."""

    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    STALE_INCARNATION = "stale_incarnation"


@dataclass(frozen=True)
class InstanceStreamStats:
    """The gate's cursor for one emitter stream. ``gap_detected`` marks
    the emitter's slice stale until its stream is replayed."""

    incarnation: int
    last_seq: int
    gap_detected: bool


@dataclass
class _StreamCursor:
    """Mutable form of :class:`InstanceStreamStats`."""

    incarnation: int
    last_seq: int = 0
    gap_detected: bool = False


class EventGate:
    """Ordering and fencing gate in front of the cache-event consumers.

    Thread-safe: one lock guards the cursor map and is held across the
    broadcast, so an emitter's batches reach the consumers in admission
    order. Consumers must therefore not call back into the gate.

    Args:
        broadcaster: Fan-out for admitted batches.
    """

    def __init__(self, broadcaster: CacheEventBroadcaster) -> None:
        self._lock = threading.Lock()
        self._broadcaster = broadcaster
        self._cursors: dict[str, _StreamCursor] = {}

    def ingest(self, batch: CacheEventBatch) -> IngestResult:
        """Offer one batch to the consumers, applying incarnation
        fencing, ``seq`` dedup, and gap detection.

        Args:
            batch: The batch to offer.

        Returns:
            Whether it was admitted (broadcast before returning), or why
            it was dropped.
        """
        with self._lock:
            cursor = self._cursors.get(batch.instance_id)
            if cursor is not None:
                if batch.incarnation < cursor.incarnation:
                    return IngestResult.STALE_INCARNATION
                if batch.incarnation > cursor.incarnation:
                    # Restart: the emitter's memory is empty, so the L1
                    # facts its previous incarnation reported are void.
                    self._broadcaster.fence_instance(batch.instance_id)
                    cursor = None
                elif batch.seq <= cursor.last_seq:
                    return IngestResult.DUPLICATE
            if cursor is None:
                cursor = _StreamCursor(incarnation=batch.incarnation)
                self._cursors[batch.instance_id] = cursor

            if batch.seq > cursor.last_seq + 1 and not cursor.gap_detected:
                cursor.gap_detected = True
                logger.warning(
                    "Event gap for instance %s (incarnation %d): "
                    "seq jumped %d -> %d; slice needs replay",
                    batch.instance_id,
                    batch.incarnation,
                    cursor.last_seq,
                    batch.seq,
                )
            cursor.last_seq = batch.seq
            self._broadcaster.broadcast(batch)
            return IngestResult.ADMITTED

    def drop_instance(self, instance_id: str) -> None:
        """Fence ``instance_id`` and forget its cursor, so a later
        reconnect starts fresh at any incarnation.

        For deregistration and heartbeat-timeout eviction.

        Args:
            instance_id: The departing instance.
        """
        with self._lock:
            self._broadcaster.fence_instance(instance_id)
            self._cursors.pop(instance_id, None)

    def stats(self) -> dict[str, InstanceStreamStats]:
        """Return a cursor snapshot keyed by ``instance_id``."""
        with self._lock:
            return {
                instance_id: InstanceStreamStats(
                    incarnation=cursor.incarnation,
                    last_seq=cursor.last_seq,
                    gap_detected=cursor.gap_detected,
                )
                for instance_id, cursor in self._cursors.items()
            }
