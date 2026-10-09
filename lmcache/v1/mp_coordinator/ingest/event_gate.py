# SPDX-License-Identifier: Apache-2.0
"""Admission stage of the coordinator's cache-event ingest layer.

Every cache event the coordinator acts on enters through
:meth:`EventGate.ingest` or :meth:`EventGate.ingest_batches`, which own
the per-emitter stream cursor and decide what reaches the consumers
holding the state.

See ``docs/design/v1/mp_coordinator/ingest.md``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import cast
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.persistence.durable_component import PersistenceType
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock

logger = init_logger(__name__)


class IngestResult(str, Enum):
    """Outcome of offering one batch to the gate; only ``ADMITTED``
    reaches the consumers."""

    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    STALE_INCARNATION = "stale_incarnation"


@dataclass(frozen=True)
class CacheEventIngestSummary:
    """Aggregate outcomes from ingesting one ordered list of batches.

    Attributes:
        applied: Batches admitted and broadcast to consumers.
        duplicates: Batches dropped because their sequence was already seen.
        stale: Batches dropped because their incarnation was outdated.
    """

    applied: int = 0
    duplicates: int = 0
    stale: int = 0


@dataclass(frozen=True)
class InstanceStreamStats:
    """The gate's cursor for one emitter stream.

    The loss counts cover the emitter's current incarnation, since this
    coordinator process began tracking it; they are not checkpointed.

    Attributes:
        incarnation: The emitter incarnation the cursor belongs to.
        last_seq: Highest ``seq`` admitted from that incarnation.
        gap_detected: The coordinator knows it is missing part of this
            emitter's slice: a batch skipped a ``seq`` or reported more
            dropped events. Cleared only when the slice is rebuilt (a new
            incarnation, or the emitter leaving).
        missing_batches: Batches that never arrived: the ``seq`` values
            skipped. The first batch of a stream the gate was not tracking
            counts none, since earlier batches were sent before it listened.
        events_dropped: Events the emitter reported dropping before they
            reached the coordinator (the increases of ``dropped_events``).
    """

    incarnation: int
    last_seq: int
    gap_detected: bool
    missing_batches: int = 0
    events_dropped: int = 0


@dataclass
class EventGateStats:
    """What the gate has seen from the emitters.

    The totals cover every emitter since this process started: unlike the
    per-stream counts they never reset when an emitter restarts or leaves,
    and they are not checkpointed.

    Attributes:
        batches_applied: Batches admitted and broadcast to the consumers.
        batches_duplicate: Batches dropped because their ``seq`` was seen.
        batches_stale: Batches dropped because their incarnation was older.
        batches_missing: Batches that never arrived (skipped ``seq``
            values in streams the gate was already tracking).
        events_dropped: Events the emitters reported dropping before they
            reached the coordinator.
        streams: Each tracked emitter's cursor, keyed by ``instance_id``.
    """

    batches_applied: int = 0
    batches_duplicate: int = 0
    batches_stale: int = 0
    batches_missing: int = 0
    events_dropped: int = 0
    streams: dict[str, InstanceStreamStats] = field(default_factory=dict)


@dataclass
class _StreamCursor:
    """Mutable form of :class:`InstanceStreamStats`.

    ``dropped_baseline`` is the ``dropped_events`` of the last admitted
    batch, or ``None`` until the gate knows it (a stream joined midway).
    """

    incarnation: int
    last_seq: int = 0
    gap_detected: bool = False
    dropped_baseline: int | None = None
    missing_batches: int = 0
    events_dropped: int = 0


class EventGate:
    """Ordering and fencing gate in front of the cache-event consumers.

    Thread-safe: one lock guards the cursor map and is held across the
    broadcast, so an emitter's batches reach the consumers in admission
    order. Consumers must therefore not call back into the gate.

    Args:
        broadcaster: Fan-out for admitted batches.
        quiesce: Held across every mutating call, so whoever captures
            durable state never sees a half-applied batch.
    """

    def __init__(
        self, broadcaster: CacheEventBroadcaster, quiesce: QuiesceLock
    ) -> None:
        self._lock = threading.Lock()
        self._broadcaster = broadcaster
        self._cursors: dict[str, _StreamCursor] = {}
        # Acquired outside self._lock on every mutating path, so a capture
        # and an ingest take the two locks in the same order.
        self._quiesce = quiesce
        # The totals; ``streams`` stays empty here and is filled from the
        # cursors when stats() takes a snapshot.
        self._stats = EventGateStats()

    def ingest(self, batch: CacheEventBatch) -> IngestResult:
        """Offer one batch to the consumers, applying incarnation
        fencing, ``seq`` dedup, and loss detection.

        Args:
            batch: The batch to offer.

        Returns:
            Whether it was admitted (broadcast before returning), or why
            it was dropped.
        """
        with self._quiesce.applying(), self._lock:
            result = self._admit(batch)
            if result == IngestResult.ADMITTED:
                self._stats.batches_applied += 1
            elif result == IngestResult.DUPLICATE:
                self._stats.batches_duplicate += 1
            else:
                self._stats.batches_stale += 1
        return result

    def ingest_batches(self, batches: list[CacheEventBatch]) -> CacheEventIngestSummary:
        """Offer ``batches`` to the gate in list order.

        Args:
            batches: Event batches ordered by their source.

        Returns:
            Counts of admitted, duplicate, and stale batches.

        Raises:
            RuntimeError: If :meth:`ingest` returns an unknown outcome.
        """
        applied = 0
        duplicates = 0
        stale = 0
        for batch in batches:
            result = self.ingest(batch)
            if result == IngestResult.ADMITTED:
                applied += 1
            elif result == IngestResult.DUPLICATE:
                duplicates += 1
            elif result == IngestResult.STALE_INCARNATION:
                stale += 1
            else:
                raise RuntimeError(f"unknown cache-event ingest result: {result!r}")
        return CacheEventIngestSummary(
            applied=applied,
            duplicates=duplicates,
            stale=stale,
        )

    def drop_instance(self, instance_id: str) -> None:
        """Fence ``instance_id`` and forget its cursor, so a later
        reconnect starts fresh at any incarnation.

        For deregistration and heartbeat-timeout eviction.

        Args:
            instance_id: The departing instance.
        """
        with self._quiesce.applying(), self._lock:
            self._broadcaster.fence_instance(instance_id)
            self._cursors.pop(instance_id, None)

    @property
    def name(self) -> str:
        """Name of the gate's section in a checkpoint."""
        return "stream_cursors"

    @property
    def persistence_type(self) -> PersistenceType:
        """Cursors describe the stream the placements came from, so they
        ride with the placements."""
        return PersistenceType.CHECKPOINT

    def capture(self) -> Mapping[str, object]:
        """Return the per-emitter cursors.

        Placements restored without these are unfenceable: fencing
        compares against a prior incarnation, and a gate with no cursor
        has nothing to compare, so a restarted server's stale L1 slice
        would be advertised forever.

        The dropped-events baseline rides along, so a restarted
        coordinator measures the next report against the last one instead
        of losing a whole report to re-learning the baseline.

        Returns:
            ``{"cursors": {instance_id: (incarnation, last_seq,
            gap_detected, dropped_baseline)}}``; ``dropped_baseline`` is
            ``None`` when not yet known.
        """
        with self._lock:
            return {
                "cursors": {
                    instance_id: (
                        cursor.incarnation,
                        cursor.last_seq,
                        cursor.gap_detected,
                        cursor.dropped_baseline,
                    )
                    for instance_id, cursor in self._cursors.items()
                }
            }

    def restore(self, state: Mapping[str, object]) -> None:
        """Load captured cursors into a gate that has admitted nothing.

        Args:
            state: A :meth:`capture` value. Cursors captured before the
                dropped-events baseline existed hold three fields; they
                restore with the baseline unknown.

        Raises:
            ValueError: If the gate already holds cursors -- a batch was
                admitted before the restore and would be overwritten.
        """
        cursors = cast(
            "Mapping[str, tuple[int, int, bool] | tuple[int, int, bool, int | None]]",
            state["cursors"],
        )
        with self._lock:
            if self._cursors:
                raise ValueError(
                    "restore() requires a gate that has admitted nothing "
                    f"(holds {len(self._cursors)} cursors)"
                )
            for instance_id, fields in cursors.items():
                incarnation, last_seq, gap_detected = fields[:3]
                self._cursors[instance_id] = _StreamCursor(
                    incarnation=incarnation,
                    last_seq=last_seq,
                    gap_detected=gap_detected,
                    dropped_baseline=fields[3] if len(fields) > 3 else None,
                )

    def stats(self) -> EventGateStats:
        """Return a snapshot of the totals and every emitter's cursor."""
        with self._lock:
            return replace(
                self._stats,
                streams={
                    instance_id: InstanceStreamStats(
                        incarnation=cursor.incarnation,
                        last_seq=cursor.last_seq,
                        gap_detected=cursor.gap_detected,
                        missing_batches=cursor.missing_batches,
                        events_dropped=cursor.events_dropped,
                    )
                    for instance_id, cursor in self._cursors.items()
                },
            )

    # -- Internals ------------------------------------------------------------

    def _admit(self, batch: CacheEventBatch) -> IngestResult:
        """Decide ``batch``'s fate and, if admitted, account its loss and
        broadcast it. Call holding the quiesce and ``self._lock``."""
        cursor = self._cursors.get(batch.instance_id)
        # Loss is measured only from a known starting point: a stream the
        # gate already tracks, including across a restart.
        tracked = cursor is not None
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
            cursor = _StreamCursor(
                incarnation=batch.incarnation,
                # A new incarnation, or a stream seen from its first batch,
                # has dropped nothing yet; one joined midway has an unknown
                # history, so its first report only sets the baseline.
                dropped_baseline=0 if tracked or batch.seq == 1 else None,
            )
            self._cursors[batch.instance_id] = cursor

        skipped = batch.seq - cursor.last_seq - 1
        if skipped > 0:
            self._mark_gap(
                cursor,
                batch,
                f"seq jumped {cursor.last_seq} -> {batch.seq}",
            )
            if tracked:
                cursor.missing_batches += skipped
                self._stats.batches_missing += skipped
        if cursor.dropped_baseline is not None:
            dropped = batch.dropped_events - cursor.dropped_baseline
            if dropped > 0:
                self._mark_gap(
                    cursor, batch, f"emitter reported {dropped} dropped events"
                )
                cursor.events_dropped += dropped
                self._stats.events_dropped += dropped
        cursor.dropped_baseline = batch.dropped_events
        cursor.last_seq = batch.seq
        self._broadcaster.broadcast(batch)
        return IngestResult.ADMITTED

    @staticmethod
    def _mark_gap(cursor: _StreamCursor, batch: CacheEventBatch, reason: str) -> None:
        """Flag ``cursor``'s slice incomplete, warning the first time."""
        if cursor.gap_detected:
            return
        cursor.gap_detected = True
        logger.warning(
            "Event loss for instance %s (incarnation %d): %s; slice needs replay",
            batch.instance_id,
            batch.incarnation,
            reason,
        )
