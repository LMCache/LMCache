# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide key directory for the MP coordinator.

Maps each :class:`ObjectKey` to its placements (instance, tier, backend,
size) across the fleet, using :class:`CacheEventBatch` streams from MP
servers. The directory is eventually consistent: lookups are hints to be
validated at the owner.

Events are processed in order per instance and only the latest L1
placements for each incarnation are kept. L2 placements persist across
restarts.

Views like per-``cache_salt`` L2 usage are maintained separately from
the same event stream.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class Placement:
    """One live placement of a key, as returned by directory lookups.

    Attributes:
        instance_id: The emitter that most recently reported the placement.
        incarnation: The reporting instance's incarnation at report time.
        tier: Tier the bytes live on (``l1`` or ``l2``).
        backend: Backend within the tier.
        size_bytes: Size the owner reported at store time.
        shared: ``True`` when the backend is a fleet-shared pool (see
            :class:`CacheEventBatch`).
    """

    instance_id: str
    incarnation: int
    tier: Tier
    backend: str
    size_bytes: int
    shared: bool = False


class ApplyResult(str, Enum):
    """Result of applying one :class:`CacheEventBatch` to the directory.

    ``APPLIED`` — the batch was applied.
    ``DUPLICATE`` — the batch's ``seq`` was already applied for the
    instance's current incarnation; the batch was dropped.
    ``STALE_INCARNATION`` — the batch carries an incarnation older than the
    instance's current one; the batch was dropped.
    """

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE_INCARNATION = "stale_incarnation"


@dataclass(frozen=True)
class InstanceDirectoryStats:
    """Directory-side bookkeeping for one reporting instance.

    Attributes:
        incarnation: The instance's current incarnation.
        last_seq: Highest batch ``seq`` applied for that incarnation.
        gap_detected: ``True`` if a ``seq`` gap was observed for the
            instance's stream.
        num_l1_keys: Number of keys the stream has reported L1
            placements for (the fencing index). Approximate for shared
            pools: a counted placement may since have been deleted or
            re-reported by another emitter.
        num_l2_keys: Number of keys currently holding an L2 placement
            whose last reporter is this instance (computed from the
            placements at read time; L2 is not fenced, so this survives
            the stream's restarts).
    """

    incarnation: int
    last_seq: int
    gap_detected: bool
    num_l1_keys: int
    num_l2_keys: int


@dataclass(frozen=True)
class DirectoryStats:
    """A point-in-time summary of directory contents.

    Attributes:
        num_keys: Keys with at least one placement.
        num_placements: Total placements across all keys.
        instances: Per-instance bookkeeping, keyed by ``instance_id``.
    """

    num_keys: int
    num_placements: int
    instances: dict[str, InstanceDirectoryStats]


@dataclass
class _KeyRecord:
    """Directory value for one key: its placements plus recency."""

    placements: list[Placement] = field(default_factory=list)
    content_hash_hex: str = ""
    last_access: float = 0.0


@dataclass
class _InstanceState:
    """Per-instance event-stream cursor and reverse index."""

    incarnation: int
    last_seq: int = 0
    gap_detected: bool = False
    keys: set[ObjectKey] = field(default_factory=set)


class KeyDirectory:
    """Thread-safe in-memory key directory built from cache events.

    Mutations arrive through :meth:`apply_batch` and :meth:`drop_instance`;
    reads through :meth:`lookup` and :meth:`stats`. Nothing is persisted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[ObjectKey, _KeyRecord] = {}
        self._instances: dict[str, _InstanceState] = {}

    def apply_batch(self, batch: CacheEventBatch) -> ApplyResult:
        """Apply one event batch to the directory.

        Applies incarnation fencing, seq dedup, and gap detection, then
        the entries. Entry application is idempotent: re-storing upserts
        the placement, deleting an absent placement is a no-op.

        Args:
            batch: The event batch to apply.

        Returns:
            Whether the batch was applied, or why it was dropped.
        """
        with self._lock:
            state = self._instances.get(batch.instance_id)
            if state is None:
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            elif batch.incarnation < state.incarnation:
                return ApplyResult.STALE_INCARNATION
            elif batch.incarnation > state.incarnation:
                # Restart: fence out the previous incarnation's placements.
                self._drop_instance_locked(batch.instance_id)
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            elif batch.seq <= state.last_seq:
                return ApplyResult.DUPLICATE

            if batch.seq > state.last_seq + 1 and not state.gap_detected:
                state.gap_detected = True
                logger.warning(
                    "Event gap for instance %s (incarnation %d): "
                    "seq jumped %d -> %d; slice needs replay",
                    batch.instance_id,
                    batch.incarnation,
                    state.last_seq,
                    batch.seq,
                )
            state.last_seq = batch.seq

            for entry in batch.entries:
                self._apply_entry_locked(state, batch, entry)
            return ApplyResult.APPLIED

    def lookup(self, keys: list[ObjectKey]) -> list[list[Placement]]:
        """Return the known placements for each requested key.

        Args:
            keys: The keys to look up.

        Returns:
            One placement list per requested key, in request order —
            empty for unknown keys. Each list is sorted by
            ``(instance_id, tier, backend)``.
        """
        with self._lock:
            results: list[list[Placement]] = []
            for key in keys:
                record = self._records.get(key)
                if record is None:
                    results.append([])
                    continue
                results.append(
                    sorted(
                        record.placements,
                        key=lambda p: (p.instance_id, p.tier.value, p.backend),
                    )
                )
            return results

    def drop_instance(self, instance_id: str) -> int:
        """Remove every **L1** placement reported by ``instance_id``.

        The instance's stream cursor is removed too, so a later reconnect
        starts fresh with any incarnation.

        Args:
            instance_id: The instance whose placements to drop.

        Returns:
            The number of placements removed.
        """
        with self._lock:
            removed = self._drop_instance_locked(instance_id)
            self._instances.pop(instance_id, None)
            return removed

    def reconcile(self, batch: CacheEventBatch) -> None:
        """Apply ``batch``'s entries without stream-cursor bookkeeping.

        Args:
            batch: The synthesized batch to apply.
        """
        with self._lock:
            state = self._instances.get(batch.instance_id)
            if state is None:
                state = _InstanceState(incarnation=batch.incarnation)
                self._instances[batch.instance_id] = state
            for entry in batch.entries:
                self._apply_entry_locked(state, batch, entry)

    def stats(self) -> DirectoryStats:
        """Return a point-in-time summary of directory contents.

        Returns:
            Key/placement counts plus per-instance stream state, keyed by
            ``instance_id``.
        """
        with self._lock:
            num_placements = 0
            l2_keys_by_instance: dict[str, int] = {}
            for record in self._records.values():
                num_placements += len(record.placements)
                for reporter in {
                    p.instance_id for p in record.placements if p.tier == Tier.L2
                }:
                    l2_keys_by_instance[reporter] = (
                        l2_keys_by_instance.get(reporter, 0) + 1
                    )
            instances = {
                instance_id: InstanceDirectoryStats(
                    incarnation=state.incarnation,
                    last_seq=state.last_seq,
                    gap_detected=state.gap_detected,
                    num_l1_keys=len(state.keys),
                    num_l2_keys=l2_keys_by_instance.get(instance_id, 0),
                )
                for instance_id, state in self._instances.items()
            }
            return DirectoryStats(
                num_keys=len(self._records),
                num_placements=num_placements,
                instances=instances,
            )

    # -- Internals (call with self._lock held) --------------------------------

    def _apply_entry_locked(
        self,
        state: _InstanceState,
        batch: CacheEventBatch,
        entry: CacheEventEntry,
    ) -> None:
        """Apply one entry of ``batch`` under the directory lock."""
        key = entry.key.to_object_key()
        if batch.event_type == CacheEventType.STORE:
            record = self._records.get(key)
            if record is None:
                record = _KeyRecord()
                self._records[key] = record
            placement = Placement(
                instance_id=batch.instance_id,
                incarnation=batch.incarnation,
                tier=batch.tier,
                backend=batch.backend,
                size_bytes=entry.size_bytes,
                shared=batch.shared,
            )
            index = self._find_placement(record.placements, batch)
            if index is None:
                record.placements.append(placement)
            else:
                record.placements[index] = placement
            if entry.content_hash_hex:
                record.content_hash_hex = entry.content_hash_hex
            record.last_access = max(record.last_access, batch.ts)
            if batch.tier == Tier.L1:
                state.keys.add(key)
        elif batch.event_type == CacheEventType.DELETE:
            record = self._records.get(key)
            if record is None:
                return
            index = self._find_placement(record.placements, batch)
            if index is not None:
                record.placements.pop(index)
            if not record.placements:
                del self._records[key]
            if batch.tier == Tier.L1 and not any(
                p.tier == Tier.L1 and p.instance_id == batch.instance_id
                for p in record.placements
            ):
                state.keys.discard(key)
        elif batch.event_type == CacheEventType.ACCESS:
            record = self._records.get(key)
            if record is not None:
                record.last_access = max(record.last_access, batch.ts)

    @staticmethod
    def _find_placement(
        placements: list[Placement], batch: CacheEventBatch
    ) -> int | None:
        """Return the index of the placement whose identity matches
        ``batch``, or ``None`` if absent."""
        for index, placement in enumerate(placements):
            if (
                placement.shared == batch.shared
                and (batch.shared or placement.instance_id == batch.instance_id)
                and placement.tier == batch.tier
                and placement.backend == batch.backend
            ):
                return index
        return None

    def _drop_instance_locked(self, instance_id: str) -> int:
        """Remove the **L1** placements ``instance_id`` reported; return
        the count. L2 placements survive: their bytes persist across the
        reporter's restarts and leave only via ``DELETE`` events."""
        state = self._instances.get(instance_id)
        if state is None:
            return 0
        removed = 0
        for key in state.keys:
            record = self._records.get(key)
            if record is None:
                continue
            kept = [
                p
                for p in record.placements
                if p.tier != Tier.L1 or p.instance_id != instance_id
            ]
            removed += len(record.placements) - len(kept)
            if kept:
                record.placements = kept
            else:
                del self._records[key]
        state.keys.clear()
        return removed
