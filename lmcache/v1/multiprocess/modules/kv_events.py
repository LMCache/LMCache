# SPDX-License-Identifier: Apache-2.0
"""Server-side KV event channel for engine workers.

The storage layer publishes key-level cache events on the observability bus
(L1 write completions and evictions, L2 stores and deletes), but only the MP
server sees them. This module records them in a bounded, sequenced log that
engine workers read with ``POLL_KV_EVENTS`` and republish through their
framework's KV event stream (vLLM ``BlockStored`` / ``BlockRemoved``), so a
KV-aware router learns the host-cache placement changes that happen inside
the server. See ``docs/design/v1/multiprocess/modules/kv_events.md``.
"""

# Standard
from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import islice
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_MEDIUM_STORAGE,
    KVEventPollResult,
    KVEventRecord,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.request_handler import request_handler

logger = init_logger(__name__)

DEFAULT_KV_EVENT_LOG_SIZE = 32768

# Internal record kind: the event bus discarded events, so records before the
# marker may be missing. Never returned to clients; ``read_after`` reports it
# as ``lost`` and drops the records preceding it.
_KIND_LOST = "lost"

# Bounds the chunk hash -> token binding cache that stamps stored records;
# covers the window between a chunk's token-binding event and its (possibly
# asynchronous L2) store event.
_TOKEN_BINDING_CACHE_SIZE = 65536


@dataclass(frozen=True)
class _ChunkBinding:
    """One chunk's token content and chain position from an ``MP_TOKENS``
    event, held until its store events arrive."""

    token_ids: tuple[int, ...]
    parent_hash: bytes | None


class KVEventLog:
    """Bounded, sequenced log of cache-event records.

    The bus drain thread appends and the request loop reads, so every method
    takes the log's lock. Records get consecutive sequence numbers starting
    at 1; once ``capacity`` is exceeded the oldest records are discarded, and
    a reader whose cursor predates the oldest retained record is told it
    lost events.

    Args:
        capacity: Maximum number of records retained.

    Raises:
        ValueError: If ``capacity`` is not positive.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive (got {capacity})")
        self._capacity = capacity
        self._records: deque[KVEventRecord] = deque(maxlen=capacity)
        self._next_seq = 1
        self._lock = threading.Lock()
        self._seen_dropped = 0
        self._lost_markers = 0

    @property
    def capacity(self) -> int:
        """Maximum number of records retained."""
        return self._capacity

    @property
    def next_seq(self) -> int:
        """Sequence number the next appended record receives."""
        with self._lock:
            return self._next_seq

    @property
    def lost_markers(self) -> int:
        """Number of event-bus loss markers recorded so far."""
        with self._lock:
            return self._lost_markers

    def __len__(self) -> int:
        """Number of records currently retained (loss markers included)."""
        with self._lock:
            return len(self._records)

    def append(
        self,
        kind: str,
        medium: str,
        model_name: str,
        block_hashes: list[bytes],
        parent_block_hash: bytes | None = None,
        token_ids: list[int] | None = None,
        block_size: int = 0,
    ) -> KVEventRecord:
        """Append one record, assigning it the next sequence number.

        Args:
            kind: ``KV_EVENT_KIND_STORED`` or ``KV_EVENT_KIND_REMOVED``.
            medium: The chunks' cache medium.
            model_name: The chunks' model.
            block_hashes: The chunk hashes the record covers.
            parent_block_hash: The preceding chunk's hash (stored records).
            token_ids: The chunk's tokens (stored records); ``None`` for none.
            block_size: Tokens per chunk.

        Returns:
            The appended record.
        """
        with self._lock:
            record = KVEventRecord(
                seq=self._next_seq,
                kind=kind,
                medium=medium,
                model_name=model_name,
                block_hashes=list(block_hashes),
                parent_block_hash=parent_block_hash,
                token_ids=list(token_ids) if token_ids else [],
                block_size=block_size,
            )
            self._next_seq += 1
            self._records.append(record)
            return record

    def note_dropped_events(self, total_dropped: int) -> None:
        """Record a loss marker if the event bus dropped events since the
        previous call.

        Dropped events may have been cache events, so readers past the marker
        must resynchronize.

        Args:
            total_dropped: The bus's cumulative dropped-event count.
        """
        with self._lock:
            if total_dropped <= self._seen_dropped:
                return
            self._seen_dropped = total_dropped
            self._lost_markers += 1
            self._records.append(
                KVEventRecord(
                    seq=self._next_seq,
                    kind=_KIND_LOST,
                    medium="",
                    model_name="",
                    block_hashes=[],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=0,
                )
            )
            self._next_seq += 1

    def read_after(
        self, cursor: int, model_name: str, max_events: int
    ) -> tuple[list[KVEventRecord], int, bool]:
        """Return the records after ``cursor`` for one model.

        Args:
            cursor: Sequence number of the last record the reader consumed,
                or 0 to start from the oldest retained record.
            model_name: Only records for this model are returned; records
                for other models still advance the cursor.
            max_events: Maximum number of records returned.

        Returns:
            ``(records, next_cursor, lost)``: the matching records (oldest
            first), the cursor to use next, and whether records the reader
            never saw were discarded (cursor older than the log, cursor from
            a different log, or an event-bus loss marker in the scanned
            range; in that case ``records`` only holds records after the
            last loss marker).

        Raises:
            ValueError: If ``cursor`` is negative or ``max_events`` is not
                positive.
        """
        if cursor < 0:
            raise ValueError(f"cursor must be >= 0 (got {cursor})")
        if max_events <= 0:
            raise ValueError(f"max_events must be positive (got {max_events})")
        with self._lock:
            if not self._records:
                last_seq = self._next_seq - 1
                return [], last_seq, cursor > last_seq
            first_seq = self._records[0].seq
            last_seq = self._records[-1].seq
            if cursor > last_seq:
                return [], last_seq, True
            # Records are trimmed oldest-first, so the reader never saw
            # anything between its cursor and the oldest retained record.
            lost = cursor + 1 < first_seq
            # Sequence numbers are consecutive, so the first unread record
            # sits at a known offset.
            start = max(0, cursor + 1 - first_seq)
            records: list[KVEventRecord] = []
            next_cursor = cursor
            for record in islice(self._records, start, None):
                next_cursor = record.seq
                if record.kind == _KIND_LOST:
                    lost = True
                    records.clear()
                    continue
                if record.model_name != model_name:
                    continue
                records.append(record)
                if len(records) >= max_events:
                    break
            return records, next_cursor, lost


class KVEventSubscriber(EventSubscriber):
    """Event-bus subscriber that feeds a :class:`KVEventLog`.

    Runs on the bus's single drain thread, so its own state needs no lock.
    Stored records need the chunk's tokens and its predecessor's hash (a
    router keys its index by them), which the store path publishes ahead of
    the write-finished events as ``MP_TOKENS`` bindings; a store whose
    binding is unknown (evicted from the binding cache, or an L2 prefetch of
    a chunk stored long ago) is counted and skipped.

    Args:
        log: The log to append to.
        bus: The bus the subscriber is registered on; polled for its
            dropped-event count so losses reach readers.
        chunk_size: Tokens per chunk, stamped on every record.
    """

    def __init__(self, log: KVEventLog, bus: EventBus, chunk_size: int) -> None:
        self._log = log
        self._bus = bus
        self._chunk_size = chunk_size
        self._bindings: OrderedDict[bytes, _ChunkBinding] = OrderedDict()
        self._unbound_stores = 0

    @property
    def unbound_stores(self) -> int:
        """Stored keys skipped because their token binding was unknown."""
        return self._unbound_stores

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the bus events this subscriber consumes."""
        return {
            EventType.MP_TOKENS: self._on_tokens,
            EventType.L1_WRITE_FINISHED: self._on_l1_stored,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_l1_stored,
            EventType.L1_KEYS_EVICTED: self._on_l1_removed,
            EventType.L2_KEYS_STORED: self._on_l2_stored,
            EventType.L2_KEYS_DELETED: self._on_l2_removed,
        }

    # -- Event handlers (bus drain thread) ------------------------------------

    def _on_tokens(self, event: Event) -> None:
        self._note_dropped_events()
        chunk_hashes: list[bytes] = event.metadata["chunk_hashes"]
        token_chunks: list[list[int]] = event.metadata["token_chunks"]
        parent_hashes: list[bytes | None] | None = event.metadata.get("parent_hashes")
        if parent_hashes is None:
            # Older producers chain only within the event: a first chunk
            # that does not start the sequence has an unknown predecessor
            # and stays unbound (its store is skipped rather than reported
            # as a sequence start).
            offsets: list[int] = event.metadata["token_offsets"]
            parent_hashes = [None] + chunk_hashes[:-1]
            if offsets and offsets[0] != 0:
                chunk_hashes = chunk_hashes[1:]
                token_chunks = token_chunks[1:]
                parent_hashes = parent_hashes[1:]
        for chunk_hash, chunk, parent_hash in zip(
            chunk_hashes, token_chunks, parent_hashes, strict=True
        ):
            self._bindings[chunk_hash] = _ChunkBinding(
                token_ids=tuple(chunk), parent_hash=parent_hash
            )
            self._bindings.move_to_end(chunk_hash)
        if len(self._bindings) <= _TOKEN_BINDING_CACHE_SIZE:
            return
        # Trim in one batch down to half the bound, so trimming stays rare.
        evicted = 0
        while len(self._bindings) > _TOKEN_BINDING_CACHE_SIZE // 2:
            self._bindings.popitem(last=False)
            evicted += 1
        logger.warning(
            "KV event token binding cache hit its %d-entry bound: evicted the "
            "%d oldest bindings; stores of those chunks completing from now "
            "on are not reported as KV events",
            _TOKEN_BINDING_CACHE_SIZE,
            evicted,
        )

    def _on_l1_stored(self, event: Event) -> None:
        self._record_stored(KV_EVENT_MEDIUM_CPU, event.metadata["keys"])

    def _on_l1_removed(self, event: Event) -> None:
        self._record_removed(KV_EVENT_MEDIUM_CPU, event.metadata["keys"])

    def _on_l2_stored(self, event: Event) -> None:
        self._record_stored(KV_EVENT_MEDIUM_STORAGE, event.metadata["keys"])

    def _on_l2_removed(self, event: Event) -> None:
        self._record_removed(KV_EVENT_MEDIUM_STORAGE, event.metadata["keys"])

    # -- Internals -------------------------------------------------------------

    def _note_dropped_events(self) -> None:
        self._log.note_dropped_events(self._bus.dropped_events_count())

    def _record_stored(self, medium: str, keys: list[ObjectKey]) -> None:
        """Append one stored record per distinct chunk in ``keys``.

        Every KV rank and object group has its own key for a chunk; one
        record per chunk is enough because a router tracks chunks, not
        shards.
        """
        self._note_dropped_events()
        seen: set[tuple[str, bytes]] = set()
        for key in keys:
            identity = (key.model_name, key.chunk_hash)
            if identity in seen:
                continue
            seen.add(identity)
            binding = self._bindings.get(key.chunk_hash)
            if binding is None:
                self._unbound_stores += 1
                continue
            self._log.append(
                kind=KV_EVENT_KIND_STORED,
                medium=medium,
                model_name=key.model_name,
                block_hashes=[key.chunk_hash],
                parent_block_hash=binding.parent_hash,
                token_ids=list(binding.token_ids),
                block_size=self._chunk_size,
            )

    def _record_removed(self, medium: str, keys: list[ObjectKey]) -> None:
        """Append one removed record per model covering the distinct chunks
        in ``keys``."""
        self._note_dropped_events()
        by_model: dict[str, list[bytes]] = {}
        seen: set[tuple[str, bytes]] = set()
        for key in keys:
            identity = (key.model_name, key.chunk_hash)
            if identity in seen:
                continue
            seen.add(identity)
            by_model.setdefault(key.model_name, []).append(key.chunk_hash)
        for model_name, block_hashes in by_model.items():
            self._log.append(
                kind=KV_EVENT_KIND_REMOVED,
                medium=medium,
                model_name=model_name,
                block_hashes=block_hashes,
                block_size=self._chunk_size,
            )


class KVEventModule:
    """Engine module serving ``POLL_KV_EVENTS`` from a bus-fed event log.

    Disabled (every poll answers ``enabled=False``) when ``log_size`` is 0 or
    the observability event bus is off, since no cache event would reach the
    log then.

    Args:
        ctx: The shared engine context.
        log_size: Number of records the log retains; 0 disables the channel.

    Raises:
        ValueError: If ``log_size`` is negative.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        log_size: int = DEFAULT_KV_EVENT_LOG_SIZE,
    ) -> None:
        if log_size < 0:
            raise ValueError(f"log_size must be >= 0 (got {log_size})")
        self._ctx = ctx
        # Distinguishes this server process's log from its predecessors'.
        self._incarnation = time.time_ns()
        self._log: KVEventLog | None = None
        self._subscriber: KVEventSubscriber | None = None
        if log_size == 0:
            logger.info("KV event channel disabled (--kv-event-log-size 0)")
            return
        if not ctx.event_bus.enabled:
            logger.warning(
                "KV event channel disabled: the observability event bus is off "
                "(--disable-observability), so engine workers cannot learn "
                "host-cache evictions for KV-aware routing"
            )
            return
        log = KVEventLog(log_size)
        self._log = log
        self._subscriber = KVEventSubscriber(log, ctx.event_bus, ctx.chunk_size)
        ctx.event_bus.register_subscriber(self._subscriber)
        register_gauge(
            "lmcache.kv_events",
            "lmcache_mp.kv_events.log_depth",
            "Cache-event records retained for engine workers to poll.",
            lambda: len(log),
        )
        logger.info(
            "KV event channel enabled: retaining %d cache-event records", log_size
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    @property
    def enabled(self) -> bool:
        """Whether the server records cache events for polling."""
        return self._log is not None

    @property
    def incarnation(self) -> int:
        """Identity of this server process's event log."""
        return self._incarnation

    def report_status(self) -> dict:
        """Return the channel's state under the ``kv_events`` key.

        Returns:
            A dict with ``enabled``, ``incarnation``, and, when enabled,
            ``log_depth``, ``log_capacity``, ``next_seq``, ``lost_markers``,
            and ``unbound_stores``.
        """
        status: dict = {"enabled": self.enabled, "incarnation": self._incarnation}
        if self._log is not None and self._subscriber is not None:
            status.update(
                {
                    "log_depth": len(self._log),
                    "log_capacity": self._log.capacity,
                    "next_seq": self._log.next_seq,
                    "lost_markers": self._log.lost_markers,
                    "unbound_stores": self._subscriber.unbound_stores,
                }
            )
        return {"kv_events": status}

    def close(self) -> None:
        """Nothing to release: the bus owns the subscriber's lifetime."""
        return

    @request_handler(RequestType.POLL_KV_EVENTS)
    def poll_kv_events(
        self, model_name: str, cursor: int, max_events: int
    ) -> KVEventPollResult:
        """Return the cache-event records after ``cursor`` for one model.

        Args:
            model_name: Only records for this model's keys are returned.
            cursor: Sequence number of the last record the caller consumed,
                or 0 on first contact.
            max_events: Maximum number of records returned (>= 1).

        Returns:
            The poll result; see :class:`KVEventPollResult`. ``enabled`` is
            ``False`` when the channel is disabled.

        Raises:
            ValueError: If ``cursor`` is negative or ``max_events`` is not
                positive.
        """
        if self._log is None:
            return KVEventPollResult(
                enabled=False,
                incarnation=self._incarnation,
                next_cursor=0,
                lost=False,
                events=[],
            )
        # Losses are detected here too, so a dropped eviction with no later
        # bus traffic still reaches the next poll.
        self._log.note_dropped_events(self._ctx.event_bus.dropped_events_count())
        records, next_cursor, lost = self._log.read_after(
            cursor, model_name, max_events
        )
        return KVEventPollResult(
            enabled=True,
            incarnation=self._incarnation,
            next_cursor=next_cursor,
            lost=lost,
            events=records,
        )
