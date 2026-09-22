# SPDX-License-Identifier: Apache-2.0
"""Management and utility operations for the MPCacheServer.

Also hosts the server-side KV event channel. The storage layer publishes
key-level cache events on the observability bus (L1 write completions and
evictions, L2 stores and deletes), but only the MP server sees them. They
are recorded here in a bounded, sequenced log that engine workers read with
``POLL_KV_EVENTS`` and republish through their framework's KV event stream
(vLLM ``BlockStored`` / ``BlockRemoved``), so a KV-aware router learns the
host-cache placement changes that happen inside the server. See
``docs/design/v1/multiprocess/modules/kv_events.md``.
"""

# Standard
from collections import OrderedDict, deque
from collections.abc import Sequence
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
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_CAPABILITY,
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_MEDIUM_STORAGE,
    BlockAllocationRecord,
    KVEventPollResult,
    KVEventRecord,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

# Internal record kind: the event bus discarded events, so records before the
# marker may be missing. Never returned to clients; ``read_after`` reports it
# as ``lost`` and drops the records preceding it.
_KIND_LOST = "lost"

# Bounds the chunk hash -> token binding cache that stamps stored records;
# covers the window between a chunk's token-binding event and its (possibly
# asynchronous L2) store event.
_TOKEN_BINDING_CACHE_SIZE = 65536


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
    ) -> KVEventRecord:
        """Append one record, assigning it the next sequence number.

        Args:
            kind: ``KV_EVENT_KIND_STORED`` or ``KV_EVENT_KIND_REMOVED``.
            medium: The chunks' cache medium.
            model_name: The chunks' model.
            block_hashes: The chunk hashes the record covers.
            parent_block_hash: The preceding chunk's hash (stored records).
            token_ids: The chunk's tokens (stored records); ``None`` for none.

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
    """

    def __init__(self, log: KVEventLog, bus: EventBus) -> None:
        self._log = log
        self._bus = bus
        self._bindings: OrderedDict[bytes, tuple[tuple[int, ...], bytes | None]] = (
            OrderedDict()
        )
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
        parent_hashes: list[bytes | None] = event.metadata["parent_hashes"]
        for chunk_hash, chunk, parent_hash in zip(
            chunk_hashes, token_chunks, parent_hashes, strict=True
        ):
            self._bindings[chunk_hash] = (tuple(chunk), parent_hash)
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
        for model_name, chunk_hash in dict.fromkeys(
            (key.model_name, key.chunk_hash) for key in keys
        ):
            binding = self._bindings.get(chunk_hash)
            if binding is None:
                self._unbound_stores += 1
                continue
            token_ids, parent_hash = binding
            self._log.append(
                kind=KV_EVENT_KIND_STORED,
                medium=medium,
                model_name=model_name,
                block_hashes=[chunk_hash],
                parent_block_hash=parent_hash,
                token_ids=list(token_ids),
            )

    def _record_removed(self, medium: str, keys: list[ObjectKey]) -> None:
        """Append one removed record per model covering the distinct chunks
        in ``keys``."""
        self._note_dropped_events()
        by_model: dict[str, list[bytes]] = {}
        for model_name, chunk_hash in dict.fromkeys(
            (key.model_name, key.chunk_hash) for key in keys
        ):
            by_model.setdefault(model_name, []).append(chunk_hash)
        for model_name, block_hashes in by_model.items():
            self._log.append(
                kind=KV_EVENT_KIND_REMOVED,
                medium=medium,
                model_name=model_name,
                block_hashes=block_hashes,
            )


class ManagementModule:
    """Handles management and utility operations for the cache engine.

    Owns the lock used during cache clearing and provides handlers for
    ping, chunk-size queries, clear, debug, and block-allocation reporting.
    Also owns the periodic reaper that evicts workers which have gone
    silent, driving the injected liveness targets.

    Args:
        ctx: The shared engine context.
        liveness_targets: Modules the reaper drives -- the transfer modules
            whose per-instance registrations are refreshed on PING and scanned
            for staleness, plus any state mirror (e.g. ``BlendModule``)
            notified via ``drop_instance_state`` when an instance is reaped.
        worker_reap_timeout_seconds: Silence budget for a ping-proven worker;
            0 disables reaping (no thread is started).
        worker_registration_grace_seconds: Silence budget for a worker that
            registered but never pinged.
        experimental_transfer: Types of experimental intermediate tensor
            transfer built in the server.
        kv_event_log_size: Cache-event records retained for engine workers
            to poll; 0 disables the KV event channel.
        advertise_kv_events: Whether the channel may be advertised through
            ``GET_EXPERIMENTAL``. False on a transport whose client cannot
            issue ``POLL_KV_EVENTS`` yet.

    Raises:
        ValueError: If ``kv_event_log_size`` is negative.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        liveness_targets: Sequence[InstanceLivenessTarget] = (),
        worker_reap_timeout_seconds: float = 0.0,
        worker_registration_grace_seconds: float = 0.0,
        experimental_transfer: Sequence[str] = (),
        kv_event_log_size: int = MPServerConfig.kv_event_log_size,
        advertise_kv_events: bool = True,
    ) -> None:
        if kv_event_log_size < 0:
            raise ValueError(
                f"kv_event_log_size must be >= 0 (got {kv_event_log_size})"
            )
        self._ctx = ctx
        self._clear_lock = threading.Lock()
        self._liveness_targets = tuple(liveness_targets)
        self._reap_timeout = worker_reap_timeout_seconds
        self._reap_grace = worker_registration_grace_seconds
        self._experimental_transfer = tuple(experimental_transfer)
        self._advertise_kv_events = advertise_kv_events
        self._build_kv_event_log(kv_event_log_size)

        # Periodic reaper, started only when reaping is enabled and there is
        # something to scan. Scans every reap_timeout/4, so an instance is
        # reaped between timeout and timeout + interval after its last signal.
        self._reaper: PeriodicThread | None = None
        if self._reap_timeout > 0 and self._liveness_targets:
            reaper = create_periodic_thread(
                name="lmcache-mp-worker-reaper",
                interval=self._reap_timeout / 4,
                execute_fn=self._reap_cycle,
                level=ThreadLevel.MEDIUM,
            )
            reaper.start()
            self._reaper = reaper

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict:
        """Return module-specific status information.

        Returns:
            A dict with the ``kv_events`` channel state, plus a
            ``worker_liveness`` summary when reaping targets are present.
        """
        status: dict = {"kv_events": self._kv_event_status()}
        if not self._liveness_targets:
            return status
        tracked = sum(t.tracked_instance_count() for t in self._liveness_targets)
        status["worker_liveness"] = {
            "enabled": self._reaper is not None,
            "reap_timeout_seconds": self._reap_timeout,
            "registration_grace_seconds": self._reap_grace,
            "tracked_instances": tracked,
        }
        return status

    def close(self) -> None:
        """Stop the reaper, if one is running."""
        if self._reaper is not None:
            self._reaper.stop()

    @request_handler(HandlerType.BLOCKING)
    def ping(self, instance_id: int | None) -> bool:
        """Respond to a ping and refresh the sender's liveness.

        Args:
            instance_id: The sender's worker instance ID, or None for an
                untracked prober (the scheduler adapter). When not None, the
                worker's last-seen time is refreshed on every liveness target.

        Returns:
            Always True.
        """
        if instance_id is not None:
            for target in self._liveness_targets:
                target.touch_instance(instance_id)
        return True

    def _reap_cycle(self) -> ThreadRunSummary:
        """Run one reaper scan: reap stale workers, drop mirrored state.

        Each reaped instance id is passed to ``drop_instance_state`` on every
        target; it is a no-op for targets that mirror nothing for that id.

        Returns:
            A summary recording how many instances were reaped this scan.
        """
        reaped: list[int] = []
        for target in self._liveness_targets:
            reaped.extend(
                target.reap_stale_instances(self._reap_timeout, self._reap_grace)
            )
        for instance_id in reaped:
            for target in self._liveness_targets:
                target.drop_instance_state(instance_id)
        return ThreadRunSummary(success=True, message=f"reaped={len(reaped)}")

    @request_handler()
    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations.

        Returns:
            The chunk size.
        """
        return self._ctx.chunk_size

    @request_handler()
    def get_experimental(self) -> list[str]:
        """Return the capabilities this server advertises.

        Returns:
            The enabled experimental intermediate tensor transfer types (see
            ``lmcache.v1.multiprocess.modules.experimental.__init__``) plus
            any advertised feature flag, such as ``KV_EVENT_CAPABILITY`` when
            the server records cache events.
        """
        capabilities = list(self._experimental_transfer)
        if self._kv_event_log is not None and self._advertise_kv_events:
            capabilities.append(KV_EVENT_CAPABILITY)
        return capabilities

    @request_handler(HandlerType.BLOCKING)
    def clear(self, force: bool = False) -> None:
        """Clear all stored KV cache data from the storage manager."""
        with self._clear_lock:
            self._ctx.storage_manager.memcheck()
            self._ctx.storage_manager.clear(force=force)
            self._ctx.storage_manager.memcheck()

    @request_handler(operation="noop")
    def debug(self) -> str:
        """Return a simple health-check string.

        Returns:
            The literal string ``"OK"``.
        """
        return "OK"

    @request_handler(
        HandlerType.BLOCKING,
        operation="report_block_allocation",
    )
    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )

    @request_handler()
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
        if self._kv_event_log is None:
            return KVEventPollResult(
                enabled=False,
                incarnation=self._kv_event_incarnation,
                next_cursor=0,
                lost=False,
                events=[],
            )
        # Losses are detected here too, so a dropped eviction with no later
        # bus traffic still reaches the next poll.
        self._kv_event_log.note_dropped_events(
            self._ctx.event_bus.dropped_events_count()
        )
        records, next_cursor, lost = self._kv_event_log.read_after(
            cursor, model_name, max_events
        )
        return KVEventPollResult(
            enabled=True,
            incarnation=self._kv_event_incarnation,
            next_cursor=next_cursor,
            lost=lost,
            events=records,
        )

    # Helper functions

    def _build_kv_event_log(self, log_size: int) -> None:
        """Start recording cache events, unless the channel cannot run.

        Args:
            log_size: Records to retain; 0 disables the channel.
        """
        # Distinguishes this server process's log from its predecessors'.
        self._kv_event_incarnation = time.time_ns()
        self._kv_event_log: KVEventLog | None = None
        self._kv_event_subscriber: KVEventSubscriber | None = None
        if log_size == 0:
            logger.info("KV event channel disabled (--kv-event-log-size 0)")
            return
        if not self._ctx.event_bus.enabled:
            logger.warning(
                "KV event channel disabled: the observability event bus is off "
                "(--disable-observability), so engine workers cannot learn "
                "host-cache evictions for KV-aware routing"
            )
            return
        log = KVEventLog(log_size)
        self._kv_event_log = log
        self._kv_event_subscriber = KVEventSubscriber(log, self._ctx.event_bus)
        self._ctx.event_bus.register_subscriber(self._kv_event_subscriber)
        register_gauge(
            "lmcache.kv_events",
            "lmcache_mp.kv_events.log_depth",
            "Cache-event records retained for engine workers to poll.",
            lambda: len(log),
        )
        logger.info(
            "KV event channel enabled: retaining %d cache-event records", log_size
        )

    def _kv_event_status(self) -> dict:
        """Return the KV event channel's state for ``report_status``."""
        status: dict = {
            "enabled": self._kv_event_log is not None,
            "incarnation": self._kv_event_incarnation,
        }
        if self._kv_event_log is not None and self._kv_event_subscriber is not None:
            status.update(
                {
                    "log_depth": len(self._kv_event_log),
                    "log_capacity": self._kv_event_log.capacity,
                    "next_seq": self._kv_event_log.next_seq,
                    "lost_markers": self._kv_event_log.lost_markers,
                    "unbound_stores": self._kv_event_subscriber.unbound_stores,
                }
            )
        return status
