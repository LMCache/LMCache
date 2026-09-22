# SPDX-License-Identifier: Apache-2.0
"""Management operations and CPU KV-event subscriptions for the MP server."""

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
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_CAPABILITY,
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    BlockAllocationRecord,
    KVEventBatch,
    KVEventRecord,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

_KIND_LOST = "lost"
_TOKEN_BINDING_CACHE_SIZE = 65536


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
        kv_event_log_size: Cache-event records retained for subscribers and
            reconnects; 0 disables the KV event channel.

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
        self._build_kv_event_log(kv_event_log_size)

        # Periodic reaper, started only when reaping is enabled and there is
        # something to scan. Scans every reap_timeout/4, so an instance is
        # reaped between timeout and timeout + interval after its last signal.
        self._reaper: PeriodicThread | None = None
        if self._reap_timeout > 0 and (
            self._liveness_targets or self._kv_events_enabled
        ):
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
        """Stop the reaper and wake all outstanding event subscriptions."""
        if self._reaper is not None:
            self._reaper.stop()
        with self._kv_event_lock:
            for stream, _ in list(self._kv_event_streams.values()):
                stream.close()

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
            with self._kv_event_lock:
                if instance_id in self._kv_event_streams:
                    stream, _ = self._kv_event_streams[instance_id]
                    self._kv_event_streams[instance_id] = (stream, time.monotonic())
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
        with self._kv_event_lock:
            now = time.monotonic()
            for instance_id, (stream, seen) in list(self._kv_event_streams.items()):
                if instance_id in reaped or now - seen > self._reap_timeout:
                    stream.close()
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
        if self._kv_events_enabled:
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
    def subscribe_kv_events(
        self, instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]:
        """Subscribe to CPU changes, replaying records after cursor.

        Sends an initial status batch, then waits for matching events or loss.
        Each worker instance owns at most one subscription. Closing it or
        reaping the worker wakes its reader. Raises ValueError for invalid
        cursor/page size. Slow subscribers share the bounded replay log.
        """
        if cursor < 0 or not 0 < max_events <= 1024:
            raise ValueError("cursor must be >= 0 and max_events must be in [1, 1024]")
        first = True

        def read(closed: threading.Event) -> KVEventBatch:
            nonlocal cursor, first
            with self._kv_event_lock:
                while not closed.is_set():
                    batch = self.read_kv_events(model_name, cursor, max_events)
                    cursor = batch.next_cursor
                    if not first and not batch.enabled:
                        raise StopIteration
                    if first or batch.events or batch.lost or not batch.enabled:
                        first = False
                        return batch
                    self._kv_event_lock.wait()
            raise StopIteration

        def cancel() -> None:
            with self._kv_event_lock:
                current = self._kv_event_streams.get(instance_id)
                if current is not None and current[0] is stream:
                    del self._kv_event_streams[instance_id]
                self._kv_event_lock.notify_all()

        stream = MessagingStream(read, cancel)
        with self._kv_event_lock:
            previous = self._kv_event_streams.get(instance_id)
            if previous is not None:
                previous[0].close()
            self._kv_event_streams[instance_id] = (stream, time.monotonic())
        return stream

    def read_kv_events(
        self, model_name: str, cursor: int, max_events: int
    ) -> KVEventBatch:
        """Read up to max_events records for model_name after cursor.

        Returns a restart/loss marker and the next cursor in KVEventBatch.
        Raises ValueError for a negative cursor or nonpositive page size.
        Disabled channels return enabled=False and no events.
        """
        if cursor < 0 or max_events <= 0:
            raise ValueError("cursor must be >= 0 and max_events must be positive")
        with self._kv_event_lock:
            records: list[KVEventRecord] = []
            next_cursor, lost = 0, False
            if self._kv_events_enabled:
                last = self._kv_event_next_seq - 1
                first = self._kv_event_log[0].seq if self._kv_event_log else last + 1
                lost = cursor + 1 < first or cursor > last
                next_cursor = min(cursor, last)
                for record in islice(
                    self._kv_event_log, max(0, cursor + 1 - first), None
                ):
                    next_cursor = record.seq
                    if record.kind == _KIND_LOST:
                        lost = True
                        records.clear()
                    elif record.model_name == model_name:
                        records.append(record)
                        if len(records) == max_events:
                            break
            return KVEventBatch(
                enabled=self._kv_events_enabled,
                incarnation=self._kv_event_incarnation,
                next_cursor=next_cursor,
                lost=lost,
                events=records,
            )

    # Helper functions

    def _build_kv_event_log(self, log_size: int) -> None:
        """Subscribe to completed CPU stores and evictions when enabled."""
        self._kv_events_enabled = log_size > 0 and bool(self._ctx.event_bus.enabled)
        self._kv_event_log: deque[KVEventRecord] = deque(maxlen=log_size)
        self._kv_event_lock = threading.Condition()
        self._kv_event_streams: dict[
            int, tuple[MessagingStream[KVEventBatch], float]
        ] = {}
        self._kv_event_incarnation = time.time_ns()
        self._kv_event_next_seq = 1
        self._kv_event_dropped = self._kv_event_lost_markers = 0
        self._kv_event_unbound_stores = 0
        self._kv_event_bindings: OrderedDict[
            bytes, tuple[tuple[int, ...], bytes | None]
        ] = OrderedDict()
        if not self._kv_events_enabled:
            return
        self._ctx.event_bus.subscribe_drops(self._record_kv_event_loss)
        for event_type in (
            EventType.MP_TOKENS,
            EventType.L1_WRITE_FINISHED,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
            EventType.L1_KEYS_EVICTED,
        ):
            self._ctx.event_bus.subscribe(event_type, self._record_kv_event)
        register_gauge(
            "lmcache.kv_events",
            "lmcache_mp.kv_events.log_depth",
            "Cache-event records retained for subscribers and reconnects.",
            lambda: len(self._kv_event_log),
        )

    def _append_kv_event(
        self,
        kind: str,
        model_name: str,
        hashes: list[bytes],
        parent: bytes | None = None,
        tokens: Sequence[int] = (),
    ) -> None:
        """Append a sequenced record while holding the event lock."""
        self._kv_event_log.append(
            KVEventRecord(
                seq=self._kv_event_next_seq,
                kind=kind,
                medium=KV_EVENT_MEDIUM_CPU,
                model_name=model_name,
                block_hashes=hashes,
                parent_block_hash=parent,
                token_ids=list(tokens),
            )
        )
        self._kv_event_next_seq += 1
        self._kv_event_lock.notify_all()

    def _record_kv_event_loss(self) -> None:
        """Wake subscribers even when the final eviction was dropped."""
        with self._kv_event_lock:
            dropped = self._ctx.event_bus.dropped_events_count()
            if dropped > self._kv_event_dropped:
                self._kv_event_dropped = dropped
                self._kv_event_lost_markers += 1
                self._append_kv_event(_KIND_LOST, "", [])

    def _record_kv_event(self, event: Event) -> None:
        """Translate bus events to ordered records; token bindings precede writes."""
        with self._kv_event_lock:
            if event.event_type == EventType.MP_TOKENS:
                for chunk_hash, tokens, parent in zip(
                    event.metadata["chunk_hashes"],
                    event.metadata["token_chunks"],
                    event.metadata["parent_hashes"],
                    strict=True,
                ):
                    self._kv_event_bindings[chunk_hash] = (tuple(tokens), parent)
                    self._kv_event_bindings.move_to_end(chunk_hash)
                if len(self._kv_event_bindings) > _TOKEN_BINDING_CACHE_SIZE:
                    while len(self._kv_event_bindings) > _TOKEN_BINDING_CACHE_SIZE // 2:
                        self._kv_event_bindings.popitem(last=False)
                return
            keys: list[ObjectKey] = event.metadata["keys"]
            by_model: dict[str, dict[bytes, None]] = {}
            for key in keys:
                by_model.setdefault(key.model_name, {})[key.chunk_hash] = None
            for model_name, hashes in by_model.items():
                if event.event_type == EventType.L1_KEYS_EVICTED:
                    self._append_kv_event(
                        KV_EVENT_KIND_REMOVED, model_name, list(hashes)
                    )
                    continue
                for chunk_hash in hashes:
                    binding = self._kv_event_bindings.get(chunk_hash)
                    if binding is None:
                        self._kv_event_unbound_stores += 1
                        continue
                    tokens, parent = binding
                    self._append_kv_event(
                        KV_EVENT_KIND_STORED,
                        model_name,
                        [chunk_hash],
                        parent,
                        tokens,
                    )

    def _kv_event_status(self) -> dict:
        """Return a consistent channel snapshot for report_status."""
        with self._kv_event_lock:
            return {
                "enabled": self._kv_events_enabled,
                "incarnation": self._kv_event_incarnation,
                "log_depth": len(self._kv_event_log),
                "log_capacity": self._kv_event_log.maxlen,
                "next_seq": self._kv_event_next_seq,
                "lost_markers": self._kv_event_lost_markers,
                "unbound_stores": self._kv_event_unbound_stores,
                "subscribers": len(self._kv_event_streams),
            }
