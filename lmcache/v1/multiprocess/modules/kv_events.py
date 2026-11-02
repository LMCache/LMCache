# SPDX-License-Identifier: Apache-2.0
"""CPU KV-event subscriptions and replay for the MP server."""

# Standard
from collections import OrderedDict, deque
from collections.abc import Sequence
from itertools import islice
import threading
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KVEventBatch,
    KVEventRecord,
)
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    InstanceLivenessTarget,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import request_handler

_KIND_LOST = "lost"
_TOKEN_BINDING_CACHE_SIZE = 65536
_ChunkKey = tuple[str, int, str, bytes]


def _chunk_key(key: ObjectKey) -> _ChunkKey:
    """Identify content independently of its required rank and object group."""
    return (
        key.model_name,
        ObjectKey.WorldSizeFromKVRank(key.kv_rank),
        key.cache_salt,
        key.chunk_hash,
    )


class CommittedKVResidency:
    """Project committed L1 copies onto complete chunks.

    The owning KVEventModule serializes every call under its event lock.
    Owners are retained only while their ObjectKey has a readable copy.

    Args:
        layouts: Registered geometry supplying every required object group;
            ObjectKey supplies the complete KV world size.
    """

    def __init__(self, layouts: LayoutDescRegistry) -> None:
        self._layouts = layouts
        self._owners: dict[ObjectKey, set[str]] = {}

    def commit(self, key: ObjectKey, l1_tag: str) -> None:
        """Record a completed readable copy; duplicate notifications are idempotent.

        Args:
            key: The committed object, including rank, group, and cache salt.
            l1_tag: The L1 manager that completed the write.
        """
        self._owners.setdefault(key, set()).add(l1_tag)

    def evict(self, key: ObjectKey, l1_tag: str) -> None:
        """Forget one L1 copy, removing owner state when its final copy is gone.

        Args:
            key: The object removed from L1.
            l1_tag: The L1 manager that removed it. Unknown copies are ignored.
        """
        owners = self._owners.get(key)
        if owners is not None:
            owners.discard(l1_tag)
            if not owners:
                del self._owners[key]

    def is_committed(self, key: ObjectKey) -> bool:
        """Return whether every required rank/group has a readable L1 copy.

        Args:
            key: Any object identifying the chunk's model, world, salt, and hash.

        Returns:
            False for missing/empty registered geometry or an incomplete chunk.
            Different L1 tags may supply different required objects.
        """
        world_size = ObjectKey.WorldSizeFromKVRank(key.kv_rank)
        groups = self._layouts.find_group_layout_descs(key.model_name, world_size)
        if not world_size or not groups:
            return False
        return all(
            ObjectKey(
                key.chunk_hash,
                key.model_name,
                ObjectKey.ComputeKVRank(world_size, rank, world_size, rank),
                group,
                key.cache_salt,
            )
            in self._owners
            for rank in range(world_size)
            for group in groups
        )

    def clear(self) -> None:
        """Discard all ownership after event loss makes the projection unreliable."""
        self._owners.clear()


class KVEventModule(DiscoverableModule, InstanceLivenessTarget):
    """Own CPU event records, token bindings, and worker subscriptions.

    Args:
        ctx: Shared engine context providing the event bus.
        kv_event_log_size: Records retained for replay; zero disables the channel.

    Raises:
        ValueError: If kv_event_log_size is negative.

    The existing management reaper refreshes and expires subscriptions through
    InstanceLivenessTarget. Subscription expiry does not reap transfer state.
    """

    module_name = "kv_events"
    # Build after both transfer classes so reverse close order stops event
    # subscriptions before any enabled transfer module releases its state.
    module_dependencies = (
        LMCacheDrivenTransferModule,
        EngineDrivenTransferModule,
    )

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the event channel; a disabled channel still serves subscribers."""
        return cls(
            build_ctx.engine_context,
            kv_event_log_size=build_ctx.mp_config.kv_event_log_size,
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        kv_event_log_size: int = MPServerConfig.kv_event_log_size,
    ) -> None:
        if kv_event_log_size < 0:
            raise ValueError(
                f"kv_event_log_size must be >= 0 (got {kv_event_log_size})"
            )
        self._ctx = ctx
        self._kv_events_enabled = kv_event_log_size > 0 and bool(
            self._ctx.event_bus.enabled
        )
        self._kv_event_log: deque[KVEventRecord] = deque(maxlen=kv_event_log_size)
        self._kv_event_lock = threading.Condition()
        self._kv_event_streams: dict[
            int, tuple[MessagingStream[KVEventBatch], float]
        ] = {}
        self._kv_event_incarnation = time.time_ns()
        self._kv_event_next_seq = 1
        self._kv_event_dropped = self._kv_event_lost_markers = 0
        self._discard_until_empty = False
        self._kv_event_unbound_stores = 0
        self._kv_event_bindings: OrderedDict[
            bytes, tuple[tuple[int, ...], bytes | None]
        ] = OrderedDict()
        self._residency = CommittedKVResidency(ctx.layout_desc_registry)
        self._announced: dict[_ChunkKey, int] = {}
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

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    @property
    def enabled(self) -> bool:
        """Whether the server records events and can advertise the channel."""
        return self._kv_events_enabled

    def report_status(self) -> dict:
        """Return a consistent snapshot under the existing kv_events status key."""
        with self._kv_event_lock:
            return {
                "kv_events": {
                    "enabled": self._kv_events_enabled,
                    "incarnation": self._kv_event_incarnation,
                    "log_depth": len(self._kv_event_log),
                    "log_capacity": self._kv_event_log.maxlen,
                    "next_seq": self._kv_event_next_seq,
                    "lost_markers": self._kv_event_lost_markers,
                    "unbound_stores": self._kv_event_unbound_stores,
                    "subscribers": len(self._kv_event_streams),
                }
            }

    def close(self) -> None:
        """Close all subscriptions and wake their readers."""
        with self._kv_event_lock:
            for stream, _ in list(self._kv_event_streams.values()):
                stream.close()

    def touch_instance(self, instance_id: int) -> None:
        """Refresh an existing subscription's last-seen time on worker PING."""
        with self._kv_event_lock:
            if instance_id in self._kv_event_streams:
                stream, _ = self._kv_event_streams[instance_id]
                self._kv_event_streams[instance_id] = (stream, time.monotonic())

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Close subscriptions silent for reap_timeout_s seconds; zero disables it.

        Return no worker IDs: subscription expiry must not reap cache registrations.
        Transfer modules own registration_grace_s and worker-registration expiry.
        """
        if reap_timeout_s > 0:
            with self._kv_event_lock:
                now = time.monotonic()
                for stream, seen in list(self._kv_event_streams.values()):
                    if now - seen > reap_timeout_s:
                        stream.close()
        return []

    def drop_instance_state(self, instance_id: int) -> None:
        """Close the subscription when a transfer module reaps instance_id."""
        with self._kv_event_lock:
            current = self._kv_event_streams.get(instance_id)
            if current is not None:
                current[0].close()

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
                self._residency.clear()
                self._announced.clear()
                self._append_kv_event(_KIND_LOST, "", [])
            # Read the drop count before the queue: later overflow stays detectable.
            self._discard_until_empty = self._ctx.event_bus.queue_depth() > 0

    def _record_kv_event(self, event: Event) -> None:
        """Translate bus events to ordered records; token bindings precede writes."""
        with self._kv_event_lock:
            if (
                self._discard_until_empty
                or self._ctx.event_bus.dropped_events_count() > self._kv_event_dropped
            ):
                self._record_kv_event_loss()
                return
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
            l1_tag: str = event.metadata["l1_tag"]
            evicted = event.event_type == EventType.L1_KEYS_EVICTED
            chunks: dict[_ChunkKey, ObjectKey] = {}
            for key in keys:
                if evicted:
                    self._residency.evict(key, l1_tag)
                else:
                    self._residency.commit(key, l1_tag)
                chunks[_chunk_key(key)] = key
            for chunk, key in chunks.items():
                announced = self._announced.get(chunk)
                if not self._residency.is_committed(key):
                    if announced is not None:
                        del self._announced[chunk]
                        self._append_kv_event(
                            KV_EVENT_KIND_REMOVED, key.model_name, [key.chunk_hash]
                        )
                    continue
                if evicted:
                    continue
                # Refresh aged-out records on a new commit so replay can recover.
                if announced is not None and announced >= self._kv_event_log[0].seq:
                    continue
                binding = self._kv_event_bindings.get(key.chunk_hash)
                if binding is None:
                    self._kv_event_unbound_stores += 1
                    continue
                tokens, parent = binding
                self._announced[chunk] = self._kv_event_next_seq
                self._append_kv_event(
                    KV_EVENT_KIND_STORED,
                    key.model_name,
                    [key.chunk_hash],
                    parent,
                    tokens,
                )
