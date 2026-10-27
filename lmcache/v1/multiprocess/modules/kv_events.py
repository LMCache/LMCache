# SPDX-License-Identifier: Apache-2.0
"""CPU KV-event subscriptions and replay for the MP server."""

# Standard
from collections import deque
from dataclasses import dataclass
from itertools import islice
import threading
import time

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_CAPABILITY,
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_SCHEMA_VERSION,
    EventBinding,
    IPCCacheServerKey,
    KVEventBatch,
    KVEventProfile,
    KVEventRecord,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
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
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE, unpack_token_ids

# Context includes TP layout and salt even though salt is unsupported initially.
ChunkIdentity = tuple[str, int, str, bytes]


@dataclass
class _BindingState:
    descriptor: EventBinding
    pending: set[str]
    advertised: bool = False


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
        module = cls(
            build_ctx.engine_context,
            kv_event_log_size=build_ctx.mp_config.kv_event_log_size,
        )
        if module.enabled:
            build_ctx.add_experimental(KV_EVENT_CAPABILITY)
        return module

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
        self._kv_event_bindings: dict[ChunkIdentity, _BindingState] = {}
        self._profiles: dict[
            tuple[str, int], tuple[KVEventProfile, frozenset[tuple[int, int]]]
        ] = {}
        self._owners: dict[ObjectKey, set[str]] = {}
        self._pending: dict[str, set[ChunkIdentity]] = {}
        self._ctx.kv_event_module = self
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

    def report_status(self) -> dict[str, dict[str, int | bool | None]]:
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
                    "unbound_stores": 0,
                    "subscribers": len(self._kv_event_streams),
                    "bindings": len(self._kv_event_bindings),
                    "live_objects": len(self._owners),
                    "pending_stores": len(self._pending),
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
        self,
        instance_id: int,
        model_name: str,
        cursor: int,
        max_events: int,
        world_size: int,
        profile: KVEventProfile,
    ) -> MessagingStream[KVEventBatch]:
        """Subscribe to CPU changes, replaying records after cursor.

        Sends an initial status batch, then waits for matching events or loss.
        Each worker instance owns at most one subscription. Closing it or
        reaping the worker wakes its reader. Raises ValueError for invalid
        cursor/page size. Slow subscribers share the bounded replay log.
        """
        if cursor < 0 or not 0 < max_events <= 1024:
            raise ValueError("cursor must be >= 0 and max_events must be in [1, 1024]")
        if (
            self.enabled
            and self._ctx.layout_desc_registry.find(model_name, world_size) is None
        ):
            raise ConnectionError("waiting for engine KV cache re-registration")
        self.register_profile(model_name, world_size, profile)
        first = True

        def read(closed: threading.Event) -> KVEventBatch:
            nonlocal cursor, first
            with self._kv_event_lock:
                while not closed.is_set():
                    batch = self.read_kv_events(
                        model_name, cursor, max_events, world_size
                    )
                    cursor = batch.next_cursor
                    if first or batch.events or batch.lost:
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
        self, model_name: str, cursor: int, max_events: int, world_size: int = 1
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
                next_cursor = max(first - 1, min(cursor, last))
                for record in islice(
                    self._kv_event_log, max(0, cursor + 1 - first), None
                ):
                    next_cursor = record.seq
                    if (
                        record.model_name == model_name
                        and record.world_size == world_size
                    ):
                        records.append(record)
                        if len(records) == max_events:
                            break
            return KVEventBatch(
                enabled=self._kv_events_enabled,
                incarnation=self._kv_event_incarnation,
                next_cursor=next_cursor,
                lost=lost,
                events=records,
                schema_version=KV_EVENT_SCHEMA_VERSION,
            )

    def register_profile(
        self,
        model_name: str,
        world_size: int,
        profile: KVEventProfile,
        *,
        attn_desc: AttnWindowDesc | None = None,
    ) -> None:
        """Validate one immutable profile against registered dense KV geometry.

        Args:
            model_name: Existing registered model namespace.
            world_size: Positive KV rank count from engine registration.
            profile: Engine-native identity semantics and block granularity.
            attn_desc: Actual candidate GPU layout during registration, before
                mutating the shared registry; None reads the registered layout.

        Raises:
            ValueError: Events are disabled, the layout is unsupported, or a
                different profile was already registered for this context.
        """
        if not self.enabled:
            raise ValueError("native KV events are disabled on this MP server")
        self._ctx.validate_native_event_storage()
        if self._ctx.chunk_size % profile.block_size:
            raise ValueError(
                "LMCache chunk size must be a multiple of native block size"
            )
        attn = (
            attn_desc
            if attn_desc is not None
            else self._ctx.layout_desc_registry.find_attn_desc(model_name, world_size)
        )
        if (
            world_size < 1
            or attn.num_object_groups == 0
            or any(w != -1 for w in attn.num_chunks_in_sw)
        ):
            raise ValueError(
                "native KV events require dense full attention "
                "and a positive KV world size"
            )
        if any(kind != "attention" for kind in attn.group_kinds):
            raise ValueError(
                "native KV events do not support recurrent or auxiliary groups"
            )
        required = frozenset(
            (ObjectKey.ComputeKVRank(world_size, rank, world_size, rank), group)
            for rank in range(world_size)
            for group in range(attn.num_object_groups)
        )
        context = (model_name, world_size)
        with self._kv_event_lock:
            candidate = (profile, required)
            if self._profiles.setdefault(context, candidate) != candidate:
                raise ValueError(
                    "native KV event profile or registered layout mismatch"
                )

    def bind_store(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey], transfer_key: str
    ) -> None:
        """Validate STORE metadata synchronously, then join it through EventBus.

        Args:
            key: STORE key with one native binding per complete chunk.
            obj_keys: Resolved object keys for the first object group.
            transfer_key: Process-unique transfer ID, used for pending cleanup.

        Raises:
            ValueError: Missing/incompatible profile, salted input, geometry or
                token mismatch, or conflicting descriptor for a live identity.

        A binding is never proof of residency. Pending entries are retained until
        the corresponding storage completion callback; live entries survive
        request cleanup.
        """
        if key.event_profile is None:
            return
        if key.cache_salt:
            raise ValueError("native KV events do not support cache_salt")
        self.register_profile(key.model_name, key.world_size, key.event_profile)
        if key.start % self._ctx.chunk_size or len(obj_keys) != len(key.event_bindings):
            raise ValueError(
                "STORE requires one native binding per complete LMCache chunk"
            )
        identities: list[ChunkIdentity] = []
        with self._kv_event_lock:
            # Validate the full request before changing any binding.
            for index, (obj_key, binding) in enumerate(
                zip(obj_keys, key.event_bindings, strict=True)
            ):
                start = key.start + index * self._ctx.chunk_size
                if (
                    binding.profile_id != key.event_profile.profile_id
                    or binding.block_size != key.event_profile.block_size
                    or len(binding.token_ids) != self._ctx.chunk_size
                    or binding.token_ids
                    != tuple(
                        unpack_token_ids(
                            key.token_bytes[
                                start * TOKEN_STRIDE : (start + self._ctx.chunk_size)
                                * TOKEN_STRIDE
                            ]
                        )
                    )
                ):
                    raise ValueError(
                        "native STORE binding profile, geometry or tokens mismatch"
                    )
                identity: ChunkIdentity = (
                    key.model_name,
                    key.world_size,
                    key.cache_salt,
                    obj_key.chunk_hash,
                )
                current = self._kv_event_bindings.get(identity)
                if current is not None and current.descriptor != binding:
                    raise ValueError("conflicting native KV event binding")
                identities.append(identity)
            for identity, binding in zip(identities, key.event_bindings, strict=True):
                state = self._kv_event_bindings.setdefault(
                    identity, _BindingState(binding, set())
                )
                state.pending.add(transfer_key)
                self._pending.setdefault(transfer_key, set()).add(identity)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [],
                    "token_chunks": [],
                    "token_offsets": [],
                    "parent_hashes": [],
                    "native_identities": tuple(identities),
                },
            )
        )

    def _record_kv_event_loss(self) -> None:
        """Invalidate completeness as well as subscribers after EventBus loss."""
        with self._kv_event_lock:
            dropped = self._ctx.event_bus.dropped_events_count()
            if dropped > self._kv_event_dropped:
                self._kv_event_dropped = dropped
                self._kv_event_lost_markers += 1
                # Drop callbacks run after the bus drains its preceding queue.
                # None of those pre-loss writes can prove post-loss residency.
                self._owners.clear()
                self._kv_event_bindings.clear()
                self._pending.clear()
                # Advancing the empty log's floor reports loss even without a
                # later write; pre-loss pages can no longer be replayed.
                self._kv_event_log.clear()
                self._kv_event_next_seq += 1
                self._kv_event_lock.notify_all()

    def _record_kv_event(self, event: Event) -> None:
        """Join native metadata with complete committed rank/group/copy state."""
        with self._kv_event_lock:
            if self._ctx.event_bus.dropped_events_count() > self._kv_event_dropped:
                # The after-drain loss barrier will clear this projection. A
                # dropped removal may already have made its owners stale.
                return
            if event.event_type == EventType.MP_TOKENS:
                for identity in event.metadata.get("native_identities", ()):
                    self._update_chunk(identity)
                pending = event.metadata.get("native_store_finished")
                if pending is not None:
                    for identity in self._pending.pop(pending, ()):
                        state = self._kv_event_bindings.get(identity)
                        if state is not None:
                            state.pending.discard(pending)
                            self._update_chunk(identity)
                return
            keys: list[ObjectKey] = event.metadata["keys"]
            owner: str = event.metadata["l1_tag"]
            for key in keys:
                world_size = ObjectKey.WorldSizeFromKVRank(key.kv_rank)
                identity = (key.model_name, world_size, key.cache_salt, key.chunk_hash)
                if event.event_type == EventType.L1_KEYS_EVICTED:
                    owners = self._owners.get(key)
                    if owners is not None:
                        owners.discard(owner)
                        if not owners:
                            del self._owners[key]
                else:
                    self._owners.setdefault(key, set()).add(owner)
                self._update_chunk(identity)

    def _update_chunk(self, identity: ChunkIdentity) -> None:
        """Emit only advertisability transitions and collect unused metadata."""
        state = self._kv_event_bindings.get(identity)
        if state is None:
            return
        model, world_size, salt, chunk_hash = identity
        _, required = self._profiles[(model, world_size)]
        resident = sum(
            bool(self._owners.get(ObjectKey(chunk_hash, model, rank, group, salt)))
            for rank, group in required
        )
        ready = resident == len(required)
        if ready != state.advertised:
            binding = state.descriptor
            self._kv_event_log.append(
                KVEventRecord(
                    seq=self._kv_event_next_seq,
                    kind=KV_EVENT_KIND_STORED if ready else KV_EVENT_KIND_REMOVED,
                    medium=KV_EVENT_MEDIUM_CPU,
                    model_name=model,
                    block_hashes=binding.block_hashes,
                    parent_block_hash=binding.parent_block_hash,
                    token_ids=binding.token_ids if ready else (),
                    block_size=binding.block_size,
                    profile_id=binding.profile_id,
                    world_size=world_size,
                    cache_salt=salt,
                    group_idx=binding.group_idx,
                    kv_cache_spec_kind=binding.kv_cache_spec_kind,
                )
            )
            self._kv_event_next_seq += 1
            self._kv_event_lock.notify_all()
            state.advertised = ready
        if not state.pending and not resident:
            del self._kv_event_bindings[identity]
