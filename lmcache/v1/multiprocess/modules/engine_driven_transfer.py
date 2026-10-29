# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from typing import Any
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    InstanceLivenessTarget,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.periodic_thread import (
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)

_TRANSFER_OBSERVATION_TIMEOUT_SECONDS = 120.0


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
        last_seen: ``time.monotonic()`` of the most recent activity from this
            instance (register, PING, prepare/commit). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window. Latched only by PING.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False


class EngineDrivenTransferModule(DiscoverableModule, InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches. Publishes MP transfer events from successful
    preparation through commit, including worker copies and RPC overhead.
    Owns a periodic task that expires abandoned observations independently of
    worker reaping; ``close`` stops it before releasing module state.

    Args:
        ctx: The shared engine context.
    """

    module_name = "engine_driven_transfer"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build unless the transfer mode excludes the engine-driven path."""
        mode = build_ctx.mp_config.supported_transfer_mode
        if mode == "lmcache_driven":
            return None
        return cls(build_ctx.engine_context)

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._engine_driven_contexts: dict[int, EngineDrivenContextEntry] = {}
        self._strategies: dict[int, TransferStrategy] = {}
        # Guards _engine_driven_contexts and _strategies together (the reaper
        # mutates them off the MQ main loop). Leaf lock, never held with
        # _pending_shm_lock.
        self._lock = threading.Lock()
        self._pending_shm_writes: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_reads: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_lock = threading.Lock()
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info
        # Values retain the START event and monotonic preparation time.
        # Includes request_id because IPCCacheServerKey equality excludes it.
        self._pending_transfers: dict[
            tuple[EventType, int, str, IPCCacheServerKey], tuple[Event, float]
        ] = {}
        self._observation_cleanup = create_periodic_thread(
            name=f"lmcache-mp-transfer-observations-{id(self):x}",
            interval=_TRANSFER_OBSERVATION_TIMEOUT_SECONDS / 4,
            execute_fn=self._cleanup_expired_transfer_observations,
            level=ThreadLevel.MEDIUM,
        )
        self._observation_cleanup.start()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        with self._lock:
            entries = dict(self._engine_driven_contexts)
        for instance_id, entry in entries.items():
            registered_non_cuda_ids.append(instance_id)
            non_cuda_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "block_size": entry.metadata.block_size,
                "use_mla": entry.metadata.use_mla,
            }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
        }

    def close(self) -> None:
        """Stop observation cleanup, release registrations and end observations."""
        self._observation_cleanup.stop()
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()
            pending = list(self._pending_transfers.values())
            self._pending_transfers.clear()
        for event, _ in pending:
            self._publish_transfer_end(event, False)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered non-GPU instances."""
        with self._lock:
            return len(self._engine_driven_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap silent registrations and release their resources.

        A ping-proven instance is judged against ``reap_timeout_s``; one that
        has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, EngineDrivenContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._engine_driven_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                entry = self._engine_driven_contexts.pop(iid)
                self._strategies.pop(iid, None)
                reaped.append((iid, entry))
        for iid, entry in reaped:
            self._release_entry(iid, entry)
            logger.warning(
                "Reaped non-GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
        return [iid for iid, _ in reaped]

    def _resolve_for_transfer(
        self, instance_id: int
    ) -> tuple[EngineDrivenContextEntry, TransferStrategy]:
        """Return (entry, strategy) for a transfer, refreshing last_seen.

        Pair-atomicity guarantees the entry exists whenever the strategy
        does. Refreshes last_seen (no latch) so an active worker is not
        reaped mid-transfer.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry and its transfer strategy.

        Raises:
            ValueError: If the instance is not registered (or was reaped).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            strategy = self._strategies.get(instance_id)
            if entry is None or strategy is None:
                raise ValueError(
                    "non-GPU context not registered (or reaped) for "
                    f"instance ID {instance_id}"
                )
            entry.last_seen = now
            return entry, strategy

    def _release_entry(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        """Release resources for a popped entry (run outside the lock).

        Closes pending transfer observations, sweeps the instance's pending
        SHM transfers and unregisters its layout descriptor.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
        with self._lock:
            stale = [k for k in self._pending_transfers if k[1] == instance_id]
            pending = [self._pending_transfers.pop(k) for k in stale]
        for event, _ in pending:
            self._publish_transfer_end(event, False)
        with self._pending_shm_lock:
            stale_writes = [k for k in self._pending_shm_writes if k[0] == instance_id]
            stale_reads = [k for k in self._pending_shm_reads if k[0] == instance_id]
            write_obj_keys = [self._pending_shm_writes.pop(k) for k in stale_writes]
            read_obj_keys = [self._pending_shm_reads.pop(k) for k in stale_reads]

        for obj_keys in write_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_write(obj_keys)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_single_group_obj_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
        """Resolve object keys for the single object group used by
        non-GPU transfers."""
        return self._ctx.resolve_obj_keys(key, [0])[0]

    def _resolve_obj_keys_by_group(
        self, key: IPCCacheServerKey, num_groups: int
    ) -> list[list[ObjectKey]]:
        """Resolve object keys for every LMCache group, group-major.

        Args:
            key: Cache key for the token range.
            num_groups: Number of LMCache groups the worker registered.

        Returns:
            ``keys[g][c]`` is chunk ``c``'s key in group ``g``
            (``ObjectKey.object_group_id == g``).
        """
        return self._ctx.resolve_obj_keys(key, list(range(num_groups)))

    def _group_keys_for(
        self, entry: "EngineDrivenContextEntry", key: IPCCacheServerKey
    ) -> "list[list[ObjectKey]] | None":
        """Return group-major keys for any explicit group layouts, else None."""
        layouts = entry.metadata.group_layouts
        if not layouts:
            return None
        return self._resolve_obj_keys_by_group(key, len(layouts))

    @request_handler()
    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                num_physical_slots).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        now = time.monotonic()
        with self._lock:
            existing = self._engine_driven_contexts.get(payload.instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered (non-GPU); refreshing liveness",
                    payload.instance_id,
                )
                return RegisterEngineDrivenContextResponse(
                    shm_name=shm_name,
                    pool_size=pool_size,
                    null_block_id=self._ctx.null_block_id,
                )

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        num_physical_slots = payload.num_physical_slots
        if num_physical_slots is None:
            # Compatibility with clients from before the physical-slot field
            # was added. Those clients require one slot per logical token.
            num_physical_slots = self._ctx.chunk_size
        elif num_physical_slots <= 0:
            raise ValueError(
                f"num_physical_slots must be positive, got {num_physical_slots}"
            )

        shape = (
            torch.Size(
                [payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        group_layouts: list[MemoryLayoutDesc] | None = None
        if payload.group_layouts:
            group_layouts = []
            for gl in payload.group_layouts:
                g_dtype = getattr(torch, gl.dtype_str, None)
                if g_dtype is None or not isinstance(g_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid group dtype_str '{gl.dtype_str}' in "
                        "engine-driven registration"
                    )
                g_slots = gl.num_physical_slots
                if g_slots is None:
                    g_slots = gl.window_tokens or self._ctx.chunk_size
                if g_slots <= 0:
                    raise ValueError(
                        f"Group num_physical_slots must be positive, got {g_slots}"
                    )
                # Plane count is per group: a two-plane K/V group can sit
                # beside a one-plane index cache. Legacy workers report 0, so
                # fall back to the top-level flag they were sized by.
                g_single_plane = gl.kv_size == 1 if gl.kv_size else payload.use_mla
                g_shape = (
                    torch.Size([gl.num_layers, g_slots, gl.hidden_dim_size])
                    if g_single_plane
                    else torch.Size([2, gl.num_layers, g_slots, gl.hidden_dim_size])
                )
                group_layouts.append(
                    MemoryLayoutDesc(shapes=[g_shape], dtypes=[g_dtype])
                )
        group_windows: list[int] | None = None
        if payload.group_layouts:
            group_windows = []
            for gl in payload.group_layouts:
                if gl.window_chunks < 0:
                    raise ValueError(
                        f"Invalid group window_chunks {gl.window_chunks} in "
                        "engine-driven registration"
                    )
                group_windows.append(gl.window_chunks or -1)
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
            group_layouts=group_layouts,
            group_windows=group_windows,
        )
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SERIALIZED, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            last_seen=now,
            has_liveness_signal=False,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
            read_owner_resolver=self._ctx.get_read_owners,
            shm_name=shm_name,
            pool_size=pool_size,
            pending_writes=self._pending_shm_writes,
            pending_reads=self._pending_shm_reads,
            pending_lock=self._pending_shm_lock,
            transfer_key_factory=self._make_transfer_key,
        )
        with self._lock:
            self._engine_driven_contexts[payload.instance_id] = entry
            self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
        )

        if group_layouts:
            # The lookup/prefetch path fans its object-group lookups out over
            # the registered AttnWindowDesc's group count. A hybrid-KV worker
            # stores and retrieves ``len(group_layouts)`` object groups, but the
            # default single-group desc makes lookup prefetch only group 0 --
            # every later group then misses at retrieve (unsafe_read finds no
            # read-locked objects) and vLLM consumes the unloaded blocks. Match
            # the desc's group count to the worker's, with each group's own
            # window and kind: sliding-window and recurrent-state groups need
            # only their trailing chunks for a hit, and the worker never stores
            # their all-null earlier chunks, so a full-prefix window would make
            # every lookup on such a model miss.
            assert group_windows is not None
            attn_desc = AttnWindowDesc(
                num_chunks_in_sw=group_windows,
                world_size=payload.world_size,
                group_kinds=tuple(
                    "recurrent" if gl.recurrent else "attention"
                    for gl in payload.group_layouts
                ),
            )
            self._ctx.layout_desc_registry.register(
                payload.model_name,
                payload.world_size,
                layout_desc,
                attn_desc=attn_desc,
                group_layout_descs=dict(enumerate(group_layouts)),
            )
        else:
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, layout_desc
            )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name,
            pool_size=pool_size,
            null_block_id=self._ctx.null_block_id,
        )

    @request_handler(operation="unregister_kv_cache_engine_driven_context")
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        with self._lock:
            entry = self._engine_driven_contexts.pop(instance_id, None)
            if entry is not None:
                self._strategies.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._release_entry(instance_id, entry)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> PrepareStoreResponse:
        """Prepare a store and begin observing a non-empty worker transfer.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            skipped_chunks: Per LMCache group, the chunk positions the worker
                will not store because every engine block in them is the null
                block. Empty means none; otherwise one entry per registered
                group. Only grouped-protocol workers send it.

        Returns:
            PrepareStoreResponse with empty slots for inline mode.

        Raises:
            ValueError: If ``skipped_chunks`` does not have one entry per
                registered group or names a chunk outside the key's range.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        group_keys = self._group_keys_for(entry, key)
        if skipped_chunks:
            if group_keys is None or len(skipped_chunks) != len(group_keys):
                raise ValueError(
                    f"skipped_chunks has {len(skipped_chunks)} groups for "
                    f"{0 if group_keys is None else len(group_keys)} registered "
                    f"groups (instance_id={instance_id})"
                )
            for gid, (skips, g_keys) in enumerate(
                zip(skipped_chunks, group_keys, strict=True)
            ):
                if any(not 0 <= idx < len(g_keys) for idx in skips):
                    raise ValueError(
                        f"skipped_chunks for group {gid} is outside its "
                        f"{len(g_keys)} chunks (instance_id={instance_id})"
                    )
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
            group_keys=group_keys,
            skipped_chunks=skipped_chunks,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
        self._start_transfer(
            key,
            instance_id,
            entry,
            EventType.MP_STORE_START,
            response.context,
            group_keys,
            skipped_chunks,
        )
        return response

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit CPU chunks and publish completion, including failed commits.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Typed inline CPU tensor payload produced by the worker.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("store_start_time", None)
        result = False
        try:
            result = strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=entry.metadata,
                resolve_obj_keys=self._resolve_single_group_obj_keys,
                group_keys=self._group_keys_for(entry, key),
            )
        finally:
            self._finish_transfer(key, instance_id, EventType.MP_STORE_START, result)
        if st is not None and result:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Prepare prefetched chunks and begin observing a successful transfer.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        group_keys = self._group_keys_for(entry, key)
        group_first_chunk: list[int] = []
        windows = entry.metadata.group_windows
        if group_keys is not None and windows is not None:
            # Serve each group's in-window suffix only, as prefetch locked it:
            # a sliding-window or recurrent group's earlier chunks are not
            # needed and were never stored.
            narrowed: list[list[ObjectKey]] = []
            for g_keys, window in zip(group_keys, windows, strict=True):
                first = 0 if window < 0 else max(0, len(g_keys) - window)
                group_first_chunk.append(first)
                narrowed.append(g_keys[first:])
            group_keys = narrowed
        response = strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
            group_keys=group_keys,
            group_first_chunk=group_first_chunk,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
        if response.success:
            self._start_transfer(
                key,
                instance_id,
                entry,
                EventType.MP_RETRIEVE_START,
                response.context,
                group_keys,
            )
        return response

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve and publish its transfer completion.

        Args:
            key: Cache key identifying the prepared retrieve.
            instance_id: Worker instance identifier.

        Returns:
            Always ``True``.
        """
        _, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = False
        try:
            result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        finally:
            self._finish_transfer(key, instance_id, EventType.MP_RETRIEVE_START, result)
        if st is not None:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

    def _start_transfer(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        entry: EngineDrivenContextEntry,
        event_type: EventType,
        context: dict[str, Any],
        group_keys: list[list[ObjectKey]] | None,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> None:
        """Observe the worker handoff after prepare, through its commit.

        CPU timestamps include RPC, worker gather/scatter and serialization,
        not just device DMA. The device label names the server's CPU endpoint.
        SHM responses report only reserved slots; empty stores need no commit.

        Args:
            key: Request and token range identifying the transfer.
            instance_id: Registered worker instance identifier.
            entry: Worker metadata defining inline object layouts.
            event_type: Store or retrieve START event type.
            context: Prepare response context, including slots for SHM.
            group_keys: Per-group object keys, already windowed for retrieve,
                or ``None`` for the single-group token range.
            skipped_chunks: Validated per-group chunk positions omitted by store.
        """
        slots = context.get("slots")
        if slots is not None:
            # SHM Path: Slots already exclude skipped chunks and describe
            # reserved bytes.
            count = len(slots)
            total_bytes = sum(slot["length"] for slot in slots)
        else:
            # Inline Path: Derive payload bytes from unskipped group
            # counts and layouts, using the already-windowed keys for retrieve.
            counts = (
                [len(keys) for keys in group_keys]
                if group_keys is not None
                else [(key.end - key.start) // self._ctx.chunk_size]
            )
            if skipped_chunks:
                counts = [
                    count - len(set(skips))
                    for count, skips in zip(counts, skipped_chunks, strict=True)
                ]
            layouts = entry.metadata.group_layouts or [entry.metadata.layout_desc]
            count = sum(counts)
            total_bytes = sum(
                n
                * sum(
                    shape.numel() * dtype.itemsize
                    for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
                )
                for n, layout in zip(counts, layouts, strict=True)
            )
        if count == 0:
            return
        count_name = (
            "stored_count"
            if event_type == EventType.MP_STORE_START
            else "retrieved_count"
        )
        event = Event(
            event_type=event_type,
            session_id=key.request_id,
            metadata={
                "device": "cpu",
                "engine_id": instance_id,
                "model_name": entry.model_name,
                "cache_salt": key.cache_salt,
                "transfer_key": next_transfer_key(key.request_id),
                "transfer_mode": "engine_driven",
                "total_bytes": total_bytes,
                "num_tokens": key.end - key.start,
                count_name: count,
            },
        )
        submitted = (
            EventType.MP_STORE_SUBMITTED
            if event_type == EventType.MP_STORE_START
            else EventType.MP_RETRIEVE_SUBMITTED
        )
        with self._lock:
            pending_key = (event_type, instance_id, key.request_id, key)
            previous = self._pending_transfers.pop(pending_key, None)
            if previous is not None:
                self._publish_transfer_end(previous[0], False)
            self._pending_transfers[pending_key] = (event, time.monotonic())
            self._ctx.event_bus.publish(
                Event(submitted, session_id=key.request_id, metadata={"device": "cpu"})
            )
            self._ctx.event_bus.publish(event)

    def _finish_transfer(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        start_type: EventType,
        success: bool,
    ) -> None:
        """Finish a prepared transfer, including failed commits."""
        with self._lock:
            pending = self._pending_transfers.pop(
                (start_type, instance_id, key.request_id, key), None
            )
        if pending is not None:
            self._publish_transfer_end(pending[0], success)

    def _publish_transfer_end(self, start: Event, success: bool) -> None:
        """Close a transfer; failed or abandoned transfers report zero volume."""
        metadata = start.metadata.copy()
        if not success:
            for name in (
                "total_bytes",
                "num_tokens",
                "stored_count",
                "retrieved_count",
            ):
                if name in metadata:
                    metadata[name] = 0
        end_type = (
            EventType.MP_STORE_END
            if start.event_type == EventType.MP_STORE_START
            else EventType.MP_RETRIEVE_END
        )
        self._ctx.event_bus.publish(
            Event(end_type, session_id=start.session_id, metadata=metadata)
        )

    def _cleanup_expired_transfer_observations(self) -> ThreadRunSummary:
        """End observations older than 120 seconds without reclaiming transfers.

        Expiry publishes a zero-volume END event independently of worker reaping
        and heartbeats. Registrations and SHM reservations stay intact: workers
        may still be accessing those buffers. A late commit finalizes storage
        normally without publishing a second END event.
        """
        now = time.monotonic()
        with self._lock:
            expired = [
                key
                for key, (_, started_at) in self._pending_transfers.items()
                if now - started_at > _TRANSFER_OBSERVATION_TIMEOUT_SECONDS
            ]
            pending = [self._pending_transfers.pop(key)[0] for key in expired]
        for event in pending:
            logger.warning(
                "Expired engine-driven transfer observation %s without commit",
                event.metadata["transfer_key"],
            )
            self._publish_transfer_end(event, False)
        return ThreadRunSummary(success=True, message=f"expired={len(pending)}")
