# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
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
from lmcache.v1.multiprocess.transfer_context.group_layout import (
    EngineDrivenGroupLayout,
)

# Local
from .server_transfer import (
    ObjectGroupKeys,
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)


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
    of CPU-serialized KV caches.

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
        """Release resources owned by this module."""
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()

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
        """Reap non-GPU registrations that have gone silent.

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

        Sweeps the instance's pending SHM transfers and unregisters its
        layout descriptor.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
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

    def _num_chunks(self, key: IPCCacheServerKey) -> int:
        """Return the number of LMCache chunks ``key`` covers."""
        return len(self._ctx.resolve_obj_keys(key, [0])[0])

    def _transfer_groups_resolver(
        self, metadata: EngineDrivenContextMetadata
    ) -> Callable[[IPCCacheServerKey], list[ObjectGroupKeys]]:
        """Return the transfer-group resolver of one registration.

        A single-group registration transfers every chunk of object group 0.
        A multi-group registration transfers, per object group, the trailing
        ``num_chunks_in_sw`` chunks only (every chunk for full attention), the
        chunks the worker gathers and scatters.
        """
        group_layout = metadata.group_layout
        if group_layout is None:

            def resolve_single_group(key: IPCCacheServerKey) -> list[ObjectGroupKeys]:
                return [
                    ObjectGroupKeys(
                        object_group_id=0,
                        first_chunk=0,
                        keys=self._ctx.resolve_obj_keys(key, [0])[0],
                        layout_desc=metadata.layout_desc,
                    )
                ]

            return resolve_single_group

        object_groups = group_layout.object_groups
        group_ids = list(range(len(object_groups)))

        def resolve_object_groups(key: IPCCacheServerKey) -> list[ObjectGroupKeys]:
            keys_per_group = self._ctx.resolve_obj_keys(key, group_ids)
            num_chunks = len(keys_per_group[0])
            groups: list[ObjectGroupKeys] = []
            for gid, (object_group, keys) in enumerate(
                zip(object_groups, keys_per_group, strict=True)
            ):
                first = object_group.first_chunk(num_chunks)
                groups.append(
                    ObjectGroupKeys(
                        object_group_id=gid,
                        first_chunk=first,
                        keys=keys[first:],
                        layout_desc=object_group.layout_desc,
                    )
                )
            return groups

        return resolve_object_groups

    def _registration_response(
        self, metadata: EngineDrivenContextMetadata
    ) -> RegisterEngineDrivenContextResponse:
        """Build the registration response for ``metadata``."""
        group_layout = metadata.group_layout
        return RegisterEngineDrivenContextResponse(
            shm_name=self._shm_pool_info["shm_name"],
            pool_size=self._shm_pool_info["pool_size"],
            num_object_groups=(
                0 if group_layout is None else len(group_layout.object_groups)
            ),
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
            null_block_id=self._ctx.null_block_id,
            chunk_size=self._ctx.chunk_size,
        )

    def _build_group_layout(
        self, payload: RegisterEngineDrivenContextPayload
    ) -> EngineDrivenGroupLayout:
        """Rebuild the worker's object groups from its KV tensor specs.

        Raises:
            ValueError: If ``engine_type`` is missing, a spec carries an
                invalid dtype name, or the specs and group infos do not
                describe a valid KV layout.
        """
        if payload.engine_type is None:
            raise ValueError("Multi-group registration requires engine_type")
        tensors = [
            torch.empty_strided(
                spec.shape,
                spec.stride,
                dtype=self._parse_dtype(spec.dtype_str),
                device="meta",
            )
            for spec in payload.kv_tensor_specs
        ]
        return EngineDrivenGroupLayout(
            tensors,
            payload.engine_group_infos,
            payload.engine_type,
            payload.layout_hints or None,
            self._ctx.chunk_size,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        """Return the torch dtype named ``dtype_str``.

        Raises:
            ValueError: If ``dtype_str`` is not a valid torch dtype name.
        """
        dtype = getattr(torch, dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )
        return dtype

    @request_handler()
    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        A payload with ``kv_tensor_specs`` registers every object group of a
        hybrid model; otherwise the scalar layout fields describe the single
        transfer group.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                num_physical_slots, and the multi-group fields).

        Raises:
            ValueError: If a dtype name is not a valid torch dtype name, or
                the multi-group fields do not describe a valid KV layout.
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
                return self._registration_response(existing.metadata)

        if payload.kv_tensor_specs:
            group_layout = self._build_group_layout(payload)
            metadata = EngineDrivenContextMetadata(
                layout_desc=group_layout.object_groups[0].layout_desc,
                block_size=payload.block_size,
                use_mla=payload.use_mla,
                group_layout=group_layout,
            )
        else:
            dtype = self._parse_dtype(payload.dtype_str)
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
            metadata = EngineDrivenContextMetadata(
                layout_desc=MemoryLayoutDesc(shapes=[shape], dtypes=[dtype]),
                block_size=payload.block_size,
                use_mla=payload.use_mla,
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

        if metadata.group_layout is None:
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, metadata.layout_desc
            )
        else:
            # Lookup, prefetch and lock release read every object group's
            # layout and window from the registry, as for LMCache-driven.
            object_groups = metadata.group_layout.object_groups
            self._ctx.layout_desc_registry.register(
                payload.model_name,
                payload.world_size,
                metadata.layout_desc,
                metadata.group_layout.attn_desc,
                group_layout_descs={
                    gid: group.layout_desc for gid, group in enumerate(object_groups)
                },
            )
            logger.info(
                "Instance %d registered %d object groups (windows=%s)",
                payload.instance_id,
                len(object_groups),
                metadata.group_layout.attn_desc.num_chunks_in_sw,
            )
        return self._registration_response(metadata)

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
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for inline mode.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_transfer_groups=self._transfer_groups_resolver(entry.metadata),
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
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
        """Commit serialized CPU chunks to storage.

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
        result = strategy.commit_store(
            key=key,
            instance_id=instance_id,
            cpu_data=cpu_data,
            context=entry.metadata,
            resolve_transfer_groups=self._transfer_groups_resolver(entry.metadata),
        )
        if st is not None and result:
            num_tokens = self._num_chunks(key) * self._ctx.chunk_size
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
        """Retrieve prefetched chunks and return serialized CPU tensors.

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
        response = strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_transfer_groups=self._transfer_groups_resolver(entry.metadata),
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
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
        """Finalize a retrieve operation.

        Args:
            key: Cache key (unused for inline transfer).
            instance_id: Worker instance identifier (unused for inline transfer).

        Returns:
            Always ``True``.
        """
        _, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            num_tokens = self._num_chunks(key) * self._ctx.chunk_size
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result
