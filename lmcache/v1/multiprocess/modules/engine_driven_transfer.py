# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVTransferMetadataWire,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupTransferMetadata,
    KVTransferMetadata,
    ObjectGroupTransferMetadata,
    build_object_group_layout_desc,
)

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)


def _kv_transfer_metadata_from_wire(
    wire: KVTransferMetadataWire,
) -> KVTransferMetadata:
    """Reconstruct a :class:`~lmcache.v1.multiprocess.transfer_plan.KVTransferMetadata`
    from its msgspec wire DTO.

    Converts primitive wire-format fields back to their runtime types:
    ``dtype_str`` → ``torch.dtype``, ``engine_kv_format_int`` →
    :class:`~lmcache.v1.platform.ops_types.EngineKVFormat`.

    Args:
        wire: The deserialized wire DTO received from the worker.

    Returns:
        A fully reconstructed immutable :class:`KVTransferMetadata` snapshot.

    Raises:
        ValueError: If any ``dtype_str`` is not a valid torch dtype name or
            any ``engine_kv_format_int`` is not a valid ``EngineKVFormat``
            value.
    """
    # Use the native lmc_ops enum when available (same int values as the pure
    # Python ops_types.EngineKVFormat, so both work with C++ op calls).
    try:
        # First Party
        import lmcache.c_ops as lmc_ops

        _EngineKVFormat = lmc_ops.EngineKVFormat
    except ImportError:
        # First Party
        from lmcache.v1.platform.ops_types import EngineKVFormat

        _EngineKVFormat = EngineKVFormat  # type: ignore[assignment]

    kernel_groups_out: list[KernelGroupTransferMetadata] = []
    for kgw in wire.kernel_groups:
        dtype = getattr(torch, kgw.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"kernel group {kgw.kernel_group_id}: invalid dtype_str "
                f"'{kgw.dtype_str}'"
            )
        try:
            engine_kv_format = _EngineKVFormat(kgw.engine_kv_format_int)
        except ValueError as exc:
            raise ValueError(
                f"kernel group {kgw.kernel_group_id}: invalid "
                f"engine_kv_format_int {kgw.engine_kv_format_int}"
            ) from exc
        kernel_groups_out.append(
            KernelGroupTransferMetadata(
                kernel_group_id=kgw.kernel_group_id,
                engine_group_id=kgw.engine_group_id,
                layer_indices=tuple(kgw.layer_indices),
                blocks_per_chunk=kgw.blocks_per_chunk,
                blocks_per_window=kgw.blocks_per_window,
                slots_per_chunk_in_window=kgw.slots_per_chunk_in_window,
                kv_size=kgw.kv_size,
                num_layers=kgw.num_layers,
                hidden_dim_size=kgw.hidden_dim_size,
                slots_per_block=kgw.slots_per_block,
                tokens_per_block=kgw.tokens_per_block,
                dtype=dtype,
                engine_kv_format=engine_kv_format,
            )
        )

    object_groups_out = tuple(
        ObjectGroupTransferMetadata(
            object_group_id=ogw.object_group_id,
            kernel_group_ids=tuple(ogw.kernel_group_ids),
            sw_size_chunks=ogw.sw_size_chunks,
        )
        for ogw in wire.object_groups
    )

    return KVTransferMetadata(
        num_chunks_in_sw=tuple(wire.num_chunks_in_sw),
        tokens_per_chunk=wire.tokens_per_chunk,
        kernel_groups=tuple(kernel_groups_out),
        object_groups=object_groups_out,
    )


def _decode_multi_group_payload_fields(
    payload: RegisterEngineDrivenContextPayload,
    legacy_layout_desc: MemoryLayoutDesc,
) -> tuple[list[MemoryLayoutDesc], AttnWindowDesc]:
    """Decode multi-group layout fields from a registration payload.

    Returns the per-object-group layout descriptors and attention-window
    descriptor encoded in ``payload``.  When the multi-group fields are absent
    (legacy single-group registration), falls back to wrapping
    ``legacy_layout_desc`` in a single-entry list with full-attention semantics.

    Args:
        payload: The decoded registration payload from the worker.
        legacy_layout_desc: Single-group layout built from the flat payload
            fields; used when ``payload.object_group_layout_shapes`` is empty.

    Returns:
        A tuple ``(object_group_layout_descs, attn_desc)`` where
        ``object_group_layout_descs`` has one entry per object group and
        ``attn_desc`` describes the attention window for each.

    Raises:
        ValueError: If the ``object_group_layout_shapes`` and
            ``object_group_layout_dtype_strs`` fields have different lengths,
            or if any dtype string is not a valid torch dtype name.
    """
    if not payload.object_group_layout_shapes:
        # Legacy single-group mode: wrap existing layout in single-element list.
        return [], DEFAULT_ATTN_WINDOW_DESC

    shapes_per_group = payload.object_group_layout_shapes
    dtypes_per_group = payload.object_group_layout_dtype_strs

    if len(shapes_per_group) != len(dtypes_per_group):
        raise ValueError(
            f"object_group_layout_shapes has {len(shapes_per_group)} entries "
            f"but object_group_layout_dtype_strs has {len(dtypes_per_group)}"
        )

    object_group_layout_descs: list[MemoryLayoutDesc] = []
    for og_idx, (group_shapes, group_dtype_strs) in enumerate(
        zip(shapes_per_group, dtypes_per_group, strict=True)
    ):
        if len(group_shapes) != len(group_dtype_strs):
            raise ValueError(
                f"object group {og_idx}: {len(group_shapes)} shapes but "
                f"{len(group_dtype_strs)} dtype strings"
            )
        dtypes: list[torch.dtype] = []
        for dt_str in group_dtype_strs:
            dt = getattr(torch, dt_str, None)
            if dt is None or not isinstance(dt, torch.dtype):
                raise ValueError(
                    f"object group {og_idx}: invalid dtype string '{dt_str}'"
                )
            dtypes.append(dt)
        sizes = [torch.Size(s) for s in group_shapes]
        object_group_layout_descs.append(MemoryLayoutDesc(shapes=sizes, dtypes=dtypes))

    # Build AttnWindowDesc from the wire num_chunks_in_sw field; fall back
    # to one full-attention group if it was not sent (should not happen with
    # a well-formed multi-group payload, but guard defensively).
    if payload.num_chunks_in_sw:
        if len(payload.num_chunks_in_sw) != len(object_group_layout_descs):
            raise ValueError(
                f"num_chunks_in_sw has {len(payload.num_chunks_in_sw)} entries "
                f"but {len(object_group_layout_descs)} object groups were decoded"
            )
        attn_desc = AttnWindowDesc(num_chunks_in_sw=list(payload.num_chunks_in_sw))
    else:
        attn_desc = AttnWindowDesc(
            num_chunks_in_sw=[-1] * len(object_group_layout_descs)
        )

    return object_group_layout_descs, attn_desc


def _validate_transfer_metadata_consistency(
    transfer_metadata: KVTransferMetadata,
    engine_group_infos: list[EngineGroupInfo],
    object_group_layout_descs: list[MemoryLayoutDesc],
    chunk_size: int,
) -> None:
    """Validate internal consistency of a reconstructed :class:`KVTransferMetadata`.

    Checks that the wire-format transfer metadata is self-consistent and
    consistent with the other multi-group registration fields.  Raises
    :exc:`ValueError` as soon as the first inconsistency is found.

    Checks performed:

    1. ``tokens_per_chunk`` matches the server's ``chunk_size``.
    2. Kernel-group IDs are contiguous from 0 (match their list index).
    3. Object-group IDs are contiguous from 0 (match their list index).
    4. Every kernel-group ID referenced by an object group is a valid index.
    5. Each object-group's ``sw_size_chunks`` matches
       ``num_chunks_in_sw[object_group_id]``.
    6. ``engine_group_infos`` and kernel groups describe the same engine groups
       and layer ordering.  Multiple kernel groups may represent one engine
       group when that engine group is split by transfer identity.
    7. Layouts rebuilt from ``build_object_group_layout_desc`` match
       ``object_group_layout_descs`` element-by-element.

    Args:
        transfer_metadata: The reconstructed transfer metadata to validate.
        engine_group_infos: Engine-group topology from the registration payload.
        object_group_layout_descs: Decoded per-object-group layout descriptors
            from the registration payload.
        chunk_size: Server chunk size in tokens; must equal
            ``transfer_metadata.tokens_per_chunk``.

    Raises:
        ValueError: If any consistency check fails.
    """
    # 1. tokens_per_chunk must match the server chunk size.
    if transfer_metadata.tokens_per_chunk != chunk_size:
        raise ValueError(
            f"transfer_metadata_wire.tokens_per_chunk "
            f"({transfer_metadata.tokens_per_chunk}) does not match "
            f"server chunk_size ({chunk_size})"
        )

    num_kernel_groups = len(transfer_metadata.kernel_groups)
    num_object_groups = len(transfer_metadata.object_groups)

    # 2. Kernel-group IDs must be contiguous from 0.
    for idx, kg in enumerate(transfer_metadata.kernel_groups):
        if kg.kernel_group_id != idx:
            raise ValueError(
                f"kernel_group at index {idx} has kernel_group_id "
                f"{kg.kernel_group_id}, expected {idx}"
            )

    # 3. Object-group IDs must be contiguous from 0.
    for idx, og in enumerate(transfer_metadata.object_groups):
        if og.object_group_id != idx:
            raise ValueError(
                f"object_group at index {idx} has object_group_id "
                f"{og.object_group_id}, expected {idx}"
            )

    # 4. Object-group kernel_group_ids must reference valid kernel groups.
    for og in transfer_metadata.object_groups:
        for kg_id in og.kernel_group_ids:
            if kg_id < 0 or kg_id >= num_kernel_groups:
                raise ValueError(
                    f"object_group {og.object_group_id} references invalid "
                    f"kernel_group_id {kg_id} "
                    f"(num_kernel_groups={num_kernel_groups})"
                )

    # 5. sw_size_chunks must match num_chunks_in_sw for each object group.
    if len(transfer_metadata.num_chunks_in_sw) != num_object_groups:
        raise ValueError(
            f"transfer_metadata.num_chunks_in_sw has "
            f"{len(transfer_metadata.num_chunks_in_sw)} entries but "
            f"{num_object_groups} object groups"
        )
    for og in transfer_metadata.object_groups:
        expected_sw = transfer_metadata.num_chunks_in_sw[og.object_group_id]
        if og.sw_size_chunks != expected_sw:
            raise ValueError(
                f"object_group {og.object_group_id}: sw_size_chunks "
                f"({og.sw_size_chunks}) does not match "
                f"num_chunks_in_sw[{og.object_group_id}] ({expected_sw})"
            )

    # 6. Engine-group metadata may be split into multiple kernel groups when
    # layers in one engine block address space need different copy kernels.
    if engine_group_infos:
        info_layers_by_engine_group: dict[int, list[int]] = {}
        info_engine_group_order: list[int] = []
        for info in engine_group_infos:
            if info.engine_group_id not in info_layers_by_engine_group:
                info_layers_by_engine_group[info.engine_group_id] = []
                info_engine_group_order.append(info.engine_group_id)
            info_layers_by_engine_group[info.engine_group_id].extend(info.layer_indices)

        kernel_layers_by_engine_group: dict[int, list[int]] = {}
        kernel_engine_group_order: list[int] = []
        for kernel_group in transfer_metadata.kernel_groups:
            if kernel_group.engine_group_id not in kernel_layers_by_engine_group:
                kernel_layers_by_engine_group[kernel_group.engine_group_id] = []
                kernel_engine_group_order.append(kernel_group.engine_group_id)
            kernel_layers_by_engine_group[kernel_group.engine_group_id].extend(
                kernel_group.layer_indices
            )

        if info_engine_group_order != kernel_engine_group_order:
            raise ValueError(
                "engine_group_infos and transfer_metadata kernel groups have "
                "different engine_group_id ordering"
            )

        for engine_group_id in info_engine_group_order:
            info_layers = info_layers_by_engine_group[engine_group_id]
            kernel_layers = kernel_layers_by_engine_group[engine_group_id]
            if info_layers != kernel_layers:
                raise ValueError(
                    f"engine_group_id {engine_group_id}: engine_group_infos "
                    f"layer_indices ({info_layers}) do not match kernel groups "
                    f"layer_indices ({kernel_layers})"
                )

    # 7. Rebuild layouts from transfer_metadata and compare with payload descs.
    if object_group_layout_descs:
        if len(object_group_layout_descs) != num_object_groups:
            raise ValueError(
                f"object_group_layout_descs has {len(object_group_layout_descs)} "
                f"entries but transfer_metadata has {num_object_groups} object groups"
            )
        for og_id, payload_desc in enumerate(object_group_layout_descs):
            rebuilt = build_object_group_layout_desc(
                transfer_metadata, chunk_size, og_id
            )
            shapes_match = rebuilt.shapes == payload_desc.shapes
            dtypes_match = rebuilt.dtypes == payload_desc.dtypes
            if not shapes_match or not dtypes_match:
                raise ValueError(
                    f"object_group {og_id}: layout rebuilt from transfer_metadata "
                    f"does not match payload layout: "
                    f"rebuilt shapes={list(rebuilt.shapes)}, "
                    f"payload shapes={list(payload_desc.shapes)}; "
                    f"rebuilt dtypes={rebuilt.dtypes}, "
                    f"payload dtypes={payload_desc.dtypes}"
                )


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


class EngineDrivenTransferModule(InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

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

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.register_kv_cache_engine_driven_context,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.PREPARE_STORE,
                self.prepare_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_STORE,
                self.commit_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.ABORT_STORE,
                self.abort_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.PREPARE_RETRIEVE,
                self.prepare_retrieve,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_RETRIEVE,
                self.commit_retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]

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
            entries = list(self._engine_driven_contexts.items())
            self._engine_driven_contexts.clear()
            self._strategies.clear()
        for instance_id, entry in entries:
            self._release_entry(instance_id, entry)

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
                # These stores never reached COMMIT_STORE, so publishing their
                # contents would expose partially written SHM data. Force-delete
                # the write-locked allocations instead of finishing the write.
                self._ctx.storage_manager.delete_l1_keys(obj_keys, force=True)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_obj_keys(
        self,
        key: IPCCacheServerKey,
        metadata: EngineDrivenContextMetadata,
    ) -> list[list[ObjectKey]]:
        """Resolve deterministic object-group keys for one transfer.

        Args:
            key: Cache key for the requested token range.
            metadata: Registered context metadata describing object groups.

        Returns:
            Object keys in object-group order, then chunk order. Legacy
            registrations return a single inner list for object group zero.
        """
        if metadata.transfer_metadata is None:
            return self._ctx.resolve_obj_keys(key, [0])
        return self._ctx.resolve_obj_keys(
            key,
            [
                object_group.object_group_id
                for object_group in metadata.transfer_metadata.object_groups
            ],
        )

    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Supports two modes:

        * **Legacy single-group** (``payload.object_group_layout_shapes`` is
          empty): builds a single :class:`~lmcache.v1.distributed.api.MemoryLayoutDesc`
          from the flat fields and registers it with full-attention semantics.
          Preserves all pre-Step-3 behaviour unchanged.
        * **Multi-group** (``payload.object_group_layout_shapes`` is non-empty):
          requires ``payload.transfer_metadata_wire`` to be present.  Reconstructs
          all per-object-group layouts from the serialised wire fields, validates
          their consistency with ``engine_group_infos`` and the server chunk size,
          and stores the full multi-group layout list on the context entry.  Object
          group 0 is still used as the primary layout for the layout-registry
          registration.

        Args:
            payload: Struct containing all registration fields.

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name,
                if multi-group fields are present but ``transfer_metadata_wire`` is
                absent, or if any consistency check on the transfer metadata fails.
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
                    shm_name=shm_name, pool_size=pool_size
                )

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        # Build legacy single-group layout from flat fields (always needed as
        # primary layout and for backward compatibility).
        legacy_shape = (
            torch.Size(
                [payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size]
            )
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size]
            )
        )
        legacy_layout_desc = MemoryLayoutDesc(shapes=[legacy_shape], dtypes=[dtype])

        # Step 3: decode multi-group fields when the payload carries them.
        object_group_layout_descs, attn_desc = _decode_multi_group_payload_fields(
            payload, legacy_layout_desc
        )

        # Reject multi-group registration without the full metadata wire DTO.
        if payload.object_group_layout_shapes and (
            payload.transfer_metadata_wire is None
        ):
            raise ValueError(
                "multi-group registration (object_group_layout_shapes is non-empty) "
                "requires transfer_metadata_wire but it is absent"
            )

        # Reconstruct the full KVTransferMetadata from the structured wire DTO
        # and run cross-field consistency checks.
        transfer_metadata = None
        if payload.transfer_metadata_wire is not None:
            transfer_metadata = _kv_transfer_metadata_from_wire(
                payload.transfer_metadata_wire
            )
            _validate_transfer_metadata_consistency(
                transfer_metadata,
                list(payload.engine_group_infos),
                object_group_layout_descs,
                self._ctx.chunk_size,
            )
        # Primary layout is object group 0.
        primary_layout_desc = (
            object_group_layout_descs[0]
            if object_group_layout_descs
            else legacy_layout_desc
        )

        metadata = EngineDrivenContextMetadata(
            layout_desc=primary_layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
            object_group_layout_descs=object_group_layout_descs,
            attn_desc=attn_desc,
            transfer_metadata=transfer_metadata,
        )
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SYNC-serialized, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            last_seen=now,
            has_liveness_signal=False,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
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
            "Registered non-GPU context for instance %d (model=%s, world_size=%d, "
            "num_object_groups=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
            len(object_group_layout_descs) if object_group_layout_descs else 1,
        )

        self._ctx.layout_desc_registry.register(
            payload.model_name, payload.world_size, primary_layout_desc, attn_desc
        )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name, pool_size=pool_size
        )

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
            PrepareStoreResponse with empty slots for pickle mode.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_obj_keys=lambda transfer_key: self._resolve_obj_keys(
                transfer_key, entry.metadata
            ),
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
        return response

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
            cpu_data: Pickled list of CPU tensors produced by the worker.

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
            resolve_obj_keys=lambda transfer_key: self._resolve_obj_keys(
                transfer_key, entry.metadata
            ),
        )
        if st is not None and result:
            num_tokens = (
                sum(
                    len(object_keys)
                    for object_keys in self._resolve_obj_keys(key, entry.metadata)
                )
                * self._ctx.chunk_size
            )
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

    @_lmcache_nvtx_annotate
    def abort_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Abort an unfinished engine-driven store.

        Args:
            key: Cache key for the token range being stored.
            instance_id: Worker instance identifier.

        Returns:
            ``True`` after the strategy releases any pending store resources.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        _, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras.pop("store_start_time", None)
        return strategy.abort_store(key=key, instance_id=instance_id)

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
            resolve_obj_keys=lambda transfer_key: self._resolve_obj_keys(
                transfer_key, entry.metadata
            ),
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
        return response

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Args:
            key: Cache key (unused for pickle).
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            Always ``True``.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            num_tokens = (
                sum(
                    len(object_keys)
                    for object_keys in self._resolve_obj_keys(key, entry.metadata)
                )
                * self._ctx.chunk_size
            )
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result
