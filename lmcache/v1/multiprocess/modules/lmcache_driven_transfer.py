# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from typing import Sequence, cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    build_staging_copies,
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupPlan,
    KernelGroupTransferMetadata,
    KVTransferMetadata,
    ObjectGroupPlan,
    TransferPlan,
    TransferPlanDirection,
    build_object_group_layout_desc,
    build_transfer_plan,
    export_kv_transfer_metadata,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    lmc_ops, "execute_object_group_transfer"
)


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.

    Note:
        Compatibility adapter for existing call sites. It exports metadata
        on demand; registration/store/retrieve should use the metadata snapshot
        cached in :class:`ContextEntry` and avoid this adapter in hot paths.
    """
    transfer_metadata = export_kv_transfer_metadata(
        cache_context.kv_layer_groups_manager,
        cache_context.lmcache_tokens_per_chunk,
    )
    return build_object_group_layout_desc(
        transfer_metadata,
        num_tokens,
        object_group_id,
    )


def _block_ids_by_engine_group(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_kernel_group: Sequence[Sequence[int]],
) -> dict[int, Sequence[int]]:
    """Normalize request block IDs from kernel-group to engine-group keys.

    LMCache-driven requests retain their established wire format: one block-ID
    sequence per kernel group.  The shared planner consumes the engine-group
    form because multiple kernel groups can address the same engine block
    space.  Such repeated entries must agree exactly.

    Args:
        transfer_metadata: Immutable metadata mapping kernel groups to engine
            groups.
        block_ids_by_kernel_group: Request block IDs in kernel-group order.

    Returns:
        Block IDs keyed by engine group ID.

    Raises:
        ValueError: If the request group count differs from metadata or
            repeated kernel groups for one engine group disagree.
    """
    if len(block_ids_by_kernel_group) != len(transfer_metadata.kernel_groups):
        raise ValueError(
            "block ID group count does not match transfer metadata: "
            f"got {len(block_ids_by_kernel_group)}, expected "
            f"{len(transfer_metadata.kernel_groups)}"
        )

    result: dict[int, Sequence[int]] = {}
    for kernel_group, block_ids in zip(
        transfer_metadata.kernel_groups, block_ids_by_kernel_group, strict=True
    ):
        existing = result.get(kernel_group.engine_group_id)
        if existing is not None and existing != block_ids:
            raise ValueError(
                "conflicting block IDs for engine group "
                f"{kernel_group.engine_group_id} from repeated kernel groups"
            )
        result[kernel_group.engine_group_id] = block_ids
    return result


def _stage_object_group_plan_block_ids(
    cache_context: BaseCacheContext,
    object_group_plan: ObjectGroupPlan,
) -> list[torch.Tensor]:
    """Stage one object group's planned block IDs on the executor device.

    Args:
        cache_context: Executor resource context that stages block-ID tensors.
        object_group_plan: Shared logical plan to bind to device resources.

    Returns:
        Device tensors in the plan's kernel-group order.
    """
    return cache_context.stage_block_ids(
        [
            list(kernel_group_plan.block_ids)
            for kernel_group_plan in object_group_plan.kernel_groups
        ]
    )


def _planned_memory_objects(
    memory_objs: Sequence[MemoryObj | None],
    object_group_plan: ObjectGroupPlan,
) -> list[MemoryObj | None]:
    """Select memory objects whose source chunks appear in a logical plan.

    Args:
        memory_objs: Objects in original source-chunk order.
        object_group_plan: Plan containing the source chunk indices to bind.

    Returns:
        Objects ordered exactly as ``object_group_plan.chunk_indices``.

    Raises:
        ValueError: If a planned chunk index cannot be mapped to an object.
    """
    try:
        return [memory_objs[chunk_idx] for chunk_idx in object_group_plan.chunk_indices]
    except IndexError as exc:
        raise ValueError(
            f"object group {object_group_plan.object_group_id} has "
            "fewer memory objects than its transfer plan requires"
        ) from exc


def _kernel_group_metadata(
    transfer_metadata: KVTransferMetadata,
    kernel_group_plan: KernelGroupPlan,
) -> KernelGroupTransferMetadata:
    """Return metadata for a plan kernel group after validating its identity."""
    kernel_group_id = kernel_group_plan.kernel_group_id
    if kernel_group_id < 0 or kernel_group_id >= len(transfer_metadata.kernel_groups):
        raise ValueError(f"invalid kernel_group_id {kernel_group_id} in transfer plan")
    kernel_group_metadata = transfer_metadata.kernel_groups[kernel_group_id]
    if (
        kernel_group_metadata.kernel_group_id != kernel_group_id
        or kernel_group_metadata.engine_group_id != kernel_group_plan.engine_group_id
    ):
        raise ValueError(
            f"transfer plan kernel group {kernel_group_id} does not match metadata"
        )
    return kernel_group_metadata


def _plan_for_request(
    transfer_metadata: KVTransferMetadata,
    block_ids_by_kernel_group: Sequence[Sequence[int]],
    num_chunks: int,
    direction: TransferPlanDirection,
    skip_first_n_tokens: int = 0,
) -> TransferPlan:
    """Build the shared logical plan for an LMCache-driven request."""
    return build_transfer_plan(
        transfer_metadata,
        _block_ids_by_engine_group(transfer_metadata, block_ids_by_kernel_group),
        num_chunks,
        direction,
        skip_first_n_tokens,
    )


def _run_object_group_transfer_plan(
    cache_context: BaseCacheContext,
    transfer_metadata: KVTransferMetadata,
    object_group_plan: ObjectGroupPlan,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    batch_size: int,
    direction: "lmc_ops.TransferDirection",
) -> None:
    """Bind one logical object-group plan to the native transfer fast path.

    Args:
        cache_context: Executor resource context with device buffers and
            kernel descriptors.
        transfer_metadata: Shared immutable per-group transfer metadata.
        object_group_plan: Logical schedule produced by
            :func:`build_transfer_plan`.
        block_ids_gpu: Staged plan block IDs in plan kernel-group order.
        memory_objs: Objects matching the plan's ``chunk_indices`` order.
        batch_size: Number of memory objects per batched copy.
        direction: H2D (retrieve) or D2H (store).

    Raises:
        ValueError: If plan bindings are invalid or a None object is found for
            H2D.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if not memory_objs:
        return
    if len(memory_objs) != len(object_group_plan.chunk_indices):
        raise ValueError("memory objects do not match object-group plan chunks")
    if len(block_ids_gpu) != len(object_group_plan.kernel_groups):
        raise ValueError("staged block IDs do not match object-group plan groups")

    is_h2d = direction == lmc_ops.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    kernel_group_specs: list["lmc_ops.KernelGroupSpec"] = []
    for kernel_group_plan, block_ids_tensor in zip(
        object_group_plan.kernel_groups, block_ids_gpu, strict=True
    ):
        group_metadata = _kernel_group_metadata(transfer_metadata, kernel_group_plan)
        kernel_group_id = kernel_group_plan.kernel_group_id
        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        temp_buffers = [
            cache_context.get_temp_kernel_group_buffer(slot, kernel_group_id)
            for slot in range(max_batch_size)
        ]
        kernel_group_specs.append(
            lmc_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                cache_context.get_shape_desc(kernel_group_id),
                group_metadata.slots_per_chunk_in_window,
                group_metadata.engine_kv_format,
                block_ids_tensor.data_ptr(),
                block_ids_tensor.numel(),
            )
        )

    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(
            slot, object_group_plan.object_group_id
        )
        for slot in range(max_batch_size)
    ]

    batch_steps: list["lmc_ops.BatchStep"] = []
    for start_object_idx in range(0, len(memory_objs), batch_size):
        memory_object_batch = memory_objs[
            start_object_idx : start_object_idx + batch_size
        ]
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        valid_memory_object_batch = cast(tuple[MemoryObj, ...], memory_object_batch)
        staging = build_staging_copies(
            valid_memory_object_batch,
            object_group_buffers[:batch_len],
            is_h2d,
        )

        launches: list["lmc_ops.LaunchVar"] = []
        for spec_index, kernel_group_plan in enumerate(object_group_plan.kernel_groups):
            start_block_pos = start_object_idx * kernel_group_plan.blocks_per_window
            end_block_pos = (
                start_object_idx + batch_len
            ) * kernel_group_plan.blocks_per_window
            launches.append(
                lmc_ops.LaunchVar(
                    spec_index,
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    (
                        kernel_group_plan.skip_first_n_blocks
                        if start_object_idx == 0
                        else 0
                    ),
                )
            )

        batch_steps.append(lmc_ops.BatchStep(staging, launches))

    if not batch_steps:
        return

    lmc_ops.execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
    )


def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    transfer_metadata: KVTransferMetadata,
    object_group_plan: ObjectGroupPlan,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    batch_size: int,
    direction: "lmc_ops.TransferDirection",
) -> None:
    """Bind a shared object-group transfer plan to executor resources.

    Args:
        cache_context: Executor resource context containing paged KV and
            staging buffers.
        transfer_metadata: Shared immutable per-group transfer metadata.
        object_group_plan: Logical work for this object group.
        block_ids_gpu: Plan block-ID tensors in plan kernel-group order.
        memory_objs: Objects in ``object_group_plan.chunk_indices`` order.
        batch_size: Number of objects to process per copy batch.
        direction: The transfer direction, H2D (retrieve) or D2H (store).

    Raises:
        ValueError: If plan bindings are invalid or a None object is found for
            H2D.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if not memory_objs:
        return
    if len(memory_objs) != len(object_group_plan.chunk_indices):
        raise ValueError("memory objects do not match object-group plan chunks")
    if len(block_ids_gpu) != len(object_group_plan.kernel_groups):
        raise ValueError("staged block IDs do not match object-group plan groups")

    if _HAS_NATIVE_OBJECT_GROUP_TRANSFER and not any(
        isinstance(mo, GDSMemoryObject) for mo in memory_objs
    ):
        _run_object_group_transfer_plan(
            cache_context,
            transfer_metadata,
            object_group_plan,
            block_ids_gpu,
            memory_objs,
            batch_size,
            direction,
        )
        return

    is_h2d = direction == lmc_ops.TransferDirection.H2D
    for start_object_idx in range(0, len(memory_objs), batch_size):
        memory_object_batch = memory_objs[
            start_object_idx : start_object_idx + batch_size
        ]
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        valid_memory_object_batch = cast(tuple[MemoryObj, ...], memory_object_batch)
        if is_h2d:
            for chunk_idx, memory_obj in enumerate(valid_memory_object_batch):
                lmcache_memcpy_async_h2d(
                    memory_obj,
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_plan.object_group_id
                    ),
                )

        for kernel_group_plan, block_ids_tensor in zip(
            object_group_plan.kernel_groups, block_ids_gpu, strict=True
        ):
            group_metadata = _kernel_group_metadata(
                transfer_metadata, kernel_group_plan
            )
            kernel_group_id = kernel_group_plan.kernel_group_id
            start_block_pos = start_object_idx * kernel_group_plan.blocks_per_window
            end_block_pos = (
                start_object_idx + batch_len
            ) * kernel_group_plan.blocks_per_window
            block_ids_curr_batch = block_ids_tensor[start_block_pos:end_block_pos]
            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )
            group_lmcache_chunk_size = group_metadata.slots_per_chunk_in_window
            tmp_gpu_buffers_batched = [
                cache_context.get_temp_kernel_group_buffer(
                    i, kernel_group_id
                ).data_ptr()
                for i in range(batch_len)
            ]
            lmc_ops.multi_layer_block_kv_transfer(
                group_kv_pointers,
                tmp_gpu_buffers_batched,
                block_ids_curr_batch,
                cache_context.device,
                direction,
                cache_context.get_shape_desc(kernel_group_id),
                group_lmcache_chunk_size,
                group_metadata.engine_kv_format,
                (kernel_group_plan.skip_first_n_blocks if start_object_idx == 0 else 0),
            )

        if not is_h2d:
            for chunk_idx, memory_obj in enumerate(valid_memory_object_batch):
                lmcache_memcpy_async_d2h(
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_plan.object_group_id
                    ),
                    memory_obj,
                )


@dataclass
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    transfer_metadata: KVTransferMetadata
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._cache_contexts.pop(iid)))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            reaped.clear()
            self._release_entries(entries)
        return reaped_ids

    def _release_entries(self, entries: list[ContextEntry]) -> None:
        """Release a batch of entries and reclaim their device memory.

        Args:
            entries: The only remaining references to the released entries.
                The list is cleared before memory is reclaimed.
        """
        if not entries:
            return
        for entry in entries:
            entry.cache_context.close()
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
        del entry
        entries.clear()
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE,
                self.register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.STORE,
                self.store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.RETRIEVE,
                self.retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        with self._lock:
            entries = list(self._cache_contexts.values())
            self._cache_contexts.clear()
        self._release_entries(entries)

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        transfer_metadata = export_kv_transfer_metadata(
            cache_context.kv_layer_groups_manager,
            self._ctx.chunk_size,
        )
        layout_desc = build_object_group_layout_desc(
            transfer_metadata,
            self._ctx.chunk_size,
            object_group_id=0,
        )
        attn_desc = transfer_metadata.build_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name, world_size, layout_desc, attn_desc
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                transfer_metadata=transfer_metadata,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes).

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), or a copy fails, the whole store is skipped and nothing is
            committed (logged at WARNING); a subsequent retrieve simply misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        transfer_metadata = entry.transfer_metadata
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = len(transfer_metadata.object_groups)
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            try:
                transfer_plan = _plan_for_request(
                    transfer_metadata,
                    gpu_block_ids,
                    num_chunks,
                    TransferPlanDirection.STORE,
                )
            except ValueError as exc:
                logger.warning(
                    "Invalid STORE block IDs for request_id=%s; skipping the store: %s",
                    key.request_id,
                    exc,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for object_group_plan, obj_keys in zip(
                    transfer_plan.object_groups,
                    obj_keys_per_obj_group,
                    strict=True,
                ):
                    obj_group_id = object_group_plan.object_group_id
                    layout_desc = build_object_group_layout_desc(
                        transfer_metadata,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        obj_keys, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (skipped by the storage manager)
                    # become None entries; the helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    planned_memory_objs = _planned_memory_objects(
                        memory_objs, object_group_plan
                    )
                    if planned_memory_objs:
                        transfer_kv_per_object_group(
                            cache_context,
                            transfer_metadata,
                            object_group_plan,
                            _stage_object_group_plan_block_ids(
                                cache_context, object_group_plan
                            ),
                            planned_memory_objs,
                            batch_size=1,
                            direction=lmc_ops.TransferDirection.D2H,
                        )

                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        list(all_dict.keys()),
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
        )

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        transfer_metadata = entry.transfer_metadata
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = len(transfer_metadata.object_groups)
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            try:
                transfer_plan = _plan_for_request(
                    transfer_metadata,
                    gpu_block_ids,
                    num_chunks,
                    TransferPlanDirection.RETRIEVE,
                    skip_first_n_tokens,
                )
            except ValueError as exc:
                logger.error(
                    "Invalid RETRIEVE block IDs for request_id=%s; "
                    "skipping the retrieve: %s",
                    key.request_id,
                    exc,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for object_group_plan, obj_keys in zip(
                    transfer_plan.object_groups,
                    obj_keys_per_obj_group,
                    strict=True,
                ):
                    with self._ctx.storage_manager.read_prefetched_results(
                        obj_keys
                    ) as memory_objs:
                        if not memory_objs or len(memory_objs) != len(obj_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in memory_objs)

                        planned_memory_objs = _planned_memory_objects(
                            memory_objs, object_group_plan
                        )
                        if planned_memory_objs:
                            transfer_kv_per_object_group(
                                cache_context,
                                transfer_metadata,
                                object_group_plan,
                                _stage_object_group_plan_block_ids(
                                    cache_context, object_group_plan
                                ),
                                planned_memory_objs,
                                batch_size=cache_context.max_batch_size,
                                direction=lmc_ops.TransferDirection.H2D,
                            )
                        # Extend only after the copy is enqueued: on exception,
                        # read_prefetched_results releases this group's locks
                        # itself, and a key must not be released twice.
                        prefetched_keys.extend(obj_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == num_chunks * num_object_groups
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )
