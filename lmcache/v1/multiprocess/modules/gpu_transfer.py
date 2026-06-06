# SPDX-License-Identifier: Apache-2.0
"""GPU-based KV cache transfer operations for the MPCacheEngine."""

# Standard
from dataclasses import dataclass
from itertools import islice
from typing import Generator
import time

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
    check_interprocess_event_support,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.group_view import LMCacheGroupView
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def get_layout_desc(gpu_context: GPUCacheContext) -> MemoryLayoutDesc:
    """Memory layout description for all KV layer groups in a GPU context."""
    num_groups = gpu_context.kv_layer_groups_manager.num_groups
    shapes = [
        gpu_context.get_storage_kv_buffer_shape(group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
        for group_idx in range(num_groups)
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def batched_iteration(lst: list, batch_size: int) -> Generator[tuple, None, None]:
    """Utility function to iterate over a list in batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.

    Raises:
        ValueError: If batch_size is less than 1.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class GPUContextEntry:
    """Registered GPU context metadata for a single worker instance.

    Args:
        gpu_context: The GPU cache context managing shape and pointers
            to vLLM GPU KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
    """

    gpu_context: GPUCacheContext
    model_name: str
    world_size: int


class GPUTransferModule:
    """Handles GPU-based KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx
        self._gpu_contexts: dict[int, GPUContextEntry] = {}

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
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    @property
    def gpu_contexts(self) -> dict[int, GPUContextEntry]:
        """Per-instance GPU context registry."""
        return self._gpu_contexts

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
        gpu_context_meta: dict[str, dict] = {}

        for instance_id, entry in self._gpu_contexts.items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.gpu_context
            gpu_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": {
                    "num_layers": ctx.num_layers,
                    "group_physical_block_sizes": ctx.group_physical_block_sizes,
                    "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                    "dtype": str(ctx.dtype),
                    "is_mla": ctx.is_mla,
                    "num_blocks": ctx.num_blocks,
                    "gpu_kv_format": ctx.gpu_kv_format_name,
                    "gpu_kv_shape": ctx.gpu_kv_shape,
                    "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                    "attention_backend": ctx.attention_backend,
                    "cache_size_per_token": ctx.cache_size_per_token(),
                },
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "gpu_context_meta": gpu_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        had_contexts = len(self._gpu_contexts) > 0
        self._gpu_contexts.clear()
        if had_contexts:
            torch_dev.empty_cache()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        group_views: list[LMCacheGroupView],
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
            group_views: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        if instance_id in self._gpu_contexts:
            logger.warning(
                "Instance %s's KV cache is already registered, "
                "skipping the new registration",
                instance_id,
            )
            return

        gpu_context = GPUCacheContext(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            group_views=group_views,
            engine_type=engine_type,
        )
        self._gpu_contexts[instance_id] = GPUContextEntry(
            gpu_context=gpu_context,
            model_name=model_name,
            world_size=world_size,
        )

        layout_desc = get_layout_desc(gpu_context)
        self._ctx.layout_desc_registry.register(model_name, world_size, layout_desc)

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        entry = self._gpu_contexts.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)
        torch_dev.empty_cache()

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
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
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        gpu_context = entry.gpu_context
        model_name = entry.model_name

        groups = gpu_context.kv_layer_groups_manager.kv_layer_groups
        num_groups = len(groups)
        blocks_per_chunk_per_group = [
            group.storage_blocks_per_chunk for group in groups
        ]

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            block_ids_per_group_gpu = gpu_context.copy_view_block_ids_to_gpu(
                gpu_block_ids
            )

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Each group has its own per-chunk block count under HMA.
            num_chunks_requested = len(obj_keys)
            if any(
                group_block_ids.shape[0]
                < num_chunks_requested * blocks_per_chunk_per_group[group_idx]
                for group_idx, group_block_ids in enumerate(block_ids_per_group_gpu)
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: need %s block "
                    "IDs per group for %d chunks; skipping the store.",
                    key.request_id,
                    [
                        num_chunks_requested * bpc
                        for bpc in blocks_per_chunk_per_group
                    ],
                    num_chunks_requested,
                )
                event.record()
                return event.ipc_handle(), False

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC event "
                    "handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                )
            )

            self._ctx.event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(gpu_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            store_succeeded = False
            try:
                layout_desc = get_layout_desc(gpu_context)
                reserved_dict = self._ctx.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                # NOTE: Store is not batched because some obj_keys may be
                # skipped (not in reserved_dict), making block_ids
                # non-contiguous. Batching would require torch.cat to
                # reassemble block_ids, negating the benefit.
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key in reserved_dict:
                        memory_obj = reserved_dict[obj_key]
                    else:
                        continue

                    # D2H per group; block IDs are already pre-trimmed by the connector.
                    for group_idx in range(num_groups):
                        group = groups[group_idx]
                        bpc = blocks_per_chunk_per_group[group_idx]
                        chunk_block_ids_gpu = block_ids_per_group_gpu[group_idx][
                            idx * bpc : (idx + 1) * bpc
                        ]
                        tmp_buffer = gpu_context.get_tmp_chunk_gpu_buffer(group_idx)
                        group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                        group_lmcache_chunk_size = group.storage_slots_per_chunk
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.D2H,
                            gpu_context.get_shape_desc(group_idx),
                            group_lmcache_chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )
                    # Store is not batched, so we always use chunk_idx=0 (single slot)
                    lmcache_memcpy_async_d2h(
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
                    )
                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(reserved_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_write",
                        list(reserved_dict.keys()),
                    )
                # All reserved MemoryObjs share one layout_desc, so per-object
                # size is identical — avoid summing N identical values.
                total_bytes = (
                    next(iter(reserved_dict.values())).get_size() * stored_count
                    if stored_count
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                        },
                    ),
                )

        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self._ctx.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_blocks_per_group: list[int] = [],
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: IPC key for the KV cache blocks (worker_id must be set).
            instance_id: GPU instance ID (PID).
            gpu_block_ids: Block IDs to retrieve into, per LMCache KV group.
            event_ipc_handle: CUDA event IPC handle for synchronization.
            skip_blocks_per_group: Leading stored-blocks to skip per group
                (APC-overlap guard). Empty -> no skip.

        Returns:
            ``(event_ipc_handle, success)``.

        Raises:
            ValueError: If no GPU context is registered for *instance_id*.
        """
        skip_blocks_per_group = skip_blocks_per_group or []
        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        gpu_context = entry.gpu_context
        model_name = entry.model_name

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            )
        )

        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(gpu_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        # Per-group blocks-per-chunk = the count the connector sends per chunk
        # (already SWA-trimmed); the server moves exactly these blocks.
        groups = gpu_context.kv_layer_groups_manager.kv_layer_groups
        blocks_per_chunk_per_group = [
            group.storage_blocks_per_chunk for group in groups
        ]

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                # Copy from CPU to GPU tmp buffers, then scatter to paged KV — per group
                # H2D copy: each memory_obj maps to its own batch slot
                for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                    )
                for group_idx, group in enumerate(groups):
                    blocks_per_chunk = blocks_per_chunk_per_group[group_idx]
                    chunk_block_ids_gpu = block_ids_per_group_gpu[group_idx][
                        start_chunk_id * blocks_per_chunk : end_chunk_id
                        * blocks_per_chunk
                    ]
                    if chunk_block_ids_gpu.shape[0] != batch_len * blocks_per_chunk:
                        # Fail closed: a short block-id slice would make the
                        # transfer kernel write out-of-bounds GPU memory.
                        raise ValueError(
                            "RETRIEVE block ID underflow: "
                            f"group_idx={group_idx} "
                            f"engine_group_idx={group.engine_group_idx} "
                            f"batch={batch_idx} "
                            f"expected={batch_len * blocks_per_chunk} "
                            f"got={chunk_block_ids_gpu.shape[0]}"
                        )
                    # APC skip: pre-computed by connector, applies to batch 0 only.
                    skip_blocks_in_chunk = (
                        skip_blocks_per_group[group_idx]
                        if batch_idx == 0 and group_idx < len(skip_blocks_per_group)
                        else 0
                    )
                    tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                    # Physical slots transferred per chunk for this group
                    # (= bpc * shape_desc.bs); satisfies the kernel's
                    # ``num_blocks_per_object * bs == lmcache_chunk_size`` check.
                    group_lmcache_chunk_size = group.storage_slots_per_chunk

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.get_shape_desc(group_idx),
                        group_lmcache_chunk_size,
                        gpu_context.gpu_kv_format_,
                        skip_blocks_in_chunk,
                    )

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            # Copy all block_ids to GPU once before the loop
            block_ids_per_group_gpu = gpu_context.copy_view_block_ids_to_gpu(
                gpu_block_ids
            )

            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            total_bytes = 0
            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    total_bytes = sum(mo.get_size() for mo in memory_objs)
                    _retrieve_loop(obj_keys, memory_objs)
                # Only set True when with-block exits normally
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                        },
                    ),
                )
        tokens_retrieved = len(obj_keys) * self._ctx.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), True
