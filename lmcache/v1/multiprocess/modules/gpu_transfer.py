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
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def get_layout_desc(
    cache_context: GPUCacheContext, num_tokens: int
) -> MemoryLayoutDesc:
    """Get the memory layout description for a given GPU context and number of tokens.

    Supports multiple KV layer groups with different shapes and dtypes.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    num_groups = cache_context.kv_layer_groups_manager.num_groups
    shapes = [
        cache_context.get_kv_buffer_shape(num_tokens, group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        cache_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
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
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The actual concrete type is whatever :func:`create_cache_context`
    returned -- currently always a :class:`GPUCacheContext`.

    Args:
        cache_context: Platform cache context managing shape and pointers
            to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
    """

    cache_context: GPUCacheContext
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
        self._cache_contexts: dict[int, ContextEntry] = {}

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
    def cache_contexts(self) -> dict[int, ContextEntry]:
        """Per-instance GPU context registry."""
        return self._cache_contexts

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

        for instance_id, entry in self._cache_contexts.items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": {
                    "num_layers": ctx.num_layers,
                    "inference_engine_logical_block_size": (
                        ctx.kv_layer_groups_manager.inference_engine_logical_block_size
                    ),
                    "group_physical_block_sizes": ctx.group_physical_block_sizes,
                    "group_compress_ratios": ctx.group_compress_ratios,
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
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        had_contexts = len(self._cache_contexts) > 0
        self._cache_contexts.clear()
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
        if instance_id in self._cache_contexts:
            logger.warning(
                "Instance %s's KV cache is already registered, "
                "skipping the new registration",
                instance_id,
            )
            return

        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            group_views=group_views,
            engine_type=engine_type,
        )
        self._cache_contexts[instance_id] = ContextEntry(
            cache_context=cache_context,
            model_name=model_name,
            world_size=world_size,
        )

        layout_desc = get_layout_desc(cache_context, self._ctx.chunk_size)
        self._ctx.layout_desc_registry.register(model_name, world_size, layout_desc)

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
        entry = self._cache_contexts.pop(instance_id, None)
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

        entry = self._cache_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.blocks_for_tokens(self._ctx.chunk_size, group_idx)
            for group_idx in range(cache_context.kv_layer_groups_manager.num_groups)
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            block_ids_per_group_gpu = cache_context.copy_view_block_ids_to_gpu(
                gpu_block_ids
            )

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete.
            if any(
                group_block_ids.shape[0] < len(obj_keys) * bpc
                for group_block_ids, bpc in zip(
                    block_ids_per_group_gpu, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "len(obj_keys) * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    len(obj_keys),
                    blocks_per_chunk,
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
                cache_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=cache_context.stream)

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
            store_succeeded = False
            try:
                layout_desc = get_layout_desc(cache_context, self._ctx.chunk_size)
                reserved_dict = self._ctx.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                # NOTE: Store is not batched because some obj_keys may be
                # skipped (not in reserved_dict), making block_ids
                # non-contiguous. Batching would require torch.cat to
                # reassemble block_ids, negating the benefit.
                num_groups = cache_context.kv_layer_groups_manager.num_groups
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key in reserved_dict:
                        memory_obj = reserved_dict[obj_key]
                    else:
                        continue

                    # Copy from GPU paged buffer to tmp buffer, then to CPU — per
                    # group. Each group uses its own block-id list (HMA).
                    for group_idx in range(num_groups):
                        bpc = blocks_per_chunk[group_idx]
                        chunk_block_ids_gpu = block_ids_per_group_gpu[group_idx][
                            idx * bpc : (idx + 1) * bpc
                        ]
                        tmp_buffer = cache_context.get_tmp_chunk_gpu_buffer(group_idx)
                        group_kv_pointers = cache_context.get_group_kv_pointers(
                            group_idx
                        )
                        # Kernel contract: ``group_lmcache_chunk_size`` here is the
                        # number of *physical* slots per chunk for this group
                        # (= logical chunk_size // compress_ratio).
                        group_lmcache_chunk_size = (
                            cache_context.get_physical_chunk_size(group_idx)
                        )
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            cache_context.device,
                            lmc_ops.TransferDirection.D2H,
                            cache_context.get_shape_desc(group_idx),
                            group_lmcache_chunk_size,
                            cache_context.gpu_kv_format_,
                            0,
                        )
                    # Store is not batched, so we always use chunk_idx=0 (single slot)
                    lmcache_memcpy_async_d2h(
                        cache_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
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
                        cache_context.cupy_stream,
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
        """
        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._cache_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name

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

        # ``skip_*_in_chunk`` is expressed in engine-block units
        # (logical tokens), which is what the kernel's
        # ``skip_blocks_in_chunk`` argument expects regardless
        # of per-group compression.
        ie_logical_block_size = (
            cache_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = cache_context.max_batch_size
            groups = cache_context.kv_layer_groups_manager.kv_layer_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self._ctx.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self._ctx.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire batch is within APC range, skip it
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self._ctx.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % ie_logical_block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to "
                        "inference_engine_logical_block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        ie_logical_block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // ie_logical_block_size,
                    )
                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                # Copy from CPU to GPU tmp buffers, then scatter to paged KV — per group
                # H2D copy: each memory_obj maps to its own batch slot
                for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        cache_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                    )
                for group_idx, group in enumerate(groups):
                    bpc = cache_context.blocks_for_tokens(
                        self._ctx.chunk_size, group_idx
                    )
                    chunk_block_ids_gpu = block_ids_per_group_gpu[group_idx][
                        start_chunk_id * bpc : end_chunk_id * bpc
                    ]
                    if chunk_block_ids_gpu.shape[0] != batch_len * bpc:
                        # Fail closed: a short block-id slice would make the
                        # transfer kernel write out-of-bounds GPU memory.
                        raise ValueError(
                            "RETRIEVE block ID underflow: "
                            f"group_idx={group_idx} "
                            f"engine_group_idx={group.engine_group_idx} "
                            f"batch={batch_idx} "
                            f"expected={batch_len * bpc} "
                            f"got={chunk_block_ids_gpu.shape[0]}"
                        )
                    group_skip_blocks = cache_context.blocks_for_tokens(
                        skip_tokens_in_chunk, group_idx
                    )
                    tmp_buffers = cache_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = cache_context.get_group_kv_pointers(group_idx)
                    group_lmcache_chunk_size = cache_context.get_physical_chunk_size(
                        group_idx
                    )

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        cache_context.device,
                        lmc_ops.TransferDirection.H2D,
                        cache_context.get_shape_desc(group_idx),
                        group_lmcache_chunk_size,
                        cache_context.gpu_kv_format_,
                        group_skip_blocks,
                    )

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            # Copy all block_ids to GPU once before the loop
            block_ids_per_group_gpu = cache_context.copy_view_block_ids_to_gpu(
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
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
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
