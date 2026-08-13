# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from itertools import islice
from typing import Any, Generator, Sequence, cast
import threading
import time

# Third Party
import msgspec
import torch

# First Party
from lmcache import device_ops, torch_dev
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
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
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
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.platform.cuda.cache_context import GPUCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    device_ops, "execute_object_group_transfer"
)
_warned_layerwise_fallback: bool = False


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
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def batched_iteration_with_skip(
    lst: Sequence,
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple], None, None]:
    """Utility function to iterate over a list in batches with an initial skip.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.
        skip_count: The number of items to skip at the start of the list.

    Yields:
        Tuples of (batch_start_idx, batch) where batch is a tuple of items
        from the list, and batch_start_idx is the "original" index of the first
        item in the batch.

    Raises:
        ValueError: If batch_size is less than 1 or skip_count is negative.

    Note:
        Batch_idx is the index of the batch in the original list, accounting
        for the skipped items. For example, if skip_count is 10 and batch_size
        is 5, the first yielded batch will have batch_start_idx=10.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    it = iter(lst)
    # Skip the initial items
    for _ in range(skip_count):
        next(it, None)
    batch_start_idx = skip_count
    while batch := tuple(islice(it, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def all_null_chunk_masks(
    block_ids: Sequence[Sequence[int]],
    object_groups: Sequence[ObjectGroupInfo],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
) -> list[list[bool]]:
    """Mark, per object group, the chunks whose engine block ids are all null.

    A chunk is null for an object group when every block id of every kernel
    group in that group is 0 (the vLLM null block). Align-mode Mamba/linear
    layers produce such chunks: only the block holding the last recurrent state
    is real, so every earlier chunk is null. These chunks must not be stored --
    the null block carries no valid KV, and object keys are content hashes, so
    committing them would serve garbage to a later prefix hit.

    Args:
        block_ids: Raw per-kernel-group engine block ids (before any downsample),
            indexed by kernel-group index.
        object_groups: The object groups, indexed by object-group id.
        blocks_per_chunk: Blocks in one chunk per kernel group, indexed by
            kernel-group index.
        num_chunks: Number of chunks in the request.

    Returns:
        ``mask[g][i]`` is True iff chunk ``i`` is all-null for object group ``g``.
    """
    masks: list[list[bool]] = []
    for group in object_groups:
        chunk_null: list[bool] = []
        for i in range(num_chunks):
            is_null = True
            for kg in group.kernel_group_indices:
                bpc = blocks_per_chunk[kg]
                if any(block_ids[kg][i * bpc : (i + 1) * bpc]):
                    is_null = False
                    break
            chunk_null.append(is_null)
        masks.append(chunk_null)
    return masks


def downsample_and_stage_block_ids(
    cache_context: BaseCacheContext,
    block_ids: list[list[int]],
) -> list[torch.Tensor]:
    """Cut the block id lists to skip the unneeded blocks in a chunk and
    stage it into GPU tensors for later use.

    This mainly targets the case where a portion of the blocks are not
    needed for every chunk, such as deepseek v4's swa cache.

    Note that the we do NOT do any object-level skipping here.

    Args:
        cache_context: The cache context containing the KV cache information.
        block_ids: The original block id lists, indexed by LMCache KV group index.

    Returns:
        The cut block id lists, indexed by LMCache KV group index.

    Note:
        This function has some coupled logic with transfer_kv_per_object_group below.
        The caller need to make sure that the block ids seen by
        transfer_kv_per_object_group are produced by this function.

    Example:
        If a model have 2 kernel groups, one is full attention with block size 32,
        one is swa attention with block size 32 and sliding window size 64, and
        LMCache has a chunk size of 128. And there are 2 chunks in total (256 tokens).

        The input will be:
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # block ids for the full attention group
          [11, 12, 13, 14, 15, 16, 17, 18], # block ids for the swa attention group
        ]

        The output will be
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # full attention group still needs all block ids
          [13, 14, 17, 18], # swa attention group only needs the last 2 block per chunk
        ]
    """
    num_kernel_groups = cache_context.kv_layer_groups_manager.num_kernel_groups
    for kernel_group_id in range(num_kernel_groups):
        subchunk_sw_size_tokens = (
            cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(
                kernel_group_id
            )
        )
        tokens_per_chunk = min(
            cache_context.lmcache_tokens_per_chunk, subchunk_sw_size_tokens
        )
        keep_blocks_per_chunk = cache_context.calculate_num_blocks(
            tokens_per_chunk, kernel_group_id
        )
        total_blocks_per_chunk = cache_context.calculate_num_blocks(
            cache_context.lmcache_tokens_per_chunk, kernel_group_id
        )

        new_block_ids = []
        old_block_ids = block_ids[kernel_group_id]
        assert len(old_block_ids) % total_blocks_per_chunk == 0, (
            f"len(block_ids[{kernel_group_id}]) should be a multiple "
            f"of total_blocks_per_chunk ({total_blocks_per_chunk}), but got "
            f"{len(old_block_ids)}"
        )

        for i in range(0, len(old_block_ids), total_blocks_per_chunk):
            chunk_block_ids = old_block_ids[i : i + total_blocks_per_chunk]
            new_block_ids.extend(chunk_block_ids[-keep_blocks_per_chunk:])

        block_ids[kernel_group_id] = new_block_ids

    # Stage the cut block ids into GPU tensors
    block_ids_gpu = cache_context.stage_block_ids(block_ids)
    return block_ids_gpu


def _recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip for a batch of chunks based
    on the blocks per chunk and blocks per sliding window WHEN the window
    size is smaller than the lmcache chunk size.

    Args:
        blocks_per_chunk: The total number of blocks in one chunk for the
            current group.
        blocks_per_window: The number of blocks in the sliding window
            for the current group. Should be less than or equal to
            blocks_per_chunk.
        blocks_to_skip: The number of blocks to skip.

    Returns:
        The re-calculated number of blocks to skip for the current batch of
        chunks.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


def _run_object_group_transfer_plan(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    kv_interleaved: bool = False,
) -> None:
    """Plan and execute one object group's transfer in a single native call.

    This is the fast path of :func:`transfer_kv_per_object_group`: it runs the
    same batched-iteration / skip logic, but instead of issuing each staging
    copy and kernel launch immediately (each a GIL release/re-acquire), it
    resolves every argument to plain pointers/scalars (the "planner", GIL held
    throughout) and hands the whole plan to ``execute_object_group_transfer``,
    which issues all of it on the stream within a single GIL release.

    Requires every object to be non-GDS (staged through the lazy-allocator
    path); the caller skips groups that contain any GDS-backed object.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The MemoryObj instances to copy. None entries are only
            valid for D2H (the batch is skipped); H2D raises.
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).

    Raises:
        ValueError: If a None entry is found in memory_objs when direction is
            H2D, or if an object's size does not match its GPU staging buffer.

    Note:
        ``kv_interleaved`` is used exclusively by the MP LMCache-driven
        layerwise mode (LMCACHE_MP_LAYERWISE_BATCH > 0).  When True the
        scatter/gather kernel treats the host buffer as per-layer
        interleaved [K0,V0, K1,V1, ...] instead of grouped
        [K0,K1,..., V0,V1,...].
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    # --- Per-kernel-group invariants, resolved once (vs. every batch before) ---
    kernel_group_specs: list[Any] = []
    spec_index_by_kg: dict[int, int] = {}
    blocks_per_chunk_by_kg: dict[int, int] = {}
    blocks_per_window_by_kg: dict[int, int] = {}
    for kernel_group_id in kernel_group_ids:
        blocks_per_chunk = cache_context.calculate_num_blocks(
            lmcache_chunk_size, kernel_group_id
        )
        tokens_per_window = min(
            lmcache_chunk_size,
            kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
        )
        blocks_per_window = cache_context.calculate_num_blocks(
            tokens_per_window, kernel_group_id
        )
        blocks_per_chunk_by_kg[kernel_group_id] = blocks_per_chunk
        blocks_per_window_by_kg[kernel_group_id] = blocks_per_window

        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        block_ids_tensor = block_ids_gpu[kernel_group_id]
        temp_buffers = [
            cache_context.get_temp_kernel_group_buffer(slot, kernel_group_id)
            for slot in range(max_batch_size)
        ]

        sd = cache_context.get_shape_desc(kernel_group_id)
        if kv_interleaved:
            sd.kv_interleaved = True

        spec_index_by_kg[kernel_group_id] = len(kernel_group_specs)
        kernel_group_specs.append(
            device_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                sd,
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                block_ids_tensor.data_ptr(),
                block_ids_tensor.numel(),
            )
        )

    # Temp object-group staging buffers (reused per batch slot, like above).
    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id)
        for slot in range(max_batch_size)
    ]

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    # --- Walk the batches in order, emitting staging + launch work per step ---
    batch_steps: list[Any] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
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
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        staging = build_staging_copies(
            memory_object_batch,
            object_group_buffers[:batch_len],
            is_h2d,
        )

        launches: list[Any] = []
        for kernel_group_id in kernel_group_ids:
            blocks_per_chunk = blocks_per_chunk_by_kg[kernel_group_id]
            blocks_per_window = blocks_per_window_by_kg[kernel_group_id]

            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            recalculated_skip_blocks = _recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            launches.append(
                device_ops.LaunchVar(
                    spec_index_by_kg[kernel_group_id],
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    recalculated_skip_blocks,
                )
            )

        batch_steps.append(device_ops.BatchStep(staging, launches))

    if not batch_steps:
        return

    execute_object_group_transfer = device_ops.execute_object_group_transfer
    execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
    )


def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    kv_interleaved: bool = False,
) -> None:
    """Helper function to transfer memory objects of a single object group
    to/from GPU, with batching support.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs to retrieve into, indexed by LMCache KV group
            index. It should satisfy `len(block_ids_gpu[i]) == len(memory_objs) *
            blocks_per_chunk[i]` for each group `i`.
            Note that the block IDs list are already on GPU.
        memory_objs: The list of MemoryObj instances to copy from. It could be
            None when allocation or retrieval fails. For store (D2H), it should
            ignore the None entry and continue copying the rest. For retrieve
            (H2D), it should raise the error and stop copying.
        object_group_id: Index of the object group being copied.
        batch_size: The number of memory objects to perform batched copy
        skip_first_n_tokens: Number of tokens to skip writing at the start of
            the retrieve range. This avoids overwriting APC-shared GPU blocks that
            may be read concurrently by other requests.
        direction: The transfer direction, H2D (retrieve) or D2H (store).
        kv_interleaved: If True, the scatter/gather kernel uses per-layer
            interleaved host buffer layout [K0,V0, K1,V1, ...].
            Only set by the MP LMCache-driven layerwise store path
            (LMCACHE_MP_LAYERWISE_BATCH > 0).

    Raises:
        ValueError: If it founds None entry in memory_objs when direction is H2D.
    Note:
        This function expects the caller to stage the block ids (list[list[int]])
        into GPU tensors and pass them in as `block_ids_gpu`.
    """
    if _HAS_NATIVE_OBJECT_GROUP_TRANSFER and not any(
        isinstance(mo, GDSMemoryObject) for mo in memory_objs
    ):
        _run_object_group_transfer_plan(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            kv_interleaved=kv_interleaved,
        )
        return

    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
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
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        # For H2D, copy from CPU to GPU tmp buffers before the kernel launch
        if is_h2d:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                lmcache_memcpy_async_h2d(
                    memory_obj,
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_id
                    ),
                )

        # Do paged KV copy
        for kernel_group_id in kernel_group_ids:
            blocks_per_chunk = cache_context.calculate_num_blocks(
                lmcache_chunk_size, kernel_group_id
            )
            tokens_per_window = min(
                lmcache_chunk_size,
                kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
            )
            blocks_per_window = cache_context.calculate_num_blocks(
                tokens_per_window, kernel_group_id
            )

            # Get the block ids for this chunk
            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            block_ids_curr_batch = block_ids_gpu[kernel_group_id][
                start_block_pos:end_block_pos
            ]

            # Re-calculate the skip blocks for this kernel group
            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            recalculated_skip_blocks = _recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            # Launch kernel
            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )
            group_lmcache_chunk_size = cache_context.get_slots_per_chunk_in_sw(
                kernel_group_id
            )
            tmp_gpu_buffers_batched = [
                cache_context.get_temp_kernel_group_buffer(
                    i, kernel_group_id
                ).data_ptr()
                for i in range(batch_len)
            ]
            sd = cache_context.get_shape_desc(kernel_group_id)
            device_ops.multi_layer_block_kv_transfer(
                group_kv_pointers,
                tmp_gpu_buffers_batched,
                block_ids_curr_batch,
                cache_context.device,
                direction,
                sd,
                group_lmcache_chunk_size,
                cache_context.get_engine_kv_format(kernel_group_id),
                recalculated_skip_blocks,
            )

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
        if not is_h2d:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                lmcache_memcpy_async_d2h(
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_id
                    ),
                    memory_obj,
                )


def transfer_kv_layerwise(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    layer_events: list,
    event_backend: EventIPCBackend,
    batch_leader_map: dict[int, int] | None = None,
    layerwise_batch: int = 1,
    event_export_callback=None,
) -> None:
    """Transfer KV cache in layer-major order, recording per-layer events.

    Instead of copying all layers for each chunk batch (chunk-major),
    this function copies all chunks for each layer (layer-major).
    After each layer's data is fully on GPU, it records the corresponding
    event so vLLM can start that layer's attention immediately.

    The real pipeline overlap is cross-process: while this server does
    H2D(i+1)+scatter(i+1), vLLM's attention(i) runs concurrently via
    IPC event synchronization (same pattern as in-process per-layer).

    Args:
        cache_context: The GPU cache context containing KV cache info.
        block_ids_gpu: GPU block IDs indexed by LMCache KV group index.
        memory_objs: List of MemoryObj instances (one per chunk).
        object_group_id: Index of the object group being transferred.
        batch_size: Max chunks per batch (for temp buffer reuse).
        skip_first_n_tokens: Tokens to skip at the start of the range.
        layer_events: List of pre-created events, one per layer.
            Events are recorded as each layer completes.
        event_backend: The event backend for recording events.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id):
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)

    # Validate allocator type once (avoid per-chunk isinstance in hot loop)
    for mo in memory_objs[num_objects_to_skip:]:
        if mo is not None and not isinstance(mo.parent(), LazyMemoryAllocator):
            raise NotImplementedError(
                "Per-layer H2D for non-LazyMemoryAllocator not yet "
                "implemented. Use LazyMemoryAllocator for per-layer transfer."
            )

    # Cache per-kernel-group invariants on first call (avoids repeated
    # property lookups, dict accesses, and object construction on hot path).
    _lw_cache_attr = "_layerwise_invariants"
    if not hasattr(cache_context, _lw_cache_attr):
        setattr(cache_context, _lw_cache_attr, {})
    _lw_cache = getattr(cache_context, _lw_cache_attr)

    cache_key = object_group_id
    if cache_key in _lw_cache:
        kg_infos, all_layers = _lw_cache[cache_key]
    else:
        kg_infos = []
        for kernel_group_id in kernel_group_ids:
            kg = kv_groups_manager.kernel_groups[kernel_group_id]
            blocks_per_chunk = cache_context.calculate_num_blocks(
                lmcache_chunk_size, kernel_group_id
            )
            tokens_per_window = min(
                lmcache_chunk_size,
                kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
            )
            blocks_per_window = cache_context.calculate_num_blocks(
                tokens_per_window, kernel_group_id
            )
            sd = kg.shape_desc
            slots_per_chunk = cache_context.get_slots_per_chunk_in_sw(kernel_group_id)
            per_kv_bytes = slots_per_chunk * kg.hidden_dim_size * sd.element_size
            per_layer_bytes = sd.kv_size * per_kv_bytes

            _tb = cast("GPUCacheContext", cache_context)._temp_buffer
            kg_byte_offset_in_obj = _tb._offset_map.get(
                (0, object_group_id, kernel_group_id)
            )
            obj_group_offset = _tb._offset_map_object_group_only.get(
                (0, object_group_id)
            )
            if kg_byte_offset_in_obj is None or obj_group_offset is None:
                raise ValueError(
                    f"Cannot find temp buffer offset for "
                    f"kernel_group={kernel_group_id}, "
                    f"object_group={object_group_id}"
                )
            kg_byte_offset = kg_byte_offset_in_obj[0] - obj_group_offset[0]

            single_layer_sd = device_ops.PageBufferShapeDesc()
            single_layer_sd.kv_size = sd.kv_size
            single_layer_sd.nl = 1
            single_layer_sd.nb = sd.nb
            single_layer_sd.bs = sd.bs
            single_layer_sd.nh = sd.nh
            single_layer_sd.hs = sd.hs
            single_layer_sd.element_size = sd.element_size
            single_layer_sd.block_stride_elems = sd.block_stride_elems
            single_layer_sd.kv_interleaved = False

            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )

            kg_infos.append(
                {
                    "kernel_group_id": kernel_group_id,
                    "kg": kg,
                    "blocks_per_chunk": blocks_per_chunk,
                    "blocks_per_window": blocks_per_window,
                    "slots_per_chunk": slots_per_chunk,
                    "per_layer_bytes": per_layer_bytes,
                    "kg_byte_offset": kg_byte_offset,
                    "sd": sd,
                    "single_layer_sd": single_layer_sd,
                    "group_kv_pointers": group_kv_pointers,
                }
            )

        all_layers = []
        for kg_info_idx, info in enumerate(kg_infos):
            kg = info["kg"]
            for local_idx, global_layer_idx in enumerate(kg.layer_indices):
                all_layers.append((kg_info_idx, local_idx, global_layer_idx))

        _lw_cache[cache_key] = (kg_infos, all_layers)

    # Sort by global layer index to ensure layer-major order
    all_layers.sort(key=lambda x: x[2])

    if not all_layers:
        return

    # Pre-compute batch plan (identical for every layer, avoid re-iteration)
    batch_plan: list[tuple[int, tuple]] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            raise ValueError(
                "MemoryObj is None for some objects in the batch during "
                "layerwise H2D transfer."
            )
        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue
        batch_plan.append((start_object_idx, memory_object_batch))

    if not batch_plan:
        # Nothing to transfer; still record 1 event so consumer doesn't hang.
        first_gl = all_layers[0][2] if all_layers else 0
        if first_gl < len(layer_events):
            event_backend.record_event(layer_events[first_gl], cache_context.stream)
        if batch_leader_map is not None:
            for _, _, global_layer_idx in all_layers:
                batch_leader_map[global_layer_idx] = first_gl
        return

    main_stream = cache_context.stream
    pin_chunk_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
    # Batch N layers per IPC event to reduce cross-process sync
    # overhead.  0 = layerwise disabled (caller should not reach here);
    # 1 = one event per layer (original behaviour); N>1 = N layers
    # transferred + scattered before events are recorded.
    layerwise_batch_size = max(1, layerwise_batch)
    use_native_layerwise_plan = _HAS_NATIVE_OBJECT_GROUP_TRANSFER

    # Flatten all valid chunks for single-launch-per-layer optimization.
    # In the existing temp buffer (4 full-chunk slots per kernel group),
    # each slot holds nl per-layer entries, giving 4*nl per-layer slots
    # total -- enough for hundreds of chunks on typical LLMs.
    all_chunks_flat: list[tuple[int, MemoryObj]] = []
    for start_object_idx, memory_object_batch in batch_plan:
        for local_idx, mo in enumerate(memory_object_batch):
            all_chunks_flat.append((start_object_idx + local_idx, mo))

    num_active_chunks = len(all_chunks_flat)

    # Pre-compute per-kernel-group block_ids and skip for single-launch path
    for info in kg_infos:
        sd = info["sd"]
        kernel_group_id = info["kernel_group_id"]
        blocks_per_window = info["blocks_per_window"]
        blocks_per_chunk = info["blocks_per_chunk"]
        per_layer_bytes = info["per_layer_bytes"]

        # Flat buffer capacity: total temp buffer bytes
        total_buffer_bytes = (
            cache_context.max_batch_size
            * cache_context.get_temp_kernel_group_buffer(0, kernel_group_id).nelement()
            * cache_context.get_temp_kernel_group_buffer(
                0, kernel_group_id
            ).element_size()
        )
        info["total_buffer_bytes"] = total_buffer_bytes
        # Per-layer check (for fallback)
        max_per_layer_slots = total_buffer_bytes // per_layer_bytes
        info["single_launch"] = num_active_chunks <= max_per_layer_slots

        if num_active_chunks > 0:
            first_obj_idx = all_chunks_flat[0][0]
            last_obj_idx = all_chunks_flat[-1][0]
            first_block = first_obj_idx * blocks_per_window
            end_block = (last_obj_idx + 1) * blocks_per_window
            info["all_block_ids"] = block_ids_gpu[kernel_group_id][
                first_block:end_block
            ]
            first_start_token = first_obj_idx * lmcache_chunk_size
            skip_tokens_first = (
                max(first_start_token, skip_first_n_tokens) - first_start_token
            )
            orig_skip = cache_context.calculate_num_blocks(
                skip_tokens_first, kernel_group_id
            )
            info["all_skip_blocks"] = _recalculate_blocks_to_skip(
                blocks_per_chunk, blocks_per_window, orig_skip
            )

    # --- Layer-major loop with N-layer merged H2D + scatter ---
    # When layerwise_batch_size > 1, N consecutive same-kernel-group layers
    # are copied with one contiguous H2D per chunk and one scatter kernel
    # with nl=N + interleaved LMCache offset, exploiting the L1 layout
    # [K0,V0,K1,V1,...] where consecutive layers are contiguous.
    num_all_layers = len(all_layers)
    layer_batch_start = 0

    while layer_batch_start < num_all_layers:
        # Determine batch: consecutive same-kg layers, up to layerwise_batch_size
        first_kg_idx = all_layers[layer_batch_start][0]
        first_local = all_layers[layer_batch_start][1]
        batch_end = layer_batch_start + 1
        while (
            batch_end < num_all_layers
            and batch_end - layer_batch_start < layerwise_batch_size
            and all_layers[batch_end][0] == first_kg_idx
            and all_layers[batch_end][1]
            == first_local + (batch_end - layer_batch_start)
        ):
            batch_end += 1

        n_in_batch = batch_end - layer_batch_start
        info = kg_infos[first_kg_idx]
        kernel_group_id = info["kernel_group_id"]
        per_layer_bytes = info["per_layer_bytes"]
        kg_byte_offset = info["kg_byte_offset"]
        group_kv_pointers = info["group_kv_pointers"]
        slots_per_chunk = info["slots_per_chunk"]
        sd = info["sd"]

        # Check if N-layer merged path can be used:
        # - Native ops available
        # - At least 1 chunk fits in staging buffer for N layers
        n_bytes = n_in_batch * per_layer_bytes
        max_chunks_per_pass = (
            info["total_buffer_bytes"] // n_bytes if n_bytes > 0 else 0
        )
        can_merge = (
            use_native_layerwise_plan
            and n_in_batch > 1
            and info.get("all_block_ids") is not None
            and max_chunks_per_pass >= 1
        )

        if can_merge:
            # --- N-layer merged path: single execute call, multiple BatchSteps
            # Same pattern as _run_object_group_transfer_plan: build one
            # KernelGroupSpec with reusable buffer slots + full block_ids,
            # then one BatchStep per chunk sub-pass (buffer reused across
            # passes, serialized on the stream).
            n_layer_sd = device_ops.PageBufferShapeDesc()
            n_layer_sd.kv_size = sd.kv_size
            n_layer_sd.nl = n_in_batch
            n_layer_sd.nb = sd.nb
            n_layer_sd.bs = sd.bs
            n_layer_sd.nh = sd.nh
            n_layer_sd.hs = sd.hs
            n_layer_sd.element_size = sd.element_size
            n_layer_sd.block_stride_elems = sd.block_stride_elems
            n_layer_sd.kv_interleaved = True

            n_layer_kv_ptrs = group_kv_pointers[first_local : first_local + n_in_batch]

            buffer_base = cache_context.get_temp_kernel_group_buffer(
                0, kernel_group_id
            ).data_ptr()
            src_layer_offset = kg_byte_offset + first_local * per_layer_bytes
            blocks_per_window = info["blocks_per_window"]

            # Fixed gpu_ptrs for max_chunks_per_pass buffer slots (reused)
            slot_gpu_ptrs = [
                buffer_base + slot * n_bytes for slot in range(max_chunks_per_pass)
            ]

            # One KernelGroupSpec for this kg batch, referencing full
            # block_ids and reusable buffer slot pointers.
            all_block_ids = info["all_block_ids"]
            layer_spec = device_ops.KernelGroupSpec(
                n_layer_kv_ptrs.data_ptr(),
                slot_gpu_ptrs,
                n_layer_sd,
                slots_per_chunk,
                cache_context.get_engine_kv_format(kernel_group_id),
                all_block_ids.data_ptr(),
                all_block_ids.numel(),
            )

            # Build one BatchStep per chunk sub-pass
            batch_steps: list = []
            chunk_pass_start = 0
            while chunk_pass_start < num_active_chunks:
                chunk_pass_end = min(
                    chunk_pass_start + max_chunks_per_pass, num_active_chunks
                )
                pass_chunks = all_chunks_flat[chunk_pass_start:chunk_pass_end]
                pass_count = len(pass_chunks)

                staging: list = []
                for buf_idx, (_, memory_obj) in enumerate(pass_chunks):
                    gpu_dst = slot_gpu_ptrs[buf_idx]
                    src_ptr = memory_obj.data_ptr + src_layer_offset
                    staging.append(
                        device_ops.StagingCopy(
                            gpu_dst,
                            src_ptr,
                            n_bytes,
                            memory_obj.meta.address,
                        )
                    )

                # Block offset within sliced block_ids for this sub-pass
                first_obj_idx = pass_chunks[0][0]
                last_obj_idx = pass_chunks[-1][0]
                base_obj_idx = all_chunks_flat[0][0]
                block_ids_offset = (first_obj_idx - base_obj_idx) * blocks_per_window
                total_blocks = (last_obj_idx + 1 - first_obj_idx) * blocks_per_window

                first_start_token = first_obj_idx * lmcache_chunk_size
                skip_tokens_first = (
                    max(first_start_token, skip_first_n_tokens) - first_start_token
                )
                pass_skip = _recalculate_blocks_to_skip(
                    info["blocks_per_chunk"],
                    blocks_per_window,
                    cache_context.calculate_num_blocks(
                        skip_tokens_first, kernel_group_id
                    ),
                )

                layer_launch = device_ops.LaunchVar(
                    0,
                    block_ids_offset,
                    total_blocks,
                    pass_count,
                    pass_skip,
                )
                batch_steps.append(device_ops.BatchStep(staging, [layer_launch]))
                chunk_pass_start = chunk_pass_end

            # Single native call for all chunk sub-passes
            device_ops.execute_object_group_transfer(
                lmcache_native.TransferDirection.H2D,
                cache_context.device,
                pin_chunk_size,
                [layer_spec],
                batch_steps,
            )
        else:
            # --- Per-layer fallback (N=1, no native ops, or buffer overflow) ---
            if n_in_batch > 1:
                global _warned_layerwise_fallback
                if not _warned_layerwise_fallback:
                    _warned_layerwise_fallback = True
                    reason = (
                        "native ops unavailable"
                        if not use_native_layerwise_plan
                        else f"staging buffer too small ({max_chunks_per_pass}"
                        f" chunks fit, need >=1)"
                    )
                    logger.warning(
                        "Layerwise merged H2D path unavailable (batch=%d, kg=%d): %s. "
                        "Falling back to %d separate per-layer H2D copies.",
                        n_in_batch,
                        kernel_group_id,
                        reason,
                        n_in_batch,
                    )
            for sub_idx in range(n_in_batch):
                layer_local_idx = first_local + sub_idx
                global_layer_idx = all_layers[layer_batch_start + sub_idx][2]
                single_layer_sd = info["single_layer_sd"]
                single_layer_kv_ptr = group_kv_pointers[
                    layer_local_idx : layer_local_idx + 1
                ]

                if info["single_launch"] and use_native_layerwise_plan:
                    buffer_base = cache_context.get_temp_kernel_group_buffer(
                        0, kernel_group_id
                    ).data_ptr()
                    all_gpu_ptrs_fb: list[int] = []
                    src_offset = kg_byte_offset + layer_local_idx * per_layer_bytes
                    staging_fb: list = []

                    for chunk_local_idx, (_, memory_obj) in enumerate(all_chunks_flat):
                        gpu_dst = buffer_base + chunk_local_idx * per_layer_bytes
                        src_ptr = memory_obj.data_ptr + src_offset
                        all_gpu_ptrs_fb.append(gpu_dst)
                        staging_fb.append(
                            device_ops.StagingCopy(
                                gpu_dst,
                                src_ptr,
                                per_layer_bytes,
                                memory_obj.meta.address,
                            )
                        )

                    all_block_ids = info["all_block_ids"]
                    layer_spec = device_ops.KernelGroupSpec(
                        single_layer_kv_ptr.data_ptr(),
                        all_gpu_ptrs_fb,
                        single_layer_sd,
                        slots_per_chunk,
                        cache_context.get_engine_kv_format(kernel_group_id),
                        all_block_ids.data_ptr(),
                        all_block_ids.numel(),
                    )
                    layer_launch = device_ops.LaunchVar(
                        0,
                        0,
                        all_block_ids.numel(),
                        len(all_gpu_ptrs_fb),
                        info["all_skip_blocks"],
                    )
                    device_ops.execute_object_group_transfer(
                        lmcache_native.TransferDirection.H2D,
                        cache_context.device,
                        pin_chunk_size,
                        [layer_spec],
                        [device_ops.BatchStep(staging_fb, [layer_launch])],
                        layerwise=True,
                    )
                elif info["single_launch"]:
                    buffer_base = cache_context.get_temp_kernel_group_buffer(
                        0, kernel_group_id
                    ).data_ptr()
                    all_gpu_ptrs_fb = []
                    src_offset = kg_byte_offset + layer_local_idx * per_layer_bytes

                    for chunk_local_idx, (_, memory_obj) in enumerate(all_chunks_flat):
                        gpu_dst = buffer_base + chunk_local_idx * per_layer_bytes
                        src_ptr = memory_obj.data_ptr + src_offset
                        device_ops.lmcache_memcpy_async(
                            gpu_dst,
                            src_ptr,
                            per_layer_bytes,
                            lmcache_native.TransferDirection.H2D,
                            memory_obj.meta.address,
                            pin_chunk_size,
                        )
                        all_gpu_ptrs_fb.append(gpu_dst)

                    device_ops.multi_layer_block_kv_transfer(
                        single_layer_kv_ptr,
                        all_gpu_ptrs_fb,
                        info["all_block_ids"],
                        cache_context.device,
                        lmcache_native.TransferDirection.H2D,
                        single_layer_sd,
                        slots_per_chunk,
                        cache_context.get_engine_kv_format(kernel_group_id),
                        info["all_skip_blocks"],
                    )
                else:
                    # Batched fallback for very long sequences
                    blocks_per_chunk = info["blocks_per_chunk"]
                    blocks_per_window = info["blocks_per_window"]
                    for start_object_idx, memory_object_batch in batch_plan:
                        batch_len = len(memory_object_batch)
                        batch_start_token = start_object_idx * lmcache_chunk_size
                        skip_tokens_in_chunk = (
                            max(batch_start_token, skip_first_n_tokens)
                            - batch_start_token
                        )
                        tmp_gpu_buffers: list[int] = []
                        for chunk_idx, memory_obj in enumerate(memory_object_batch):
                            full_kg_buffer = cache_context.get_temp_kernel_group_buffer(
                                chunk_idx, kernel_group_id
                            )
                            gpu_dst = full_kg_buffer.data_ptr()
                            src_offset = (
                                kg_byte_offset + layer_local_idx * per_layer_bytes
                            )
                            src_ptr = memory_obj.data_ptr + src_offset
                            device_ops.lmcache_memcpy_async(
                                gpu_dst,
                                src_ptr,
                                per_layer_bytes,
                                lmcache_native.TransferDirection.H2D,
                                memory_obj.meta.address,
                                pin_chunk_size,
                            )
                            tmp_gpu_buffers.append(gpu_dst)

                        orig_skip_blocks = cache_context.calculate_num_blocks(
                            skip_tokens_in_chunk, kernel_group_id
                        )
                        recalculated_skip_blocks = _recalculate_blocks_to_skip(
                            blocks_per_chunk,
                            blocks_per_window,
                            orig_skip_blocks,
                        )
                        start_block_pos = start_object_idx * blocks_per_window
                        end_block_pos = (
                            start_object_idx + batch_len
                        ) * blocks_per_window
                        block_ids_curr = block_ids_gpu[kernel_group_id][
                            start_block_pos:end_block_pos
                        ]

                        device_ops.multi_layer_block_kv_transfer(
                            single_layer_kv_ptr,
                            tmp_gpu_buffers,
                            block_ids_curr,
                            cache_context.device,
                            lmcache_native.TransferDirection.H2D,
                            single_layer_sd,
                            slots_per_chunk,
                            cache_context.get_engine_kv_format(kernel_group_id),
                            recalculated_skip_blocks,
                        )

        # Record 1 IPC event per batch (all layers in batch share it)
        first_gl = all_layers[layer_batch_start][2]
        if first_gl < len(layer_events):
            event_backend.record_event(layer_events[first_gl], main_stream)
        if batch_leader_map is not None:
            for sub_idx in range(n_in_batch):
                gl = all_layers[layer_batch_start + sub_idx][2]
                batch_leader_map[gl] = first_gl

        # Stream the event handle to the worker immediately so it can
        # start attention on these layers while later batches transfer.
        if event_export_callback is not None and first_gl < len(layer_events):
            event_export_callback(first_gl, n_in_batch, layer_events[first_gl])

        layer_batch_start = batch_end


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
            # Same handler as RETRIEVE; the dedicated request type only
            # differs in that its protocol has streaming=True, so the MQ
            # server allocates a StreamingSink for it (layerwise path only).
            HandlerSpec(
                RequestType.RETRIEVE_LAYERWISE,
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
    ) -> int:
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
                return self._ctx.layerwise_batch

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
        kv_groups_manager = cache_context.kv_layer_groups_manager
        num_object_groups = kv_groups_manager.num_object_groups
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        # One layout per object group, also in the single-group case: no
        # None special-casing downstream (group 0 maps to the merged layout).
        group_layout_descs = {
            gid: get_layout_desc(
                cache_context, self._ctx.chunk_size, object_group_id=gid
            )
            for gid in range(num_object_groups)
        }
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs=group_layout_descs,
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )
        return self._ctx.layerwise_batch

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
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Chunks whose block ids are all the null block (e.g. align-mode
            # Mamba chunks holding no real state) carry no valid KV and must not
            # be committed. Computed on the raw block ids before downsampling
            # mutates them.
            skipped_chunks = all_null_chunk_masks(
                gpu_block_ids,
                cache_context.kv_layer_groups_manager.object_groups,
                blocks_per_chunk,
                num_chunks,
            )

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

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

            # Worker 0 only: bindings depend on token content alone, so one
            # report covers every rank's keys. Published before finish_write
            # is enqueued so the token bindings precede the write-finished
            # events on the bus.
            if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                EventType.MP_TOKENS
            ):
                self._publish_token_bindings(key, obj_keys_per_obj_group[0])

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
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    skip_mask = skipped_chunks[obj_group_id]
                    keys_to_reserve = [
                        k for i, k in enumerate(obj_keys) if not skip_mask[i]
                    ]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        keys_to_reserve, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (all-null chunks skipped above, or
                    # skipped by the storage manager) become None entries; the
                    # helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    # kv_interleaved: when layerwise loading is enabled
                    # (LMCACHE_MP_LAYERWISE_BATCH > 0), store D2H writes
                    # in per-layer interleaved layout [K0,V0,K1,V1,...]
                    # so layerwise retrieve can do single-memcpy per layer.
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmcache_native.TransferDirection.D2H,
                        kv_interleaved=self._ctx.layerwise_loading,
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
        *,
        streaming_sink=None,
    ) -> tuple[bytes | list[bytes], bool]:
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
        # Derive layerwise from the request type: streaming_sink is
        # non-None iff the request was RETRIEVE_LAYERWISE.
        layerwise = streaming_sink is not None

        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
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

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            # Per object group, the prefetch only locked the in-window suffix
            # (the last ``num_chunks_in_sw`` chunks; the whole prefix for full
            # attention, where the value is < 0). Read and transfer only those.
            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(num_chunks - skip for skip in group_skips)

            # Create per-layer events if layerwise mode is enabled
            layer_events = []
            if layerwise:
                num_total_layers = sum(
                    kgr.num_layers
                    for kgr in cache_context.kv_layer_groups_manager.kernel_groups
                )
                layer_events = [
                    event_backend.create_event(cache_context.device)
                    for _ in range(num_total_layers)
                ]
            batch_leader_map: dict[int, int] = {}

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    with self._ctx.storage_manager.read_prefetched_results(
                        in_window_keys
                    ) as window_objs:
                        if not window_objs or len(window_objs) != len(in_window_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in window_objs)

                        # None-pad the skipped prefix to full length so the
                        # transfer's ``num_objects_to_skip`` and block-id slicing
                        # line up unchanged; the None entries are never read.
                        memory_objs: list[MemoryObj | None] = [None] * skip + list(
                            window_objs
                        )

                        if layerwise:
                            # Build streaming callback if sink available
                            _export_cb = None
                            if streaming_sink is not None:

                                def _export_cb(
                                    first_layer,
                                    count,
                                    event,
                                    _sink=streaming_sink,
                                    _eb=event_backend,
                                    _cc=cache_context,
                                ):
                                    handle = _eb.export_event(event, _cc.device)
                                    _sink.send_partial(
                                        msgspec.msgpack.encode(
                                            (first_layer, count, handle)
                                        )
                                    )

                            transfer_kv_layerwise(
                                cache_context,
                                block_ids_per_group_gpu,
                                memory_objs,
                                object_group_id=obj_group_id,
                                batch_size=cache_context.max_batch_size,
                                skip_first_n_tokens=skip_first_n_tokens,
                                layer_events=layer_events,
                                event_backend=event_backend,
                                batch_leader_map=batch_leader_map,
                                layerwise_batch=self._ctx.layerwise_batch,
                                event_export_callback=_export_cb,
                            )
                        else:
                            transfer_kv_per_object_group(
                                cache_context,
                                block_ids_per_group_gpu,
                                memory_objs,
                                object_group_id=obj_group_id,
                                batch_size=cache_context.max_batch_size,
                                skip_first_n_tokens=skip_first_n_tokens,
                                direction=lmcache_native.TransferDirection.H2D,
                            )
                        # Extend only after the copy is enqueued: on exception,
                        # read_prefetched_results releases this group's locks
                        # itself, and a key must not be released twice.
                        prefetched_keys.extend(in_window_keys)
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
                    if len(prefetched_keys) == expected_retained
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

        if layerwise and layer_events:
            if streaming_sink is not None:
                # Events were already streamed via partials; return
                # empty list so the final ZMQ response only carries
                # the success flag.
                return [], retrieve_succeeded
            # Export deduplicated handles as flat list[bytes] (msgspec
            # compatible). Layers sharing a batch get the same handle
            # bytes; consumer deduplicates on import.
            exported_cache: dict[int, bytes] = {}
            layer_event_handles: list[bytes] = []
            for gl in range(len(layer_events)):
                leader = batch_leader_map.get(gl, gl)
                if leader not in exported_cache:
                    exported_cache[leader] = event_backend.export_event(
                        layer_events[leader], cache_context.device
                    )
                layer_event_handles.append(exported_cache[leader])
            return layer_event_handles, retrieve_succeeded
        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Publish one ``MP_TOKENS`` event for ``key``'s chunks.

        Pairs each complete chunk in ``[key.start, key.end)`` with its
        ObjectKey chunk hash and token position. Must be called at store
        submission, before the write-finished events reach the bus, so the
        cache-event subscriber can stamp them onto the STORE entries. A
        store that later fails leaves only unused cache entries.

        Args:
            key: The IPC key of the store being submitted.
            obj_keys: One ObjectKey per complete chunk, in chunk order.
        """
        # Complete chunks in [key.start, key.end) paired with the absolute
        # position of each chunk's first token. Prefix-chained chunk hashes
        # imply a position without revealing it, so it is reported here. A
        # trailing partial chunk has no stored KV to bind to.
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [
            token_ids[offset : offset + chunk_size] for offset in token_offsets
        ]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys "
                "vs %d complete chunks in [%d, %d)",
                key.request_id,
                len(obj_keys),
                len(token_chunks),
                key.start,
                key.end,
            )
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [obj_key.chunk_hash for obj_key in obj_keys],
                    "token_chunks": token_chunks,
                    "token_offsets": token_offsets,
                },
            )
        )
