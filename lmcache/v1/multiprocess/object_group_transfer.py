# SPDX-License-Identifier: Apache-2.0
"""Object-group KV transfer for the multiprocess server.

Per-kernel-group gather and scatter between the engine's paged KV cache and
LMCache memory objects, driven by the cache context's ``KVLayerGroupsManager``:
block-id downsampling for sub-chunk sliding windows, skip recalculation, and
the per-object-group transfer plan the copy kernels run.

Two copy paths are available, chosen per object group by
:func:`direct_transfer_supported`: the direct path moves each chunk straight
between the pinned host object and the paged buffers with one
``cudaMemcpyBatchAsync`` per chunk, and is used whenever the build, the driver
and the object group's layout allow it; everything else stages each chunk
through a GPU temp buffer and scatters it with the block transfer kernel. See
``docs/design/v1/multiprocess/lmcache_driven_transfer_copy_paths.md``.
"""

# Standard
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from typing import Any, Generator, Protocol, Sequence, cast

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gds_context import (
    GDSContext,
    SlabDirection,
    get_l1_gds_context,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    build_staging_copies,
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    device_ops, "execute_object_group_transfer"
)
_HAS_TRANSFER_PHASE_TIMING: bool = hasattr(device_ops, "pop_completed_phase_timings")
# Whether the copy engine can run the direct transfer at all: the compiled
# extension exports ``execute_direct_copy_transfer`` (built against CUDA >=
# 12.8 or HIP >= 7.15) and ``cudaMemcpyBatchAsync`` / ``hipMemcpyBatchAsync``
# is usable on this runtime and driver (on ROCm, only when opted in via
# ``LMCACHE_ROCM_ENABLE_BATCH_MEMCPY``). Resolved once -- ``device_ops`` is
# native-bound during ``import lmcache``, before this module is imported.
# ``batch_memcpy_supported`` and ``direct_copy_format_supported`` are
# native-only and exported together with ``execute_direct_copy_transfer``, so
# the ``hasattr`` check guards them.
_HAS_BATCH_MEMCPY_ASYNC: bool = (
    hasattr(device_ops, "execute_direct_copy_transfer")
    and device_ops.batch_memcpy_supported()
)
# Layouts already reported as ineligible, so a model whose format never
# qualifies logs once instead of once per request.
_direct_copy_rejected_formats: set[str] = set()


class _GDSStream(Protocol):
    """CUDA slot stream ordering work behind an already-recorded input event."""

    def wait_event(self, event: object) -> None: ...

    def query(self) -> bool: ...


@dataclass
class _GDSSlot:
    slot: int
    object_index: int
    skip_tokens: int
    memory_obj: GDSMemoryObject
    context: GDSContext
    scatter_submitted: bool = False


def direct_transfer_supported(
    cache_context: BaseCacheContext,
    object_group_id: int,
    memory_objs: Sequence[MemoryObj | None],
    block_ids_host: Sequence[Sequence[int]],
) -> bool:
    """Whether one object group can be moved by the copy engine.

    The direct path (see :func:`run_direct_transfer`) needs host block ids, a
    non-GDS object set, and a token-major layout whose paged block is one
    contiguous run in every kernel group of the object group. Callers must
    also have checked :data:`_HAS_BATCH_MEMCPY_ASYNC`; this function assumes
    the native entry point exists and only inspects the request.

    An ineligible layout is logged once per format, since a model whose KV
    layout never qualifies would otherwise log on every request.

    Args:
        cache_context: The cache context of the registered KV cache.
        object_group_id: Index of the object group being transferred.
        memory_objs: The objects of the transfer (None entries allowed).
        block_ids_host: Downsampled host block ids, indexed by kernel group;
            empty when the caller only has device tensors.

    Returns:
        True when the direct copy path can serve this object group.
    """
    if not block_ids_host:
        return False
    if any(isinstance(mo, GDSMemoryObject) for mo in memory_objs):
        logger.debug(
            "Object group %d has GDS-backed objects; using the kernel path",
            object_group_id,
        )
        return False

    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    for kernel_group_id in object_group.kernel_group_indices:
        engine_kv_format = cache_context.get_engine_kv_format(kernel_group_id)
        if not device_ops.direct_copy_format_supported(engine_kv_format):
            format_name = str(engine_kv_format)
            if format_name not in _direct_copy_rejected_formats:
                _direct_copy_rejected_formats.add(format_name)
                logger.warning(
                    "Layout %s of kernel group %d is not eligible for the direct "
                    "copy path (only token-major layouts whose paged block is one "
                    "contiguous run qualify); using the block transfer kernel",
                    format_name,
                    kernel_group_id,
                )
            return False
    return True


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


def recalculate_blocks_to_skip(
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
    *,
    transfer_key: str,
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
        transfer_key: Identity of this store/retrieve operation, echoed back
            on every phase-timing sample (a request issues several transfers,
            so the request id cannot identify one).

    Raises:
        ValueError: If a None entry is found in memory_objs when direction is
            H2D, or if an object's size does not match its GPU staging buffer.
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

        spec_index_by_kg[kernel_group_id] = len(kernel_group_specs)
        kernel_group_specs.append(
            device_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                cache_context.get_shape_desc(kernel_group_id),
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
            recalculated_skip_blocks = recalculate_blocks_to_skip(
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

    # Time the phases only when a subscriber consumes the samples. An older
    # compiled extension has neither the keywords nor anything to consume
    # them, so fall back to the untimed legacy signature.
    timing_kwargs = (
        {
            "phase_timing_enabled": is_observability_enabled()
            and get_event_bus().has_subscribers(EventType.MP_TRANSFER_PHASE_SAMPLES),
            # Echoed back verbatim on each sample; the transfer's identity.
            "session_id": transfer_key,
        }
        if _HAS_TRANSFER_PHASE_TIMING
        else {}
    )
    device_ops.execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
        **timing_kwargs,
    )


def run_direct_transfer(
    cache_context: BaseCacheContext,
    block_ids_host: Sequence[Sequence[int]],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Plan and execute one object group's transfer through the copy engine.

    Direct-copy counterpart of :func:`_run_object_group_transfer_plan`: the
    same window skip / ``skip_first_n_tokens`` logic, but instead of staging
    each chunk through the GPU temp buffer and launching the block transfer
    kernel, each (kv plane, block) of a chunk is copied straight between the
    pinned host object and the paged buffer (``execute_direct_copy_transfer``,
    single GIL release for the whole group): one 2-D copy over all layers when
    the kernel group's layers sit at a constant pitch (vLLM's block-outermost
    layouts), else one ``cudaMemcpyBatchAsync`` entry per layer.

    The caller must have checked :data:`_HAS_BATCH_MEMCPY_ASYNC` and
    :func:`direct_transfer_supported`.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_host: Downsampled host block ids, indexed by kernel group,
            ``blocks_per_window`` entries per chunk.
        memory_objs: The MemoryObj instances to copy. None entries are only
            valid for D2H (the chunk is skipped); H2D raises.
        object_group_id: Index of the object group being copied.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).

    Raises:
        ValueError: If a None entry is found in memory_objs when direction is
            H2D, or if an object has not been allocated.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D

    group_specs: list[Any] = []
    blocks_per_chunk_by_kg: list[int] = []
    blocks_per_window_by_kg: list[int] = []
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
        blocks_per_chunk_by_kg.append(blocks_per_chunk)
        blocks_per_window_by_kg.append(blocks_per_window)
        group_specs.append(
            device_ops.DirectCopyGroupSpec(
                cache_context.get_kernel_group_kv_pointer_list(kernel_group_id),
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_kernel_group_offset_in_object(
                    object_group_id, kernel_group_id
                ),
                list(block_ids_host[kernel_group_id]),
            )
        )

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)

    objects: list[Any] = []
    for chunk_idx in range(num_objects_to_skip, len(memory_objs)):
        memory_obj = memory_objs[chunk_idx]
        if memory_obj is None:
            if is_h2d:
                raise ValueError(
                    f"MemoryObj is None for chunk {chunk_idx}, cannot perform H2D copy"
                )
            continue
        if memory_obj.raw_tensor is None:
            raise ValueError(
                "memory_obj.raw_tensor is None; ensure the MemoryObj has been "
                "allocated."
            )
        chunk_start_token = chunk_idx * lmcache_chunk_size
        chunk_end_token = chunk_start_token + lmcache_chunk_size
        effective_start = max(chunk_start_token, skip_first_n_tokens)
        if effective_start >= chunk_end_token:
            continue
        skip_tokens_in_chunk = effective_start - chunk_start_token

        skip_blocks: list[int] = []
        for position, kernel_group_id in enumerate(kernel_group_ids):
            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            skip_blocks.append(
                recalculate_blocks_to_skip(
                    blocks_per_chunk_by_kg[position],
                    blocks_per_window_by_kg[position],
                    orig_skip_blocks,
                )
            )
        objects.append(
            device_ops.DirectCopyObject(
                memory_obj.data_ptr,
                memory_obj.meta.address,
                memory_obj.get_size(),
                chunk_idx,
                skip_blocks,
            )
        )

    if not objects:
        return

    device_ops.execute_direct_copy_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        group_specs,
        objects,
    )


def _copy_kv_batch(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    object_group_id: int,
    start_object_idx: int,
    batch_len: int,
    skip_tokens_in_chunk: int,
    direction: "lmcache_native.TransferDirection",
    slot_start: int = 0,
) -> None:
    """Gather or scatter a batch using its original object/block mapping.

    Args:
        cache_context: Owner of engine buffers and staging slots.
        block_ids_gpu: Read-only block IDs for the whole request.
        object_group_id: Object layout to transfer.
        start_object_idx: Original index, including skipped prefix objects.
        batch_len: Number of consecutive objects handled by this kernel.
        skip_tokens_in_chunk: Protected prefix in this batch.
        direction: Gather (D2H) or scatter (H2D).
        slot_start: First independently owned staging slot.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    kernel_group_ids = kv_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
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
        recalculated_skip_blocks = recalculate_blocks_to_skip(
            blocks_per_chunk,
            blocks_per_window,
            orig_skip_blocks,
        )

        # Launch kernel
        group_kv_pointers = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        group_lmcache_chunk_size = cache_context.get_slots_per_chunk_in_sw(
            kernel_group_id
        )
        tmp_gpu_buffers_batched = [
            cache_context.get_temp_kernel_group_buffer(
                slot_start + i, kernel_group_id
            ).data_ptr()
            for i in range(batch_len)
        ]
        device_ops.multi_layer_block_kv_transfer(
            group_kv_pointers,
            tmp_gpu_buffers_batched,
            block_ids_curr_batch,
            cache_context.device,
            direction,
            cache_context.get_shape_desc(kernel_group_id),
            group_lmcache_chunk_size,
            cache_context.get_engine_kv_format(kernel_group_id),
            recalculated_skip_blocks,
        )


def _validate_gds_batch(
    memory_objects: Sequence[MemoryObj | None], cache_context: BaseCacheContext
) -> None:
    """Check all GDS owners before scatter or successful store publication."""
    owners = {
        obj.get_l1_manager()
        for obj in memory_objects
        if isinstance(obj, GDSMemoryObject)
    }
    first_error: Exception | None = None
    for owner in owners:
        try:
            get_l1_gds_context(owner).synchronize_and_check(
                cache_context.device, cache_context.stream
            )
        except Exception as error:
            logger.exception("GDS completion failed for owner %s", owner)
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _run_gds_concurrent(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    skip_first_n_tokens: int,
    skip_objects: int,
    direction: "lmcache_native.TransferDirection",
    block_ids_host: Sequence[Sequence[int]],
) -> bool:
    """Reap/refill at most four independent stream-ordered GDS slots.

    A shared slab scope serializes groups from different cache contexts. Within
    that scope, the pending queue bounds both operations and useful bytes. Reads
    are validated before scatter; stores gather before I/O on the same stream.
    Every slot remains owned until its final stream wait completes. This runs on
    the existing blocking affinity handler, never the transport event loop.
    """
    chunk_size = cache_context.lmcache_tokens_per_chunk
    objects = [
        mo
        for index, mo in enumerate(memory_objs)
        if mo is not None
        and index >= skip_objects
        and (index + 1) * chunk_size > skip_first_n_tokens
    ]
    if not objects or cache_context.device.type != "cuda":
        return False
    if not all(isinstance(mo, GDSMemoryObject) for mo in objects):
        return False
    contexts = {
        mo.get_l1_manager(): get_l1_gds_context(mo.get_l1_manager()) for mo in objects
    }
    if not all(context.io_concurrency for context in contexts.values()):
        return False
    is_read = direction == lmcache_native.TransferDirection.H2D
    if is_read:
        # Concurrent scatter must preserve the serial ordering of aliased blocks.
        if not block_ids_host:
            return False
        manager = cache_context.kv_layer_groups_manager
        first_object = max(skip_objects, skip_first_n_tokens // chunk_size)
        for group_id in manager.object_groups[object_group_id].kernel_group_indices:
            window_tokens = min(
                chunk_size, manager.get_subchunk_sw_size_tokens(group_id)
            )
            blocks = cache_context.calculate_num_blocks(window_tokens, group_id)
            destinations = block_ids_host[group_id][
                first_object * blocks : len(memory_objs) * blocks
            ]
            if len(destinations) != len(set(destinations)):
                return False
    budget = min(context.max_inflight_bytes for context in contexts.values())
    if any(mo.get_size() > budget for mo in objects):
        raise ValueError("GDS object exceeds the configured in-flight byte budget")

    with cache_context.gds_transfer_scope(), ExitStack() as scopes:
        for owner in sorted(contexts, key=lambda owner: -1 if owner is None else owner):
            scopes.enter_context(contexts[owner].io_scope())
        count = min(
            cache_context.max_batch_size,
            min(c.io_concurrency for c in contexts.values()),
        )
        streams = [cache_context.get_gds_transfer_stream(slot) for slot in range(count)]
        for context in contexts.values():
            for stream in streams:
                context.register_stream(cache_context.device, stream)
        ready = cache_context.record_gds_transfer_ready()
        free_slots = list(range(count))
        pending: deque[_GDSSlot] = deque()
        inflight_bytes = 0

        def finish_oldest() -> int:
            """Validate I/O and finish scatter before this slot can be reused."""
            if is_read:
                for entry in pending:
                    stream = streams[entry.slot]
                    if entry.scatter_submitted:
                        continue
                    if entry is not pending[0] and not cast(_GDSStream, stream).query():
                        continue
                    entry.context.synchronize_and_check(cache_context.device, stream)
                    with torch_dev.stream(stream):
                        _copy_kv_batch(
                            cache_context,
                            block_ids_gpu,
                            object_group_id,
                            entry.object_index,
                            1,
                            entry.skip_tokens,
                            direction,
                            entry.slot,
                        )
                    entry.scatter_submitted = True
                entry = pending[0]
                platform_stream.synchronize_stream(
                    cache_context.device, streams[entry.slot]
                )
            else:
                entry = pending[0]
                entry.context.synchronize_and_check(
                    cache_context.device, streams[entry.slot]
                )
            pending.popleft()
            free_slots.append(entry.slot)
            return entry.memory_obj.get_size()

        try:
            for index in range(skip_objects, len(memory_objs)):
                obj = memory_objs[index]
                skip = max(0, skip_first_n_tokens - index * chunk_size)
                if skip >= chunk_size:
                    continue
                if obj is None:
                    if is_read:
                        raise ValueError("Missing GDS object during retrieve")
                    continue
                assert isinstance(obj, GDSMemoryObject)  # Eligibility checked above.
                nbytes = obj.get_size()
                while not free_slots or inflight_bytes + nbytes > budget:
                    inflight_bytes -= finish_oldest()
                entry = _GDSSlot(
                    free_slots.pop(), index, skip, obj, contexts[obj.get_l1_manager()]
                )
                # Track ownership before gather or I/O can enqueue partially and fail.
                pending.append(entry)
                inflight_bytes += nbytes
                stream = streams[entry.slot]
                with torch_dev.stream(stream):
                    cast(_GDSStream, stream).wait_event(ready)
                    if not is_read:
                        _copy_kv_batch(
                            cache_context,
                            block_ids_gpu,
                            object_group_id,
                            index,
                            1,
                            skip,
                            direction,
                            entry.slot,
                        )
                    entry.context.transfer_async(
                        obj,
                        cache_context.get_temp_object_group_buffer(
                            entry.slot, object_group_id
                        ),
                        SlabDirection.READ if is_read else SlabDirection.WRITE,
                    )
            while pending:
                inflight_bytes -= finish_oldest()
        finally:
            # Keep the original error; every partially submitted owner/stream drains.
            for entry in pending:
                try:
                    entry.context.synchronize_and_check(
                        cache_context.device, streams[entry.slot]
                    )
                except Exception:
                    logger.exception("GDS slot cleanup failed after transfer error")
    return True


def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    transfer_key: str,
    block_ids_host: Sequence[Sequence[int]] = (),
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
        transfer_key: Identity of this store/retrieve operation, echoed back on
            every phase-timing sample; see _run_object_group_transfer_plan.
        block_ids_host: The same downsampled block ids as ``block_ids_gpu``,
            as host lists indexed by kernel group. Required for the direct
            copy path (see :func:`run_direct_transfer`); leave empty to force
            the kernel path.

    Raises:
        ValueError: If it founds None entry in memory_objs when direction is H2D.
    Note:
        This function expects the caller to stage the block ids (list[list[int]])
        into GPU tensors and pass them in as `block_ids_gpu`.
        Opt-in concurrent GDS streams reuse independently owned staging slots
        with one object per kernel, including stores. They return after validated I/O
        and completed GPU work. Call from a blocking affinity handler, never
        the transport event loop. The default path remains stream ordered.
    """
    if _HAS_BATCH_MEMCPY_ASYNC and direct_transfer_supported(
        cache_context, object_group_id, memory_objs, block_ids_host
    ):
        run_direct_transfer(
            cache_context,
            block_ids_host,
            memory_objs,
            object_group_id,
            skip_first_n_tokens,
            direction,
        )
        return

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
            transfer_key=transfer_key,
        )
        return

    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
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

    if _run_gds_concurrent(
        cache_context,
        block_ids_gpu,
        memory_objs,
        object_group_id,
        skip_first_n_tokens,
        num_objects_to_skip,
        direction,
        block_ids_host,
    ):
        return

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
            try:
                for chunk_idx, memory_obj in enumerate(memory_object_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        cache_context.get_temp_object_group_buffer(
                            chunk_idx, object_group_id
                        ),
                    )
            except Exception:
                try:
                    _validate_gds_batch(memory_object_batch, cache_context)
                except Exception:
                    logger.exception("GDS read cleanup failed after submission error")
                raise
            _validate_gds_batch(memory_object_batch, cache_context)

        _copy_kv_batch(
            cache_context,
            block_ids_gpu,
            object_group_id,
            start_object_idx,
            batch_len,
            skip_tokens_in_chunk,
            direction,
        )

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
        if not is_h2d:
            try:
                for chunk_idx, memory_obj in enumerate(memory_object_batch):
                    lmcache_memcpy_async_d2h(
                        cache_context.get_temp_object_group_buffer(
                            chunk_idx, object_group_id
                        ),
                        memory_obj,
                    )
            except Exception:
                try:
                    _validate_gds_batch(memory_object_batch, cache_context)
                except Exception:
                    logger.exception("GDS write cleanup failed after submission error")
                raise
            _validate_gds_batch(memory_object_batch, cache_context)
