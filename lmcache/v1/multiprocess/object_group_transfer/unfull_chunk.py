# SPDX-License-Identifier: Apache-2.0
"""Transfer of a request's trailing unfull chunk."""

# Standard
from itertools import accumulate
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


def has_unfull_chunk_lead_block(
    cache_context: BaseCacheContext, kernel_group_id: int
) -> bool:
    """Return whether a kernel group's unfull-chunk objects have a lead block.

    Args:
        cache_context: The cache context containing the KV cache information.
        kernel_group_id: Index of the kernel group.

    Returns:
        True for a sliding-window group whose window is narrower than a chunk.

    Notes:
        Such a group keeps only the last window of blocks per chunk. An
        unfull chunk can end inside a block, and the window of the next token
        then also needs the block before them (the lead block; NIXL keeps the
        same ``cdiv(W, T) + 1`` blocks for P/D).
    """
    return (
        cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(
            kernel_group_id
        )
        < cache_context.lmcache_tokens_per_chunk
    )


def get_unfull_chunk_layout_desc(
    cache_context: BaseCacheContext, object_group_id: int
) -> MemoryLayoutDesc:
    """Return the memory layout of an object group's unfull-chunk objects.

    Args:
        cache_context: The cache context containing the KV cache information.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        The object group's full-chunk layout followed by one single-block
        entry per kernel group with a lead block (see
        :func:`has_unfull_chunk_lead_block`); the full-chunk layout itself
        when there is none.

    Notes:
        Assumes the ``2LTD`` object layout: the token (slot) dimension is the
        second to last of every shape, so a one-block entry replaces it with
        the block's slots. Entries are appended, so every regular region keeps
        its full-chunk offset.
    """
    kv_groups_manager = cache_context.kv_layer_groups_manager
    kernel_group_ids = kv_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
    shapes: list[torch.Size] = []
    dtypes: list[torch.dtype] = []
    for kernel_group_id in kernel_group_ids:
        shape, dtype = cache_context.get_kernel_group_shape_dtype(
            cache_context.lmcache_tokens_per_chunk, kernel_group_id
        )
        shapes.append(shape)
        dtypes.append(dtype)
    for index, kernel_group_id in enumerate(kernel_group_ids):
        if has_unfull_chunk_lead_block(cache_context, kernel_group_id):
            slots_per_block = kv_groups_manager.kernel_groups[
                kernel_group_id
            ].slots_per_block
            shapes.append(
                torch.Size((*shapes[index][:-2], slots_per_block, shapes[index][-1]))
            )
            dtypes.append(dtypes[index])
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def trim_and_stage_unfull_chunk_block_ids(
    cache_context: BaseCacheContext,
    block_ids: list[list[int]],
    num_tokens: int,
    skip_first_n_tokens: int = 0,
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Fit an unfull chunk's block id lists to the chunk layout and stage them.

    Args:
        cache_context: The cache context containing the KV cache information.
        block_ids: One block id list per kernel group, covering only the
            unfull chunk. Fitted in place.
        num_tokens: Tokens in the unfull chunk.
        skip_first_n_tokens: Leading tokens of the unfull chunk not to write
            on retrieve: the engine's prefix cache already holds them in
            blocks other requests read. A multiple of every group's block
            size, below ``num_tokens``.

    Returns:
        The fitted block id lists staged on GPU, the number of leading
        blocks to skip (padding plus blocks below ``skip_first_n_tokens``),
        and the number of lead blocks to copy (0 or 1), all indexed by
        kernel group.

    Raises:
        ValueError: If a kernel group has no block id.

    Notes:
        Each list keeps its last ``keep`` blocks (the group's per-chunk
        window in blocks), plus the lead block when one is needed. The copy
        kernel needs exactly that many per object, so a shorter list is
        padded at the front and the padding is skipped
        (``skip_prefix_n_blocks``). Store and retrieve fit lists the same
        way, so block ``i`` maps to the same object slot on both sides.
    """
    kv_groups_manager = cache_context.kv_layer_groups_manager
    skip_blocks: list[int] = []
    lead_blocks: list[int] = []
    for kernel_group_id in range(kv_groups_manager.num_kernel_groups):
        if not block_ids[kernel_group_id]:
            raise ValueError(f"unfull chunk has no block for group {kernel_group_id}")
        window_tokens = kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id)
        keep = cache_context.calculate_num_blocks(
            min(cache_context.lmcache_tokens_per_chunk, window_tokens),
            kernel_group_id,
        )
        lead = 0
        if has_unfull_chunk_lead_block(cache_context, kernel_group_id):
            tokens_per_block = kv_groups_manager.kernel_groups[
                kernel_group_id
            ].tokens_per_block
            # The next token attends to [num_tokens - W + 1, num_tokens).
            window_blocks = (
                -(-num_tokens // tokens_per_block)
                - max(0, num_tokens - window_tokens + 1) // tokens_per_block
            )
            lead = int(window_blocks > keep)
        group_block_ids = block_ids[kernel_group_id]
        kept = group_block_ids[-(keep + lead) :]
        engine_held_blocks = cache_context.calculate_num_blocks(
            skip_first_n_tokens, kernel_group_id
        )
        # A lead block the engine already holds is not copied at all.
        if lead and len(group_block_ids) - len(kept) < engine_held_blocks:
            kept, lead = kept[1:], 0
        padding = keep + lead - len(kept)
        block_ids[kernel_group_id] = kept[:1] * padding + kept
        skip_blocks.append(
            padding
            + max(0, engine_held_blocks - (len(group_block_ids) - len(kept)) - lead)
        )
        lead_blocks.append(lead)
    return cache_context.stage_block_ids(block_ids), skip_blocks, lead_blocks


def get_unfull_chunk_staging_buffer(
    cache_context: BaseCacheContext, object_group_id: int
) -> torch.Tensor | None:
    """Return the staging region an object group's unfull-chunk object uses.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        object_group_id: Index of the object group being copied.

    Returns:
        The object group's first (batch, object-group) slot when no kernel
        group has a lead block. Otherwise the start of the staging buffer,
        sized to the object; None when the object does not fit there.

    Notes:
        An object with lead-block regions is larger than one slot, so it
        stages at the start of the staging buffer
        (:meth:`BaseCacheContext.get_temp_staging_buffer`), spanning the
        following slots; transfers on the stream run in order, so nothing
        else uses them meanwhile. The buffer is registered for GPUDirect
        Storage, so a GDS-backed L1 can hold the object.
    """
    kernel_group_ids = cache_context.kv_layer_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
    if not any(
        has_unfull_chunk_lead_block(cache_context, kernel_group_id)
        for kernel_group_id in kernel_group_ids
    ):
        return cache_context.get_temp_object_group_buffer(0, object_group_id)
    layout = get_unfull_chunk_layout_desc(cache_context, object_group_id)
    return cache_context.get_temp_staging_buffer(
        sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
        )
    )


def transfer_unfull_chunk(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    skip_blocks: list[int],
    lead_blocks: list[int],
    memory_obj: MemoryObj,
    object_group_id: int,
    object_buffer: torch.Tensor,
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Copy one object group's unfull-chunk object to or from the GPU.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: Block ids from :func:`trim_and_stage_unfull_chunk_block_ids`.
        skip_blocks: Leading padding blocks per kernel group, from the same call.
        lead_blocks: Lead blocks per kernel group, from the same call.
        memory_obj: The unfull chunk's object for this object group.
        object_group_id: Index of the object group being copied.
        object_buffer: The object's staging region, from
            :func:`get_unfull_chunk_staging_buffer`.
        direction: H2D (retrieve) or D2H (store).

    Notes:
        Always uses the staging kernel path.
    """
    kernel_group_ids = cache_context.kv_layer_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    lead_group_ids = [
        kernel_group_id
        for kernel_group_id in kernel_group_ids
        if has_unfull_chunk_lead_block(cache_context, kernel_group_id)
    ]
    if lead_group_ids:
        layout = get_unfull_chunk_layout_desc(cache_context, object_group_id)
        offsets = list(
            accumulate(
                (
                    shape.numel() * dtype.itemsize
                    for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
                ),
                initial=0,
            )
        )
        regular_ptrs = [
            object_buffer.data_ptr() + offset
            for offset in offsets[: len(kernel_group_ids)]
        ]
        lead_ptrs = {
            kernel_group_id: object_buffer.data_ptr() + offset
            for kernel_group_id, offset in zip(
                lead_group_ids, offsets[len(kernel_group_ids) : -1], strict=True
            )
        }
    else:
        regular_ptrs = [
            cache_context.get_temp_kernel_group_buffer(0, kernel_group_id).data_ptr()
            for kernel_group_id in kernel_group_ids
        ]
        lead_ptrs = {}

    if is_h2d:
        lmcache_memcpy_async_h2d(memory_obj, object_buffer)
    for kernel_group_id, regular_ptr in zip(
        kernel_group_ids, regular_ptrs, strict=True
    ):
        group_block_ids = block_ids_gpu[kernel_group_id]
        lead = lead_blocks[kernel_group_id]
        if lead:
            device_ops.multi_layer_block_kv_transfer(
                cache_context.get_kernel_group_kv_pointers(kernel_group_id),
                [lead_ptrs[kernel_group_id]],
                group_block_ids[:lead],
                cache_context.device,
                direction,
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.kv_layer_groups_manager.kernel_groups[
                    kernel_group_id
                ].slots_per_block,
                cache_context.get_engine_kv_format(kernel_group_id),
                0,
            )
        device_ops.multi_layer_block_kv_transfer(
            cache_context.get_kernel_group_kv_pointers(kernel_group_id),
            [regular_ptr],
            group_block_ids[lead:],
            cache_context.device,
            direction,
            cache_context.get_shape_desc(kernel_group_id),
            cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
            cache_context.get_engine_kv_format(kernel_group_id),
            skip_blocks[kernel_group_id],
        )
    if not is_h2d:
        lmcache_memcpy_async_d2h(object_buffer, memory_obj)


def split_unfull_chunk_block_ids(
    block_ids: list[list[int]],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
    has_unfull_chunk: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split a transfer's block IDs into its full chunks' and its unfull chunk's.

    The unfull chunk's block IDs follow the ``num_chunks`` full chunks' in
    every kernel group.

    Args:
        block_ids: Block IDs per kernel group for the whole transfer.
        blocks_per_chunk: Blocks in one full chunk per kernel group.
        num_chunks: Number of full chunks in the transfer.
        has_unfull_chunk: Whether the transfer ends with an unfull chunk.

    Returns:
        ``(full_chunk_block_ids, unfull_chunk_block_ids)``; the second is
        empty when ``has_unfull_chunk`` is False.
    """
    if not has_unfull_chunk:
        return block_ids, []
    full_lengths = [num_chunks * bpc for bpc in blocks_per_chunk]
    return (
        [ids[:n] for ids, n in zip(block_ids, full_lengths, strict=True)],
        [ids[n:] for ids, n in zip(block_ids, full_lengths, strict=True)],
    )
