# SPDX-License-Identifier: Apache-2.0
"""Pick the copy path for one object group and run it."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    _HAS_BATCH_MEMCPY_ASYNC,
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    batched_iteration_with_skip,
    recalculate_blocks_to_skip,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
    _run_object_group_transfer_plan,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


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
            recalculated_skip_blocks = recalculate_blocks_to_skip(
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

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
        if not is_h2d:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                lmcache_memcpy_async_d2h(
                    cache_context.get_temp_object_group_buffer(
                        chunk_idx, object_group_id
                    ),
                    memory_obj,
                )
