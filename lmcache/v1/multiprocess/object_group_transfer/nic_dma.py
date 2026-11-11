# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: chunks a peer owns, read straight into GPU memory.

Each chunk's object starts its own transfer-channel read
(:meth:`BorrowedMemoryObj.read_into`) into its slot of the cache context's
staging buffer, the same slots the other engines stage into; the block
transfer kernel then scatters the slots into the paged KV blocks. The
staging buffer must have been registered with the channel as a region
(:func:`set_staging_region`).

Those reads are not ordered by the stream, so the host orders them: before
the reads of a batch it waits for everything queued on the stream, which
includes the kernel of the previous batch, and it waits for the reads before
queuing the batch's kernel.
"""

# Standard
from dataclasses import dataclass
from typing import Sequence
import time
import weakref

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.memory_management import BorrowedMemoryObj, MemoryObj, PendingRead
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    batched_iteration_with_skip,
    recalculate_blocks_to_skip,
    resolve_chunk_range,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.stream import CompletionEvent, record_completion_event
import lmcache.lmcache_native as lmcache_native

# Host-side pause between polls of in-flight reads and of stream events.
_POLL_INTERVAL_S = 0.00002


@dataclass(frozen=True)
class StagingRegion:
    """A cache context's staging buffer as a transfer-channel region.

    Attributes:
        region_id: The id the channel returned when the buffer was registered.
        base_ptr: The address the region starts at.
    """

    region_id: int
    base_ptr: int


_staging_regions: "weakref.WeakKeyDictionary[BaseCacheContext, StagingRegion]" = (
    weakref.WeakKeyDictionary()
)


def set_staging_region(
    cache_context: BaseCacheContext, region: StagingRegion | None
) -> None:
    """Tell the engine which channel region ``cache_context``'s staging
    buffer is, or that it is no longer registered.

    The caller registers the buffer with the transfer channel and keeps it
    registered while a region is set. Every staging slot must start on the
    channel's alignment for a read to land in it.

    Args:
        cache_context: The cache context whose staging buffer was registered.
        region: The region, or None to forget it.
    """
    if region is None:
        _staging_regions.pop(cache_context, None)
    else:
        _staging_regions[cache_context] = region


def _wait_for_event(event: CompletionEvent) -> None:
    """Block until the stream work before ``event`` has finished."""
    while not event.is_complete():
        time.sleep(_POLL_INTERVAL_S)


def _wait_for_reads(reads: list[tuple[BorrowedMemoryObj, PendingRead]]) -> None:
    """Block until every read settled, then raise if one failed.

    Waiting for all of them matters: an abandoned read would still be landing
    in the staging buffer when it is next used.

    Raises:
        RuntimeError: If a read failed; the peer may have dropped the object
            or the lease ended.
    """
    failed: BorrowedMemoryObj | None = None
    while reads:
        running = []
        for memory_obj, read in reads:
            if not read.is_finished():
                running.append((memory_obj, read))
            elif not read.succeeded() and failed is None:
                failed = memory_obj
        reads = running
        if reads:
            time.sleep(_POLL_INTERVAL_S)
    if failed is not None:
        raise RuntimeError(
            f"Read of the peer object at offset {failed.meta.address:#x} failed"
        )


def transfer(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    transfer_key: str,
    block_ids_host: Sequence[Sequence[int]],
    chunk_range: tuple[int, int],
) -> None:
    """Retrieve a run of chunks a peer owns.

    Blocks the caller until the last read of the run finished; the kernels
    are left queued on the stream.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The object group's objects, one per chunk.
        object_group_id: Index of the object group being copied.
        batch_size: Number of chunks read and scattered together.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: Must be H2D; a peer's object is never a store target.
        transfer_key: Unused.
        block_ids_host: Unused.
        chunk_range: ``[start, end)`` chunks this engine moves.

    Raises:
        ValueError: If ``direction`` is not H2D, or a chunk in the range has
            no object or one a peer does not own.
        RuntimeError: If the cache context's staging buffer is not registered
            with the transfer channel, or a read failed.
    """
    if direction != lmcache_native.TransferDirection.H2D:
        raise ValueError("Objects a peer owns can only be retrieved")
    region = _staging_regions.get(cache_context)
    if region is None:
        raise RuntimeError(
            "Retrieving objects a peer owns needs the staging buffer registered "
            "with the transfer channel"
        )
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    kernel_group_ids = kv_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
    device = cache_context.device
    stream = cache_context.stream

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id):
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
    start_chunk, end_chunk = resolve_chunk_range(
        len(memory_objs), chunk_range, num_objects_to_skip
    )

    # An earlier transfer may still be using the staging buffer.
    staging_idle = record_completion_event(device, stream)
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs[:end_chunk], batch_size, skip_count=start_chunk
    ):
        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue
        skip_tokens_in_chunk = effective_start - batch_start_token

        _wait_for_event(staging_idle)
        reads: list[tuple[BorrowedMemoryObj, PendingRead]] = []
        try:
            for chunk_idx, memory_obj in enumerate(memory_object_batch):
                if not isinstance(memory_obj, BorrowedMemoryObj):
                    raise ValueError(
                        "The NIC DMA engine moves only objects a peer owns, "
                        f"got {memory_obj!r}"
                    )
                slot = cache_context.get_temp_object_group_buffer(
                    chunk_idx, object_group_id
                )
                offset = slot.data_ptr() - region.base_ptr
                reads.append(
                    (memory_obj, memory_obj.read_into(region.region_id, offset))
                )
        finally:
            _wait_for_reads(reads)

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
            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window
            device_ops.multi_layer_block_kv_transfer(
                cache_context.get_kernel_group_kv_pointers(kernel_group_id),
                [
                    cache_context.get_temp_kernel_group_buffer(
                        chunk_idx, kernel_group_id
                    ).data_ptr()
                    for chunk_idx in range(batch_len)
                ],
                block_ids_gpu[kernel_group_id][start_block_pos:end_block_pos],
                device,
                direction,
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                recalculate_blocks_to_skip(
                    blocks_per_chunk,
                    blocks_per_window,
                    cache_context.calculate_num_blocks(
                        skip_tokens_in_chunk, kernel_group_id
                    ),
                ),
            )
        staging_idle = record_completion_event(device, stream)
