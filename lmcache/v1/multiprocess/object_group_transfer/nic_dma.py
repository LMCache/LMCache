# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: chunks a peer owns, read straight into GPU memory.

Each chunk's object starts its own transfer-channel read
(:meth:`BorrowedMemoryObj.read_into`) into a slot of the cache context's
landing buffer, which is registered with the channel as a GPU region; the
block transfer kernel then scatters the slots into the paged KV blocks.

Those reads are not ordered by the stream, so the host orders them: it
drains the stream before the first read, waits for the reads of a batch
before queuing its kernel, and waits for the kernel that last read a bank
before reads land in that bank again. Alternating banks let the reads of one
batch overlap the kernel of the previous one.
"""

# Standard
from typing import Sequence
import time

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


def _wait_for_event(event: CompletionEvent) -> None:
    """Block until the stream work before ``event`` has finished."""
    while not event.is_complete():
        time.sleep(_POLL_INTERVAL_S)


def _wait_for_reads(reads: list[tuple[BorrowedMemoryObj, PendingRead]]) -> None:
    """Block until every read settled, then raise if one failed.

    Waiting for all of them matters: an abandoned read would still be landing
    in the bank when it is next used.

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
        batch_size: Unused; a batch is one bank, ``max_batch_size`` chunks.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: Must be H2D; a peer's object is never a store target.
        transfer_key: Unused.
        block_ids_host: Unused.
        chunk_range: ``[start, end)`` chunks this engine moves.

    Raises:
        ValueError: If ``direction`` is not H2D, or a chunk in the range has
            no object or one a peer does not own.
        RuntimeError: If the cache context has no landing buffer, or a read
            failed.
    """
    if direction != lmcache_native.TransferDirection.H2D:
        raise ValueError("Objects a peer owns can only be retrieved")
    num_banks = cache_context.landing_banks
    if num_banks == 0:
        raise RuntimeError(
            "Retrieving objects a peer owns needs a landing buffer registered "
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

    # An earlier transfer's kernels may still be reading the banks.
    _wait_for_event(record_completion_event(device, stream))
    bank_events: list[CompletionEvent | None] = [None] * num_banks
    bank = 0
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs[:end_chunk], cache_context.max_batch_size, skip_count=start_chunk
    ):
        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue
        skip_tokens_in_chunk = effective_start - batch_start_token

        bank_event = bank_events[bank]
        if bank_event is not None:
            _wait_for_event(bank_event)
        slots = [
            cache_context.get_landing_slot(bank, chunk_idx, object_group_id)
            for chunk_idx in range(batch_len)
        ]
        reads: list[tuple[BorrowedMemoryObj, PendingRead]] = []
        try:
            for memory_obj, slot in zip(memory_object_batch, slots, strict=True):
                if not isinstance(memory_obj, BorrowedMemoryObj):
                    raise ValueError(
                        "The NIC DMA engine moves only objects a peer owns, "
                        f"got {memory_obj!r}"
                    )
                reads.append(
                    (memory_obj, memory_obj.read_into(slot.region_id, slot.offset))
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
            kernel_group_offset = cache_context.get_kernel_group_offset_in_object(
                object_group_id, kernel_group_id
            )
            device_ops.multi_layer_block_kv_transfer(
                cache_context.get_kernel_group_kv_pointers(kernel_group_id),
                [slot.buffer.data_ptr() + kernel_group_offset for slot in slots],
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
        bank_events[bank] = record_completion_event(device, stream)
        bank = (bank + 1) % num_banks
