# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: chunks a peer owns, read into the staging buffer over the
transfer channel and scattered by the block transfer kernel.

The reads are not ordered by the stream, so the host orders them: per batch
it waits for the stream, then for the reads, then queues the kernel.
"""

# Standard
from dataclasses import dataclass
from typing import Sequence
import time
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import BorrowedMemoryObj, MemoryObj, PendingRead
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    batched_iteration_with_skip,
    resolve_chunk_range,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    launch_batch_kernels,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.stream import synchronize_stream
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Pause between polls of in-flight reads.
_POLL_INTERVAL_S = 0.00002
# Reads still pending after this long are logged once per batch.
_SLOW_READ_WARNING_S = 5.0


@dataclass(frozen=True)
class StagingRegion:
    """A cache context's staging buffer as a transfer-channel region.

    Attributes:
        region_id: The channel's id for the region.
        base_ptr: Address the region starts at.
        size: Bytes in the region.
    """

    region_id: int
    base_ptr: int
    size: int


_staging_regions: "weakref.WeakKeyDictionary[BaseCacheContext, StagingRegion]" = (
    weakref.WeakKeyDictionary()
)


def set_staging_region(cache_context: BaseCacheContext, region: StagingRegion) -> None:
    """Record which channel region ``cache_context``'s staging buffer is.

    The caller registers the buffer and keeps it registered until
    :func:`clear_staging_region`. Every staging slot must start on the
    channel's alignment.

    Args:
        cache_context: The cache context whose staging buffer was registered.
        region: The region.
    """
    _staging_regions[cache_context] = region


def clear_staging_region(cache_context: BaseCacheContext) -> None:
    """Forget ``cache_context``'s staging region; a no-op when none is set.

    Args:
        cache_context: The cache context whose staging buffer is deregistered.
    """
    _staging_regions.pop(cache_context, None)


def _start_read(
    cache_context: BaseCacheContext,
    region: StagingRegion,
    memory_obj: MemoryObj | None,
    chunk_idx: int,
    object_group_id: int,
) -> tuple[BorrowedMemoryObj, PendingRead]:
    """Start reading one chunk into staging slot ``chunk_idx``.

    Raises:
        ValueError: If the object is not one a peer owns, its size differs
            from the slot's, or the slot is outside ``region``.
    """
    if not isinstance(memory_obj, BorrowedMemoryObj):
        raise ValueError(
            f"The NIC DMA engine moves only objects a peer owns, got {memory_obj!r}"
        )
    slot = cache_context.get_temp_object_group_buffer(chunk_idx, object_group_id)
    if memory_obj.get_size() != slot.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={memory_obj.get_size()}, "
            f"staging slot nbytes={slot.nbytes}"
        )
    offset = slot.data_ptr() - region.base_ptr
    if offset < 0 or offset + slot.nbytes > region.size:
        raise ValueError(
            f"Staging slot at offset {offset} is outside the registered region "
            f"of {region.size} bytes"
        )
    return memory_obj, memory_obj.read_into(region.region_id, offset)


def _settle(
    reads: list[tuple[BorrowedMemoryObj, PendingRead]],
) -> BorrowedMemoryObj | None:
    """Block until every read stopped; return an object whose read failed.

    All are awaited: an abandoned read would land in a reused slot. There is
    no deadline here; :class:`PendingRead` requires every read to stop. Slow
    reads are logged once.
    """
    failed: BorrowedMemoryObj | None = None
    started = time.monotonic()
    warned = False
    while reads:
        running = []
        for memory_obj, read in reads:
            if not read.is_finished():
                running.append((memory_obj, read))
            elif not read.succeeded() and failed is None:
                failed = memory_obj
        reads = running
        if not reads:
            break
        if not warned and time.monotonic() - started > _SLOW_READ_WARNING_S:
            logger.warning(
                "Still waiting for %d peer reads after %.0f s, first at offset %#x",
                len(reads),
                _SLOW_READ_WARNING_S,
                reads[0][0].meta.address,
            )
            warned = True
        time.sleep(_POLL_INTERVAL_S)
    return failed


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

    Blocks until the run's last read finished; the kernels stay queued. One
    thread must drive a cache context's stream and staging slots, which the
    handlers' client affinity gives.

    Args:
        cache_context: The cache context of the KV cache.
        block_ids_gpu: GPU block IDs, indexed by kernel group.
        memory_objs: The object group's objects, one per chunk.
        object_group_id: Index of the object group.
        batch_size: Chunks per batch.
        skip_first_n_tokens: Tokens not written at the start of the range;
            chunks it covers entirely are not read.
        direction: Must be H2D.
        transfer_key: Unused.
        block_ids_host: Unused.
        chunk_range: ``[start, end)`` chunks to move.

    Raises:
        ValueError: If ``direction`` is not H2D, a chunk in the range is not
            an object a peer owns, or an object does not fit its staging slot.
        RuntimeError: If the staging buffer is not registered with the
            transfer channel, or a read failed.
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

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id):
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
    start_chunk, end_chunk = resolve_chunk_range(
        len(memory_objs), chunk_range, num_objects_to_skip
    )

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

        # Earlier work on the stream may still use the staging slots.
        synchronize_stream(cache_context.device, cache_context.stream)
        reads: list[tuple[BorrowedMemoryObj, PendingRead]] = []
        try:
            # The kernel skips the chunks before the first one it writes.
            for chunk_idx in range(
                skip_tokens_in_chunk // lmcache_chunk_size, batch_len
            ):
                reads.append(
                    _start_read(
                        cache_context,
                        region,
                        memory_object_batch[chunk_idx],
                        chunk_idx,
                        object_group_id,
                    )
                )
        finally:
            failed = _settle(reads)
        if failed is not None:
            raise RuntimeError(
                f"Read of the peer object at offset {failed.meta.address:#x} failed"
            )
        launch_batch_kernels(
            cache_context,
            block_ids_gpu,
            object_group_id,
            start_object_idx,
            batch_len,
            skip_tokens_in_chunk,
            direction,
        )
