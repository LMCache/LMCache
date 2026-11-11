# SPDX-License-Identifier: Apache-2.0
"""GPU DMA engine: chunks held in host memory (DRAM, DevDAX)."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    _HAS_BATCH_MEMCPY_ASYNC,
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
    _run_object_group_transfer_plan,
    run_staged_loop,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


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
    """Move a run of host-memory chunks, by the fastest path available.

    The direct copy path is taken when the build, the driver and the layout
    allow it; otherwise the chunks are staged and scattered, as one native
    plan when the extension provides it.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The object group's objects, one per chunk.
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).
        transfer_key: Identity of this store/retrieve operation.
        block_ids_host: The block ids as host lists; empty when the caller
            only has device tensors.
        chunk_range: ``[start, end)`` chunks this engine moves.

    Raises:
        ValueError: If a None entry is found in the range when direction is H2D.
    """
    start, end = chunk_range
    if _HAS_BATCH_MEMCPY_ASYNC and direct_transfer_supported(
        cache_context, object_group_id, memory_objs[start:end], block_ids_host
    ):
        run_direct_transfer(
            cache_context,
            block_ids_host,
            memory_objs,
            object_group_id,
            skip_first_n_tokens,
            direction,
            chunk_range=chunk_range,
        )
    elif _HAS_NATIVE_OBJECT_GROUP_TRANSFER:
        _run_object_group_transfer_plan(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            transfer_key=transfer_key,
            chunk_range=chunk_range,
        )
    else:
        run_staged_loop(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            chunk_range=chunk_range,
        )
