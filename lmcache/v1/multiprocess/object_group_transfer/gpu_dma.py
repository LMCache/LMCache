# SPDX-License-Identifier: Apache-2.0
"""GPU DMA engine: chunks held in host memory (DRAM, DevDAX)."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    batch_memcpy_available,
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    native_plan_available,
    run_object_group_transfer_plan,
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
    """Move a run of host-memory chunks: direct copy if usable, else staged.

    Args:
        cache_context: Cache context of the KV cache.
        block_ids_gpu: Block IDs on the GPU, per kernel group.
        memory_objs: The group's objects, one per chunk.
        object_group_id: Object group index.
        batch_size: Chunks per batch.
        skip_first_n_tokens: Leading tokens not written.
        direction: H2D (retrieve) or D2H (store).
        transfer_key: Operation id for phase timing.
        block_ids_host: Block IDs as host lists; empty rules out the direct copy.
        chunk_range: ``[start, end)`` chunks to move.

    Raises:
        ValueError: If the range has a None entry on retrieve.
    """
    start, end = chunk_range
    if batch_memcpy_available() and direct_transfer_supported(
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
    elif native_plan_available():
        run_object_group_transfer_plan(
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
