# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: chunks held in storage-backed objects."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer.staged import run_staged_loop
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
    """Move a run of storage-backed chunks through the staging buffer.

    The staging copy of such an object is a file read or write against the
    GPU staging buffer, issued on the stream like a host copy.

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
