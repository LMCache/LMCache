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
    """Move a run of storage-backed chunks: staged, one batch at a time.

    Args are those of ``transfer_kv_per_object_group``; ``chunk_range`` is
    the ``[start, end)`` chunks to move.

    Raises:
        ValueError: If the range has a None entry on retrieve.
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
