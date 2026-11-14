# SPDX-License-Identifier: Apache-2.0
"""Split one object group's chunks by transfer engine and run each engine."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import gpu_dma, nic_dma, nvme_dma
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    split_into_runs,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


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
    """Transfer one object group's chunks, each by the engine its L1 object
    selects. Runs are issued in chunk order on the context's stream.

    Args:
        cache_context: Cache context of the KV cache.
        block_ids_gpu: Block IDs on the GPU, per kernel group.
        memory_objs: One object per chunk; a None entry raises on retrieve and
            skips its batch on store.
        object_group_id: Object group index.
        batch_size: Chunks per batch.
        skip_first_n_tokens: Leading tokens not written, to leave shared blocks
            alone.
        direction: H2D (retrieve) or D2H (store).
        transfer_key: Operation id for phase timing.
        block_ids_host: Block IDs as host lists; empty forces the kernel path.

    Raises:
        ValueError: If a None entry is found on retrieve, or a store targets
            an object a peer owns.
        RuntimeError: If a chunk a peer owns cannot be read.
    """
    for engine, start, end in split_into_runs(memory_objs):
        if engine is TransferEngine.NVME_DMA:
            transfer = nvme_dma.transfer
        elif engine is TransferEngine.NIC_DMA:
            transfer = nic_dma.transfer
        else:
            transfer = gpu_dma.transfer
        transfer(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            transfer_key=transfer_key,
            block_ids_host=block_ids_host,
            chunk_range=(start, end),
        )
