# SPDX-License-Identifier: Apache-2.0
"""Split one object group's chunks by transfer engine and run each engine."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import gpu_dma, nvme_dma
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
    """Helper function to transfer memory objects of a single object group
    to/from GPU, with batching support.

    Each chunk is moved by the engine its L1 object calls for (see
    :func:`~lmcache.v1.multiprocess.object_group_transfer.engines.select_engine`),
    so a group that mixes object types does not fall back as a whole. The
    engines run one after another on the context's stream.

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
    for engine, start, end in split_into_runs(memory_objs):
        transfer = (
            nvme_dma.transfer if engine is TransferEngine.NVME_DMA else gpu_dma.transfer
        )
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
