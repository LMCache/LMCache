# SPDX-License-Identifier: Apache-2.0
"""Transfer one object group's full chunks: split them by transfer engine,
then pick each run's copy path."""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.cache_context.base import BaseCacheContext
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    batch_memcpy_available,
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    split_into_runs,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    native_plan_available,
    run_object_group_transfer_plan,
    run_staged_loop,
)
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

    Host-memory runs take the direct copy if usable, else the native plan if
    the extension has it; every other run is staged one batch at a time.

    Args:
        cache_context: Cache context of the KV cache.
        block_ids_gpu: Block IDs on the GPU, per kernel group. The caller
            stages them with :func:`downsample_and_stage_block_ids`; for each
            kernel group ``i``, ``len(block_ids_gpu[i])`` is
            ``len(memory_objs)`` times the blocks kept per chunk in that group.
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
        ValueError: If a None entry is found on retrieve.
        NotImplementedError: If an object is owned by a peer.
    """
    for engine, start, end in split_into_runs(memory_objs):
        chunk_range = (start, end)
        is_host_memory = engine is TransferEngine.GPU_DMA
        if (
            is_host_memory
            and batch_memcpy_available()
            and direct_transfer_supported(
                cache_context, object_group_id, memory_objs[start:end], block_ids_host
            )
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
        elif is_host_memory and native_plan_available():
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
                engine=engine,
                chunk_range=chunk_range,
            )
