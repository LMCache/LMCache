# SPDX-License-Identifier: Apache-2.0
"""Transfer one object group's full chunks: split them by transfer engine,
then pick each run's copy path.

A retrieve's extra streams are tied to the caller's stream, the one the
request is ordered on:

- gate: the scatter stream waits for the caller's stream, so no kernel runs
  before the block IDs are staged and the engine has released the blocks.
- join: the caller's stream waits for every stream the call used, so what
  the caller queues next runs after the last load and the last kernel.
"""

# Standard
from typing import Any, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    batch_memcpy_available,
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    split_into_runs,
    staging_lane,
)
from lmcache.v1.multiprocess.object_group_transfer.staged import (
    native_plan_available,
    run_object_group_transfer_plan,
    run_staged_loop,
)
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.staging_lane import StagingLane
import lmcache.lmcache_native as lmcache_native


def _lane(
    cache_context: BaseCacheContext,
    lanes: dict[TransferEngine, StagingLane],
    engine: TransferEngine,
    scatter_stream: Any,
) -> StagingLane:
    """Return ``engine``'s staging lane; the call's first lane gates the
    scatter stream behind the caller's stream."""
    lane = lanes.get(engine)
    if lane is None:
        if not lanes:
            device = cache_context.device
            platform_stream.record_completion_event(
                device, platform_stream.current_stream(device)
            ).wait_on(scatter_stream)
        lane = lanes[engine] = staging_lane(cache_context, engine)
    return lane


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
    selects. Runs are issued in chunk order; work the caller queues next on
    its stream runs after all of them.

    Host-memory runs take the direct copy if usable, else the native plan if
    the extension has it. A retrieve on a context with a scatter stream
    stages every run of another engine on that engine's lane; the rest is
    staged one batch at a time on the caller's stream.

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
    lanes: dict[TransferEngine, StagingLane] = {}
    try:
        _transfer_runs(
            lanes,
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            transfer_key,
            block_ids_host,
        )
    finally:
        # Join, also when the call raises: a queued load must finish before
        # the caller releases the objects it reads.
        if lanes:
            device = cache_context.device
            caller_stream = platform_stream.current_stream(device)
            for stream in (
                *(lane.stream for lane in lanes.values()),
                cache_context.get_scatter_stream(),
            ):
                platform_stream.record_completion_event(device, stream).wait_on(
                    caller_stream
                )


def _transfer_runs(
    lanes: dict[TransferEngine, StagingLane],
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    transfer_key: str,
    block_ids_host: Sequence[Sequence[int]],
) -> None:
    """Issue every run; ``lanes`` collects the lanes the call used."""
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
            # Lanes are for retrieves from the tiers slower than host memory,
            # whose staged copy already runs near the link's rate.
            scatter_stream = (
                cache_context.get_scatter_stream()
                if direction == lmcache_native.TransferDirection.H2D
                and not is_host_memory
                else None
            )
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
                lane=_lane(cache_context, lanes, engine, scatter_stream)
                if scatter_stream is not None
                else None,
                scatter_stream=scatter_stream,
            )
