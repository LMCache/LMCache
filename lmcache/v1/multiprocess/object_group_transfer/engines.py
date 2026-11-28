# SPDX-License-Identifier: Apache-2.0
"""Transfer engines: which one moves an object, decided from its L1 type.

An engine only moves bytes between objects and staging regions; what the
regions hold (a full chunk, an unfull chunk) is the caller's business.
"""

# Standard
from typing import Protocol, Sequence
import enum

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer.gpu_dma import GpuDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.nic_dma import NicDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.nvme_dma import NvmeDmaEngine
from lmcache.v1.multiprocess.object_group_transfer.transfer_step import (
    OffStreamEngine,
    TransferStep,
    hand_off_step,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


class TransferEngine(enum.Enum):
    """How an L1 object's bytes travel to or from GPU memory."""

    GPU_DMA = "gpu_dma"
    """Host memory (DRAM, DevDAX): a host/GPU copy."""
    NVME_DMA = "nvme_dma"
    """Storage-backed object: a file read or write against GPU memory."""
    NIC_DMA = "nic_dma"
    """Object a peer owns: a transfer-channel read into GPU memory. Off the
    stream, and not implemented yet."""


def select_engine(memory_obj: MemoryObj) -> TransferEngine:
    """Return the engine for ``memory_obj``'s type.

    Args:
        memory_obj: An L1 object.

    Returns:
        The engine that moves objects of that type.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        return TransferEngine.NVME_DMA
    if isinstance(memory_obj, BorrowedMemoryObj):
        return TransferEngine.NIC_DMA
    return TransferEngine.GPU_DMA


class DMAEngine(Protocol):
    """What every transfer engine implements.

    An engine only moves bytes between objects and staging regions; it
    knows nothing about chunks, block ids or kernels.
    """

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Move objects to or from their staging regions.

        Args:
            memory_objs: Objects of this engine's type.
            buffers: One staging region per object.
            direction: H2D fills the regions, D2H fills the objects.
        """
        ...


_ENGINES: dict[TransferEngine, DMAEngine] = {
    TransferEngine.GPU_DMA: GpuDmaEngine(),
    TransferEngine.NVME_DMA: NvmeDmaEngine(),
    TransferEngine.NIC_DMA: NicDmaEngine(),
}


def stage(
    engine: TransferEngine,
    memory_objs: Sequence[MemoryObj],
    buffers: Sequence[torch.Tensor],
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Move objects to or from their staging regions with ``engine``.

    On return the copies are queued on the stream ahead of anything queued
    later.

    Args:
        engine: The engine of every object in ``memory_objs``.
        memory_objs: The objects.
        buffers: One staging region per object.
        direction: H2D fills the regions, D2H fills the objects.

    Raises:
        ValueError: If a host-memory object and its region differ in size.
        NotImplementedError: If an object is owned by a peer.
    """
    _ENGINES[engine].stage(memory_objs, buffers, direction)


OFF_STREAM_ENGINES: dict[TransferEngine, OffStreamEngine | None] = {
    TransferEngine.NIC_DMA: None,
}
"""Off-stream engine kinds and the engine that runs each one's retrieve
steps; None if it is not implemented yet."""


def run_step(
    engine: TransferEngine, cache_context: BaseCacheContext, step: TransferStep
) -> None:
    """Run one retrieve step with ``engine``.

    An on-stream engine stages the step's objects into the context's own
    staging and queues its kernels now. An off-stream engine's step is
    handed to the retrieve the transfer belongs to (``retrieve_transfer``),
    neither staged nor launched here.

    Args:
        engine: The engine of every object of the step.
        cache_context: The retrieve's context.
        step: The step.

    Raises:
        ValueError: If a host-memory object and its region differ in size.
        NotImplementedError: If the engine is off-stream and the transfer
            does not run inside ``retrieve_transfer``.
    """
    if engine in OFF_STREAM_ENGINES:
        hand_off_step(OFF_STREAM_ENGINES[engine], step)
        return
    stage(
        engine,
        step.memory_objs,
        step.targets(cache_context),
        lmcache_native.TransferDirection.H2D,
    )
    step.launch(cache_context, step.block_ids)


def split_into_runs(
    memory_objs: Sequence[MemoryObj | None],
) -> list[tuple[TransferEngine, int, int]]:
    """Split a group's chunks into consecutive same-engine runs.

    Args:
        memory_objs: One object per chunk; a None entry joins the next
            object's run, or the previous run at the end.

    Returns:
        ``(engine, start, end)`` per run, in chunk order.
    """
    engines: list[TransferEngine | None] = [
        None if memory_obj is None else select_engine(memory_obj)
        for memory_obj in memory_objs
    ]
    following: TransferEngine | None = None
    for index in range(len(engines) - 1, -1, -1):
        if engines[index] is None:
            engines[index] = following
        following = engines[index]
    runs: list[tuple[TransferEngine, int, int]] = []
    for index, engine in enumerate(engines):
        # Trailing None entries have no following object.
        current = engine or (runs[-1][0] if runs else TransferEngine.GPU_DMA)
        if runs and runs[-1][0] is current:
            runs[-1] = (current, runs[-1][1], index + 1)
        else:
            runs.append((current, index, index + 1))
    return runs
