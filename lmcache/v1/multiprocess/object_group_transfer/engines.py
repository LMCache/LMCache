# SPDX-License-Identifier: Apache-2.0
"""Transfer engines: which one moves an object, decided from its L1 type.

An engine only moves bytes between objects and staging regions; what the
regions hold (a full chunk, an unfull chunk) is the caller's business.
"""

# Standard
from typing import Sequence, cast
import enum

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)
from lmcache.v1.multiprocess.object_group_transfer import gpu_dma, nic_dma, nvme_dma
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native


class TransferEngine(enum.Enum):
    """How an L1 object's bytes travel to or from GPU memory."""

    GPU_DMA = "gpu_dma"
    """Host memory (DRAM, DevDAX): a host/GPU copy."""
    NVME_DMA = "nvme_dma"
    """Storage-backed object: a file read or write against GPU memory."""
    NIC_DMA = "nic_dma"
    """Object a peer owns: a transfer-channel read into GPU memory."""


def select_engine(memory_obj: MemoryObj) -> TransferEngine:
    """Return the engine for ``memory_obj``'s type.

    Args:
        memory_obj: An L1 object.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        return TransferEngine.NVME_DMA
    if isinstance(memory_obj, BorrowedMemoryObj):
        return TransferEngine.NIC_DMA
    return TransferEngine.GPU_DMA


def stage(
    engine: TransferEngine,
    cache_context: BaseCacheContext,
    memory_objs: Sequence[MemoryObj],
    buffers: Sequence[torch.Tensor],
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Move objects to or from their staging regions with ``engine``.

    On return the bytes are in place, or the copies are queued on the
    stream ahead of anything queued later.

    Args:
        engine: The engine of every object in ``memory_objs``.
        cache_context: Cache context owning the staging regions.
        memory_objs: The objects.
        buffers: One staging region per object.
        direction: H2D fills the regions, D2H fills the objects.

    Raises:
        ValueError: If an object and its region differ in size, or a store
            targets an object a peer owns.
        RuntimeError: If an object a peer owns cannot be read.
    """
    if engine is TransferEngine.GPU_DMA:
        gpu_dma.stage(cache_context, memory_objs, buffers, direction)
    elif engine is TransferEngine.NVME_DMA:
        nvme_dma.stage(
            cache_context,
            cast(Sequence[GDSMemoryObject], memory_objs),
            buffers,
            direction,
        )
    else:
        nic_dma.stage(cache_context, memory_objs, buffers, direction)


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
