# SPDX-License-Identifier: Apache-2.0
"""Which transfer engine moves a chunk, decided from its L1 object."""

# Standard
from typing import Sequence
import enum

# First Party
from lmcache.v1.memory_management import (
    BorrowedMemoryObj,
    GDSMemoryObject,
    MemoryObj,
)


class TransferEngine(enum.Enum):
    """How a chunk's bytes travel between its L1 object and GPU memory."""

    GPU_DMA = "gpu_dma"
    """Host memory (DRAM, DevDAX): a host/GPU copy."""
    NVME_DMA = "nvme_dma"
    """Storage-backed object: a file read or write against GPU memory."""
    NIC_DMA = "nic_dma"
    """Object a peer owns: a transfer-channel read into GPU memory."""


def select_engine(memory_obj: MemoryObj) -> TransferEngine:
    """Return the engine that moves ``memory_obj``.

    Args:
        memory_obj: The chunk's L1 object.

    Returns:
        The engine for the object's type.
    """
    if isinstance(memory_obj, GDSMemoryObject):
        return TransferEngine.NVME_DMA
    if isinstance(memory_obj, BorrowedMemoryObj):
        return TransferEngine.NIC_DMA
    return TransferEngine.GPU_DMA


def split_into_runs(
    memory_objs: Sequence[MemoryObj | None],
) -> list[tuple[TransferEngine, int, int]]:
    """Split an object group's chunks into consecutive same-engine runs.

    Args:
        memory_objs: One object per chunk. A None entry joins the next
            object's run, or the previous run at the end.

    Returns:
        ``(engine, start, end)`` per run, covering every chunk in order.
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
