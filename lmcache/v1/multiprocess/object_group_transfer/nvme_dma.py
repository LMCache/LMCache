# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: objects held in a GDS slab.

A retrieve loads slab objects with blocking reads on host threads when the
slab is configured with reader threads (``--gds-l1-read-threads``; the
default 0 keeps the stream-ordered read). The caller queues device work for a batch
only after its reads have returned: the read does device work of its own
(cuFile compat mode), and kernels queued ahead behind a host-written signal
deadlocked with it.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Sequence, cast
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_l1_gds_context
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
import lmcache.lmcache_native as lmcache_native


def read_threads(memory_obj: MemoryObj) -> int:
    """Return the reader threads configured for the object's slab (0: none)."""
    slab_obj = cast(GDSMemoryObject, memory_obj)
    return get_l1_gds_context(slab_obj.get_l1_manager()).read_threads


def read_slab_object(
    memory_obj: MemoryObj, buffer: torch.Tensor, device: torch.device
) -> None:
    """Read one slab object into its staging region on the calling thread.

    Raises:
        OSError: If the read fails or is short.
    """
    slab_obj = cast(GDSMemoryObject, memory_obj)
    # The GDS library needs the device's context on the reading thread.
    with torch_dev.device(device):
        get_l1_gds_context(slab_obj.get_l1_manager()).read_blocking(slab_obj, buffer)


class NvmeDmaEngine:
    """Stages slab-backed objects with GDS reads and writes."""

    def __init__(self) -> None:
        self._readers: ThreadPoolExecutor | None = None
        self._readers_lock = threading.Lock()

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Queue one slab read or write per object on the current stream.

        Args:
            memory_objs: Slab-backed objects (``GDSMemoryObject``).
            buffers: One registered staging region per object.
            direction: H2D reads the slab into the regions, D2H writes the
                regions to the slab.
        """
        slab_direction = (
            SlabDirection.READ
            if direction == lmcache_native.TransferDirection.H2D
            else SlabDirection.WRITE
        )
        for memory_obj, buffer in zip(memory_objs, buffers, strict=True):
            slab_obj = cast(GDSMemoryObject, memory_obj)
            get_l1_gds_context(slab_obj.get_l1_manager()).transfer_async(
                slab_obj, buffer, slab_direction
            )

    def loads_from_host(self, memory_objs: Sequence[MemoryObj]) -> bool:
        """Return whether these objects are read on host threads: every
        object's slab is configured with reader threads."""
        return all(read_threads(memory_obj) > 0 for memory_obj in memory_objs)

    def load_from_host(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Read the objects into their staging regions on the reader threads
        and return when every read has ended.

        Raises:
            OSError: If a read failed or was short.
        """
        with self._readers_lock:
            if self._readers is None:
                self._readers = ThreadPoolExecutor(
                    read_threads(memory_objs[0]),
                    thread_name_prefix="lmcache-gds-read",
                )
        reads = [
            self._readers.submit(read_slab_object, memory_obj, buffer, device)
            for memory_obj, buffer in zip(memory_objs, buffers, strict=True)
        ]
        # No reader may still write the regions when the caller sees the
        # failure, so wait for all of them before raising.
        wait(reads)
        for read in reads:
            read.result()
