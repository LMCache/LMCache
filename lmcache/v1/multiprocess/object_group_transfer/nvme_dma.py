# SPDX-License-Identifier: Apache-2.0
"""NVMe DMA engine: objects held in a GDS slab.

Two ways to load a slab object into a staging region:

- **On the stream** (:meth:`NvmeDmaEngine.stage`): one stream-ordered GDS
  read per object. The submitting call can block for the whole read (cuFile
  in compat mode), and the reads of a stream run one after another.
- **On host threads** (:meth:`NvmeDmaEngine.load_from_host`): reader threads
  issue blocking slab reads side by side, and the call returns when the
  batch has landed. This is the default when the backend has a blocking
  read; ``LMCACHE_GDS_READ_THREADS`` sets the number of reader threads (2 by
  default, 0 turns host-thread loads off).

Device work for a batch is queued only after its reads have returned. A
blocking read does device work of its own (cuFile in compat mode copies the
bytes to the device on an internal stream), and the driver may run that
work behind a pending doorbell wait; a stream that waited for a host thread
to signal the end of the read would deadlock with the read itself.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Sequence, cast
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gds_context import SlabDirection, get_l1_gds_context
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

ENV_GDS_READ_THREADS = "LMCACHE_GDS_READ_THREADS"
_DEFAULT_READ_THREADS = 2


def gds_read_threads() -> int:
    """Return the number of host threads that read slab objects.

    Returns:
        ``LMCACHE_GDS_READ_THREADS`` if set (0 turns host-thread loads off),
        else 2.

    Raises:
        ValueError: If the variable is not a non-negative integer.
    """
    raw = os.environ.get(ENV_GDS_READ_THREADS)
    if raw is None:
        return _DEFAULT_READ_THREADS
    threads = int(raw)
    if threads < 0:
        raise ValueError(f"{ENV_GDS_READ_THREADS} must be >= 0, got {raw!r}")
    return threads


def read_slab_object(memory_obj: MemoryObj, buffer: torch.Tensor) -> None:
    """Read one slab object into its staging region on the calling thread.

    Args:
        memory_obj: A slab-backed object (``GDSMemoryObject``).
        buffer: Its registered staging region.

    Raises:
        OSError: If the read fails or is short.
    """
    slab_obj = cast(GDSMemoryObject, memory_obj)
    get_l1_gds_context(slab_obj.get_l1_manager()).read_blocking(slab_obj, buffer)


def _read_on_device(
    memory_obj: MemoryObj, buffer: torch.Tensor, device: torch.device
) -> None:
    # The GDS library needs the device's context on the reading thread.
    with torch_dev.device(device):
        read_slab_object(memory_obj, buffer)


class NvmeDmaEngine:
    """Stages slab-backed objects with GDS reads and writes."""

    reads_files = True

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
        """Return whether these objects are read on host threads: reader
        threads are enabled and every object's slab has a blocking read.

        Args:
            memory_objs: Slab-backed objects (``GDSMemoryObject``).
        """
        if gds_read_threads() == 0:
            return False
        return all(
            get_l1_gds_context(
                cast(GDSMemoryObject, memory_obj).get_l1_manager()
            ).supports_blocking_read
            for memory_obj in memory_objs
        )

    def load_from_host(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Read the objects into their staging regions on the reader threads
        and return when every read has ended.

        Args:
            memory_objs: Slab-backed objects for which
                :meth:`loads_from_host` is True.
            buffers: One registered staging region per object.
            device: The device that owns the regions.

        Raises:
            OSError: If a read failed or was short; the regions then hold
                incomplete data.
        """
        readers = self._get_readers()
        reads = [
            readers.submit(_read_on_device, memory_obj, buffer, device)
            for memory_obj, buffer in zip(memory_objs, buffers, strict=True)
        ]
        # Wait for all of them before raising, so that no reader still
        # writes the regions when the caller handles the failure.
        wait(reads)
        for read in reads:
            read.result()

    def _get_readers(self) -> ThreadPoolExecutor:
        with self._readers_lock:
            if self._readers is None:
                self._readers = ThreadPoolExecutor(
                    gds_read_threads(), thread_name_prefix="lmcache-gds-read"
                )
            return self._readers
