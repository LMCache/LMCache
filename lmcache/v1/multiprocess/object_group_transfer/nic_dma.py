# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: reads objects a peer owns into the staging buffer over
the transfer channel.

The reads are outside the stream, so the host orders them: wait for the
stream, then wait for the reads.
"""

# Standard
from dataclasses import dataclass
from typing import Sequence
import time
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import BorrowedMemoryObj, MemoryObj, PendingRead
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.stream import synchronize_stream
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Pause between polls of in-flight reads.
_POLL_INTERVAL_S = 0.00002
# Reads still pending after this long are logged once per call.
_SLOW_READ_WARNING_S = 5.0


@dataclass(frozen=True)
class StagingRegion:
    """A cache context's staging buffer as a transfer-channel region.

    Attributes:
        region_id: The channel's region id.
        base_ptr: Start address.
        size: Size in bytes.
    """

    region_id: int
    base_ptr: int
    size: int


_staging_regions: "weakref.WeakKeyDictionary[BaseCacheContext, StagingRegion]" = (
    weakref.WeakKeyDictionary()
)


def set_staging_region(cache_context: BaseCacheContext, region: StagingRegion) -> None:
    """Record the channel region of ``cache_context``'s staging buffer.

    The caller keeps the buffer registered until
    :func:`clear_staging_region`; every region read into must start on the
    channel's alignment.

    Args:
        cache_context: The cache context.
        region: Its staging region.
    """
    _staging_regions[cache_context] = region


def clear_staging_region(cache_context: BaseCacheContext) -> None:
    """Forget ``cache_context``'s staging region, if any.

    Args:
        cache_context: The cache context.
    """
    _staging_regions.pop(cache_context, None)


def _start_read(
    region: StagingRegion,
    memory_obj: MemoryObj,
    buffer: torch.Tensor,
) -> tuple[BorrowedMemoryObj, PendingRead]:
    """Start reading one object into ``buffer``.

    Raises:
        ValueError: If the object is not peer-owned, differs in size from
            ``buffer``, or ``buffer`` is outside ``region``.
    """
    if not isinstance(memory_obj, BorrowedMemoryObj):
        raise ValueError(
            f"The NIC DMA engine moves only objects a peer owns, got {memory_obj!r}"
        )
    if memory_obj.get_size() != buffer.nbytes:
        raise ValueError(
            f"Size mismatch: memory_obj nbytes={memory_obj.get_size()}, "
            f"staging region nbytes={buffer.nbytes}"
        )
    offset = buffer.data_ptr() - region.base_ptr
    if offset < 0 or offset + buffer.nbytes > region.size:
        raise ValueError(
            f"Staging region at offset {offset} is outside the registered region "
            f"of {region.size} bytes"
        )
    return memory_obj, memory_obj.read_into(region.region_id, offset)


def _settle(
    reads: list[tuple[BorrowedMemoryObj, PendingRead]],
) -> BorrowedMemoryObj | None:
    """Wait for every read to stop; return an object whose read failed.

    No read is abandoned (it could land in a reused slot) and there is no
    deadline, so the transport must stop every read. Slow reads are logged
    once.
    """
    failed: BorrowedMemoryObj | None = None
    started = time.monotonic()
    warned = False
    while reads:
        running = []
        for memory_obj, read in reads:
            if not read.is_finished():
                running.append((memory_obj, read))
            elif not read.succeeded() and failed is None:
                failed = memory_obj
        reads = running
        if not reads:
            break
        if not warned and time.monotonic() - started > _SLOW_READ_WARNING_S:
            logger.warning(
                "Still waiting for %d peer reads after %.0f s, first at offset %#x",
                len(reads),
                _SLOW_READ_WARNING_S,
                reads[0][0].meta.address,
            )
            warned = True
        time.sleep(_POLL_INTERVAL_S)
    return failed


def stage(
    cache_context: BaseCacheContext,
    memory_objs: Sequence[MemoryObj],
    buffers: Sequence[torch.Tensor],
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Read objects a peer owns into their staging regions.

    Blocks until the stream is idle and the reads finished. Needs one
    thread per cache context, which client affinity gives.

    Args:
        cache_context: Cache context owning the staging buffer.
        memory_objs: Peer-owned objects.
        buffers: One staging region per object.
        direction: Must be H2D.

    Raises:
        ValueError: If ``direction`` is not H2D, an object is not
            peer-owned, or an object does not fit its region.
        RuntimeError: If the staging buffer is not registered with the
            transfer channel, or a read failed.
    """
    if direction != lmcache_native.TransferDirection.H2D:
        raise ValueError("Objects a peer owns can only be retrieved")
    region = _staging_regions.get(cache_context)
    if region is None:
        raise RuntimeError(
            "Retrieving objects a peer owns needs the staging buffer registered "
            "with the transfer channel"
        )
    # Earlier work on the stream may still use the staging regions.
    synchronize_stream(cache_context.device, cache_context.stream)
    reads: list[tuple[BorrowedMemoryObj, PendingRead]] = []
    try:
        for memory_obj, buffer in zip(memory_objs, buffers, strict=True):
            reads.append(_start_read(region, memory_obj, buffer))
    finally:
        failed = _settle(reads)
    if failed is not None:
        raise RuntimeError(
            f"Read of the peer object at offset {failed.meta.address:#x} failed"
        )
