# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    create_transfer_channel_context,
)

# Import the implementations so they self-register their factories.
import lmcache.v1.distributed.transfer_channel.impl  # noqa: F401

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

__all__ = [
    "TransferChannelAddress",
    "TransferChannelTransferResult",
    "TransferChannelContext",
    "TransferChannelServer",
    "TransferChannelClient",
    "initialize_transfer_channel_context",
    "get_transfer_channel_context",
    "delete_transfer_channel_context",
    "StagingRegion",
    "enable_staging_buffers",
    "disable_staging_buffers",
    "staging_alignment",
    "register_staging_buffer",
    "unregister_staging_buffer",
    "find_staging_region",
]

logger = init_logger(__name__)

_context: TransferChannelContext | None = None
_context_lock = threading.Lock()


def initialize_transfer_channel_context(
    transfer_channel_type: str,
    l1_memory_desc: "L1MemoryDesc",
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> TransferChannelContext:
    """Create the global transfer channel context.

    Args:
        transfer_channel_type: Registered implementation name, such as
            ``"nixl"`` or ``"mooncake"``.
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's singleton server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity (the
            key peers store its reverse client under).
        **kwargs: Implementation-specific keyword arguments forwarded to the
            factory.

    Returns:
        The created context (also retrievable via ``get_transfer_channel_context``).

    Raises:
        RuntimeError: If a context has already been initialized.
        ValueError: If no factory is registered for ``transfer_channel_type``.
    """
    global _context
    with _context_lock:
        if _context is not None:
            raise RuntimeError(
                "Transfer channel context already initialized; call "
                "delete_transfer_channel_context() first."
            )
        _context = create_transfer_channel_context(
            transfer_channel_type=transfer_channel_type,
            l1_memory_desc=l1_memory_desc,
            listen_url=listen_url,
            advertise_url=advertise_url,
            **kwargs,
        )
        return _context


def get_transfer_channel_context() -> TransferChannelContext:
    """Get the global transfer channel context.

    Raises:
        RuntimeError: If the context has not been initialized yet.
    """
    with _context_lock:
        if _context is None:
            raise RuntimeError("Transfer channel context not initialized.")
        return _context


def delete_transfer_channel_context() -> None:
    """Delete the global transfer channel context, if it exists."""
    global _context
    with _context_lock:
        if _context is not None:
            _context.close()
            _context = None


# GPU-direct borrowing state, guarded by ``_context_lock``:
# - the staging target: enabled by the P2P controller once the context exists,
#   disabled when it closes; read by GPU cache contexts as they start.
# - the staging regions: added and removed by each GPU cache context for its
#   own staging buffer; read by the NIC DMA engine on every borrowed retrieve.
_staging_target: tuple[TransferChannelContext, int] | None = None


@dataclass(frozen=True)
class StagingRegion:
    """A GPU cache context's staging buffer as a transfer-channel region.

    Attributes:
        context: The transfer channel context the buffer is registered with.
        region_id: The context's region id.
        base_ptr: Start address.
        size: Size in bytes.
        device: The device holding the buffer.
        stream: The device stream whose queued work uses the buffer; its
            type depends on the device backend.
    """

    context: TransferChannelContext
    region_id: int
    base_ptr: int
    size: int
    device: torch.device
    stream: object


_staging_regions: dict[int, StagingRegion] = {}


def enable_staging_buffers(context: TransferChannelContext, align_bytes: int) -> None:
    """Let GPU cache contexts register their staging buffers with ``context``.

    Args:
        context: The transfer channel context to register with.
        align_bytes: The alignment its regions need, the L1 alignment.
    """
    global _staging_target
    with _context_lock:
        _staging_target = (context, align_bytes)


def disable_staging_buffers() -> None:
    """Stop new staging buffers from registering."""
    global _staging_target
    with _context_lock:
        _staging_target = None


def staging_alignment() -> int:
    """Return the alignment staging-buffer slots need; 1 when disabled."""
    with _context_lock:
        return 1 if _staging_target is None else _staging_target[1]


def register_staging_buffer(
    buffer: torch.Tensor, stream: object
) -> StagingRegion | None:
    """Register a staging buffer so peer reads can land in it.

    Args:
        buffer: The flat staging tensor, aligned to :func:`staging_alignment`.
        stream: The device stream whose queued work uses the buffer.

    Returns:
        The region to pass to :func:`unregister_staging_buffer`, or None when
        staging buffers are disabled.
    """
    with _context_lock:
        target = _staging_target
    if target is None:
        return None
    region = StagingRegion(
        target[0],
        target[0].register_memory_region(
            buffer.data_ptr(), buffer.nbytes, buffer.device.index
        ),
        buffer.data_ptr(),
        buffer.nbytes,
        buffer.device,
        stream,
    )
    with _context_lock:
        _staging_regions[region.base_ptr] = region
    return region


def unregister_staging_buffer(region: StagingRegion) -> None:
    """Forget and deregister a staging buffer.

    Args:
        region: The region :func:`register_staging_buffer` returned.
    """
    with _context_lock:
        if _staging_regions.get(region.base_ptr) is region:
            del _staging_regions[region.base_ptr]
    try:
        region.context.deregister_memory_region(region.region_id)
    except KeyError:
        # The context was closed first, which released every region.
        logger.debug("Staging region %d was already released", region.region_id)


def find_staging_region(ptr: int) -> StagingRegion | None:
    """Return the registered staging region containing ``ptr``, if any.

    Args:
        ptr: An address inside a staging buffer.
    """
    with _context_lock:
        for region in _staging_regions.values():
            if region.base_ptr <= ptr < region.base_ptr + region.size:
                return region
    return None
