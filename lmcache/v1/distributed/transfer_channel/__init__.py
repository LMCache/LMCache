# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_ap import L1MemoryDesc

__all__ = [
    "TransferChannelAddress",
    "TransferChannelReadResult",
    "TransferChannelContext",
    "TransferChannelServer",
    "TransferChannelClient",
    "initialize_transfer_channel_context",
    "get_transfer_channel_context",
    "delete_transfer_channel_context",
]

_context: TransferChannelContext | None = None
_context_lock = threading.Lock()


def initialize_transfer_channel_context(
    transfer_channel_type: str,
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> TransferChannelContext:
    """Create the global transfer channel context.

    Args:
        transfer_channel_type: Currently only ``"nixl"`` is supported.
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's singleton server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity (the
            key peers store its reverse client under).

    Returns:
        The created context (also retrievable via ``get_transfer_channel_context``).
    """
    global _context
    with _context_lock:
        if _context is not None:
            raise RuntimeError(
                "Transfer channel context already initialized; call "
                "delete_transfer_channel_context() first."
            )
        if transfer_channel_type == "nixl":
            # Third Party
            from transfer_channel.nixl_impl import NixlTransferChannelContext

            _context = NixlTransferChannelContext(
                l1_memory_desc=l1_memory_desc,
                listen_url=listen_url,
                advertise_url=advertise_url,
                backends=kwargs.get("backends"),
            )
        else:
            raise ValueError(
                f"Unsupported transfer_channel_type: {transfer_channel_type!r}"
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
