# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

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
from lmcache.v1.distributed.transfer_channel.factory import (
    create_transfer_channel_context,
)

# Import the implementations so they self-register their factories.
import lmcache.v1.distributed.transfer_channel.impl  # noqa: F401

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.multiprocess.config import P2PConfig

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
    transfer_channel_type: "str | P2PConfig",
    l1_memory_desc: "L1MemoryDesc",
    listen_url: str | None = None,
    advertise_url: str | None = None,
    **kwargs,
) -> TransferChannelContext:
    """Create the global transfer channel context.

    Args:
        transfer_channel_type: Registered implementation name, such as
            ``"nixl"`` or ``"verbs"``, or the P2P configuration from which
            the implementation settings are derived.
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's singleton server binds to.
            Required when ``transfer_channel_type`` is a string.
        advertise_url: ``host:port`` this peer advertises as its identity (the
            key peers store its reverse client under). Required when
            ``transfer_channel_type`` is a string.
        **kwargs: Implementation-specific keyword arguments forwarded to the
            factory.

    Returns:
        The created context (also retrievable via ``get_transfer_channel_context``).

    Raises:
        RuntimeError: If a context has already been initialized.
        TypeError: If the string-based form omits either URL.
        ValueError: If no factory is registered for ``transfer_channel_type``.
    """
    if isinstance(transfer_channel_type, str):
        channel_type = transfer_channel_type
        if listen_url is None or advertise_url is None:
            raise TypeError(
                "listen_url and advertise_url are required when initializing "
                "a transfer channel by type"
            )
    else:
        p2p_config = transfer_channel_type
        channel_type = p2p_config.transfer_engine
        listen_url = p2p_config.effective_listen_url
        advertise_url = p2p_config.advertise_url
        if channel_type == "verbs":
            kwargs.update(
                device_name=p2p_config.rdma_device,
                port_num=p2p_config.rdma_port,
                gid_index=p2p_config.rdma_gid_index,
                gid_indices=p2p_config.rdma_gid_indices,
                queue_depth=p2p_config.rdma_queue_depth,
                handshake_timeout_ms=p2p_config.rdma_handshake_timeout_ms,
            )

    global _context
    with _context_lock:
        if _context is not None:
            raise RuntimeError(
                "Transfer channel context already initialized; call "
                "delete_transfer_channel_context() first."
            )
        _context = create_transfer_channel_context(
            transfer_channel_type=channel_type,
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
