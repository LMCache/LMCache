# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import threading
import weakref

# First Party
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
    "acquire_transfer_channel_client",
    "release_transfer_channel_client",
]

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


# Peer clients shared by several adapters of the same peer (a removal can be
# deferred while a re-added adapter already uses the client): users per peer
# url of each context, guarded by ``_clients_lock``. A separate lock, as
# acquiring may connect to the peer.
_client_users: "weakref.WeakKeyDictionary[TransferChannelContext, dict[str, int]]" = (
    weakref.WeakKeyDictionary()
)
_clients_lock = threading.Lock()


def acquire_transfer_channel_client(
    context: TransferChannelContext, peer_advertise_url: str
) -> TransferChannelClient:
    """Get the client for a peer and count the caller as one of its users.

    Every call must be paired with :func:`release_transfer_channel_client`.

    Args:
        context: The transfer channel context.
        peer_advertise_url: The peer's ``host:port``.

    Returns:
        The context's client for the peer.
    """
    with _clients_lock:
        client = context.get_transfer_channel_client(peer_advertise_url)
        users = _client_users.setdefault(context, {})
        users[peer_advertise_url] = users.get(peer_advertise_url, 0) + 1
    return client


def release_transfer_channel_client(
    context: TransferChannelContext, peer_advertise_url: str
) -> None:
    """Drop one user of a peer's client; the last one removes the client.

    Releasing a peer with no users does nothing.

    Args:
        context: The transfer channel context.
        peer_advertise_url: The peer's ``host:port``.
    """
    with _clients_lock:
        users = _client_users.get(context, {})
        count = users.get(peer_advertise_url, 0)
        if count == 0:
            return
        if count > 1:
            users[peer_advertise_url] = count - 1
            return
        del users[peer_advertise_url]
        context.remove_transfer_channel_client(peer_advertise_url)
