# SPDX-License-Identifier: Apache-2.0
"""Abstract base classes for the transfer channel abstraction.

Main classes:
* ``TransferChannelContext`` -- the global singleton that owns the underlying
  transfer engine, maintains the registered L1 memory, and translates L1 addresses
  into transfer-channel-specific addresses.
* ``TransferChannelServer`` -- listens for client connections and exchanges the
  metadata during the handshake.
* ``TransferChannelClient`` -- performs the read operations against a peer's
  memory. Only reads are supported for P2P for now.
"""

# Standard
import abc

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)

L1_REGION = 0
"""Region id of the L1 buffer every context registers at construction."""


class TransferChannelServer(metaclass=abc.ABCMeta):
    """Listens for incoming client connections and exchanges transfer metadata."""

    @abc.abstractmethod
    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        """Creates the server listening on ``listen_url``.

        Args:
            listen_url: The ``host:port`` to bind and listen on (e.g.
                ``0.0.0.0:7600``).
            advertise_url: The ``host:port`` this peer is reachable at and
                announces in its outgoing handshakes.
            l1_memory_desc: Describes the L1 memory region to register.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Free the resources created by the server."""
        raise NotImplementedError


class TransferChannelClient(metaclass=abc.ABCMeta):
    """Performs read operations from a remote peer's memory."""

    @abc.abstractmethod
    def __init__(self, transfer_channel_server_url: str) -> None:
        """Connect to the server at ``transfer_channel_server_url``."""
        raise NotImplementedError

    @abc.abstractmethod
    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
        *,
        local_region: int = L1_REGION,
    ) -> int:
        """Read data from the remote addresses into the local addresses.

        Args:
            local_addresses: addresses in the local region ``local_region``.
            remote_addresses: addresses in the remote peer's L1 buffer.
                Must have the same length as ``local_addresses``.
            local_region: The local region the destinations belong to: the
                L1 buffer by default, or a region returned by
                ``TransferChannelContext.register_device_region``.

        Returns:
            A task id, to be passed to ``query_read_status``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """Query the status of a previously submitted read.

        Args:
            task_id: The task id returned by ``submit_read``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Terminate in-flight requests, deregister from the server, free resources."""
        raise NotImplementedError


class TransferChannelContext(metaclass=abc.ABCMeta):
    """Global singleton that owns the server singleton, and all clients."""

    @abc.abstractmethod
    def get_transfer_channel_server(self) -> TransferChannelServer:
        """Return the transfer channel server singleton."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> TransferChannelClient:
        """Get or create a client that is connected to the peer identified by
        ``peer_advertise_url``.

        Args:
            peer_advertise_url: The ``host:port`` url that can connects to the
                peer.

        Notes:
            For some bi-directional transport libraries (like NIXL), the client
            might be created passively when the current context (server) is
            connected by the remote peer.
            This behavior is per implementation and transparent to the caller.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Discard the client for ``peer_advertise_url`` and free its resources.

        Call this when reads from the peer are no longer needed (e.g. the
        owning L2 adapter is being removed). Any task ids previously returned by
        that client become invalid. A later ``get_transfer_channel_client`` for
        the same peer returns a fresh client.

        Calling this for a peer with no current client does nothing.

        Args:
            peer_advertise_url: The ``host:port`` of the peer, as passed to
                ``get_transfer_channel_client``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        """Translate ``(offset, size)`` L1 objects into the transfer-channel-specific
        addresses.

        Args:
            lmcache_addresses: A list of tuples of the form ``(offset, size)``, where
                ``offset`` is the offset into the L1 buffer, and ``size`` is the
                size of the region.

        Returns:
            A list of transfer-channel-specific addresses that can be used in the
                client read operations. Must have the same length as
                ``lmcache_addresses``.
        """
        raise NotImplementedError

    def supports_device_regions(self) -> bool:
        """Return whether :meth:`register_device_region` is implemented.

        Lets orchestration refuse a deployment that needs reads into device
        memory before any buffer exists to register.
        """
        return False

    def register_device_region(self, ptr: int, size: int, device_index: int) -> int:
        """Register a device-memory buffer as an additional local read target.

        Peers never read from it; it only receives reads this context submits
        with ``submit_read(..., local_region=<id>)``, so nothing is exchanged
        with peers. Offsets within the region follow the L1 buffer's
        alignment contract. Implementations that cannot read into device
        memory keep this default.

        Args:
            ptr: Device address of the buffer.
            size: Size in bytes; a multiple of the L1 alignment.
            device_index: Ordinal of the device holding the buffer.

        Returns:
            The region id to pass as ``local_region``.

        Raises:
            ValueError: If ``ptr`` or ``size`` is not aligned.
            NotImplementedError: If the transport cannot read into device
                memory.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot read into device memory; GPU-direct "
            "borrowing needs a transfer engine that registers device regions"
        )

    def unregister_device_region(self, region_id: int) -> None:  # noqa: B027
        """Release a region from ``register_device_region``.

        Reads into the region must have finished. Unknown ids are ignored.

        Args:
            region_id: Id returned by ``register_device_region``.
        """

    @abc.abstractmethod
    def get_num_connected_clients(self) -> int:
        """Return the number of currently connected clients."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Stop and clean up all servers and clients; free context resources."""
        raise NotImplementedError
