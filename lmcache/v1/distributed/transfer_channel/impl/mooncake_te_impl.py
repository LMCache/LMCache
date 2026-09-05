# SPDX-License-Identifier: Apache-2.0
"""Mooncake Transfer Engine implementation of the transfer channel abstraction."""

# Standard
from typing import Union
import os
import threading

# Third Party
from mooncake.engine import TransferEngine
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
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
    register_transfer_channel_factory,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

logger = init_logger(__name__)

# Timeout for each blocking recv during the client handshake. Without it a
# misconfigured/unreachable server url would block the connecting thread
# forever. Hard-coded for now.
_HANDSHAKE_TIMEOUT_MS = 60_000

INVALID_BATCH_ID = (1 << 64) - 1


############################################################
# Helper functions
############################################################
def _parse_url(url: str) -> tuple[str, int]:
    """Parse a ``host:port`` (optionally ``tcp://host:port``) into (host, port)."""
    stripped = url.split("://", 1)[-1]
    host, _, port = stripped.rpartition(":")
    if not host or not port:
        raise ValueError(f"Invalid transfer channel url: {url!r} (expected host:port)")
    return host, int(port)


def collect_env_var() -> tuple[str, str]:
    """Collect Mooncake Transfer Engine environment variables.
        By default, the protocol is ``tcp`` and the device ID is empty (CPU).
    Returns:
        tuple[str, str]: A tuple containing the protocol and device ID.
    """
    mc_te_protocol = os.getenv("MC_TE_PROTOCOL", "tcp")
    mc_te_device_id = os.getenv("MC_TE_DEVICE", "")
    return mc_te_protocol, mc_te_device_id


############################################################
# Handshake messages (msgspec, tagged union)
############################################################
class HandshakeMsgBase(msgspec.Struct, tag=True):
    pass


class InitReq(HandshakeMsgBase):
    advertise_url: str
    session_id: str
    buffer_base_ptr: int


class InitResp(HandshakeMsgBase):
    advertise_url: str
    session_id: str
    buffer_base_ptr: int


HandshakeMsg = Union[InitReq, InitResp]


############################################################
# Client
############################################################
class MooncakeTeTransferChannelClient(TransferChannelClient):
    def __init__(
        self,
        context: "MooncakeTeTransferChannelContext",
        remote_session_id: str,
        remote_buffer_ptr: int,
    ):
        self._ctx = context
        self._remote_session_id = remote_session_id
        self._remote_buffer_ptr = remote_buffer_ptr

        self._task_counter = 0
        # task_id -> (xfer_handle, remote_addresses)
        self._tasks: dict[int, tuple] = {}
        self._lock = threading.Lock()

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit a read transfer from the remote addresses to the local addresses.

        Args:
            local_addresses: The local addresses to read into.
            remote_addresses: The remote addresses to read from.

        Returns:
            A unique task ID for this transfer.
        """
        if len(local_addresses) != len(remote_addresses):
            raise ValueError(
                "local_addresses and remote_addresses must have equal length "
                f"({len(local_addresses)} != {len(remote_addresses)})"
            )
        local_ptrs = []
        remote_ptrs = []
        data_lengths = []
        for a in zip(local_addresses, remote_addresses, strict=True):
            local_ptrs.append(a[0].offset + self._ctx.l1_memory_desc.ptr)
            remote_ptrs.append(a[1].offset + self._remote_buffer_ptr)
            if a[0].size != a[1].size:
                raise ValueError("local and remote sizes must match")
            data_lengths.append(a[0].size)
        batch_id = self._ctx.mooncake_te_engine.batch_transfer_async_read(
            self._remote_session_id,
            local_ptrs,
            remote_ptrs,
            data_lengths,
        )

        if batch_id == 0 or batch_id == INVALID_BATCH_ID:
            raise RuntimeError(
                "Failed to submit async read. "
                f"Peer {self._remote_session_id} is probably not running, "
                "not reachable, or the remote metadata is invalid."
            )

        if batch_id < 0 or batch_id >= (1 << 63):
            raise RuntimeError(f"Invalid batch ID returned: {batch_id}")

        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (batch_id, list(remote_addresses))
        return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        """
        Query the status of a previously submitted read transfer.

        Args:
            task_id: The ID of the transfer task to query.

        Returns:
            A TransferChannelReadResult indicating whether the transfer is finished
            and which objects succeeded (if finished).
        """
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown read task id: {task_id}")
            batch_id, remote_addresses = self._tasks[task_id]

        status = self._ctx.mooncake_te_engine.transfer_check_status(batch_id)
        if status == 0:
            return TransferChannelReadResult(finished=False, succeeded_mask=[])

        # Transfer still in progress, continue polling
        with self._lock:
            self._tasks.pop(task_id, None)
        if status == 1:
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[True] * len(remote_addresses)
            )
        elif status == -1:
            logger.error(
                f"Transfer failed {self._remote_session_id} batch_id {batch_id}"
            )
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[False] * len(remote_addresses)
            )
        elif status == -2:
            logger.error(
                f"Transfer timed out {self._remote_session_id} batch_id {batch_id}"
            )
            return TransferChannelReadResult(
                finished=True, succeeded_mask=[False] * len(remote_addresses)
            )

        raise RuntimeError(f"unexpected mooncake transfer engine status: {status}")

    def close(self) -> None:
        """
        Close the transfer channel, releasing any pending tasks and the remote handle.

        Note:
            Mooncake Transfer Engine does not need anything when we close the client
            because reads do not create a remote handle that needs to be released.
        """
        pass


############################################################
# Server
############################################################
class MooncakeTeTransferChannelServer(TransferChannelServer):
    """
    Note:
        The server uses a ZMQ REP socket for the metadata handshake.
        It's not using LMCache MQ because we think that's an overkill
        and we don't want to register the transfer-channel specific
        functions into the global LMCache MQ.
    """

    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
        context: "MooncakeTeTransferChannelContext",
    ) -> None:
        self._ctx = context
        self._listen_url = listen_url
        self._advertise_url = advertise_url
        self._l1_memory_desc = l1_memory_desc

        self._running = True
        self._socket = self._ctx.zmq_context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        host, port = _parse_url(listen_url)
        self._socket.bind(f"tcp://{host}:{port}")

        self._thread = threading.Thread(
            target=self._serve_loop, name="tc-mooncake_te-server", daemon=True
        )
        self._thread.start()

    def _serve_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=1000))  # ms
                if self._socket not in events:
                    continue
                req_bytes = self._socket.recv()
                req = msgspec.msgpack.decode(req_bytes, type=HandshakeMsg)
                resp = self._handle_msg(req)
                self._socket.send(msgspec.msgpack.encode(resp))
            except Exception:  # noqa: BLE001
                if self._running:
                    logger.exception("Error in transfer channel server loop")

    def _handle_msg(self, req: HandshakeMsg) -> HandshakeMsg:
        if isinstance(req, InitReq):
            # Learn the connecting peer's agent (idempotent on repeat).
            logger.info(
                f"initialized transfer channel server with mooncake transfer engine "
                f"{self._ctx.advertise_host}:{self._ctx.mooncake_te_port}"
            )
            self._ctx.register_client(
                key=req.advertise_url,
                client=MooncakeTeTransferChannelClient(
                    context=self._ctx,
                    remote_session_id=req.session_id,
                    remote_buffer_ptr=req.buffer_base_ptr,
                ),
            )
            return InitResp(
                advertise_url=self._ctx.advertise_url,
                buffer_base_ptr=self._ctx.l1_memory_desc.ptr,
                session_id=f"{self._ctx.advertise_host}:{self._ctx.mooncake_te_port}",
            )
        else:
            raise ValueError(f"Unexpected handshake message: {type(req)}")

    def close(self) -> None:
        """
        Close the transfer channel server, stopping the serve loop and
        closing the socket.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._socket.close(linger=0)


############################################################
# Context
############################################################
class MooncakeTeTransferChannelContext(TransferChannelContext):
    """Owns the single mooncake_te agent, the registered L1 buffer, and support
    the address translation.
    """

    def __init__(
        self,
        l1_memory_desc: L1MemoryDesc,
        listen_url: str,
        advertise_url: str,
    ) -> None:
        """
        Creates the transfer channel context using mooncake_te.

        Args:
            l1_memory_desc: The description of the local L1 memory buffer to register.
            listen_url: The URL to listen on for incoming connections.
            advertise_url: The URL to advertise to peers for them to connect to us.
        """
        self._l1_memory_desc = l1_memory_desc
        self.listen_url = listen_url
        self.advertise_url = advertise_url
        host, port = _parse_url(advertise_url)
        self._advertise_host = host
        host, port = _parse_url(listen_url)
        self._listen_host = host
        self._mooncake_te_engine = TransferEngine()
        self._protocol, self._device_id = collect_env_var()
        ret = self._mooncake_te_engine.initialize(
            self._listen_host,
            "P2PHANDSHAKE",
            self._protocol,
            self._device_id,
        )
        if ret != 0:
            raise RuntimeError(f"Failed to initialize mooncake_te engine, ret={ret}")
        logger.info(
            f"mooncake_te engine created, mooncake transfer engine port "
            f"{self._mooncake_te_engine.get_rpc_port()}, protocol: "
            f"{self._protocol}, device: {self._device_id}"
        )
        self._mooncake_te_port = self._mooncake_te_engine.get_rpc_port()
        # Register the whole L1 buffer once (CPU/DRAM, fixed mooncake_te dev_id=0).
        ptr, size = l1_memory_desc.ptr, l1_memory_desc.size
        self._mooncake_te_engine.register_memory(ptr, size)

        # Build + prep a page-granular local xfer dlist over the whole buffer.
        self.zmq_context = zmq.Context.instance()

        # Clients keyed by the peer's advertised url. Populated either actively
        # (we dialed the peer) or reactively (the peer connected to our server).
        self._clients: dict[str, MooncakeTeTransferChannelClient] = {}
        self._lock = threading.Lock()

        # Exactly one server per context, bound eagerly at construction.
        self._server = MooncakeTeTransferChannelServer(
            listen_url=listen_url,
            advertise_url=advertise_url,
            l1_memory_desc=l1_memory_desc,
            context=self,
        )

    @property
    def advertise_host(self) -> str:
        """Return the host advertised to transfer-channel peers."""
        return self._advertise_host

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        """Return the registered L1 memory-region description."""
        return self._l1_memory_desc

    @property
    def mooncake_te_engine(self) -> TransferEngine:
        """Return the Mooncake Transfer Engine owned by this context."""
        return self._mooncake_te_engine

    @property
    def mooncake_te_port(self) -> int:
        """Return the Mooncake Transfer Engine RPC port."""
        return self._mooncake_te_port

    ############################################################
    # Address translation
    ############################################################
    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        """
        Validate the given LMCache addresses (offset, size) against the
        registered L1 memory region and convert it to TransferChannelAddress

        Args:
            lmcache_addresses: List of (offset, size) tuples representing the LMCache
                addresses.

        Returns:
            A list of TransferChannelAddress corresponding to the given LMCache
            addresses.
        """
        size = self._l1_memory_desc.size
        out = []
        for offset, obj_size in lmcache_addresses:
            if offset < 0 or offset + obj_size > size:
                raise ValueError(
                    f"Object [{offset:#x}, {offset + obj_size:#x}) is outside the "
                    f"registered L1 region [0x0, {size:#x})"
                )
            out.append(TransferChannelAddress(offset=offset, size=obj_size))
        return out

    ############################################################
    # Server / client management
    ############################################################
    def get_transfer_channel_server(self) -> MooncakeTeTransferChannelServer:
        return self._server

    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> MooncakeTeTransferChannelClient:
        with self._lock:
            client = self._clients.get(peer_advertise_url)
            if client is not None:
                logger.error("found client")
                return client

        # Not yet known: actively connect to the peer and perform the handshake.
        client = self._connect(peer_advertise_url)
        return self.register_client(peer_advertise_url, client)

    def get_num_connected_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def register_client(
        self, key: str, client: MooncakeTeTransferChannelClient
    ) -> MooncakeTeTransferChannelClient:
        """
        Register a client for the given key (peer advertise url).

        A client already registered for ``key`` is kept; ``client`` is then a
        redundant duplicate (the active connect and the peer's inbound
        connection can race) and is closed.

        Args:
            key: The peer advertise url to register the client under.
            client: A freshly created MooncakeTeTransferChannelClient for the peer.

        Returns:
            The canonical client for ``key``: the previously registered one if
            present, otherwise ``client``.
        """
        with self._lock:
            existing = self._clients.get(key)
            if existing is None:
                self._clients[key] = client
                return client
            if existing is client:
                return client

        logger.debug("Reusing existing transfer channel client for %s", key)
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error closing duplicate transfer channel client for %s", key
            )
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Discard the client for ``peer_advertise_url`` and free its resources.

        Call this when reads from the peer are no longer needed (e.g. the
        owning L2 adapter is being removed). Any task ids previously returned by
        that client become invalid. A later ``get_transfer_channel_client`` for
        the same peer returns a fresh client. The peer is not notified.

        Calling this for a peer with no current client does nothing.

        Args:
            peer_advertise_url: The ``host:port`` of the peer, as passed to
                ``get_transfer_channel_client``.
        """
        with self._lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception(
                "Error closing transfer channel client for %s", peer_advertise_url
            )

    ############################################################
    # Cleanup
    ############################################################
    def close(self) -> None:
        with self._lock:
            server = self._server
            clients = list(self._clients.values())
            self._clients.clear()

        if server is not None:
            server.close()
        for client in clients:
            client.close()

        ret = self._mooncake_te_engine.unregister_memory(self._l1_memory_desc.ptr)
        if ret != 0:
            logger.warning("Failed to unregister the Mooncake memory buffer")
        else:
            logger.info("Unregistered memory buffer.")

    ############################################################
    # Helper functions
    ############################################################
    def _recv_handshake(self, socket: "zmq.Socket", server_url: str) -> bytes:
        """Receive one handshake reply, mapping a timeout to a clear error.

        Args:
            socket: The REQ socket awaiting the server's reply.
            server_url: The peer url being dialed (for the error message).

        Returns:
            The raw reply bytes.

        Raises:
            TimeoutError: If no reply arrives within the handshake timeout
                (e.g. the url is wrong/unreachable or the port is blocked).
        """
        try:
            return socket.recv()
        except zmq.Again as err:
            raise LMCacheTimeoutError(
                f"Timed out after {_HANDSHAKE_TIMEOUT_MS / 1000:.0f}s waiting for "
                f"a transfer-channel handshake reply from {server_url!r}. Check "
                f"that the peer is running and that the url/port is correct and "
                f"reachable."
            ) from err

    def _connect(self, server_url: str) -> MooncakeTeTransferChannelClient:
        socket = self.zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)
        host, port = _parse_url(server_url)
        socket.connect(f"tcp://{host}:{port}")
        try:
            # Stage 1: exchange agent metadata.
            session_id = f"{self._advertise_host}:{self._mooncake_te_port}"
            logger.info(
                "initiate connection to server %s with local session id %s",
                server_url,
                session_id,
            )
            socket.send(
                msgspec.msgpack.encode(
                    InitReq(
                        advertise_url=self.advertise_url,
                        session_id=session_id,
                        buffer_base_ptr=self._l1_memory_desc.ptr,
                    )
                )
            )
            init_resp = msgspec.msgpack.decode(
                self._recv_handshake(socket, server_url), type=HandshakeMsg
            )
            assert isinstance(init_resp, InitResp)
            remote_session_id = init_resp.session_id
        finally:
            socket.close(linger=0)

        return MooncakeTeTransferChannelClient(
            context=self,
            remote_session_id=remote_session_id,
            remote_buffer_ptr=init_resp.buffer_base_ptr,
        )


############################################################
# Factory registration
############################################################
def create_mooncake_te_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> MooncakeTeTransferChannelContext:
    """Create a ``MooncakeTeTransferChannelContext``.

    Args:
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity.

    Returns:
        A new ``MooncakeTeTransferChannelContext`` instance.
    """
    # logger.info(l1_memory_desc, listen_url, advertise_url, kwargs.get("backends"))
    return MooncakeTeTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
    )


register_transfer_channel_factory(
    "mooncake_te", create_mooncake_te_transfer_channel_context
)
