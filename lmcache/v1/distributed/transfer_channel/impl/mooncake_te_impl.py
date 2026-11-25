# SPDX-License-Identifier: Apache-2.0
"""Mooncake Transfer Engine implementation of the transfer channel abstraction."""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
import os
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.common.errors import LMCacheTimeoutError
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)

if TYPE_CHECKING:
    # Third Party
    from mooncake.engine import TransferEngine

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


def _supports_whole_batch_status_check(engine: "TransferEngine") -> bool:
    """Return whether ``engine`` provides non-blocking whole-batch status checks.

    Mooncake introduced ``batch_transfer_check_status(batch_ids)`` alongside the
    whole-batch ``transfer_check_status`` fix. Older bindings only check task 0
    in ``transfer_check_status`` and do not expose ``batch_transfer_check_status``.
    """
    return callable(getattr(engine, "batch_transfer_check_status", None))


def _best_effort_free_batches(engine: "TransferEngine", batch_ids: list[int]) -> None:
    """Release ``batch_ids`` without blocking when ``free_batch_id`` is available."""
    free_batch_id = getattr(engine, "free_batch_id", None)
    if not callable(free_batch_id):
        return
    for batch_id in batch_ids:
        try:
            free_batch_id(batch_id)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.warning(
                "Failed to free Mooncake batch_id %s during cleanup",
                batch_id,
                exc_info=True,
            )


@dataclass
class _PendingTransferTask:
    """In-flight Mooncake batch handle(s) for a single transfer task.

    When the engine supports whole-batch status checks, one batch covers all
    objects (``pending_batches`` maps ``batch_id -> None``). On legacy engines
    where ``transfer_check_status`` inspects only task 0, multi-buffer transfers
    are split into single-buffer batches (``batch_id -> object_index``).
    """

    object_count: int
    pending_batches: dict[int, int | None]
    succeeded_mask: list[bool]


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
        self._tasks: dict[int, _PendingTransferTask] = {}
        self._lock = threading.Lock()
        self._warned_legacy_split = False

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
        local_region: int | None = None,
    ) -> int:
        """Submit a read transfer from the remote addresses to the local addresses.

        Args:
            local_addresses: The local addresses to read into.
            remote_addresses: The remote addresses to read from.
            local_region: Must be None (L1).

        Returns:
            A unique task ID for this transfer.
        """
        if local_region is not None:
            raise ValueError("mooncake_te only reads into the L1 buffer")
        return self._submit_transfer("READ", local_addresses, remote_addresses)

    def submit_write(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit a write transfer from the local addresses to the remote addresses.

        Args:
            local_addresses: The local addresses to send from.
            remote_addresses: The remote addresses to write into.

        Returns:
            A unique task ID for this transfer.
        """
        return self._submit_transfer("WRITE", local_addresses, remote_addresses)

    def _apply_polled_statuses(
        self,
        task: _PendingTransferTask,
        polled: list[tuple[int, int]],
    ) -> int | None:
        """Record terminal batch statuses on ``task``; return unexpected code if any.

        Must be called while holding ``self._lock``.
        """
        unexpected_status: int | None = None
        for batch_id, status in polled:
            if batch_id not in task.pending_batches:
                continue
            if status == 0:
                continue
            obj_idx = task.pending_batches.pop(batch_id)
            if status == 1:
                if obj_idx is None:
                    task.succeeded_mask = [True] * task.object_count
                else:
                    task.succeeded_mask[obj_idx] = True
            elif status == -1:
                logger.error(
                    "Transfer failed %s batch_id %s",
                    self._remote_session_id,
                    batch_id,
                )
            elif status == -2:
                logger.error(
                    "Transfer timed out %s batch_id %s",
                    self._remote_session_id,
                    batch_id,
                )
            elif unexpected_status is None:
                unexpected_status = status
        return unexpected_status

    def query_status(self, task_id: int) -> TransferChannelTransferResult:
        """Query the status of a previously submitted read or write.

        Args:
            task_id: The ID of the transfer task to query.

        Returns:
            A TransferChannelTransferResult indicating whether the transfer is
            finished and which objects succeeded (if finished).

        Raises:
            KeyError: ``task_id`` is unknown or was already reported finished.
            RuntimeError: The engine reported an unknown status code.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown transfer task id: {task_id}")
            if not task.pending_batches:
                self._tasks.pop(task_id, None)
                return TransferChannelTransferResult(
                    finished=True, succeeded_mask=list(task.succeeded_mask)
                )
            pending_ids = list(task.pending_batches.keys())

        engine = self._ctx.mooncake_te_engine
        if _supports_whole_batch_status_check(engine):
            statuses = engine.batch_transfer_check_status(pending_ids)
            if len(statuses) != len(pending_ids):
                with self._lock:
                    self._tasks.pop(task_id, None)
                raise RuntimeError(
                    "batch_transfer_check_status returned "
                    f"{len(statuses)} statuses, expected {len(pending_ids)}"
                )
            polled = list(zip(pending_ids, statuses, strict=True))
        else:
            polled = []
            try:
                for batch_id in pending_ids:
                    polled.append((batch_id, engine.transfer_check_status(batch_id)))
            except Exception:
                # Never re-poll batch handles that already returned a terminal
                # status (Mooncake frees handles immediately on terminal check).
                if polled:
                    with self._lock:
                        if self._tasks.get(task_id) is task:
                            self._apply_polled_statuses(task, polled)
                raise

        with self._lock:
            if self._tasks.get(task_id) is not task:
                raise KeyError(f"Unknown transfer task id: {task_id}")
            unexpected_status = self._apply_polled_statuses(task, polled)
            if unexpected_status is not None:
                self._tasks.pop(task_id, None)
                raise RuntimeError(
                    f"unexpected mooncake transfer engine status: {unexpected_status}"
                )
            if task.pending_batches:
                return TransferChannelTransferResult(finished=False, succeeded_mask=[])
            self._tasks.pop(task_id, None)
            return TransferChannelTransferResult(
                finished=True, succeeded_mask=list(task.succeeded_mask)
            )

    def close(self) -> None:
        """
        Close the transfer channel, releasing any pending tasks and the remote handle.
        """
        with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        batch_ids = [
            batch_id for task in tasks for batch_id in task.pending_batches.keys()
        ]
        if batch_ids:
            _best_effort_free_batches(self._ctx.mooncake_te_engine, batch_ids)

    def _validate_batch_id(self, batch_id: int, operation: str) -> None:
        if batch_id == 0 or batch_id == INVALID_BATCH_ID:
            raise RuntimeError(
                f"Failed to submit async {operation.lower()}. "
                f"Peer {self._remote_session_id} is probably not running, "
                "not reachable, or the remote metadata is invalid."
            )
        if batch_id < 0 or batch_id >= (1 << 63):
            raise RuntimeError(f"Invalid batch ID returned: {batch_id}")

    def _submit_transfer(
        self,
        operation: str,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit one batched transfer and return its task id.

        Args:
            operation: ``"READ"`` (remote to local) or ``"WRITE"`` (local to
                remote).
            local_addresses: The local addresses of the transfer.
            remote_addresses: The remote addresses of the transfer.

        Raises:
            ValueError: The two address lists differ in length, or an object
                has different sizes on the two sides.
            RuntimeError: The engine refused the batch.
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

        object_count = len(remote_addresses)
        if object_count == 0:
            task = _PendingTransferTask(
                object_count=0,
                pending_batches={},
                succeeded_mask=[],
            )
            with self._lock:
                task_id = self._task_counter
                self._task_counter += 1
                self._tasks[task_id] = task
            return task_id

        engine = self._ctx.mooncake_te_engine
        submit = (
            engine.batch_transfer_async_write
            if operation == "WRITE"
            else engine.batch_transfer_async_read
        )

        pending_batches: dict[int, int | None] = {}
        if _supports_whole_batch_status_check(engine) or object_count == 1:
            batch_id = submit(
                self._remote_session_id,
                local_ptrs,
                remote_ptrs,
                data_lengths,
            )
            self._validate_batch_id(batch_id, operation)
            pending_batches[batch_id] = None
        else:
            # Legacy Mooncake TE bindings check only task index 0 in
            # transfer_check_status(batch_id). Split multi-buffer transfers into
            # one single-buffer batch per object so completion polling tracks
            # every buffer non-blockingly.
            if not self._warned_legacy_split:
                self._warned_legacy_split = True
                logger.warning(
                    "Mooncake TransferEngine lacks batch_transfer_check_status; "
                    "splitting multi-buffer transfers into per-buffer batches "
                    "to ensure whole-transfer completion for peer %s",
                    self._remote_session_id,
                )
            try:
                for idx, (local_ptr, remote_ptr, length) in enumerate(
                    zip(local_ptrs, remote_ptrs, data_lengths, strict=True)
                ):
                    batch_id = submit(
                        self._remote_session_id,
                        [local_ptr],
                        [remote_ptr],
                        [length],
                    )
                    self._validate_batch_id(batch_id, operation)
                    if batch_id in pending_batches:
                        raise RuntimeError(f"Duplicate batch ID returned: {batch_id}")
                    pending_batches[batch_id] = idx
            except Exception:
                if pending_batches:
                    _best_effort_free_batches(engine, list(pending_batches.keys()))
                raise

        task = _PendingTransferTask(
            object_count=object_count,
            pending_batches=pending_batches,
            succeeded_mask=[False] * object_count,
        )
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = task
        return task_id


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
                "initialized transfer channel server with mooncake transfer engine "
                "%s:%s",
                self._ctx.advertise_host,
                self._ctx.mooncake_te_port,
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

        Import the optional SDK only when creating this context, so discovering
        transfer-channel factories does not load Mooncake's native libraries.

        Args:
            l1_memory_desc: The description of the local L1 memory buffer to register.
            listen_url: The URL to listen on for incoming connections.
            advertise_url: The URL to advertise to peers for them to connect to us.

        Raises:
            ImportError: If the Mooncake SDK or one of its dependencies cannot
                be imported.
        """
        # Third Party
        from mooncake.engine import TransferEngine

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
            "mooncake_te engine created, mooncake transfer engine port "
            "%s, protocol: %s, device: %s",
            self._mooncake_te_engine.get_rpc_port(),
            self._protocol,
            self._device_id,
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
    def mooncake_te_engine(self) -> "TransferEngine":
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
        region: int | None = None,
    ) -> list[TransferChannelAddress]:
        """
        Validate the given LMCache addresses (offset, size) against the
        registered L1 memory region and convert it to TransferChannelAddress

        Args:
            lmcache_addresses: List of (offset, size) tuples representing the LMCache
                addresses.
            region: Must be None (L1).

        Returns:
            A list of TransferChannelAddress corresponding to the given LMCache
            addresses.
        """
        if region is not None:
            raise ValueError("mooncake_te only reads into the L1 buffer")
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
