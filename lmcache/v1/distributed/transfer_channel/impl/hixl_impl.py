# SPDX-License-Identifier: Apache-2.0
"""HIXL-backed implementation of the transfer channel abstraction."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Lock, Thread
from typing import TYPE_CHECKING
from uuid import uuid4
import ctypes
import json
import os
import re
import socket

# Third Party
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
    TransferChannelTransferResult,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

if TYPE_CHECKING:
    # Third Party
    from lmcache_ascend.hixl_npu_comms import (
        Hixl,
        TransferOp,
        TransferOpDesc,
        TransferStatus,
    )

logger = init_logger(__name__)

_ENV_PREFIX = "LMCACHE_HIXL_"
_HANDSHAKE_TIMEOUT_MS = 60_000
_DISCONNECT_TIMEOUT_MS = 1000
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


############################################################
# Helper functions
############################################################
def _parse_channel_url(url: str) -> tuple[str, int]:
    """Parse host:port or tcp://host:port, preserving IPv6 brackets.

    Raise ValueError for an invalid address or port; do not resolve the host.
    """
    match = re.fullmatch(r"(?:tcp://)?(\[[^\]]+\]|[^:/\s?#@]+):([0-9]+)", url)
    try:
        if match is None:
            raise ValueError("expected host:port or tcp://host:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid Channel address {url!r}: {exc}") from exc
    return host, number


def _validate_engine_id(engine_id: str) -> None:
    """Require IPv4:port or [IPv6]:port with a positive listening port.

    Raise ValueError for an invalid IP literal or port.
    """
    match = re.fullmatch(r"(\[[^\]]+\]|[^:]+):([0-9]+)", engine_id)
    try:
        if match is None:
            raise ValueError("expected IP:port or [IPv6]:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        else:
            socket.inet_pton(socket.AF_INET, host)
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid HIXL engine_id {engine_id!r}: {exc}") from exc


def _validate_metadata(metadata: HixlChannelMetadata) -> None:
    """Validate peer addresses and memory range; raise ValueError if invalid."""
    _parse_channel_url(metadata.advertise_url)
    _validate_engine_id(metadata.engine_id)
    _validate_memory_range(metadata.buffer_base_ptr, metadata.buffer_size)


def _read_int(suffix: str, default: str, minimum: int, maximum: int) -> int:
    """Read LMCACHE_HIXL_<suffix> as a decimal integer within inclusive bounds.

    Use default when unset; raise ValueError for an invalid value.
    """
    name = _ENV_PREFIX + suffix
    value = os.environ.get(name, default)
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a decimal integer, got {value!r}")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value!r}"
        )
    return number


def _read_config() -> _HixlConfig:
    """Read HIXL environment settings; raise ValueError if invalid."""
    engine_id = os.environ.get(_ENV_PREFIX + "ENGINE_ID", "")
    _validate_engine_id(engine_id)
    return _HixlConfig(
        engine_id=engine_id,
        device_id=_read_int("DEVICE_ID", "0", 0, _INT32_MAX),
        connect_timeout_ms=_read_int("CONNECT_TIMEOUT_MS", "10000", 1, _INT32_MAX),
        comm_listen_port=(
            _read_int("COMM_LISTEN_PORT", "", 1, 65535)
            if _ENV_PREFIX + "COMM_LISTEN_PORT" in os.environ
            else None
        ),
    )


def _validate_memory_range(ptr: int, size: int) -> None:
    """Require a positive Host range representable by native address types.

    Raise ValueError for invalid values or address overflow.
    """
    if (
        type(ptr) is not int
        or type(size) is not int
        or not 0 < ptr <= _UINTPTR_MAX
        or not 0 < size <= _SIZE_MAX
        or size > _UINTPTR_MAX - ptr
    ):
        raise ValueError(f"Invalid Host memory range: ptr={ptr!r}, size={size!r}")


def _validate_address_range(
    address: TransferChannelAddress, buffer_size: int, *, label: str
) -> None:
    """Require a positive object range within the Host pool.

    Raise ValueError with label if the offset or size is invalid.
    """
    offset, size = address.offset, address.size
    if type(offset) is not int or type(size) is not int:
        raise ValueError(f"{label}: offset and size must be integers, not bool")
    if offset < 0 or size <= 0:
        raise ValueError(
            f"{label}: require offset >= 0 and size > 0, got {offset}, {size}"
        )
    if offset > buffer_size or size > buffer_size - offset:
        raise ValueError(
            f"{label}: range offset={offset}, size={size} "
            f"exceeds buffer_size={buffer_size}"
        )


@dataclass(frozen=True)
class _HixlConfig:
    """Engine identity, logical device and connection settings for one Context.

    comm_listen_port optionally overrides the native communication-resource
    port; it is separate from the engine ID and the Channel handshake port.
    """

    engine_id: str
    device_id: int
    connect_timeout_ms: int
    comm_listen_port: int | None

    def initialize_options(self) -> dict[str, str]:
        """Return HIXL_CS/device-RoCE options with explicit connections.

        The registered L1 is used directly, without a HIXL staging pool.
        """
        comm: dict[str, str | int] = {
            "comm_resource_config.protocol_desc": "roce:device",
        }
        if self.comm_listen_port is not None:
            comm["comm_resource_config.listen_port"] = self.comm_listen_port
        return {
            "LocalCommRes": '{"version":"1.3"}',
            "BufferPool": "0:0",
            "AutoConnect": "0",
            "GlobalResourceConfig": json.dumps(comm),
        }


class HixlChannelMetadata(msgspec.Struct, frozen=True):
    """Registered Host range and peer identity; contains no native handles.

    Context admission validates the identity fields before creating a Client.
    The base and size describe the entire registered allocation in bytes.
    """

    advertise_url: str
    engine_id: str
    buffer_base_ptr: int
    buffer_size: int
    instance_id: str


############################################################
# Handshake messages (msgspec, tagged union)
############################################################
class HandshakeMsgBase(msgspec.Struct, tag=True):
    """Share tagged MessagePack encoding across HIXL handshake messages."""


class InitReq(HandshakeMsgBase):
    """Request peer admission using the sender's registered Host metadata."""

    metadata: HixlChannelMetadata


class InitResp(HandshakeMsgBase):
    """Return local metadata after reverse peer admission succeeds."""

    metadata: HixlChannelMetadata


HandshakeMsg = InitReq | InitResp


############################################################
# Client
############################################################
class HixlTransferChannelClient(TransferChannelClient):
    """Manage READ/WRITE tasks over a Context-owned peer connection.

    The caller keeps source data unchanged and destination ranges exclusively
    writable until native accesses end, including after a native error.
    """

    def __init__(
        self, context: HixlTransferChannelContext, peer_metadata: HixlChannelMetadata
    ) -> None:
        """Create a Client for an admitted peer.

        Args:
            context: Owner of the Engine, local pool registration and connections.
            peer_metadata: Validated peer metadata from the Context.

        Raises:
            RuntimeError: The Ascend native binding cannot be loaded.
        """
        self._ctx = context
        self._local_metadata = context.metadata
        self._peer_metadata = peer_metadata

        try:
            # Third Party
            from lmcache_ascend.hixl_npu_comms import (
                HixlError,
                TransferOp,
                TransferOpDesc,
                TransferStatus,
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HIXL requires compatible LMCache-Ascend native bindings with "
                "HixlError, TransferOp, TransferOpDesc and TransferStatus"
            ) from exc

        self._native_error = HixlError
        self._operation = TransferOp
        self._descriptor = TransferOpDesc
        self._status = TransferStatus
        self._lock = Lock()
        self._closed = False
        self._task_counter = 0
        # Client task_id -> (native req, object count for the result mask).
        self._tasks: dict[int, tuple[int, int]] = {}

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Read remote objects into local destinations.

        Args:
            local_addresses: Nonempty list of local destination offsets/sizes.
            remote_addresses: Matching remote source offsets/sizes.

        Returns:
            A task ID for query_status, shared with WRITE tasks.

        Raises:
            ValueError: Address ranges or batch dimensions are invalid.
            RuntimeError: The Client/Engine is closed, ACL setup fails or req is null.
            HixlError: Native submission fails; no task ID is registered.
        """
        return self._submit_transfer(
            self._operation.READ, local_addresses, remote_addresses
        )

    def submit_write(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Write local objects into caller-authorized remote destinations.

        This does not allocate or publish remote KV objects, or roll back writes.

        Args:
            local_addresses: Nonempty list of local source offsets/sizes.
            remote_addresses: Matching remote destination offsets/sizes.

        Returns:
            A task ID for query_status, shared with READ tasks.

        Raises:
            ValueError: Address ranges or batch dimensions are invalid.
            RuntimeError: The Client/Engine is closed, ACL setup fails or req is null.
            HixlError: Native submission fails; no task ID is registered.
        """
        return self._submit_transfer(
            self._operation.WRITE, local_addresses, remote_addresses
        )

    def query_status(self, task_id: int) -> TransferChannelTransferResult:
        """Poll a task and consume it on completion or native query failure.

        Args:
            task_id: ID returned by submit_read or submit_write.

        Returns:
            An unfinished result with an empty mask while waiting; otherwise a
            uniform success/failure mask. HIXL reports only whole-request status.

        Raises:
            KeyError: The task is unknown or has been consumed or closed.
            HixlError: Native query fails; the task is consumed.
            RuntimeError: ACL/Engine preparation fails (task retained), or an
                unsupported terminal status is returned (task consumed).
        """
        with self._lock:
            req, count = self._tasks[task_id]
            try:
                status = self._ctx.query_transfer(req)
            except self._native_error:
                # A native query error consumes the HIXL req. ACL preparation
                # errors propagate separately and leave this task available.
                del self._tasks[task_id]
                raise
            if status == self._status.WAITING:
                return TransferChannelTransferResult(finished=False, succeeded_mask=[])

            # HIXL removes every non-WAITING req from its index, even for an
            # unsupported status. Never query that native handle again.
            del self._tasks[task_id]
            if status == self._status.COMPLETED:
                return TransferChannelTransferResult(
                    finished=True, succeeded_mask=[True] * count
                )
            if status == self._status.FAILED:
                return TransferChannelTransferResult(
                    finished=True, succeeded_mask=[False] * count
                )
            raise RuntimeError(f"Unsupported HIXL transfer status: {status!r}")

    def close(self) -> None:
        """Close the Client and discard task records without cancelling transfers."""
        with self._lock:
            self._closed = True
            self._tasks.clear()

    def _submit_transfer(
        self,
        operation: TransferOp,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Build descriptors, submit the batch and register a Client task ID."""
        if len(local_addresses) != len(remote_addresses):
            raise ValueError("Local and remote address count must match")
        if not local_addresses:
            raise ValueError("Transfer address batch must not be empty")
        if len(local_addresses) > _UINT32_MAX:
            raise ValueError("Transfer descriptor count exceeds uint32 representation")

        descriptors = []
        for index, (local, remote) in enumerate(
            zip(local_addresses, remote_addresses, strict=True)
        ):
            _validate_address_range(
                local,
                self._local_metadata.buffer_size,
                label=f"local_addresses[{index}]",
            )
            _validate_address_range(
                remote,
                self._peer_metadata.buffer_size,
                label=f"remote_addresses[{index}]",
            )
            if local.size != remote.size:
                raise ValueError(
                    f"pair[{index}]: local size {local.size} "
                    f"!= remote size {remote.size}"
                )
            descriptors.append(
                self._descriptor(
                    local_addr=self._local_metadata.buffer_base_ptr + local.offset,
                    remote_addr=self._peer_metadata.buffer_base_ptr + remote.offset,
                    len=local.size,
                )
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("HIXL Client is closed")
            count = len(descriptors)
            req = self._ctx.submit_transfer(
                self._peer_metadata.engine_id, operation, descriptors
            )
            if req == 0:
                raise RuntimeError("HIXL TransferAsync returned a null request")
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (req, count)
            return task_id


############################################################
# Server
############################################################
class HixlTransferChannelServer(TransferChannelServer):
    """Exchange peer metadata over a ZMQ REP socket.

    A failed request stops the service loop.
    """

    def __init__(self, context: HixlTransferChannelContext, listen_url: str) -> None:
        """Bind the metadata listener and start its service thread.

        Args:
            context: Owner with registered local metadata and peer connections.
            listen_url: Channel address to bind, independent of the HIXL Engine.

        Raises:
            ValueError: The address is invalid.
            RuntimeError: The service thread cannot start.
            zmq.ZMQError: Socket setup or binding fails.
        """
        self._ctx = context
        host, port = _parse_channel_url(listen_url)
        self._running = True
        self._socket = self._ctx.zmq_context.socket(zmq.REP)
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            if host.startswith("["):
                self._socket.setsockopt(zmq.IPV6, 1)
            self._socket.bind(f"tcp://{host}:{port}")
            self._thread = Thread(
                target=self._serve_loop, name="tc-hixl-server", daemon=True
            )
            self._thread.start()
        except Exception:
            self._socket.close(linger=0)
            raise

    def close(self) -> None:
        """Wait for the service thread to exit, then close its socket."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        self._socket.close(linger=0)

    def _serve_loop(self) -> None:
        """Receive, dispatch and reply until stopped or a request fails."""
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=1000))
                if self._socket not in events:
                    continue
                req_bytes = self._socket.recv()
                req = msgspec.msgpack.decode(req_bytes, type=HandshakeMsg)
                resp = self._handle_msg(req)
                self._socket.send(msgspec.msgpack.encode(resp))
            except Exception:
                if self._running:
                    logger.exception("Error in HIXL transfer channel server loop")
                # Without an error reply, REP cannot receive another request
                # after a handler fails. Leave recovery to the caller.
                self._running = False

    def _handle_msg(self, req: HandshakeMsg) -> HandshakeMsg:
        """Create and register the peer Client before returning local metadata."""
        if isinstance(req, InitReq):
            client = self._ctx.create_client(req.metadata)
            self._ctx.register_client(req.metadata.advertise_url, client)
            return InitResp(self._ctx.metadata)
        raise ValueError(f"Unexpected handshake message: {type(req)}")


############################################################
# Context
############################################################
class HixlTransferChannelContext(TransferChannelContext):
    """Own the HIXL Engine, Host registration and peer connections.

    The caller owns the fixed ACL Host L1 allocation and ACL runtime, and must
    keep them alive through cleanup. Stop local and remote accesses before close.
    """

    def __init__(
        self, l1_memory_desc: L1MemoryDesc, listen_url: str, advertise_url: str
    ) -> None:
        """Initialize HIXL, register the Host L1 range and start the Server.

        Args:
            l1_memory_desc: Entire fixed ACL Host allocation.
            listen_url: Local handshake bind address.
            advertise_url: Reachable handshake address published to peers.

        Raises:
            ValueError: Configuration, addresses or the L1 range are invalid.
            RuntimeError: Dependencies, ACL setup, registration or Server startup fail.
            HixlError: Native initialization or registration fails.
        """
        self._config = _read_config()
        _validate_memory_range(l1_memory_desc.ptr, l1_memory_desc.size)
        _parse_channel_url(listen_url)
        _parse_channel_url(advertise_url)
        self._l1_memory_desc = l1_memory_desc
        # Serialize peer admission and native cleanup, not submissions/queries.
        self._engine_lock = Lock()
        # Only a successfully initialized Engine is published here.
        self._engine: Hixl | None = None
        self._mem_handle: int | None = None
        # Successful Connect metadata, retained for reuse and shutdown, not health.
        self._connections: dict[str, HixlChannelMetadata] = {}
        self.zmq_context = zmq.Context.instance()
        self._clients_lock = Lock()
        self._clients: dict[str, HixlTransferChannelClient] = {}
        self._closed = False  # Admission flag protected by _clients_lock.
        self._server: HixlTransferChannelServer | None = None

        try:
            # Third Party
            from lmcache_ascend.hixl_npu_comms import (
                MEM_HOST,
                Hixl,
            )
            from lmcache_ascend.v1.hixl_runtime import (
                prepare_acl_context,
                use_acl_context,
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HIXL Context requires compatible LMCache-Ascend HIXL bindings "
                "and prepare_acl_context/use_acl_context runtime helpers"
            ) from exc

        self._use_acl_context: Callable[[int], AbstractContextManager[None]] = (
            use_acl_context
        )
        options = self._config.initialize_options()
        self._acl_context = prepare_acl_context(self._config.device_id)
        with self._engine_lock, self._use_acl_context(self._acl_context):
            engine = Hixl()
            try:
                engine.initialize(self._config.engine_id, options)
            except BaseException:
                # Release the local owner in the ACL scope even on interruption.
                # Initialize's internal partial cleanup remains native-owned.
                engine = None
                raise
            self._engine = engine
            del engine  # Cleanup must not leave a local owner outside its scope.

        try:
            # Initialize may change the current context: use a fresh scope.
            with self._engine_lock, self._use_acl_context(self._acl_context):
                handle = self._engine.register_mem(
                    l1_memory_desc.ptr, l1_memory_desc.size, MEM_HOST
                )
                if handle == 0:
                    raise RuntimeError("HIXL RegisterMem returned a null handle")
                self._mem_handle = handle
            self._metadata = HixlChannelMetadata(
                advertise_url=advertise_url,
                engine_id=self._config.engine_id,
                buffer_base_ptr=l1_memory_desc.ptr,
                buffer_size=l1_memory_desc.size,
                instance_id=str(uuid4()),
            )
            # Incoming admission can only start after all local state is ready.
            self._server = HixlTransferChannelServer(self, listen_url)
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self) -> HixlChannelMetadata:
        """Return the local identity and registered L1 range."""
        return self._metadata

    ############################################################
    # Server / client management
    ############################################################
    def get_transfer_channel_server(self) -> HixlTransferChannelServer:
        """Return this Context's metadata Server.

        Raises:
            RuntimeError: The Context is closed or the Server is unavailable.
        """
        with self._clients_lock:
            if self._closed or self._server is None:
                raise RuntimeError("HIXL metadata server is not available")
            return self._server

    def get_transfer_channel_client(
        self, peer_advertise_url: str
    ) -> HixlTransferChannelClient:
        """Return a cached Client or connect and register one for the peer.

        Args:
            peer_advertise_url: Peer handshake address and Client lookup key.

        Returns:
            The Client registered for the peer address.

        Raises:
            ValueError: The address, response or peer metadata is invalid.
            RuntimeError: The Context is closed or Client setup fails.
            LMCacheTimeoutError: The handshake reply times out.
            HixlError: Native Connect fails.
            zmq.ZMQError: The handshake transport fails.
        """
        with self._clients_lock:
            if self._closed:
                raise RuntimeError("HIXL Context is closed")
            client = self._clients.get(peer_advertise_url)
            if client is not None:
                return client

        client = self._connect(peer_advertise_url)
        return self.register_client(peer_advertise_url, client)

    def create_client(self, metadata: HixlChannelMetadata) -> HixlTransferChannelClient:
        """Create a Client, establishing or reusing the peer connection.

        Args:
            metadata: Peer identity and registered Host range.

        Returns:
            A new Client to pass to register_client.

        Raises:
            ValueError: Metadata is invalid or conflicts with a known peer.
            RuntimeError: The Context is closed or Client setup fails.
            HixlError: Native Connect fails.
        """
        _validate_metadata(metadata)
        with self._clients_lock:
            if self._closed:
                raise RuntimeError("HIXL Context is closed")
        peer_metadata = self._ensure_connection(metadata)
        return HixlTransferChannelClient(self, peer_metadata)

    def register_client(
        self, key: str, client: HixlTransferChannelClient
    ) -> HixlTransferChannelClient:
        """Register a Client, closing redundant duplicates.

        Args:
            key: Peer advertise URL used for lookup and removal.
            client: Candidate from this Context's create_client method.

        Returns:
            The existing Client for key, or the newly registered candidate.

        Raises:
            RuntimeError: The Context is closing or closed; the candidate is
                closed without being registered.
        """
        with self._clients_lock:
            if self._closed:
                existing = None
            else:
                existing = self._clients.get(key)
                if existing is None:
                    self._clients[key] = client
                    return client
                if existing is client:
                    return client

        if existing is None:
            client.close()
            raise RuntimeError("HIXL Context is closed")

        logger.debug("Reusing existing transfer channel client for %s", key)
        try:
            client.close()
        except Exception:
            logger.exception(
                "Error closing duplicate transfer channel client for %s", key
            )
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Remove and close a Client, retaining its shared HIXL connection.

        Unknown peers are ignored. This does not drain native transfers.

        Args:
            peer_advertise_url: Peer address used for Client lookup.
        """
        with self._clients_lock:
            client = self._clients.pop(peer_advertise_url, None)
        if client is None:
            return
        try:
            client.close()
        except Exception:
            logger.exception(
                "Error closing transfer channel client for %s", peer_advertise_url
            )

    def get_num_connected_clients(self) -> int:
        """Return the registered Client count, not a network health measurement."""
        with self._clients_lock:
            return len(self._clients)

    def submit_transfer(
        self,
        remote_engine_id: str,
        operation: TransferOp,
        descriptors: list[TransferOpDesc],
    ) -> int:
        """Submit a validated batch in the Engine's ACL context.

        Called by a registered Client while holding its task lock.

        Args:
            remote_engine_id: Admitted peer's Engine ID.
            operation: Native READ or WRITE operation.
            descriptors: Nonempty batch validated by the Client.

        Returns:
            The native request handle.

        Raises:
            RuntimeError: The Engine is closed or ACL preparation fails.
            HixlError: Native submission fails.
        """
        if self._engine is None:
            raise RuntimeError("HIXL engine is closed")
        with self._use_acl_context(self._acl_context):
            return self._engine.transfer_async(remote_engine_id, operation, descriptors)

    def query_transfer(self, req: int) -> TransferStatus:
        """Query a native request in the Engine's ACL context.

        Called by a registered Client while holding its task lock.

        Args:
            req: Native request handle from the Client's task table.

        Returns:
            The native transfer status.

        Raises:
            RuntimeError: The Engine is closed or ACL preparation fails.
            HixlError: Native query fails.
        """
        if self._engine is None:
            raise RuntimeError("HIXL engine is closed")
        with self._use_acl_context(self._acl_context):
            return self._engine.get_transfer_status(req)

    ############################################################
    # Address translation
    ############################################################
    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate local byte ranges and convert them to Channel addresses.

        Args:
            lmcache_addresses: Local (offset, size) pairs within the L1 pool.

        Returns:
            Byte-offset addresses in input order, without alignment adjustment.

        Raises:
            ValueError: An offset/size is invalid or outside the local pool.
        """
        addresses = []
        for index, (offset, size) in enumerate(lmcache_addresses):
            address = TransferChannelAddress(offset=offset, size=size)
            _validate_address_range(
                address, self.metadata.buffer_size, label=f"local_addresses[{index}]"
            )
            addresses.append(address)
        return addresses

    ############################################################
    # Cleanup
    ############################################################
    def close(self) -> None:
        """Close the Server and Clients and clean up HIXL resources.

        Call after local and remote transfers have stopped; this does not drain
        transfers or release the caller's L1 memory and ACL runtime.
        """
        with self._clients_lock:
            self._closed = True
            server = self._server
        # A Server may be in admission and need either lock: join before taking
        # a snapshot or releasing its resources, with no Context lock held.
        if server is not None:
            server.close()
        with self._clients_lock:
            self._server = None
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

        with self._engine_lock:
            if self._engine is None:
                return
            for engine_id in list(self._connections):
                try:
                    with self._use_acl_context(self._acl_context):
                        try:
                            self._engine.disconnect(
                                engine_id, timeout_ms=_DISCONNECT_TIMEOUT_MS
                            )
                        finally:
                            # Finalize owns any resources left by a failed call.
                            # Retain the record if ACL scope entry itself fails.
                            self._connections.pop(engine_id)
                except RuntimeError:
                    logger.exception("Failed to disconnect HIXL peer %s", engine_id)
            if self._mem_handle is not None:
                try:
                    with self._use_acl_context(self._acl_context):
                        try:
                            self._engine.deregister_mem(self._mem_handle)
                        finally:
                            # A native failure has uncertain consumption; do not
                            # replay the handle. Finalize owns remaining cleanup.
                            self._mem_handle = None
                except RuntimeError:
                    logger.exception(
                        "Failed to deregister HIXL L1 range at %s",
                        self._l1_memory_desc.ptr,
                    )
            try:
                with self._use_acl_context(self._acl_context):
                    try:
                        self._engine.finalize()
                    finally:
                        self._mem_handle = None
                        self._engine = None
                        self._connections.clear()
            except RuntimeError:
                logger.exception(
                    "Failed to finalize HIXL engine %s", self._config.engine_id
                )

    ############################################################
    # Helper functions
    ############################################################
    def _ensure_connection(self, metadata: HixlChannelMetadata) -> HixlChannelMetadata:
        """Reuse matching metadata or connect a new peer under the Engine lock."""
        with self._engine_lock:
            if self._engine is None:
                raise RuntimeError("HIXL engine is closed")
            for peer_metadata in self._connections.values():
                if (
                    peer_metadata.engine_id == metadata.engine_id
                    or peer_metadata.advertise_url == metadata.advertise_url
                ):
                    if peer_metadata != metadata:
                        raise ValueError("Conflicting HIXL peer metadata")
                    return peer_metadata
            with self._use_acl_context(self._acl_context):
                self._engine.connect(
                    metadata.engine_id, timeout_ms=self._config.connect_timeout_ms
                )
            # Unknown ALREADY_CONNECTED is not evidence of consistent admission.
            self._connections[metadata.engine_id] = metadata
            return metadata

    def _recv_handshake(self, socket: zmq.Socket, server_url: str) -> bytes:
        """Receive a reply, mapping a timeout to the shared MP timeout error.

        Args:
            socket: REQ socket awaiting a handshake reply.
            server_url: Peer URL used in the error message.

        Returns:
            The raw MessagePack reply.

        Raises:
            LMCacheTimeoutError: No reply arrived within the handshake timeout.
            zmq.ZMQError: Receiving the reply fails for another reason.
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

    def _connect(self, server_url: str) -> HixlTransferChannelClient:
        """Exchange metadata and create a Client for subsequent registration."""
        host, port = _parse_channel_url(server_url)
        socket = self.zmq_context.socket(zmq.REQ)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)
            if host.startswith("["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(f"tcp://{host}:{port}")
            socket.send(msgspec.msgpack.encode(InitReq(self.metadata)))
            response = msgspec.msgpack.decode(
                self._recv_handshake(socket, server_url), type=HandshakeMsg
            )
            if not isinstance(response, InitResp):
                raise ValueError("Expected HIXL InitResp")
        finally:
            socket.close(linger=0)

        return self.create_client(response.metadata)


############################################################
# Factory registration
############################################################
def create_hixl_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs: object,
) -> HixlTransferChannelContext:
    """Create a HIXL Context using environment settings.

    Args:
        l1_memory_desc: Fixed ACL Host L1 region to register.
        listen_url: Local handshake bind address.
        advertise_url: Reachable handshake address published to peers.
        **kwargs: Unused factory arguments.

    Returns:
        The initialized Context.

    Raises:
        ValueError: Configuration, addresses or the L1 range are invalid.
        RuntimeError: Dependencies, ACL setup, registration or Server startup fail.
        HixlError: Native initialization or registration fails.
    """
    return HixlTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
    )


register_transfer_channel_factory("hixl", create_hixl_transfer_channel_context)
