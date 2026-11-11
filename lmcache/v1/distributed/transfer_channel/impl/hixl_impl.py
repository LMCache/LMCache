# SPDX-License-Identifier: Apache-2.0
"""HIXL Host L1 lifecycle, metadata handshake and peer management.

Context supports native READ/WRITE submission and request-status queries;
factory wiring is not available. Importing this module does not load the
Ascend extension or initialize the NPU.
"""

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


def _parse_channel_url(url: str) -> tuple[str, int]:
    """Parse a handshake address without resolving its host.

    Args:
        url: host:port or tcp://host:port, with brackets for IPv6 literals.

    Returns:
        Host and positive port, preserving IPv6 brackets.

    Raises:
        ValueError: The URL or port is invalid.
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
    """Require a listening HIXL Engine identity for this bidirectional backend.

    Args:
        engine_id: IPv4:port or [IPv6]:port, without a URL scheme.

    Raises:
        ValueError: The IP literal or positive listening port is invalid.
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
    """Validate peer identity and range before any native admission.

    Args:
        metadata: Typed handshake metadata; values are not trusted by decoding.

    Raises:
        ValueError: An address or memory range is invalid.
    """
    _parse_channel_url(metadata.advertise_url)
    _validate_engine_id(metadata.engine_id)
    _validate_memory_range(metadata.buffer_base_ptr, metadata.buffer_size)


def _read_int(suffix: str, default: str, minimum: int, maximum: int) -> int:
    """Read one bounded decimal integer from the HIXL environment.

    Args:
        suffix: Variable name after LMCACHE_HIXL_.
        default: String used when the variable is unset.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound.

    Returns:
        The parsed integer.

    Raises:
        ValueError: The value is not decimal or is outside the bounds.
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
    """Snapshot backend configuration before loading native dependencies.

    Returns:
        A validated configuration with an explicit listening engine ID.

    Raises:
        ValueError: ENGINE_ID is missing/invalid or an integer is invalid.
    """
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
    """Reject a Host pool that cannot be represented by native address types.

    Args:
        ptr: Raw base address of an already registered Host allocation.
        size: Registered byte length; no alignment or rounding is applied.

    Raises:
        ValueError: Either value is invalid or the range overflows uintptr_t.
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
    """Check one object lies within a previously validated Host pool.

    Args:
        address: Object offset and byte length relative to its pool.
        buffer_size: Validated registered pool size.
        label: Address position to include in error messages.

    Raises:
        ValueError: The offset or length is invalid or outside the pool.
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


class HandshakeMsgBase(msgspec.Struct, tag=True):
    """Share tagged MessagePack encoding across HIXL handshake messages."""


class InitReq(HandshakeMsgBase):
    """Request peer admission using the sender's registered Host metadata."""

    metadata: HixlChannelMetadata


class InitResp(HandshakeMsgBase):
    """Return local metadata after reverse peer admission succeeds."""

    metadata: HixlChannelMetadata


HandshakeMsg = InitReq | InitResp


class HixlTransferChannelClient(TransferChannelClient):
    """Borrow a peer connection and manage a shared READ/WRITE task ID space.

    Context owns registration, connections and engine lifetime. The caller
    retains both data regions until native accesses have ended; this Client
    neither owns the allocations nor grants permission to overwrite a peer.
    """

    def __init__(
        self, context: HixlTransferChannelContext, peer_metadata: HixlChannelMetadata
    ) -> None:
        """Create a Client for an admitted peer without connecting or allocating.

        Args:
            context: Owner providing the native-call boundary and local pool.
            peer_metadata: Validated metadata returned by this Context's peer setup.

        Raises:
            RuntimeError: The required optional Ascend binding cannot be loaded.
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
        """Read remote objects into local destinations; return a Client task ID.

        Args:
            local_addresses: Nonempty list of local destination offsets/sizes.
            remote_addresses: Matching remote source offsets/sizes.

        Returns:
            An ID in the shared READ/WRITE space, queried with query_status.

        Raises:
            ValueError: A range, pair length or batch size is invalid.
            RuntimeError: The Client/connection is unusable or req is null.
            HixlError: Native submission failed; no task ID is published.

        Note:
            Native errors do not prove no bytes were accessed. No retry occurs.
            Source and destination must remain valid until accesses have ended.
        """
        return self._submit_transfer(
            self._operation.READ, local_addresses, remote_addresses
        )

    def submit_write(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Write local objects into remote destinations; return a Client task ID.

        Args:
            local_addresses: Nonempty list of local source offsets/sizes.
            remote_addresses: Matching remote destination offsets/sizes. The
                caller must hold permission to overwrite these ranges.

        Returns:
            An ID in the shared READ/WRITE space, queried with query_status.

        Raises:
            ValueError: A range, pair length or batch size is invalid.
            RuntimeError: The Client/connection is unusable or req is null.
            HixlError: Native submission failed; no task ID is published.

        Note:
            This does not reserve or publish remote KV objects. Failure does not
            roll back writes. Retain source bytes unchanged and destinations
            exclusively writable until native accesses have ended.
        """
        return self._submit_transfer(
            self._operation.WRITE, local_addresses, remote_addresses
        )

    def query_status(self, task_id: int) -> TransferChannelTransferResult:
        """Poll a READ or WRITE once, consuming terminal requests at most once.

        Args:
            task_id: ID returned by this Client's submit_read or submit_write.

        Returns:
            Unfinished with an empty mask while waiting; otherwise a uniform
            success/failure mask. HIXL reports request-level status, not which
            individual objects were modified before a failure.

        Raises:
            KeyError: The ID is unknown, consumed, or invalidated by close.
            HixlError: Native query failed; this task is consumed and the error
                propagates without invalidating the peer or other tasks.
            RuntimeError: Context preparation failed (task retained), or an
                unsupported native terminal status was returned (task consumed).
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
        """Invalidate local tasks and reject new submissions; safe to repeat.

        This does not cancel or drain native accesses, free allocations, or
        disconnect a shared peer. Context owns connection/engine cleanup.
        A returned close is not permission to reuse or free in-flight buffers.
        """
        with self._lock:
            self._closed = True
            self._tasks.clear()

    def _submit_transfer(
        self,
        operation: TransferOp,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Build descriptors, submit the batch and publish a Client task ID.

        Address validation and conversion do not modify task state. Hold the
        Client lock across native submission and registration so close cannot
        invalidate the Client between them. Query consumes tasks under the same
        lock. Context takes its Engine lock inside the native-call boundary.
        WRITE uses the same local/remote address fields as READ.
        """
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


class HixlTransferChannelServer(TransferChannelServer):
    """Exchange peer metadata over a ZMQ REP socket.

    Bind before starting the service thread, as in NIXL and Mooncake TE.
    A failed request stops the loop; there is no error response or recovery.
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
        """Stop the service before closing its socket; sequentially repeatable.

        The caller must not hold Context locks or run on the service thread.
        HIXL Connect may outlast NIXL/TE's five-second join, so wait for the
        current handler to return before closing the socket and Engine.
        This does not drain native transfers.
        """
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


class HixlTransferChannelContext(TransferChannelContext):
    """Own an Engine and Host registration; borrow the L1 and ACL context.

    It does not allocate L1 memory, create an ACL context or own the runtime.
    The L1 owner must supply a fully live, fixed ACL Host allocation and retain
    it through deregistration. Users must quiesce local and remote accesses
    before close; the engine lock alone does not provide that guarantee.
    """

    def __init__(
        self, l1_memory_desc: L1MemoryDesc, listen_url: str, advertise_url: str
    ) -> None:
        """Initialize HIXL and register the exact supplied Host L1 range.

        Args:
            l1_memory_desc: Entire fixed Host allocation, not a lazy prefix.
            listen_url: Local handshake bind address.
            advertise_url: Reachable handshake address published in metadata.

        Raises:
            ValueError: Configuration, addresses or the L1 range are invalid.
            RuntimeError: Native/runtime dependencies cannot load, ACL setup
                fails, RegisterMem returns a null handle, or Server startup fails.
            HixlError: Native initialization or registration fails.

        Note:
            The initialization thread keeps the selected logical device.
            Failed construction rolls back acquired Engine/registration
            ownership without freeing the caller's memory or ACL context.
        """
        self._config = _read_config()
        _validate_memory_range(l1_memory_desc.ptr, l1_memory_desc.size)
        _parse_channel_url(listen_url)
        _parse_channel_url(advertise_url)
        self._l1_memory_desc = l1_memory_desc
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
        """Return immutable identity and bounds of the registered local L1.

        This snapshot is not a liveness check and remains readable after close.
        """
        return self._metadata

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
        """Return a cached Client or handshake and register one for the peer.

        Args:
            peer_advertise_url: The peer's advertised Channel address, used
                unchanged as the Client collection key.

        Returns:
            The canonical Client registered for that address.

        Raises:
            ValueError: The address, response or peer metadata is invalid.
            RuntimeError: The Context is closed or peer connection is unusable.
            LMCacheTimeoutError: Waiting for the handshake reply times out.
            HixlError: Native Connect fails.
            zmq.ZMQError: The handshake transport fails.

        Remove a Client through the Context before getting a replacement.
        A cached Client is not a live connectivity probe.
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
        """Create a Client over a validated, Context-owned HIXL connection.

        Args:
            metadata: Peer identity and complete registered Host range.

        Returns:
            A new Client for the caller to pass to register_client.

        Raises:
            ValueError: Metadata is invalid or conflicts with a known peer.
            RuntimeError: The Context/connection is unusable or bindings fail.
            HixlError: Native Connect fails; no new record is published.

        Shared by active and passive handshakes. A Client construction failure
        leaves the connected Engine peer owned by Context for later cleanup.
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
        """Keep the first Client for a peer and close redundant duplicates.

        Args:
            key: Peer advertise URL used unchanged for lookup and removal.
            client: Client created by this Context's create_client method.

        Returns:
            The existing Client for key, or client if none was registered.

        Callers must finish active handshakes before closing Context. Shutdown
        joins the Server before clearing Clients registered by inbound requests.
        Closing a duplicate does not disconnect its shared HIXL connection.
        """
        with self._clients_lock:
            existing = self._clients.get(key)
            if existing is None:
                self._clients[key] = client
                return client
            if existing is client:
                return client

        logger.debug("Reusing existing transfer channel client for %s", key)
        try:
            client.close()
        except Exception:
            logger.exception(
                "Error closing duplicate transfer channel client for %s", key
            )
        return existing

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        """Remove the logical Client, retaining the Context-owned connection.

        Args:
            peer_advertise_url: Peer address used for Client lookup.

        Unknown peers are a no-op. Old task IDs become invalid; a later get
        creates a fresh Client. Neither removal nor close drains native access.
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

        Args:
            remote_engine_id: Engine ID from this Context's admitted peer metadata.
            operation: Native READ or WRITE operation.
            descriptors: Nonempty batch already validated by the Client.

        Returns:
            The native request handle for Client task registration.

        Raises:
            RuntimeError: The Engine is closed or ACL preparation fails.
            HixlError: Native submission fails.

        Errors propagate without disconnecting the peer or retrying the batch.
        The Client owns task IDs and checks the returned handle.
        """
        with self._engine_lock:
            if self._engine is None:
                raise RuntimeError("HIXL engine is closed")
            with self._use_acl_context(self._acl_context):
                return self._engine.transfer_async(
                    remote_engine_id, operation, descriptors
                )

    def query_transfer(self, req: int) -> TransferStatus:
        """Query one native request without changing peer or other task state.

        Args:
            req: Native request handle returned by this Context's submit_transfer
                and retained in the calling Client's task table.

        Returns:
            The native status for the Client to translate and consume.

        Raises:
            RuntimeError: The Engine is closed or ACL preparation fails.
            HixlError: Native query fails; the Client consumes this task.

        ACL preparation failures happen before querying and leave the task
        available. No synthetic FAILED status, disconnect or reconnect occurs.
        """
        with self._engine_lock:
            if self._engine is None:
                raise RuntimeError("HIXL engine is closed")
            with self._use_acl_context(self._acl_context):
                return self._engine.get_transfer_status(req)

    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate local byte ranges without alignment or address adjustment.

        Args:
            lmcache_addresses: Local (offset, size) pairs within the L1 pool.

        Returns:
            Public byte-offset addresses in input order; empty for empty input.
            These values do not keep the underlying memory alive.

        Raises:
            ValueError: An offset/length is invalid or outside the local pool.
        """
        addresses = []
        for index, (offset, size) in enumerate(lmcache_addresses):
            address = TransferChannelAddress(offset=offset, size=size)
            _validate_address_range(
                address, self.metadata.buffer_size, label=f"local_addresses[{index}]"
            )
            addresses.append(address)
        return addresses

    def close(self) -> None:
        """Stop admission, close Clients, disconnect, deregister and finalize.

        Call only after local and remote accesses have stopped. Sequential
        calls are safe; no cancellation, draining or automatic retry is added.
        ACL/native RuntimeErrors are logged so rollback preserves the original
        construction error. A cleanup attempt is not proof of successful free.

        Each native operation gets a fresh ACL scope. After attempting Finalize,
        release the Engine reference inside that scope. If scope entry fails,
        retain unprocessed ownership rather than call native cleanup in the
        wrong context. The caller must keep the borrowed runtime alive.
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
        """Exchange metadata and create a Client for subsequent registration.

        No Context lock spans the remote metadata wait. A lost reply or local
        Connect failure does not roll back the peer's completed reverse setup.
        """
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
