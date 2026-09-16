# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from unittest.mock import create_autospec
import asyncio
import socket
import struct
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.server.__main__ import LMCacheServer
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

_SERVER_WAIT_SECONDS = 3.0
_TINY_SHAPE = torch.Size([1, 1, 2, 8])
_TINY_FORMAT = MemoryFormat.KV_2LTD
_TINY_DTYPE = torch.uint8
ConnectionHandler = Callable[[socket.socket], None]


class TinyTensorAllocator(MemoryAllocatorInterface):
    """Allocate small CPU ``TensorMemoryObj`` instances for connector tests."""

    def __init__(self) -> None:
        """Initialize the allocator with no outstanding objects."""
        self._next_address = 0
        self._freed: list[MemoryObj] = []
        self._freed_ids: set[int] = set()
        self.fail_next_allocation = False

    @property
    def freed_count(self) -> int:
        """Return the number of distinct objects released through this allocator."""
        return len(self._freed)

    def allocate(
        self,
        shapes: torch.Size | list[torch.Size],
        dtypes: torch.dtype | list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: str | None = None,
    ) -> MemoryObj | None:
        """Allocate one CPU tensor object, or return ``None`` for an injected miss."""
        del allocator_type
        if self.fail_next_allocation:
            self.fail_next_allocation = False
            return None

        shape, dtype = self._single_shape_and_dtype(shapes, dtypes)
        byte_count = shape.numel() * dtype.itemsize
        metadata = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=self._next_address,
            phy_size=byte_count,
            ref_count=1,
            fmt=fmt,
            shapes=[shape],
            dtypes=[dtype],
        )
        self._next_address += byte_count
        return TensorMemoryObj(
            torch.empty(byte_count, dtype=torch.uint8), metadata, parent_allocator=self
        )

    def batched_allocate(
        self,
        shapes: torch.Size | list[torch.Size],
        dtypes: torch.dtype | list[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: str | None = None,
    ) -> list[MemoryObj] | None:
        """Allocate a fixed number of test objects using the one-object contract."""
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        memory_objs: list[MemoryObj] = []
        for _ in range(batch_size):
            memory_obj = self.allocate(shapes, dtypes, fmt, allocator_type)
            if memory_obj is None:
                for allocated_obj in memory_objs:
                    allocated_obj.ref_count_down()
                return None
            memory_objs.append(memory_obj)
        return memory_objs

    def free(self, memory_obj: MemoryObj, allocator_type: str | None = None) -> None:
        """Record and invalidate one released object exactly once."""
        del allocator_type
        if not memory_obj.is_valid() or id(memory_obj) in self._freed_ids:
            raise AssertionError("TinyTensorAllocator received a duplicate free")
        memory_obj.invalidate()
        self._freed_ids.add(id(memory_obj))
        self._freed.append(memory_obj)

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        """Release every object in ``memory_objs`` through ``free``."""
        del update_stats
        for memory_obj in memory_objs:
            self.free(memory_obj, allocator_type)

    @staticmethod
    def _single_shape_and_dtype(
        shapes: torch.Size | list[torch.Size], dtypes: torch.dtype | list[torch.dtype]
    ) -> tuple[torch.Size, torch.dtype]:
        """Validate the one-group shape and dtype used by this tiny allocator."""
        shape_list = [shapes] if isinstance(shapes, torch.Size) else shapes
        dtype_list = [dtypes] if isinstance(dtypes, torch.dtype) else dtypes
        if len(shape_list) != 1 or len(dtype_list) != 1:
            raise ValueError("TinyTensorAllocator supports one tensor group")
        return shape_list[0], dtype_list[0]


class PayloadStore:
    """Keep protocol payloads by serialized cache key for the loopback server."""

    def __init__(self) -> None:
        """Initialize an empty loopback payload store."""
        self._payloads: dict[str, bytes] = {}

    def put(self, key: CacheEngineKey, payload: bytes) -> None:
        """Store ``payload`` under ``key`` exactly as received on the wire."""
        self._payloads[key.to_string()] = payload

    def get(self, key: CacheEngineKey) -> bytes | None:
        """Return the stored payload for ``key``, if the key exists."""
        return self._payloads.get(key.to_string())


class LoopbackScriptServer:
    """Run a bounded sequence of real TCP protocol handlers on loopback."""

    def __init__(
        self, handlers: list[ConnectionHandler], port: int | None = None
    ) -> None:
        """Bind ``port`` or a unique port and retain ordered connection handlers."""
        self._handlers = handlers
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0 if port is None else port))
        self._listener.listen(len(handlers) or 1)
        self._listener.settimeout(0.1)
        self._thread = threading.Thread(
            target=self._serve, name="lm-reconnect-loopback"
        )
        self._started = threading.Event()
        self._stopping = threading.Event()
        self._failure: BaseException | None = None
        self._accepted_connections = 0

    @property
    def port(self) -> int:
        """Return the operating-system-assigned loopback TCP port."""
        return cast(int, self._listener.getsockname()[1])

    @property
    def accepted_connections(self) -> int:
        """Return how many client connections reached this loopback service."""
        return self._accepted_connections

    def start(self) -> None:
        """Start the server thread and wait until it can accept connections."""
        self._thread.start()
        if not self._started.wait(_SERVER_WAIT_SECONDS):
            raise TimeoutError("loopback server did not start")

    def close(self) -> None:
        """Stop the server and surface any unexpected handler failure."""
        self._stopping.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(_SERVER_WAIT_SECONDS)
        if self._thread.is_alive():
            raise TimeoutError("loopback server thread did not stop")
        if self._failure is not None:
            raise RuntimeError("loopback server handler failed") from self._failure

    def _serve(self) -> None:
        """Accept each scripted connection in order until stopped or exhausted."""
        self._started.set()
        try:
            for handler in self._handlers:
                connection = self._accept_one()
                if connection is None:
                    return
                self._accepted_connections += 1
                connection.settimeout(_SERVER_WAIT_SECONDS)
                try:
                    handler(connection)
                finally:
                    connection.close()
        except BaseException as exc:
            if not self._stopping.is_set():
                self._failure = exc
        finally:
            try:
                self._listener.close()
            except OSError:
                pass

    def _accept_one(self) -> socket.socket | None:
        """Accept one client, observing an explicit stop request between waits."""
        while not self._stopping.is_set():
            try:
                connection, _ = self._listener.accept()
                return connection
            except socket.timeout:
                continue
            except OSError:
                if self._stopping.is_set():
                    return None
                raise
        return None


def _new_backend(allocator: TinyTensorAllocator) -> LocalCPUBackend:
    """Build a spec-constrained backend mock with real config and metadata."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2, local_cpu=True, lmcache_instance_id="reconnect-test"
    )
    metadata = LMCacheMetadata(
        model_name="reconnect-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=_TINY_DTYPE,
        kv_shape=(1, 1, 2, 1, 8),
        chunk_size=2,
    )
    backend = cast(LocalCPUBackend, create_autospec(LocalCPUBackend, instance=True))
    backend.config = config
    backend.metadata = metadata
    backend.allocate.side_effect = allocator.allocate
    return backend


def _new_key() -> CacheEngineKey:
    """Create the stable tiny cache key used by all reconnect scenarios."""
    return CacheEngineKey(
        model_name="reconnect-test",
        world_size=1,
        worker_id=0,
        chunk_hash=3565,
        dtype=_TINY_DTYPE,
    )


def _new_memory_obj(allocator: TinyTensorAllocator, payload: bytes) -> MemoryObj:
    """Allocate a real test memory object populated with ``payload`` bytes."""
    memory_obj = allocator.allocate(_TINY_SHAPE, _TINY_DTYPE, _TINY_FORMAT)
    if memory_obj is None:
        raise RuntimeError("test allocator unexpectedly refused a source object")
    if len(payload) != len(memory_obj.byte_array):
        raise ValueError("payload must exactly fill the tiny test object")
    memoryview(memory_obj.byte_array).cast("B")[:] = payload
    return memory_obj


def _release(memory_obj: MemoryObj | None) -> None:
    """Release a caller-owned memory object if it remains valid."""
    if memory_obj is not None and memory_obj.is_valid():
        memory_obj.ref_count_down()


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or fail if the peer closes the frame early."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining} frame bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_request(connection: socket.socket) -> ClientMetaMessage:
    """Read and decode one real client metadata frame from ``connection``."""
    return ClientMetaMessage.deserialize(
        _recv_exact(connection, ClientMetaMessage.packlength())
    )


def _expect_request(
    connection: socket.socket, expected_command: ClientCommand
) -> ClientMetaMessage:
    """Read one request and require the expected public protocol command."""
    message = _read_request(connection)
    if message.command != expected_command:
        raise AssertionError(
            f"expected {expected_command.name}, got {message.command.name}"
        )
    return message


def _send_response(
    connection: socket.socket,
    code: ServerReturnCode,
    payload: bytes = b"",
    declared_length: int | None = None,
) -> None:
    """Send a real server header and an optional tiny tensor payload."""
    response_length = len(payload) if declared_length is None else declared_length
    if response_length < len(payload):
        raise ValueError("declared response length cannot truncate payload")
    connection.sendall(
        ServerMetaMessage(
            code=code,
            length=response_length,
            fmt=_TINY_FORMAT,
            dtype=_TINY_DTYPE,
            shape=_TINY_SHAPE,
        ).serialize()
    )
    if payload:
        connection.sendall(payload)


def _serve_one_request(store: PayloadStore, connection: socket.socket) -> None:
    """Handle one PUT, GET, or EXIST request through the real wire protocol."""
    message = _read_request(connection)
    if message.command == ClientCommand.PUT:
        store.put(message.key, _recv_exact(connection, message.length))
        return
    if message.command == ClientCommand.GET:
        payload = store.get(message.key)
        if payload is None:
            _send_response(connection, ServerReturnCode.FAIL)
        else:
            _send_response(connection, ServerReturnCode.SUCCESS, payload)
        return
    if message.command == ClientCommand.EXIST:
        code = (
            ServerReturnCode.SUCCESS
            if store.get(message.key) is not None
            else ServerReturnCode.FAIL
        )
        _send_response(connection, code)
        return
    raise AssertionError(f"unexpected client command {message.command.name}")


def _make_request_handler(store: PayloadStore, request_count: int) -> ConnectionHandler:
    """Create a handler that serves exactly ``request_count`` normal requests."""

    def handler(connection: socket.socket) -> None:
        """Serve the requested number of complete wire-format RPCs."""
        for _ in range(request_count):
            _serve_one_request(store, connection)

    return handler


def _force_reset(connection: socket.socket) -> None:
    """Close a connection with TCP RST so a following write observes EPIPE-like I/O."""
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()


def _make_reset_after_header_handler(
    expected_command: ClientCommand,
) -> ConnectionHandler:
    """Create a handler that reads one metadata frame then drops the TCP connection."""

    def handler(connection: socket.socket) -> None:
        """Read the expected header and force an immediate connection failure."""
        _expect_request(connection, expected_command)
        _force_reset(connection)

    return handler


def _make_header_then_wait_for_close_handler(
    expected_command: ClientCommand, header_received: threading.Event
) -> ConnectionHandler:
    """Create a handler that confirms one header then waits for peer closure."""

    def handler(connection: socket.socket) -> None:
        """Record the header boundary and wait until recovery closes the socket."""
        _expect_request(connection, expected_command)
        header_received.set()
        connection.recv(1)

    return handler


def _make_half_body_get_handler(
    payload: bytes, first_body_bytes: int
) -> ConnectionHandler:
    """Create a GET handler that sends a header and only a prefix of its body."""
    if first_body_bytes <= 0 or first_body_bytes >= len(payload):
        raise ValueError("first_body_bytes must be a strict payload prefix")

    def handler(connection: socket.socket) -> None:
        """Emit an incomplete successful GET response then send TCP FIN."""
        _expect_request(connection, ClientCommand.GET)
        _send_response(
            connection,
            ServerReturnCode.SUCCESS,
            payload[:first_body_bytes],
            declared_length=len(payload),
        )
        connection.shutdown(socket.SHUT_WR)

    return handler


def _make_get_then_wait_for_close_handler(payload: bytes) -> ConnectionHandler:
    """Create a GET handler whose stale body must be discarded before another RPC."""

    def handler(connection: socket.socket) -> None:
        """Send one full body, then wait for the client to discard its connection."""
        _expect_request(connection, ClientCommand.GET)
        _send_response(connection, ServerReturnCode.SUCCESS, payload)
        try:
            connection.recv(1)
        except ConnectionResetError:
            return

    return handler


def _make_wait_for_close_handler() -> ConnectionHandler:
    """Create a handler used to observe that connector close closes its socket."""

    def handler(connection: socket.socket) -> None:
        """Block only until the peer closes or the bounded socket timeout expires."""
        connection.recv(1)

    return handler


@pytest.mark.no_shared_allocator
class TestLMCServerConnectorReconnect:
    """Public API regression coverage for reconnecting LMC server connectors."""

    @pytest.mark.asyncio
    async def test_control_put_get_and_exists_use_the_real_protocol(self) -> None:
        """A healthy connection preserves the existing PUT, GET, and EXIST contract."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        payload = bytes(range(_TINY_SHAPE.numel()))
        key = _new_key()
        server = LoopbackScriptServer([_make_request_handler(store, request_count=3)])
        source = _new_memory_obj(allocator, payload)
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        received: MemoryObj | None = None
        try:
            await connector.put(key, source)
            assert await connector.exists(key)
            received = await connector.get(key)
            assert received is not None
            assert bytes(received.byte_array) == payload
            assert server.accepted_connections == 1
        finally:
            _release(received)
            _release(source)
            await connector.close()
            server.close()

    @pytest.mark.asyncio
    async def test_exists_recovers_after_a_dropped_response_header(self) -> None:
        """A dropped EXIST response is replayed on a new TCP connection."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        key = _new_key()
        store.put(key, bytes(range(_TINY_SHAPE.numel())))
        server = LoopbackScriptServer(
            [
                _make_reset_after_header_handler(ClientCommand.EXIST),
                _make_request_handler(store, request_count=1),
            ]
        )
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        try:
            assert await connector.exists(key)
            assert server.accepted_connections == 2
        finally:
            await connector.close()
            server.close()

    @pytest.mark.asyncio
    async def test_get_recovers_after_half_body_and_releases_old_object(self) -> None:
        """A short GET body is retried and its old allocation is released."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        payload = bytes(range(_TINY_SHAPE.numel()))
        key = _new_key()
        store.put(key, payload)
        server = LoopbackScriptServer(
            [
                _make_half_body_get_handler(payload, first_body_bytes=5),
                _make_request_handler(store, request_count=1),
            ]
        )
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        received: MemoryObj | None = None
        try:
            received = await connector.get(key)
            assert received is not None
            assert bytes(received.byte_array) == payload
            assert allocator.freed_count == 1
            assert server.accepted_connections == 2
        finally:
            _release(received)
            await connector.close()
            server.close()
        assert allocator.freed_count == 2

    @pytest.mark.asyncio
    async def test_put_replays_the_same_payload_after_an_injected_write_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body-write reset retries the original encoded PUT on a new connection."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        payload = bytes(range(_TINY_SHAPE.numel()))
        key = _new_key()
        header_received = threading.Event()
        server = LoopbackScriptServer(
            [
                _make_header_then_wait_for_close_handler(
                    ClientCommand.PUT, header_received
                ),
                _make_request_handler(store, request_count=2),
            ]
        )
        source = _new_memory_obj(allocator, payload)
        server.start()
        loop = asyncio.get_running_loop()
        real_sock_sendall = loop.sock_sendall
        send_call_count = 0

        async def send_with_one_injected_reset(
            client_socket: socket.socket, data: bytes
        ) -> None:
            """Delegate all sends except the first PUT body write to real I/O."""
            nonlocal send_call_count
            send_call_count += 1
            if send_call_count == 2:
                header_seen = await asyncio.to_thread(
                    header_received.wait, _SERVER_WAIT_SECONDS
                )
                if not header_seen:
                    raise TimeoutError("loopback server did not receive PUT header")
                raise ConnectionResetError("injected PUT body connection reset")
            await real_sock_sendall(client_socket, data)

        monkeypatch.setattr(loop, "sock_sendall", send_with_one_injected_reset)
        connector = LMCServerConnector("127.0.0.1", server.port, loop, backend)
        try:
            await connector.put(key, source)
            assert await connector.exists(key)
            assert store.get(key) == payload
            assert server.accepted_connections == 2
            assert send_call_count == 4
        finally:
            _release(source)
            await connector.close()
            server.close()

    @pytest.mark.asyncio
    async def test_allocation_refusal_discards_old_body_before_next_get(self) -> None:
        """An allocation miss closes the old stream before a GET gets fresh bytes."""
        allocator = TinyTensorAllocator()
        allocator.fail_next_allocation = True
        backend = _new_backend(allocator)
        store = PayloadStore()
        stale_payload = bytes(range(_TINY_SHAPE.numel()))
        fresh_payload = bytes(range(_TINY_SHAPE.numel(), _TINY_SHAPE.numel() * 2))
        key = _new_key()
        store.put(key, fresh_payload)
        server = LoopbackScriptServer(
            [
                _make_get_then_wait_for_close_handler(stale_payload),
                _make_request_handler(store, request_count=1),
            ]
        )
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        received: MemoryObj | None = None
        try:
            assert await connector.get(key) is None
            received = await connector.get(key)
            assert received is not None
            assert bytes(received.byte_array) == fresh_payload
            assert server.accepted_connections == 2
        finally:
            _release(received)
            await connector.close()
            server.close()

    @pytest.mark.asyncio
    async def test_close_rejects_future_public_operations_without_reconnecting(
        self,
    ) -> None:
        """After close, a public RPC reports closure and opens no new socket."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        server = LoopbackScriptServer([_make_wait_for_close_handler()])
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        try:
            await connector.close()
            with pytest.raises(RuntimeError, match="closed"):
                await connector.exists(_new_key())
        finally:
            await connector.close()
            server.close()
        assert server.accepted_connections == 1

    @pytest.mark.asyncio
    async def test_refused_reconnect_attempts_are_bounded_then_a_later_rpc_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exhausted reconnect is finite and a later public RPC can recover."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        key = _new_key()
        store.put(key, bytes(range(_TINY_SHAPE.numel())))
        first_server = LoopbackScriptServer(
            [_make_reset_after_header_handler(ClientCommand.EXIST)]
        )
        first_server.start()
        server_port = first_server.port
        loop = asyncio.get_running_loop()
        real_sock_connect = loop.sock_connect
        reject_connections = True
        refused_connect_count = 0

        async def connect_with_refusals(
            fresh_socket: socket.socket, address: tuple[str, int]
        ) -> None:
            """Refuse candidate reconnect sockets until the test enables recovery."""
            nonlocal refused_connect_count
            if reject_connections:
                refused_connect_count += 1
                raise ConnectionRefusedError("injected reconnect refusal")
            await real_sock_connect(fresh_socket, address)

        monkeypatch.setattr(loop, "sock_connect", connect_with_refusals)
        connector = LMCServerConnector("127.0.0.1", server_port, loop, backend)
        recovery_server: LoopbackScriptServer | None = None
        try:
            with pytest.raises(ConnectionError):
                await connector.exists(key)
            assert refused_connect_count == 2

            first_server.close()
            reject_connections = False
            recovery_server = LoopbackScriptServer(
                [_make_request_handler(store, request_count=1)],
                port=server_port,
            )
            recovery_server.start()
            assert await connector.exists(key)
            assert recovery_server.accepted_connections == 1
        finally:
            await connector.close()
            if recovery_server is not None:
                recovery_server.close()
            first_server.close()

    @pytest.mark.asyncio
    async def test_invalid_response_is_not_retried_and_next_request_recovers(
        self,
    ) -> None:
        """An invalid reply propagates once, without preserving its old stream."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        key = _new_key()
        store = PayloadStore()
        store.put(key, bytes(range(_TINY_SHAPE.numel())))

        def send_invalid_response(connection: socket.socket) -> None:
            _expect_request(connection, ClientCommand.EXIST)
            response = ServerMetaMessage(
                ServerReturnCode.SUCCESS, 0, _TINY_FORMAT, _TINY_DTYPE, _TINY_SHAPE
            ).serialize()
            connection.sendall(struct.pack("i", 999) + response[4:])
            assert connection.recv(1) == b""

        server = LoopbackScriptServer(
            [send_invalid_response, _make_request_handler(store, request_count=1)]
        )
        server.start()
        connector = LMCServerConnector(
            "127.0.0.1", server.port, asyncio.get_running_loop(), backend
        )
        try:
            with pytest.raises(ValueError, match="999"):
                await connector.exists(key)
            assert server.accepted_connections == 1
            assert await connector.exists(key)
            assert server.accepted_connections == 2
        finally:
            await connector.close()
            server.close()

    def test_incomplete_old_put_cannot_overwrite_a_completed_retry(self) -> None:
        """Actual server handlers must not publish an old PUT that ends early."""
        server = LMCacheServer("127.0.0.1", 0, "cpu")
        server.server_socket.settimeout(_SERVER_WAIT_SECONDS)
        address = server.server_socket.getsockname()
        clients: list[socket.socket] = []
        accepted: list[socket.socket] = []
        workers = ThreadPoolExecutor(max_workers=2)
        key = _new_key()
        payload = bytes(range(_TINY_SHAPE.numel()))
        put_header = ClientMetaMessage(
            ClientCommand.PUT, key, len(payload), _TINY_FORMAT, _TINY_DTYPE, _TINY_SHAPE
        ).serialize()
        try:
            for _ in range(2):
                client = socket.create_connection(address, _SERVER_WAIT_SECONDS)
                clients.append(client)
                peer, _ = server.server_socket.accept()
                peer.settimeout(_SERVER_WAIT_SECONDS)
                accepted.append(peer)
            old_request = workers.submit(server.handle_client, accepted[0])
            new_request = workers.submit(server.handle_client, accepted[1])
            clients[0].sendall(put_header + payload[:5])
            clients[1].sendall(put_header + payload)
            # An EXIST on the same stream proves the new complete PUT was stored.
            clients[1].sendall(
                ClientMetaMessage(
                    ClientCommand.EXIST, key, 0, _TINY_FORMAT, _TINY_DTYPE, _TINY_SHAPE
                ).serialize()
            )
            response = ServerMetaMessage.deserialize(
                _recv_exact(clients[1], ServerMetaMessage.packlength())
            )
            assert response.code == ServerReturnCode.SUCCESS
            # Only now permit the old incomplete PUT to return EOF and finish.
            clients[0].shutdown(socket.SHUT_WR)
            old_request.result(timeout=_SERVER_WAIT_SECONDS)
            stored = server.data_store.get(key)
            assert stored is not None
            assert stored.data == payload
            clients[1].sendall(
                ClientMetaMessage(
                    ClientCommand.GET, key, 0, _TINY_FORMAT, _TINY_DTYPE, _TINY_SHAPE
                ).serialize()
            )
            response = ServerMetaMessage.deserialize(
                _recv_exact(clients[1], ServerMetaMessage.packlength())
            )
            assert response.code == ServerReturnCode.SUCCESS
            assert _recv_exact(clients[1], response.length) == payload
            clients[1].shutdown(socket.SHUT_WR)
            new_request.result(timeout=_SERVER_WAIT_SECONDS)
        finally:
            for client in clients:
                client.close()
            workers.shutdown(wait=True)
            for peer in accepted:
                peer.close()
            server.server_socket.close()
            server.data_store.close()

    @pytest.mark.asyncio
    async def test_cancelling_fresh_connect_closes_its_socket_and_later_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancellation closes the fresh reconnect FD and leaves a later RPC usable."""
        allocator = TinyTensorAllocator()
        backend = _new_backend(allocator)
        store = PayloadStore()
        key = _new_key()
        store.put(key, bytes(range(_TINY_SHAPE.numel())))
        first_server = LoopbackScriptServer(
            [_make_reset_after_header_handler(ClientCommand.EXIST)]
        )
        first_server.start()
        server_port = first_server.port
        loop = asyncio.get_running_loop()
        real_sock_connect = loop.sock_connect
        connect_started = asyncio.Event()
        release_connect = asyncio.Event()
        captured_socket: socket.socket | None = None
        delay_connect = True

        async def delay_fresh_connect(
            fresh_socket: socket.socket, address: tuple[str, int]
        ) -> None:
            """Pause one fresh connection at the I/O boundary until cancellation."""
            nonlocal captured_socket
            if delay_connect:
                captured_socket = fresh_socket
                connect_started.set()
                await release_connect.wait()
                return
            await real_sock_connect(fresh_socket, address)

        monkeypatch.setattr(loop, "sock_connect", delay_fresh_connect)
        connector = LMCServerConnector("127.0.0.1", server_port, loop, backend)
        recovery_server: LoopbackScriptServer | None = None
        rpc_task: asyncio.Task[bool] | None = None
        try:
            rpc_task = asyncio.create_task(connector.exists(key))
            await asyncio.wait_for(connect_started.wait(), _SERVER_WAIT_SECONDS)
            rpc_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await rpc_task
            if captured_socket is None:
                raise AssertionError("test did not observe a fresh reconnect socket")
            assert captured_socket.fileno() == -1

            first_server.close()
            delay_connect = False
            recovery_server = LoopbackScriptServer(
                [_make_request_handler(store, request_count=1)],
                port=server_port,
            )
            recovery_server.start()
            assert await connector.exists(key)
            assert recovery_server.accepted_connections == 1
        finally:
            if rpc_task is not None:
                rpc_task.cancel()
                await asyncio.gather(rpc_task, return_exceptions=True)
            release_connect.set()
            await connector.close()
            if recovery_server is not None:
                recovery_server.close()
            first_server.close()
