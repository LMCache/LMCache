# SPDX-License-Identifier: Apache-2.0
"""Behavior the native gRPC server must share with the grpcio servers.

The shared contract (every RPC, handler concurrency, affinity, stream
backpressure) is covered by ``test_grpc_transport.py`` for all modes. These
tests pin the parts a grpcio server gets from grpcio itself: status codes,
cancellation, deadlines, binding, shutdown, message size and out-of-tree
services.
"""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
import importlib
import shutil
import socket
import sys
import tempfile
import threading
import time

# Third Party
import grpc
import psutil
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.client import GrpcMultiprocessClient
from lmcache.v1.multiprocess.transport.grpc_impl.native_server import (
    GrpcNativeMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_base import (
    GrpcMultiprocessServerBase,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)

pytest.importorskip("pygrpc_native")

extension_service_pb2 = importlib.import_module(
    "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.extension_service_pb2"
)

_PING = "/lmcache.mp.ManagementService/Ping"
_SUBSCRIBE = "/lmcache.mp.KVEventService/SubscribeKvEvents"
_SERVER_MODULE_CALL = "/lmcache.mp.ExtensionService/ServerModuleCall"


def _serve(modules: list[Any], **overrides: Any) -> GrpcMultiprocessServerBase:
    settings: dict[str, Any] = {
        "transport": "grpc",
        "host": "127.0.0.1",
        "port": 0,
        "max_cpu_workers": 2,
        "max_gpu_workers": 1,
        "grpc_server_mode": "native",
    }
    server = build_grpc_request_server(
        modules, MPServerConfig(**(settings | overrides))
    )
    server.start()
    return server


def _client(server: GrpcMultiprocessServerBase) -> GrpcMultiprocessClient:
    return GrpcMultiprocessClient(  # type: ignore[abstract]
        f"grpc://127.0.0.1:{server.bound_port}"
    )


def _status(error: BaseException) -> tuple[grpc.StatusCode, str]:
    call = cast(grpc.Call, error)
    return call.code(), call.details()


def _echo(client: GrpcMultiprocessClient, payload: bytes) -> bytes:
    response = client.server_module_call(
        ServerModuleCallRequest(method="echo", payload=payload)
    ).result(30)
    return response.payload


class _EchoModule:
    @request_handler(HandlerType.BLOCKING)
    def server_module_call(
        self, request: ServerModuleCallRequest
    ) -> ServerModuleCallResponse:
        return ServerModuleCallResponse(
            success=True, payload=request.payload[::-1], error=""
        )


def _idle_stream(closed: threading.Event) -> MessagingStream[KVEventBatch]:
    """A stream whose reads wait until it is closed."""

    def read(stop: threading.Event) -> KVEventBatch:
        stop.wait()
        raise StopIteration

    return MessagingStream(read, closed.set)


def _stream_module(make: Callable[[], Any]) -> object:
    class Module:
        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            return cast(MessagingStream[KVEventBatch], make())

    return Module()


def _first_stream_error(
    server: GrpcMultiprocessServerBase,
) -> tuple[grpc.StatusCode, str]:
    client = _client(server)
    try:
        stream = client.subscribe_kv_events(1, "model", 0, 8)
        with pytest.raises(grpc.RpcError) as info:
            next(stream)
        return _status(info.value)
    finally:
        client.close()


@pytest.fixture
def blocked() -> Iterator[threading.Event]:
    release = threading.Event()
    yield release
    release.set()


def test_handler_error_is_unknown_with_grpcio_message() -> None:
    """A handler exception is UNKNOWN with grpcio's message, and frees its worker."""

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            raise RuntimeError("handler failed")

    server = _serve([Module()])
    client = _client(server)
    try:
        # More failures than workers.
        for _ in range(3):
            with pytest.raises(grpc.RpcError) as info:
                client.ping(1).result(5)
            assert _status(info.value) == (
                grpc.StatusCode.UNKNOWN,
                "Exception calling application: handler failed",
            )
    finally:
        client.close()
        server.close()


def test_missing_handler_is_unimplemented() -> None:
    """A generated method without a module handler is UNIMPLEMENTED."""
    server = _serve([])
    client = _client(server)
    try:
        with pytest.raises(grpc.RpcError) as info:
            client.ping(1).result(5)
        assert _status(info.value) == (
            grpc.StatusCode.UNIMPLEMENTED,
            "ping is not enabled on this server",
        )
    finally:
        client.close()
        server.close()


def test_unknown_method_is_unimplemented() -> None:
    """A path no service defines is UNIMPLEMENTED."""
    server = _serve([])
    try:
        with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            with pytest.raises(grpc.RpcError) as info:
                channel.unary_unary("/lmcache.mp.NoSuchService/Call")(b"", timeout=5)
        assert _status(info.value)[0] is grpc.StatusCode.UNIMPLEMENTED
    finally:
        server.close()


def test_undecodable_request_is_internal() -> None:
    """Bytes that are not the request message are INTERNAL, as in grpcio."""

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return True

    server = _serve([Module()])
    try:
        with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            with pytest.raises(grpc.RpcError) as info:
                channel.unary_unary(_PING)(b"\xff\xff\xff", timeout=5)
        assert _status(info.value)[0] is grpc.StatusCode.INTERNAL
    finally:
        server.close()


def test_handler_returning_the_wrong_kind_is_unknown() -> None:
    """A unary handler returning a stream, or the reverse, fails the call."""
    closed = threading.Event()

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return cast(bool, _idle_stream(closed))

    server = _serve([Module()])
    client = _client(server)
    try:
        with pytest.raises(grpc.RpcError) as info:
            client.ping(1).result(5)
        assert _status(info.value)[0] is grpc.StatusCode.UNKNOWN
        assert closed.is_set()
    finally:
        client.close()
        server.close()

    server = _serve([_stream_module(lambda: None)])
    try:
        assert _first_stream_error(server)[0] is grpc.StatusCode.UNKNOWN
    finally:
        server.close()


def test_streaming_handler_errors_map_like_grpcio() -> None:
    """ValueError from a streaming handler is INVALID_ARGUMENT; an error while
    reading its stream is INTERNAL."""

    def invalid() -> None:
        raise ValueError("bad cursor")

    def broken() -> MessagingStream[KVEventBatch]:
        def read(stop: threading.Event) -> KVEventBatch:
            raise RuntimeError("bus gone")

        return MessagingStream(read, lambda: None)

    for make, expected in (
        (invalid, (grpc.StatusCode.INVALID_ARGUMENT, "bad cursor")),
        (broken, (grpc.StatusCode.INTERNAL, "bus gone")),
    ):
        server = _serve([_stream_module(make)])
        try:
            assert _first_stream_error(server) == expected
        finally:
            server.close()


def test_serialized_handlers_run_one_at_a_time() -> None:
    """SERIALIZED handlers never overlap, across methods too."""
    lock = threading.Lock()
    running = peak = 0

    def enter() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.02)
        with lock:
            running -= 1

    class Module:
        @request_handler()
        def get_chunk_size(self) -> int:
            enter()
            return 256

        @request_handler()
        def negotiate_chunk_size(self, required_chunk_alignment: int) -> int:
            enter()
            return required_chunk_alignment

    server = _serve([Module()], max_cpu_workers=4)
    client = _client(server)
    try:
        futures = [client.get_chunk_size() for _ in range(4)]
        futures += [client.negotiate_chunk_size(16) for _ in range(4)]
        for future in futures:
            future.result(5)
    finally:
        client.close()
        server.close()
    assert peak == 1


def test_cancelled_stream_closes_the_handler_stream() -> None:
    """A client leaving an idle stream closes the handler's MessagingStream."""
    started = threading.Event()
    closed = threading.Event()

    class Module:
        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            started.set()
            return _idle_stream(closed)

    server = _serve([Module()])
    client = _client(server)
    try:
        stream = client.subscribe_kv_events(1, "model", 0, 8)
        assert started.wait(5)
        stream.close()
        assert closed.wait(5)
    finally:
        client.close()
        server.close()


def test_stream_cancelled_before_the_handler_returns_is_closed() -> None:
    """A stream the client left while its handler was still running is closed
    as soon as the handler returns it."""
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Module:
        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            entered.set()
            assert release.wait(5)
            return _idle_stream(closed)

    server = _serve([Module()])
    client = _client(server)
    try:
        stream = client.subscribe_kv_events(1, "model", 0, 8)
        assert entered.wait(5)
        stream.close()
        # Give the cancel time to reach the server first; if it arrives
        # later the stream must be closed all the same.
        time.sleep(0.2)
        release.set()
        assert closed.wait(5)
    finally:
        release.set()
        client.close()
        server.close()


def test_deadline_ends_unary_and_streaming_calls(blocked: threading.Event) -> None:
    """grpc-timeout ends both kinds of call, and closes the handler's stream."""
    closed = threading.Event()

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            blocked.wait(10)
            return True

        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            return _idle_stream(closed)

    server = _serve([Module()])
    try:
        with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            with pytest.raises(grpc.RpcError) as info:
                channel.unary_unary(_PING)(b"", timeout=0.3)
            assert _status(info.value)[0] is grpc.StatusCode.DEADLINE_EXCEEDED
            stream = channel.unary_stream(_SUBSCRIBE)(b"", timeout=0.3)
            with pytest.raises(grpc.RpcError) as info:
                next(stream)
            assert _status(info.value)[0] is grpc.StatusCode.DEADLINE_EXCEEDED
        assert closed.wait(5)
    finally:
        blocked.set()
        server.close()


def test_close_fails_calls_in_flight_and_is_idempotent(
    blocked: threading.Event,
) -> None:
    """Calls still running at close() get UNAVAILABLE; close() runs twice."""
    entered = threading.Event()

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            entered.set()
            blocked.wait(10)
            return True

    server = _serve([Module()])
    client = _client(server)
    try:
        future = client.ping(1)
        assert entered.wait(5)
        server.close()
        server.close()
        with pytest.raises(grpc.RpcError) as info:
            future.result(5)
        assert _status(info.value)[0] is grpc.StatusCode.UNAVAILABLE
    finally:
        blocked.set()
        client.close()
        server.close()


def test_start_twice_is_rejected() -> None:
    """A server starts once."""
    server = _serve([])
    try:
        with pytest.raises(RuntimeError, match="already started"):
            server.start()
    finally:
        server.close()


def test_bind_failures_raise_and_leave_no_threads() -> None:
    """A failed bind raises like the grpcio servers and starts no thread."""

    def fail_to_bind() -> None:
        with pytest.raises(RuntimeError, match="Failed to bind"):
            GrpcNativeMultiprocessServer("grpc://nonexistent.invalid:0", 1, 1, 1)
        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        try:
            port = taken.getsockname()[1]
            with pytest.raises(RuntimeError, match="Failed to bind"):
                GrpcNativeMultiprocessServer(f"grpc://127.0.0.1:{port}", 1, 1, 1)
        finally:
            taken.close()

    # The first failure may start grpcio's process-wide threads.
    fail_to_bind()
    process = psutil.Process()
    before = process.num_threads()
    for _ in range(3):
        fail_to_bind()
    # OS threads, so the Rust runtime counts too. Name resolution can leave a
    # short-lived helper thread behind, so wait for the count to settle.
    deadline = time.monotonic() + 5
    while process.num_threads() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process.num_threads() <= before


def test_missing_rust_module_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without pygrpc-native the server says which package to install."""
    monkeypatch.setitem(sys.modules, "pygrpc_native", None)
    with pytest.raises(RuntimeError, match="pip install pygrpc-native"):
        GrpcNativeMultiprocessServer("grpc://127.0.0.1:0", 1, 1, 1)


def test_localhost_is_served_on_both_address_families() -> None:
    """Like grpc-core, a host is bound on every address it resolves to."""
    addresses = {info[4][0] for info in socket.getaddrinfo("localhost", None)}
    if not {"127.0.0.1", "::1"} <= addresses:
        pytest.skip("localhost does not resolve to both 127.0.0.1 and ::1")
    server = _serve([_EchoModule()], host="localhost")
    try:
        for host in ("127.0.0.1", "[::1]"):
            client = GrpcMultiprocessClient(  # type: ignore[abstract]
                f"grpc://{host}:{server.bound_port}"
            )
            try:
                assert _echo(client, b"ab") == b"ba"
            finally:
                client.close()
    finally:
        server.close()


def test_unix_socket_forms_and_restart_after_a_stale_file() -> None:
    """Both unix target forms work, a stale socket file does not block the
    bind, and close() removes the file."""
    # pytest's tmp_path can exceed the 104-byte socket path limit on macOS.
    directory = Path(tempfile.mkdtemp(prefix="lmc", dir="/tmp"))
    path = directory / "s.sock"
    request = extension_service_pb2.ServerModuleCallRequest(
        method="echo", payload=b"ab"
    )
    try:
        stale = socket.socket(socket.AF_UNIX)
        stale.bind(str(path))
        stale.close()  # leaves the file behind, as a killed server would
        for url in (f"grpc+unix://{path}", f"unix:{path}"):
            server = GrpcNativeMultiprocessServer(url, 1, 1, 1)
            server.add_modules([_EchoModule()])
            server.start()
            assert server.bound_port == 1
            # grpc-core sends the socket path as :authority, which the front
            # rejects as invalid HTTP/2, so name the authority explicitly.
            channel = grpc.insecure_channel(
                f"unix:{path}", options=[("grpc.default_authority", "localhost")]
            )
            try:
                call = channel.unary_unary(
                    _SERVER_MODULE_CALL,
                    request_serializer=extension_service_pb2.ServerModuleCallRequest.SerializeToString,
                    response_deserializer=extension_service_pb2.ServerModuleCallResponse.FromString,
                )
                assert call(request, timeout=5).payload == b"ba"
            finally:
                channel.close()
                server.close()
            assert not path.exists()
    finally:
        shutil.rmtree(directory)


def test_large_messages_round_trip() -> None:
    """16 MiB, the size of an inline KV chunk, goes both ways."""
    payload = bytes(range(256)) * (64 * 1024)
    server = _serve([_EchoModule()])
    client = _client(server)
    try:
        assert _echo(client, payload) == payload[::-1]
    finally:
        client.close()
        server.close()


def test_gzip_requests_are_accepted() -> None:
    """Like grpcio servers, gzip-compressed requests are accepted."""
    server = _serve([_EchoModule()])
    request = extension_service_pb2.ServerModuleCallRequest(
        method="echo", payload=b"ab" * 4096
    )
    try:
        with grpc.insecure_channel(
            f"127.0.0.1:{server.bound_port}", compression=grpc.Compression.Gzip
        ) as channel:
            call = channel.unary_unary(
                _SERVER_MODULE_CALL,
                request_serializer=extension_service_pb2.ServerModuleCallRequest.SerializeToString,
                response_deserializer=extension_service_pb2.ServerModuleCallResponse.FromString,
            )
            assert call(request, timeout=5).payload == b"ba" * 4096
    finally:
        server.close()


def test_out_of_tree_grpcio_services_are_served() -> None:
    """Services registered on the grpc.Server work next to generated ones."""

    def reverse(request: bytes, context: grpc.ServicerContext) -> bytes:
        return request[::-1]

    def count(request: bytes, context: grpc.ServicerContext) -> Iterator[bytes]:
        for i in range(int(request)):
            yield str(i).encode()

    class Module:
        def register_grpc_services(self, server: grpc.Server) -> None:
            server.add_generic_rpc_handlers(
                (
                    grpc.method_handlers_generic_handler(
                        "ext.Echo",
                        {
                            "Reverse": grpc.unary_unary_rpc_method_handler(reverse),
                            "Count": grpc.unary_stream_rpc_method_handler(count),
                        },
                    ),
                )
            )

    server = _serve([Module(), _EchoModule()])
    client = _client(server)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            assert channel.unary_unary("/ext.Echo/Reverse")(b"abc", timeout=5) == b"cba"
            assert list(channel.unary_stream("/ext.Echo/Count")(b"3", timeout=5)) == [
                b"0",
                b"1",
                b"2",
            ]
            with pytest.raises(grpc.RpcError) as info:
                channel.unary_unary("/ext.Echo/Missing")(b"", timeout=5)
            assert _status(info.value)[0] is grpc.StatusCode.UNIMPLEMENTED
        assert _echo(client, b"xy") == b"yx"
    finally:
        client.close()
        server.close()


def test_out_of_tree_services_may_register_until_start() -> None:
    """The registrar gets a real grpc.Server, and services added to it any
    time before start() are served."""

    def reverse(request: bytes, context: grpc.ServicerContext) -> bytes:
        return request[::-1]

    class Module:
        server: grpc.Server | None = None

        def register_grpc_services(self, server: grpc.Server) -> None:
            assert isinstance(server, grpc.Server)
            self.server = server

    module = Module()
    server = GrpcNativeMultiprocessServer("grpc://127.0.0.1:0", 1, 1, 1)
    try:
        server.add_modules([module])
        assert module.server is not None
        module.server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "ext.Late",
                    {"Reverse": grpc.unary_unary_rpc_method_handler(reverse)},
                ),
            )
        )
        server.start()
        with grpc.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            assert channel.unary_unary("/ext.Late/Reverse")(b"ab", timeout=5) == b"ba"
    finally:
        server.close()


def test_out_of_tree_socket_is_only_created_for_registrars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a registrar no private socket is made; with one it is removed
    on close()."""
    # pytest's tmp_path can exceed the 104-byte socket path limit on macOS.
    tmp_path = Path(tempfile.mkdtemp(prefix="lmc", dir="/tmp"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    class Registrar:
        def register_grpc_services(self, server: grpc.Server) -> None:
            pass

    server = _serve([_EchoModule()])
    try:
        assert list(tmp_path.iterdir()) == []
    finally:
        server.close()

    server = _serve([Registrar()])
    try:
        assert list(tmp_path.iterdir()) != []
    finally:
        server.close()
    assert list(tmp_path.iterdir()) == []
    tmp_path.rmdir()


def test_out_of_tree_socket_bind_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private socket path over the OS limit fails start() clearly, and
    close() still cleans up."""
    long_dir = Path(tempfile.mkdtemp(prefix="lmc", dir="/tmp")) / ("d" * 120)
    long_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_dir))

    class Registrar:
        def register_grpc_services(self, server: grpc.Server) -> None:
            pass

    server = GrpcNativeMultiprocessServer("grpc://127.0.0.1:0", 1, 1, 1)
    try:
        server.add_modules([Registrar()])
        with pytest.raises(RuntimeError, match="out-of-tree gRPC services"):
            server.start()
    finally:
        server.close()
        assert list(long_dir.iterdir()) == []
        shutil.rmtree(long_dir.parent)


def test_failing_stream_close_callback_does_not_stop_the_server() -> None:
    """A handler stream whose close callback raises fails only that stream."""
    started = threading.Event()

    def broken_close() -> None:
        raise RuntimeError("close failed")

    class Module:
        @request_handler(HandlerType.BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return True

        @request_handler()
        def subscribe_kv_events(
            self, instance_id: int, model_name: str, cursor: int, max_events: int
        ) -> MessagingStream[KVEventBatch]:
            started.set()

            def read(stop: threading.Event) -> KVEventBatch:
                stop.wait()
                raise StopIteration

            return MessagingStream(read, broken_close)

    server = _serve([Module()])
    client = _client(server)
    try:
        stream = client.subscribe_kv_events(1, "model", 0, 8)
        assert started.wait(5)
        stream.close()
        assert client.ping(1).result(5) is True
    finally:
        client.close()
        server.close()


def test_close_survives_a_failing_stream_close_callback() -> None:
    """close() still closes the other streams and the pools if one raises."""
    made: list[MessagingStream[KVEventBatch]] = []
    closed = threading.Event()

    def broken_close() -> None:
        raise RuntimeError("close failed")

    def read(stop: threading.Event) -> KVEventBatch:
        stop.wait()
        raise StopIteration

    def make() -> MessagingStream[KVEventBatch]:
        if made:
            made.append(_idle_stream(closed))
        else:
            made.append(MessagingStream(read, broken_close))
        return made[-1]

    server = cast(GrpcNativeMultiprocessServer, _serve([_stream_module(make)]))
    client = _client(server)
    streams = [client.subscribe_kv_events(i, "model", 0, 8) for i in (1, 2)]
    try:
        deadline = time.monotonic() + 5
        while len(server._streams) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(server._streams) == 2
        server.close()
        assert closed.is_set()
        with pytest.raises(RuntimeError):
            server._normal_pool.submit(lambda: None)
    finally:
        for stream in streams:
            stream.close()
        client.close()
        server.close()


def test_non_blocking_handlers_are_unimplemented() -> None:
    """NON_BLOCKING handlers are rejected, as on the sync and aio servers."""

    class Module:
        @request_handler(HandlerType.NON_BLOCKING)
        def ping(self, instance_id: int | None) -> bool:
            return True

    server = _serve([Module()])
    client = _client(server)
    try:
        with pytest.raises(grpc.RpcError) as info:
            client.ping(1).result(5)
        assert _status(info.value) == (
            grpc.StatusCode.UNIMPLEMENTED,
            "NON_BLOCKING handlers are not supported",
        )
    finally:
        client.close()
        server.close()


def test_calls_before_add_modules_are_unimplemented() -> None:
    """A server started before add_modules() serves no method yet."""
    server = GrpcNativeMultiprocessServer("grpc://127.0.0.1:0", 1, 1, 1)
    server.start()
    client = _client(server)
    try:
        with pytest.raises(grpc.RpcError) as info:
            client.ping(1).result(5)
        assert _status(info.value) == (
            grpc.StatusCode.UNIMPLEMENTED,
            "Method not found!",
        )
    finally:
        client.close()
        server.close()


def test_start_after_close_is_rejected() -> None:
    """A closed server cannot be started again."""
    server = GrpcNativeMultiprocessServer("grpc://127.0.0.1:0", 1, 1, 1)
    server.close()
    with pytest.raises(RuntimeError, match="closed"):
        server.start()


def test_exports_grpc_server_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    # First Party
    from lmcache.v1.mp_observability import otel_init

    pygrpc_native = pytest.importorskip("pygrpc_native")
    if not hasattr(pygrpc_native.NativeGrpcFront, "metrics"):
        pytest.skip("needs pygrpc-native 0.3.0 or later")
    monkeypatch.setattr(otel_init, "_GRPC_METRICS_ENABLED", True)
    monkeypatch.setattr(otel_init, "_METRICS_SOURCES", [])
    server = _serve([_EchoModule()])
    client = _client(server)
    try:
        for _ in range(3):
            assert _echo(client, b"ab") == b"ba"
        (source,) = otel_init._METRICS_SOURCES
        for _ in range(100):
            metrics = {metric.name: metric for metric in source()}
            if metrics and metrics["grpc.server.call.duration"].data.data_points:
                break
            time.sleep(0.01)
    finally:
        client.close()
        server.close()

    (started,) = metrics["grpc.server.call.started"].data.data_points
    (duration,) = metrics["grpc.server.call.duration"].data.data_points
    method = _SERVER_MODULE_CALL.lstrip("/")
    assert started.attributes == {"grpc.method": method}
    assert started.value == 3
    assert duration.attributes == {"grpc.method": method, "grpc.status": "OK"}
    assert duration.count == sum(duration.bucket_counts) == 3
    assert otel_init._METRICS_SOURCES == []
