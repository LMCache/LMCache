# SPDX-License-Identifier: Apache-2.0
"""Authentication tests for the multiprocess gRPC transport."""

# Standard
from dataclasses import dataclass
import threading

# Third Party
import grpc
import pytest

# First Party
from lmcache.security.api_key import APIKeyVerifier
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.transport.grpc_impl import create_request_client
from lmcache.v1.multiprocess.transport.grpc_impl.aio_server import (
    GrpcAioMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_base import (
    GrpcMultiprocessServerBase,
)


@dataclass
class _Calls:
    unary: int = 0
    streaming: int = 0


class _AuthenticationTestModule:
    def __init__(self, calls: _Calls) -> None:
        self._calls = calls

    @request_handler(operation="noop")
    def noop(self) -> str:
        self._calls.unary += 1
        return "ok"

    @request_handler()
    def subscribe_kv_events(
        self, instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]:
        del instance_id, model_name, cursor, max_events
        self._calls.streaming += 1
        sent = False

        def read(closed: threading.Event) -> KVEventBatch:
            nonlocal sent
            if closed.is_set() or sent:
                raise StopIteration
            sent = True
            return KVEventBatch(False, 1, 0, False, [])

        return MessagingStream(read, lambda: None)


@pytest.fixture(
    params=[GrpcMultiprocessServer, GrpcAioMultiprocessServer], ids=["sync", "aio"]
)
def server_class(request: pytest.FixtureRequest) -> type[GrpcMultiprocessServerBase]:
    """Run authentication regressions against both gRPC server modes."""
    return request.param


def _start_server(
    accepted_keys: list[str],
    calls: _Calls,
    server_class: type[GrpcMultiprocessServerBase],
) -> GrpcMultiprocessServerBase:
    server = server_class(
        "grpc://127.0.0.1:0",
        max_cpu_workers=1,
        max_gpu_workers=1,
        grpc_server_workers=2,
        authentication=APIKeyVerifier(accepted_keys),
    )
    server.add_modules([_AuthenticationTestModule(calls)])
    server.start()
    return server


def _assert_unauthenticated(error: grpc.RpcError) -> None:
    assert error.code() is grpc.StatusCode.UNAUTHENTICATED
    assert error.details() == "Invalid or missing API key"


@pytest.mark.parametrize("client_key", ["current", "next"])
def test_grpc_authentication_accepts_unary_and_streaming_requests(
    server_class: type[GrpcMultiprocessServerBase], client_key: str
) -> None:
    """The client interceptor injects credentials for both RPC cardinalities."""
    calls = _Calls()
    server = _start_server(["current", "next"], calls, server_class)
    client = create_request_client(
        f"grpc://127.0.0.1:{server.bound_port}", api_key=client_key
    )
    try:
        assert client.noop().result(timeout=3) == "ok"
        stream = client.subscribe_kv_events(1, "model", 0, 1)
        assert next(stream).enabled is False
        stream.close()
        assert calls == _Calls(unary=1, streaming=1)
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("client_key", [None, "wrong"])
def test_grpc_authentication_rejects_before_dispatch(
    client_key: str | None, server_class: type[GrpcMultiprocessServerBase]
) -> None:
    """Missing and incorrect metadata never reaches unary or stream handlers."""
    calls = _Calls()
    server = _start_server(["secret"], calls, server_class)
    client = create_request_client(
        f"grpc://127.0.0.1:{server.bound_port}", api_key=client_key
    )
    try:
        with pytest.raises(grpc.RpcError) as unary_error:
            client.noop().result(timeout=3)
        _assert_unauthenticated(unary_error.value)

        stream = client.subscribe_kv_events(1, "model", 0, 1)
        with pytest.raises(grpc.RpcError) as stream_error:
            next(stream)
        _assert_unauthenticated(stream_error.value)
        stream.close()
        assert calls == _Calls()
    finally:
        client.close()
        server.close()


def test_grpc_authentication_disabled_preserves_compatibility(
    server_class: type[GrpcMultiprocessServerBase],
) -> None:
    """A server without keys accepts an existing unauthenticated client."""
    calls = _Calls()
    server = _start_server([], calls, server_class)
    client = create_request_client(f"grpc://127.0.0.1:{server.bound_port}")
    try:
        assert client.noop().result(timeout=3) == "ok"
        assert calls.unary == 1
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize(
    "method",
    [
        "/lmcache.mp.ManagementService/Noop",
        "/lmcache.mp.KVEventService/SubscribeKvEvents",
    ],
)
def test_grpc_authentication_precedes_request_deserialization(
    server_class: type[GrpcMultiprocessServerBase], method: str
) -> None:
    """Invalid protobuf bytes still fail as unauthenticated before decoding."""
    calls = _Calls()
    server = _start_server(["secret"], calls, server_class)
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    rpc = (
        channel.unary_stream
        if method.endswith("SubscribeKvEvents")
        else channel.unary_unary
    )
    call = rpc(
        method,
        request_serializer=lambda _request: b"not-a-protobuf-message",
    )
    try:
        with pytest.raises(grpc.RpcError) as error:
            result = call(object(), timeout=3)
            if method.endswith("SubscribeKvEvents"):
                next(result)
        _assert_unauthenticated(error.value)
        assert calls.unary == 0
    finally:
        channel.close()
        server.close()
