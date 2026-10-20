# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request server behavior tests."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, cast
import socket
import threading
import time

# Third Party
import grpc
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventBatch, KVEventProfile
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Test helpers
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


@dataclass
class _DispatchState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    max_active: int = 0
    calls: int = 0


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


@pytest.mark.parametrize("request_transport", REQUEST_TRANSPORTS)
def test_serialized_handlers_run_one_at_a_time(
    request_transport: RequestTransport,
) -> None:
    """SERIALIZED handlers must retain the single-main-loop execution contract."""
    state = _DispatchState()

    class SyncModule(EngineModule):
        @property
        def context(self) -> MPCacheServerContext:
            """The dispatch-only test does not use an engine context."""
            raise NotImplementedError

        def report_status(self) -> dict:
            """Return the empty status of this test module."""
            return {}

        def close(self) -> None:
            """Release no resources for this test module."""

        @request_handler()
        def noop(self) -> str:
            with state.lock:
                state.active += 1
                state.calls += 1
                state.max_active = max(state.max_active, state.active)
            try:
                time.sleep(0.05)
            finally:
                with state.lock:
                    state.active -= 1
            return "ok"

    worker_count = 8
    server_url = request_server_url(request_transport, _unused_tcp_port())
    config = request_server_config(request_transport, server_url)
    server = create_request_server([SyncModule()], config)
    server.start()
    clients: list[RequestClient] = []
    try:
        clients = [RequestClientFactory.create(server_url) for _ in range(worker_count)]
        barrier = threading.Barrier(worker_count)

        def call_noop(client: RequestClient) -> str:
            barrier.wait(timeout=5)
            return cast(str, client.noop().result(timeout=5))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(call_noop, client) for client in clients]
            assert [future.result(timeout=10) for future in futures] == [
                "ok"
            ] * worker_count
    finally:
        for client in clients:
            client.close()
        server.close()

    assert state.calls == worker_count
    assert state.max_active == 1


@pytest.mark.parametrize(
    "request_transport,mode", [("zmq", "sync"), ("grpc", "sync"), ("grpc", "aio")]
)
@pytest.mark.parametrize("handler_type", [HandlerType.SERIALIZED, HandlerType.BLOCKING])
@pytest.mark.parametrize("error_type", [ValueError, TypeError, ConnectionError])
@pytest.mark.parametrize("streaming", [False, True])
def test_handler_errors_preserve_transport_and_later_requests(
    request_transport: RequestTransport,
    mode: Literal["sync", "aio"],
    handler_type: HandlerType,
    error_type: type[Exception],
    streaming: bool,
) -> None:
    """Validation errors fail explicitly; reconnectable errors remain distinct."""

    class Module:
        @request_handler(handler_type)
        def ping(self, instance_id: int | None) -> bool:
            if instance_id == 1:
                raise error_type("rejected request")
            return True

        @request_handler(handler_type)
        def subscribe_kv_events(
            self,
            instance_id: int,
            model_name: str,
            cursor: int,
            max_events: int,
            world_size: int,
            profile: KVEventProfile,
        ) -> MessagingStream[KVEventBatch]:
            raise error_type("rejected request")

    url = request_server_url(request_transport, _unused_tcp_port())
    config = request_server_config(request_transport, url)
    config.grpc_server_mode = mode
    server = create_request_server([cast(EngineModule, Module())], config)
    server.start()
    client = RequestClientFactory.create(url)
    stream = None
    try:
        if streaming:
            stream = client.subscribe_kv_events(
                1, "model", 0, 1, 1, KVEventProfile("test", 1)
            )
        expected = (
            grpc.RpcError
            if request_transport == "grpc"
            and not (streaming and error_type is ConnectionError)
            else ConnectionError
            if error_type is ConnectionError
            else ValueError
        )
        with pytest.raises(expected) as caught:
            if stream is not None:
                next(stream)
            else:
                client.ping(1).result(timeout=5)
        if isinstance(caught.value, grpc.RpcError):
            assert caught.value.code() == (
                grpc.StatusCode.UNAVAILABLE
                if error_type is ConnectionError
                else grpc.StatusCode.INVALID_ARGUMENT
            )
        assert client.ping(None).result(timeout=5)
    finally:
        if stream is not None:
            stream.close()
        client.close()
        server.close()
