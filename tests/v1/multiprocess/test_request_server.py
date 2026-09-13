# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request server behavior tests."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast
import socket
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.request_handler import request_handler
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
def test_sync_handlers_are_serialized(request_transport: RequestTransport) -> None:
    """SYNC handlers must retain the single-main-loop execution contract."""
    state = _DispatchState()

    class SyncModule:
        @request_handler(RequestType.NOOP)
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
        clients = [
            RequestClientFactory.create(server_url) for _ in range(worker_count)
        ]
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
