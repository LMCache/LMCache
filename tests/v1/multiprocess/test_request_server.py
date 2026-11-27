# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request server behavior tests."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast
import threading
import time

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
    build_grpc_request_server,
)
from lmcache.v1.transport.request_client import RequestClient


@dataclass
class _DispatchState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    max_active: int = 0
    calls: int = 0


def test_serialized_handlers_run_one_at_a_time() -> None:
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
    config = MPServerConfig(host="127.0.0.1", port=0)
    server = build_grpc_request_server([SyncModule()], config)
    server_url = f"grpc://127.0.0.1:{server.bound_port}"
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
