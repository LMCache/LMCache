# SPDX-License-Identifier: Apache-2.0
"""Tests for server-module handlers served on the client's affinity thread."""

# Standard
from typing import Any, Literal
from unittest.mock import MagicMock
import queue
import threading

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleRouter,
    build_server_module_router,
    server_module_handler,
)
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.request_handler import (
    HandlerType,
    get_request_handler_options,
    request_handler,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server


def _ok(payload: bytes) -> ServerModuleCallResponse:
    return ServerModuleCallResponse(success=True, payload=payload, error="")


def _failed(error: str) -> ServerModuleCallResponse:
    return ServerModuleCallResponse(success=False, payload=b"", error=error)


def _request(method: str, payload: bytes = b"") -> ServerModuleCallRequest:
    return ServerModuleCallRequest(method=method, payload=payload)


class _LayerwisePlugin:
    """Plugin whose affinity call reports progress while it runs."""

    def __init__(self) -> None:
        self.ctx = MagicMock(name="ctx")
        self.frames: queue.Queue[bytes] = queue.Queue()
        self.gate = threading.Event()

    @property
    def context(self) -> Any:
        return self.ctx

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        self.gate.set()

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def noop(self) -> str:
        """Stands in for store/retrieve: report the client's affinity thread."""
        return threading.current_thread().name

    @server_module_handler("fake.echo")
    def echo(self, payload: bytes) -> bytes:
        return b"echo:" + payload

    @server_module_handler("fake.retrieve", requires_client_affinity=True)
    def retrieve(self, payload: bytes) -> bytes:
        """Push one frame per layer, then hold the thread until released."""
        for layer in range(int(payload)):
            self.frames.put(b"layer-%d" % layer)
        assert self.gate.wait(timeout=5)
        return threading.current_thread().name.encode()

    @server_module_handler("fake.affinity_fail", requires_client_affinity=True)
    def affinity_fail(self, payload: bytes) -> bytes:
        raise RuntimeError(payload.decode())


def _router(plugin: _LayerwisePlugin) -> ExtServerModuleRouter:
    router = build_server_module_router(plugin.context, [plugin])
    assert router is not None
    return router


def test_affinity_call_is_a_blocking_client_affinity_handler() -> None:
    options = get_request_handler_options(
        ExtServerModuleRouter.server_module_affinity_call
    )
    unary = get_request_handler_options(ExtServerModuleRouter.server_module_call)

    assert options is not None
    assert options.handler_type == HandlerType.BLOCKING
    assert options.requires_client_affinity
    assert unary is not None and not unary.requires_client_affinity


def test_router_dispatches_each_kind_to_its_own_rpc() -> None:
    plugin = _LayerwisePlugin()
    plugin.gate.set()
    router = _router(plugin)

    assert router.server_module_affinity_call(_request("fake.retrieve", b"1")) == (
        _ok(threading.current_thread().name.encode())
    )
    assert plugin.frames.get_nowait() == b"layer-0"
    assert router.server_module_affinity_call(
        _request("fake.affinity_fail", b"boom")
    ) == _failed("boom")

    assert router.server_module_call(_request("fake.retrieve")) == _failed(
        "server module method fake.retrieve is served by server_module_affinity_call"
    )
    assert router.server_module_affinity_call(_request("fake.echo")) == _failed(
        "server module method fake.echo is served by server_module_call"
    )
    assert router.server_module_affinity_call(_request("fake.missing")) == _failed(
        "unknown server module method: fake.missing"
    )


def test_router_status_lists_affinity_methods() -> None:
    router = _router(_LayerwisePlugin())

    assert router.report_status() == {
        "server_module_plugins": {
            "methods": ["fake.echo"],
            "affinity_methods": ["fake.affinity_fail", "fake.retrieve"],
        }
    }


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_affinity_call_runs_in_order_over_real_transport(
    transport: Literal["zmq", "grpc"],
) -> None:
    """The call runs on the client's affinity thread, can report progress
    before it returns, and holds back the client's later affinity requests."""
    plugin = _LayerwisePlugin()
    router = _router(plugin)
    config = MPServerConfig(
        transport=transport,
        host="127.0.0.1",
        port=0,
        max_cpu_workers=1,
        max_gpu_workers=4,
        grpc_server_workers=4,
    )
    server: Any = create_request_server([plugin, router], config)
    if transport == "grpc":
        url = f"grpc://127.0.0.1:{server.bound_port}"
    else:
        url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    server.start()
    client = RequestClientFactory.create(url, context=zmq.Context.instance())
    try:
        affinity_thread = client.noop().result(timeout=5)
        assert "affinity" in affinity_thread

        retrieve = client.server_module_affinity_call(_request("fake.retrieve", b"2"))
        # Frames arrive while the call still holds the affinity thread.
        assert plugin.frames.get(timeout=5) == b"layer-0"
        assert plugin.frames.get(timeout=5) == b"layer-1"
        assert not retrieve.query()

        # A later affinity request from the same client waits behind it.
        queued = client.noop()
        assert not queued.wait(timeout=0.2)
        plugin.gate.set()
        assert retrieve.result(timeout=5) == _ok(affinity_thread.encode())
        assert queued.result(timeout=5) == affinity_thread

        assert client.server_module_affinity_call(
            _request("fake.affinity_fail", b"boom")
        ).result(timeout=5) == _failed("boom")
        assert client.server_module_affinity_call(_request("fake.echo")).result(
            timeout=5
        ) == _failed("server module method fake.echo is served by server_module_call")
    finally:
        plugin.close()
        client.close()
        server.close()
