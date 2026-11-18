# SPDX-License-Identifier: Apache-2.0
"""Tests for ``get_server_info`` and the worker's ``auto`` transfer mode."""

# Standard
from collections.abc import Callable, Iterator
from typing import Any, Literal
from unittest.mock import MagicMock
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.client.transport.errors import (
    MPRequestError,
    MPServerUnavailableError,
)
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.client.worker_transfer import (
    ENV_MP_TRANSFER_MODE,
    resolve_mp_transfer_mode,
)
from lmcache.v1.common.custom_types import ServerCapabilities
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_module import ModuleBuildContext
from lmcache.v1.multiprocess.modules import management
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.platform.transfer_mode import MPTransferMode
from lmcache.v1.transport.futures import MessagingFuture
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)

LMCACHE_DRIVEN = MPTransferMode.LMCACHE_DRIVEN
ENGINE_DRIVEN = MPTransferMode.ENGINE_DRIVEN

Serve = Callable[[RequestTransport, list[Any]], RequestClient]
ServerMode = Literal["lmcache_driven", "engine_driven", "auto"]


def _management(supported_transfer_mode: ServerMode) -> ManagementModule:
    mp_config = MPServerConfig(supported_transfer_mode=supported_transfer_mode)
    module = ManagementModule.create(
        ModuleBuildContext(MagicMock(), mp_config, CoordinatorConfig())
    )
    assert isinstance(module, ManagementModule)
    return module


def _answering(response: Any = None, error: BaseException | None = None) -> Any:
    """Return a request client whose ``get_server_info`` is already settled."""
    future: MessagingFuture[Any] = MessagingFuture()
    if error is None:
        future.set_result(response)
    else:
        future.set_exception(error)
    client = MagicMock(spec=RequestClient)
    client.get_server_info.return_value = future
    return client


@pytest.fixture(autouse=True)
def _no_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_MP_TRANSFER_MODE, raising=False)


@pytest.fixture
def server_device(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Set the device type the in-process server reports."""

    def set_device(device_type: str) -> None:
        monkeypatch.setattr(management, "torch_device_type", device_type)

    return set_device


@pytest.fixture
def serve() -> Iterator[Serve]:
    """Start real request servers from the given modules; yield their clients."""
    started: list[tuple[Any, RequestClient]] = []

    def start(transport: RequestTransport, modules: list[Any]) -> RequestClient:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = request_server_url(transport, port)
        server = create_request_server(modules, request_server_config(transport, url))
        server.start()
        client = RequestClientFactory.create(url)
        started.append((server, client))
        return client

    yield start
    for server, client in started:
        client.close()
        server.close()


@pytest.mark.parametrize(
    ("supported_transfer_mode", "loaded"),
    [
        ("lmcache_driven", ["lmcache_driven"]),
        ("engine_driven", ["engine_driven"]),
        ("auto", ["lmcache_driven", "engine_driven"]),
    ],
)
def test_get_server_info_reports_device_and_loaded_modes(
    server_device: Callable[[str], None],
    supported_transfer_mode: ServerMode,
    loaded: list[str],
) -> None:
    server_device("cpu")

    info = _management(supported_transfer_mode).get_server_info()

    assert info == ServerCapabilities(
        device_type="cpu", supported_transfer_modes=loaded
    )


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
@pytest.mark.parametrize(
    ("server_device_type", "supported_transfer_mode", "expected"),
    [
        # A CPU-only server cannot open CUDA IPC handles (#4186).
        ("cpu", "auto", ENGINE_DRIVEN),
        # A server on the worker's device type keeps today's default.
        ("cuda", "auto", LMCACHE_DRIVEN),
        # ... unless it did not load the lmcache-driven module.
        ("cuda", "engine_driven", ENGINE_DRIVEN),
    ],
)
def test_auto_follows_what_the_server_can_serve(
    server_device: Callable[[str], None],
    serve: Serve,
    transport: RequestTransport,
    server_device_type: str,
    supported_transfer_mode: ServerMode,
    expected: MPTransferMode,
) -> None:
    server_device(server_device_type)
    client = serve(transport, [_management(supported_transfer_mode)])

    assert resolve_mp_transfer_mode("cuda", client) is expected


def test_auto_rejects_a_cpu_server_that_loads_only_lmcache_driven(
    server_device: Callable[[str], None], serve: Serve
) -> None:
    server_device("cpu")
    client = serve("zmq", [_management("lmcache_driven")])

    with pytest.raises(ValueError, match="--supported-transfer-mode auto"):
        resolve_mp_transfer_mode("cuda", client)


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
def test_auto_keeps_the_device_default_against_a_server_without_the_rpc(
    serve: Serve, transport: RequestTransport
) -> None:
    # A server that predates get_server_info: ZMQ answers operation_not_enabled
    # and gRPC UNIMPLEMENTED.
    client = serve(transport, [])

    assert resolve_mp_transfer_mode("cuda", client) is LMCACHE_DRIVEN


def test_auto_keeps_the_device_default_when_the_server_never_answers() -> None:
    # Older ZMQ servers leave an unknown operation unanswered.
    client = _answering(error=TimeoutError("Future result not available"))

    assert resolve_mp_transfer_mode("cuda", client) is LMCACHE_DRIVEN


@pytest.mark.parametrize(
    "error",
    [
        MPRequestError("INTERNAL", "boom"),
        MPServerUnavailableError("UNAVAILABLE", "connection refused"),
    ],
)
def test_auto_raises_other_server_failures(error: BaseException) -> None:
    client = _answering(error=error)

    with pytest.raises(type(error)):
        resolve_mp_transfer_mode("cuda", client)


def test_explicit_mode_is_not_checked_against_the_server() -> None:
    client = _answering(ServerCapabilities("cpu", ["engine_driven"]))

    mode = resolve_mp_transfer_mode("cuda", client, "lmcache_driven")

    assert mode is LMCACHE_DRIVEN
    client.get_server_info.assert_not_called()


def test_auto_does_not_ask_when_the_device_defaults_to_engine_driven() -> None:
    client = _answering(ServerCapabilities("cuda", ["lmcache_driven"]))

    assert resolve_mp_transfer_mode("cpu", client) is ENGINE_DRIVEN
    client.get_server_info.assert_not_called()
