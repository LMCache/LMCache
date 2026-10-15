# SPDX-License-Identifier: Apache-2.0
"""Tests for scheme-based multiprocess request client selection."""

# Standard
from pathlib import Path
from unittest.mock import MagicMock
import importlib

# Third Party
import grpc
import pytest

# First Party
from lmcache.v1.client.transport import grpc_impl, zmq_impl
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.transport import server_factory
from lmcache.v1.multiprocess.transport.grpc_impl.native_server import (
    GrpcNativeMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native

KEY_ENV_VARS = (
    "LMCACHE_API_KEY",
    "LMCACHE_API_KEY_FILE",
    "LMCACHE_HTTP_API_KEY",
    "LMCACHE_HTTP_API_KEY_FILE",
)


@pytest.fixture(autouse=True)
def clear_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not inherit credentials from the test runner's environment."""
    for variable in KEY_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize(
    ("server_url", "normalized_url"),
    [
        ("localhost:5555", "tcp://localhost:5555"),
        ("TCP://localhost:5555", "tcp://localhost:5555"),
        ("ipc:///tmp/lmcache.sock", "ipc:///tmp/lmcache.sock"),
        ("inproc://lmcache", "inproc://lmcache"),
    ],
)
def test_factory_selects_zmq_by_scheme(
    monkeypatch: pytest.MonkeyPatch,
    server_url: str,
    normalized_url: str,
) -> None:
    client = MagicMock(name="zmq_request_client")
    create = MagicMock(return_value=client)
    context = object()
    monkeypatch.setattr(zmq_impl, "create_request_client", create)

    result = RequestClientFactory.create(server_url, context=context)

    assert result is client
    create.assert_called_once_with(normalized_url, context=context, api_key=None)


def test_factory_passes_explicit_zmq_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic clients can provide one request credential explicitly."""
    client = MagicMock(name="zmq_request_client")
    create = MagicMock(return_value=client)
    monkeypatch.setattr(zmq_impl, "create_request_client", create)

    result = RequestClientFactory.create("tcp://localhost:5555", api_key="secret")

    assert result is client
    create.assert_called_once_with(
        "tcp://localhost:5555", context=None, api_key="secret"
    )


@pytest.mark.parametrize("variable", KEY_ENV_VARS)
def test_zmq_client_ignores_http_key_environment(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """Inherited HTTP keys, including unreadable paths, do not affect ZMQ."""
    monkeypatch.setenv(variable, "http-only-key-or-nonexistent-file")
    create = MagicMock()
    monkeypatch.setattr(zmq_impl, "create_request_client", create)

    result = RequestClientFactory.create("tcp://localhost:5555")

    assert result is create.return_value
    create.assert_called_once_with("tcp://localhost:5555", context=None, api_key=None)


@pytest.mark.parametrize("source", ["cli", *KEY_ENV_VARS])
def test_request_server_factory_builds_zmq_server_with_configured_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
) -> None:
    """Configured keys reach the ZMQ server instead of being downgraded."""
    key_file = tmp_path / "key"
    key_file.write_text("request-secret\n")
    config = MPServerConfig(transport="zmq")
    if source == "cli":
        config.api_key_files = [str(key_file)]
    else:
        monkeypatch.setenv(
            source, str(key_file) if source.endswith("_FILE") else "request-secret"
        )
    zmq_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.zmq_impl.server"
    )
    build = MagicMock()
    monkeypatch.setattr(zmq_server_module, "build_zmq_request_server", build)

    result = create_request_server([], config)

    assert result is build.return_value
    build.assert_called_once_with([], config)


@pytest.mark.parametrize(
    ("server_url", "normalized_url"),
    [
        ("grpc://localhost:5555", "grpc://localhost:5555"),
        ("GRPC+UNIX:///tmp/lmcache.sock", "grpc+unix:///tmp/lmcache.sock"),
    ],
)
def test_factory_selects_grpc_by_scheme(
    monkeypatch: pytest.MonkeyPatch,
    server_url: str,
    normalized_url: str,
) -> None:
    client = MagicMock(name="grpc_request_client")
    create = MagicMock(return_value=client)
    context = object()
    monkeypatch.setattr(grpc_impl, "create_request_client", create)

    result = RequestClientFactory.create(server_url, context=context)

    assert result is client
    create.assert_called_once_with(normalized_url, context=context, api_key=None)


def test_factory_passes_explicit_grpc_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programmatic gRPC clients can provide a request credential."""
    client = MagicMock(name="grpc_request_client")
    create = MagicMock(return_value=client)
    monkeypatch.setattr(grpc_impl, "create_request_client", create)

    result = RequestClientFactory.create("grpc://localhost:5555", api_key="secret")

    assert result is client
    create.assert_called_once_with(
        "grpc://localhost:5555", context=None, api_key="secret"
    )


def test_zmq_client_rejects_empty_explicit_api_key() -> None:
    """An empty explicit credential is a configuration error, not anonymous access."""
    with pytest.raises(ValueError, match="must not be empty"):
        zmq_impl.create_request_client("tcp://localhost:5555", api_key="")


@pytest.mark.parametrize("server_url", ["", "grpc://", "http://localhost:5555"])
def test_factory_rejects_invalid_or_unsupported_urls(server_url: str) -> None:
    with pytest.raises(ValueError):
        RequestClientFactory.create(server_url)


def test_request_server_factory_selects_grpc_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grpc_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl.server_builder"
    )
    server = MagicMock(name="grpc_request_server")
    build = MagicMock(return_value=server)
    modules: list[EngineModule] = []
    config = MPServerConfig(transport="grpc")
    monkeypatch.setattr(grpc_server_module, "build_grpc_request_server", build)

    result = create_request_server(modules, config)

    assert result is server
    build.assert_called_once_with(modules, config)


def test_request_server_factory_builds_grpc_server_with_configured_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gRPC server with configured keys is built, not rejected."""
    monkeypatch.setenv("LMCACHE_API_KEY", "secret")
    grpc_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl.server_builder"
    )
    build = MagicMock()
    monkeypatch.setattr(grpc_server_module, "build_grpc_request_server", build)

    result = create_request_server([], MPServerConfig(transport="grpc"))

    assert result is build.return_value
    build.assert_called_once()


def test_request_server_factory_selects_zmq_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zmq_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.zmq_impl.server"
    )
    server = MagicMock(name="zmq_request_server")
    build = MagicMock(return_value=server)
    modules: list[EngineModule] = []
    config = MPServerConfig(transport="zmq")
    monkeypatch.setattr(zmq_server_module, "build_zmq_request_server", build)

    result = create_request_server(modules, config)

    assert result is server
    build.assert_called_once_with(modules, config)


@requires_pygrpc_native
@pytest.mark.parametrize("source", ["cli", *KEY_ENV_VARS])
def test_native_grpc_enforces_configured_authentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    """Native mode authenticates with keys from every shared server source."""
    key_file = tmp_path / "key"
    key_file.write_text("native-secret\n")
    config = MPServerConfig(
        transport="grpc", grpc_server_mode="native", host="127.0.0.1", port=0
    )
    if source == "cli":
        config.api_key_files = [str(key_file)]
    else:
        monkeypatch.setenv(
            source, str(key_file) if source.endswith("_FILE") else "native-secret"
        )
    server = create_request_server([], config)
    assert isinstance(server, GrpcNativeMultiprocessServer)
    server.start()
    # Raw channel: the request client would read the key from the environment.
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    noop = channel.unary_unary("/lmcache.mp.ManagementService/Noop")
    try:
        with pytest.raises(grpc.RpcError) as rejected:
            noop(b"", timeout=3)
        assert rejected.value.code() is grpc.StatusCode.UNAUTHENTICATED
        # No module serves noop, so getting past authentication means UNIMPLEMENTED.
        with pytest.raises(grpc.RpcError) as unserved:
            noop(b"", timeout=3, metadata=[("authorization", "Bearer native-secret")])
        assert unserved.value.code() is grpc.StatusCode.UNIMPLEMENTED
    finally:
        channel.close()
        server.close()
