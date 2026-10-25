# SPDX-License-Identifier: Apache-2.0
"""Tests for scheme-based multiprocess request client selection."""

# Standard
from pathlib import Path
from unittest.mock import MagicMock
import importlib

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.transport import grpc_impl, server_factory, zmq_impl
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

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
    create.assert_called_once_with(normalized_url, context=context)


def test_factory_rejects_zmq_api_key_until_supported() -> None:
    """A configured credential cannot be silently ignored on another transport."""
    with pytest.raises(ValueError, match="not yet supported by ZMQ"):
        RequestClientFactory.create("tcp://localhost:5555", api_key="secret")


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
    create.assert_called_once_with("tcp://localhost:5555", context=None)


@pytest.mark.parametrize("source", ["cli", *KEY_ENV_VARS])
def test_request_server_factory_preserves_zmq_with_http_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    source: str,
) -> None:
    """HTTP-only key configurations keep ZMQ available with an honest warning."""
    key_file = tmp_path / "key"
    key_file.write_text("http-only-secret\n")
    config = MPServerConfig(transport="zmq")
    if source == "cli":
        config.api_key_files = [str(key_file)]
    else:
        monkeypatch.setenv(
            source, str(key_file) if source.endswith("_FILE") else "http-only-secret"
        )
    zmq_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.zmq_impl.server"
    )
    build = MagicMock()
    monkeypatch.setattr(zmq_server_module, "build_zmq_request_server", build)
    server_factory.logger.addHandler(caplog.handler)
    try:
        result = create_request_server([], config)
    finally:
        server_factory.logger.removeHandler(caplog.handler)

    assert result is build.return_value
    build.assert_called_once_with([], config)
    assert "ZMQ requests remain unauthenticated" in caplog.text
    assert "http-only-secret" not in caplog.text


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
