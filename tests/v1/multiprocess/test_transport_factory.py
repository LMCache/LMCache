# SPDX-License-Identifier: Apache-2.0
"""Tests for scheme-based multiprocess request client selection."""

# Standard
from unittest.mock import MagicMock
import importlib

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.transport import grpc_impl, zmq_impl
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server


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
    create.assert_called_once_with(normalized_url, context=context)


@pytest.mark.parametrize("server_url", ["", "grpc://", "http://localhost:5555"])
def test_factory_rejects_invalid_or_unsupported_urls(server_url: str) -> None:
    with pytest.raises(ValueError):
        RequestClientFactory.create(server_url)


def test_request_server_factory_passes_grpc_service_registrars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grpc_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl.server"
    )
    server = MagicMock(name="grpc_request_server")
    build = MagicMock(return_value=server)
    modules: list[EngineModule] = []
    config = MPServerConfig(transport="grpc")
    grpc_registrars = (MagicMock(name="grpc_registrar"),)
    zmq_registrars = (MagicMock(name="zmq_registrar"),)
    monkeypatch.setattr(grpc_server_module, "build_grpc_request_server", build)

    result = create_request_server(
        modules,
        config,
        grpc_service_registrars=grpc_registrars,
        zmq_service_registrars=zmq_registrars,
    )

    assert result is server
    build.assert_called_once_with(
        modules,
        config,
        service_registrars=grpc_registrars,
    )


def test_request_server_factory_passes_zmq_service_registrars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zmq_server_module = importlib.import_module(
        "lmcache.v1.multiprocess.transport.zmq_impl.server"
    )
    server = MagicMock(name="zmq_request_server")
    build = MagicMock(return_value=server)
    modules: list[EngineModule] = []
    config = MPServerConfig(transport="zmq")
    grpc_registrars = (MagicMock(name="grpc_registrar"),)
    zmq_registrars = (MagicMock(name="zmq_registrar"),)
    monkeypatch.setattr(zmq_server_module, "build_zmq_request_server", build)

    result = create_request_server(
        modules,
        config,
        grpc_service_registrars=grpc_registrars,
        zmq_service_registrars=zmq_registrars,
    )

    assert result is server
    build.assert_called_once_with(
        modules,
        config,
        service_registrars=zmq_registrars,
    )
