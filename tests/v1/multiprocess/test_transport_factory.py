# SPDX-License-Identifier: Apache-2.0
"""Tests for scheme-based multiprocess request client selection."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport import factory as transport_factory
from lmcache.v1.multiprocess.transport import grpc_impl, zmq_impl
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


@pytest.mark.parametrize(
    ("server_url", "normalized_url"),
    [
        ("localhost:5555", "tcp://localhost:5555"),
        ("TCP://localhost:5555", "tcp://localhost:5555"),
        ("ipc:///tmp/lmcache.sock", "ipc:///tmp/lmcache.sock"),
        ("inproc://lmcache", "inproc://lmcache"),
    ],
)
def test_factory_temporarily_forces_zmq_schemes_to_grpc(
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


def test_configured_zmq_path_remains_available_without_ci_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock(name="zmq_request_client")
    create = MagicMock(return_value=client)
    monkeypatch.setattr(
        transport_factory,
        "effective_transport",
        lambda configured_transport: configured_transport,
    )
    monkeypatch.setattr(zmq_impl, "create_request_client", create)

    result = RequestClientFactory.create("tcp://localhost:5555")

    assert result is client
    create.assert_called_once_with("tcp://localhost:5555", context=None)


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
