# SPDX-License-Identifier: Apache-2.0
"""A failed handler must show up in the server's own log."""

# Standard
from typing import Any, cast
import logging
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.errors import MPRequestError
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server


class FailingModule:
    @request_handler(HandlerType.BLOCKING)
    def clear(self, force: bool = False) -> None:
        raise RuntimeError("handler failed on purpose")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("mode", ["sync", "aio"])
def test_failed_handler_is_logged(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    grpc_logger = logging.getLogger("grpc")
    monkeypatch.setattr(grpc_logger, "handlers", list(grpc_logger.handlers))
    monkeypatch.setattr(grpc_logger, "propagate", grpc_logger.propagate)
    monkeypatch.setattr(grpc_logger, "level", grpc_logger.level)
    monkeypatch.setenv("LMCACHE_LOG_LEVEL", "ERROR")

    port = free_port()
    config = MPServerConfig(
        transport="grpc",
        host="127.0.0.1",
        port=port,
        grpc_server_mode=cast(Any, mode),
    )
    server = create_request_server([cast(Any, FailingModule())], config)
    server.start()
    client = RequestClientFactory.create(f"grpc://127.0.0.1:{port}")
    try:
        with pytest.raises(MPRequestError):
            client.clear(False).result(timeout=10)
    finally:
        client.close()
        server.close()

    assert "RuntimeError: handler failed on purpose" in capsys.readouterr().err
