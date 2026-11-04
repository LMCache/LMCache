# SPDX-License-Identifier: Apache-2.0
"""A gRPC client against a real ``lmcache`` server process that crashes."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport.errors import MPServerUnavailableError
from lmcache.v1.multiprocess.transport.grpc_impl.client import GrpcMultiprocessClient
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native
import lmcache

# Spawned servers must import the same lmcache tree as this process: wheel or checkout.
SOURCE_ROOT = Path(lmcache.__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="uses SIGSTOP on a real server"
)


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start(mode: str, port: int, log: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if part
    )
    args = ["lmcache.v1.multiprocess.server", "--transport", "grpc"]
    args += ["--grpc-server-mode", mode, "--host", "127.0.0.1", "--port", str(port)]
    args += ["--l1-size-gb", "0.1", "--eviction-policy", "LRU"]
    with open(log, "ab") as out:
        return subprocess.Popen(
            [sys.executable, "-m", *args],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )


def _wait_ready(
    client: GrpcMultiprocessClient, server: subprocess.Popen, log: Path
) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert server.poll() is None, log.read_text()
        try:
            if client.ping(1).result(2):
                return
        except (TimeoutError, ConnectionError):
            pass
    pytest.fail("the server did not start: " + log.read_text())


@pytest.fixture
def servers() -> Iterator[list[subprocess.Popen]]:
    started: list[subprocess.Popen] = []
    yield started
    for server in started:
        if server.poll() is None:
            server.kill()
            server.wait(10)


@pytest.mark.parametrize(
    "mode", ["sync", "aio", pytest.param("native", marks=requires_pygrpc_native)]
)
def test_client_recovers_after_the_server_crashes(
    mode: str, tmp_path: Path, servers: list[subprocess.Popen]
) -> None:
    """The call in flight when the server dies fails, a call made while it is
    down waits for the restarted server, and the same client keeps working."""
    port = _unused_port()
    log = tmp_path / "server.log"
    servers.append(server := _start(mode, port, log))
    client = GrpcMultiprocessClient(f"grpc://127.0.0.1:{port}")  # type: ignore[abstract]
    try:
        _wait_ready(client, server, log)
        # Stopping the process first keeps the call in flight when it dies.
        server.send_signal(signal.SIGSTOP)
        in_flight = client.ping(1)
        time.sleep(0.5)  # let the request reach the stopped server
        server.kill()
        server.wait(10)
        with pytest.raises(MPServerUnavailableError):
            in_flight.result(10)

        waiting = client.ping(1)
        servers.append(_start(mode, port, log))
        assert waiting.result(90), log.read_text()
        assert client.ping(1).result(5)
    finally:
        client.close()
