# SPDX-License-Identifier: Apache-2.0
"""A real gRPC server process exits once it has shut down."""

# Standard
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
from lmcache.v1.multiprocess.transport.grpc_impl.client import GrpcMultiprocessClient
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native
import lmcache

# Spawned servers must import the same lmcache tree as this process: wheel or checkout.
SOURCE_ROOT = Path(lmcache.__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="gRPC runtime metrics, which kept the process alive, are Linux-only",
)


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_until_ready(args: list[str], port: int, log: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if part
    )
    with open(log, "wb") as out:
        server = subprocess.Popen(
            [sys.executable, "-m", *args, "--host", "127.0.0.1", "--port", str(port)]
            + ["--l1-size-gb", "0.1", "--eviction-policy", "LRU"],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    client = GrpcMultiprocessClient(f"grpc://127.0.0.1:{port}")  # type: ignore[abstract]
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert server.poll() is None, log.read_text()
            try:
                if client.ping(1).result(2):
                    return server
            except (TimeoutError, ConnectionError):
                pass
    finally:
        client.close()
    server.kill()
    raise AssertionError("the server did not start: " + log.read_text())


def _assert_exits(server: subprocess.Popen, sig: signal.Signals, log: Path) -> None:
    server.send_signal(sig)
    try:
        server.wait(30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(10)
        pytest.fail("the server shut down but did not exit: " + log.read_text())


@pytest.mark.parametrize(
    "mode", ["sync", "aio", pytest.param("native", marks=requires_pygrpc_native)]
)
def test_server_exits_on_sigterm(mode: str, tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    args = ["lmcache.v1.multiprocess.server", "--transport", "grpc"]
    server = _run_until_ready(args + ["--grpc-server-mode", mode], _unused_port(), log)
    _assert_exits(server, signal.SIGTERM, log)


def test_http_server_exits_on_ctrl_c(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    args = ["lmcache.cli.main", "server", "--transport", "grpc"]
    args += ["--http-port", str(_unused_port())]
    server = _run_until_ready(args, _unused_port(), log)
    _assert_exits(server, signal.SIGINT, log)
