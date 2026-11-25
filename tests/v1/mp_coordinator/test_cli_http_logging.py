# SPDX-License-Identifier: Apache-2.0
"""The HTTP logging flags of ``lmcache coordinator`` and ``lmcache server``,
end to end.

The unit tests stop at the arguments handed to uvicorn, so they would still
pass if uvicorn ignored them. These cases start the real processes, send one
request, and read what the process logged.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Third Party
import pytest

_START_TIMEOUT = 120.0
_STOP_TIMEOUT = 30.0
_PROBE_PATH = "/http-logging-probe"
_ACCESS_LINE = f'"GET {_PROBE_PATH} HTTP/1.1" 404'
_STARTUP_LINE = "Uvicorn running on"
_API_KEY_ENV_VARS = (
    "LMCACHE_API_KEY",
    "LMCACHE_API_KEY_FILE",
    "LMCACHE_HTTP_API_KEY",
    "LMCACHE_HTTP_API_KEY_FILE",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _coordinator_command(http_port: int) -> list[str]:
    return ["coordinator", "--port", str(http_port), "--disable-metrics"]


def _server_command(http_port: int) -> list[str]:
    return [
        "server",
        "--port",
        str(_free_port()),
        "--http-port",
        str(http_port),
        "--l1-size-gb",
        "0.0625",
        "--eviction-policy",
        "LRU",
    ]


def _probe(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _probe_when_serving(url: str, process: subprocess.Popen[bytes]) -> int:
    deadline = time.monotonic() + _START_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"process exited before serving (return code {process.returncode})"
            )
        try:
            return _probe(url)
        except urllib.error.URLError:
            time.sleep(0.2)
    raise AssertionError(f"{url} was not served within the timeout")


def _log_after_one_request(command: list[str], http_port: int, log_path: Path) -> str:
    """Run ``lmcache <command>``, request an unknown path once, stop the
    process, and return everything it logged."""
    env = {k: v for k, v in os.environ.items() if k not in _API_KEY_ENV_VARS}
    env.update(LMCACHE_DISABLE_BANNER="1", LMCACHE_TRACK_USAGE="false")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "lmcache.cli.main", *command],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{http_port}{_PROBE_PATH}"
            assert _probe_when_serving(url, process) == 404
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return log_path.read_text(errors="replace")


@pytest.mark.parametrize(
    "build_command",
    [
        pytest.param(_coordinator_command, id="coordinator"),
        pytest.param(_server_command, id="server"),
    ],
)
@pytest.mark.parametrize(
    ("flags", "logs_access", "logs_startup"),
    [
        pytest.param([], True, True, id="defaults"),
        pytest.param(["--no-http-access-log"], False, True, id="no-access-log"),
        pytest.param(["--http-log-level", "warning"], False, False, id="warning"),
    ],
)
def test_http_logging_flags_control_what_is_logged(
    tmp_path: Path,
    build_command: Callable[[int], list[str]],
    flags: list[str],
    logs_access: bool,
    logs_startup: bool,
) -> None:
    http_port = _free_port()
    log = _log_after_one_request(
        [*build_command(http_port), *flags], http_port, tmp_path / "process.log"
    )

    assert (_ACCESS_LINE in log) is logs_access
    assert (_STARTUP_LINE in log) is logs_startup
