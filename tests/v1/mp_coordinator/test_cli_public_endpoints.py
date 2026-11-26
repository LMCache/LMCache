# SPDX-License-Identifier: Apache-2.0
"""The coordinator CLI's public-path setting, end to end.

The middleware tests call the app directly, so they would still pass if the CLI
dropped `--public-endpoint` or `create_app` forgot to forward it. These cases
start the real `lmcache coordinator` process and probe it over HTTP instead:
default list, an explicit replacement, and the explicit empty list.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_KEY = "fleet-key"
_START_TIMEOUT = 60.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _status(url: str, key: str | None = None) -> int:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _wait_until_serving(base_url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + _START_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"coordinator exited before serving (return code {process.returncode})"
            )
        try:
            _status(f"{base_url}/healthz")
            return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise AssertionError(f"coordinator did not serve {base_url} within the timeout")


@contextmanager
def _coordinator(tmp_path: Path, extra_args: list[str]) -> Iterator[str]:
    """Run the coordinator CLI with an API key and yield its base URL."""
    key_file = tmp_path / "fleet-key"
    key_file.write_text(_KEY + "\n", encoding="utf-8")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, LMCACHE_DISABLE_BANNER="1")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "coordinator",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key-file",
            str(key_file),
            *extra_args,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_until_serving(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


class TestDefaultList:
    def test_only_the_probe_is_public(self, tmp_path: Path) -> None:
        with _coordinator(tmp_path, []) as base_url:
            assert _status(f"{base_url}/healthz") == 200
            assert _status(f"{base_url}/metrics") == 401

    def test_the_key_reaches_metrics(self, tmp_path: Path) -> None:
        with _coordinator(tmp_path, []) as base_url:
            assert _status(f"{base_url}/metrics", _KEY) == 200


class TestExplicitList:
    def test_naming_both_paths_publishes_both(self, tmp_path: Path) -> None:
        args = [
            "--public-endpoint",
            "/healthz",
            "--public-endpoint",
            "/metrics",
        ]
        with _coordinator(tmp_path, args) as base_url:
            assert _status(f"{base_url}/healthz") == 200
            assert _status(f"{base_url}/metrics") == 200

    def test_the_list_replaces_the_default(self, tmp_path: Path) -> None:
        """Naming only /metrics must not leave the probe public."""
        with _coordinator(tmp_path, ["--public-endpoint", "/metrics"]) as base_url:
            assert _status(f"{base_url}/healthz") == 401
            assert _status(f"{base_url}/metrics") == 200
            assert _status(f"{base_url}/healthz", _KEY) == 200


class TestEmptyList:
    def test_every_path_requires_the_key(self, tmp_path: Path) -> None:
        with _coordinator(tmp_path, ["--no-public-endpoints"]) as base_url:
            assert _status(f"{base_url}/healthz") == 401
            assert _status(f"{base_url}/metrics") == 401
            assert _status(f"{base_url}/healthz", _KEY) == 200
            assert _status(f"{base_url}/metrics", _KEY) == 200
