# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for graceful MP server shutdown on SIGTERM."""

# Standard
from pathlib import Path
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

# Third Party
import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[3]
STARTUP_TIMEOUT_SECONDS = 90.0
SHUTDOWN_TIMEOUT_SECONDS = 60.0
SHM_SIZE_BYTES = 64 << 20


def _unused_tcp_port() -> int:
    """Reserve an unused local TCP port for the server subprocess."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not torch.cuda.is_available(),
    reason="Mixed-allocator POSIX SHM requires Linux and an accelerator device",
)
def test_cli_sigterm_unlinks_shm_pool(tmp_path: Path) -> None:
    """A ready MP CLI must unlink its named SHM pool after SIGTERM."""
    shm_dir = Path("/dev/shm")
    if shutil.disk_usage(shm_dir).free < 2 * SHM_SIZE_BYTES:
        pytest.skip("insufficient /dev/shm capacity for the regression test")

    log_path = tmp_path / "mp-server.log"
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), python_path) if part
    )
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(_unused_tcp_port()),
        "--l1-size-gb",
        "0.0625",
        "--no-l1-use-lazy",
        "--supported-transfer-mode",
        "auto",
        "--eviction-policy",
        "LRU",
        "--disable-observability",
    ]

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    shm_path = shm_dir / f"lmcache_l1_pool_{process.pid}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"MP server exited before becoming ready:\n"
                    f"{log_path.read_text(errors='replace')}"
                )
            log = log_path.read_text(errors="replace")
            if shm_path.exists() and "LMCache cache server is running" in log:
                break
            time.sleep(0.2)
        else:
            pytest.fail(
                f"MP server did not become ready:\n"
                f"{log_path.read_text(errors='replace')}"
            )

        process.send_signal(signal.SIGTERM)
        return_code = process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)

        assert return_code == 0
        assert not shm_path.exists(), "named SHM pool leaked after SIGTERM"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if shm_path.exists():
            shm_path.unlink()
