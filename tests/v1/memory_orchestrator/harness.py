# SPDX-License-Identifier: Apache-2.0
"""Spawn real ``python -m lmcache.v1.memory_orchestrator`` processes for tests."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time

# First Party
from lmcache.v1 import memory_orchestrator

REGION_ID = "pool-test"
STARTUP_TIMEOUT_S = 60.0
EXIT_TIMEOUT_S = 30.0

# Root of the ``lmcache`` tree under test; spawned orchestrators import it too.
SOURCE_ROOT = Path(memory_orchestrator.__file__).resolve().parents[3]


def free_port() -> int:
    """Return a TCP port that was free a moment ago on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Orchestrator:
    """A spawned orchestrator process and its startup marker."""

    process: subprocess.Popen[bytes]
    state_dir: Path
    region_id: str
    log_path: Path | None

    @property
    def marker_path(self) -> Path:
        return self.state_dir / f"{self.region_id}.marker"

    def marker(self) -> dict[str, object]:
        return json.loads(self.marker_path.read_text())

    @property
    def endpoint(self) -> str:
        return str(self.marker()["endpoint"])

    def log(self) -> str:
        return self.log_path.read_text() if self.log_path else ""

    def wait_for_marker(self) -> None:
        """Wait until this process wrote its marker; raise if it exited first."""
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"orchestrator exited with {self.process.returncode}:\n{self.log()}"
                )
            try:
                if self.marker()["pid"] == self.process.pid:
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # not created yet, or being written
            time.sleep(0.05)
        self.kill()
        raise RuntimeError(
            f"no startup marker within {STARTUP_TIMEOUT_S}s:\n{self.log()}"
        )

    def stop(self) -> int | None:
        """SIGTERM a running process and return its exit code; no-op otherwise."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.kill()
        return self.process.returncode

    def kill(self) -> None:
        """SIGKILL the process, as a crash would, and reap it."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


def spawn_orchestrator(
    state_dir: Path,
    *,
    capacity_bytes: int,
    alignment_bytes: int = 4096,
    visibility_mode: str = "coherent",
    max_batch_entries: int | None = None,
    port: int = 0,
    region_id: str = REGION_ID,
    log_path: Path | None = None,
) -> Orchestrator:
    """Start an orchestrator process without waiting for it to come up.

    Args:
        state_dir: Directory of the startup marker; created if missing.
        capacity_bytes: Region capacity.
        alignment_bytes: Extent alignment.
        visibility_mode: Region visibility mode.
        max_batch_entries: ``--max-batch-entries``; the server default if None.
        port: Port to listen on; 0 picks one.
        region_id: Region identity.
        log_path: File that receives stdout and stderr; inherited if None.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "lmcache.v1.memory_orchestrator",
        "--region-id",
        region_id,
        "--capacity-bytes",
        str(capacity_bytes),
        "--alignment",
        str(alignment_bytes),
        "--visibility-mode",
        visibility_mode,
        "--listen",
        f"127.0.0.1:{port}",
        "--state-dir",
        str(state_dir),
    ]
    if max_batch_entries is not None:
        argv += ["--max-batch-entries", str(max_batch_entries)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if path
    )
    if log_path is None:
        process = subprocess.Popen(argv, env=env)
    else:
        with open(log_path, "w") as log:
            process = subprocess.Popen(
                argv, env=env, stdout=log, stderr=subprocess.STDOUT
            )
    return Orchestrator(process, state_dir, region_id, log_path)


def start_orchestrator(state_dir: Path, **kwargs: object) -> Orchestrator:
    """Spawn an orchestrator and wait until it serves; see ``spawn_orchestrator``.

    Raises:
        RuntimeError: The process exited or did not come up in time.
    """
    orchestrator = spawn_orchestrator(state_dir, **kwargs)  # type: ignore[arg-type]
    orchestrator.wait_for_marker()
    return orchestrator


@contextmanager
def running_orchestrator(state_dir: Path, **kwargs: object) -> Iterator[Orchestrator]:
    """Run an orchestrator for the duration of a ``with`` block."""
    orchestrator = start_orchestrator(state_dir, **kwargs)
    try:
        yield orchestrator
    finally:
        orchestrator.stop()
