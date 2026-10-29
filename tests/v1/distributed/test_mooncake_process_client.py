# SPDX-License-Identifier: Apache-2.0
"""Exercise real worker processes and shared mappings without CUDA or Mooncake.

Run on Linux: uv run --with pytest -m pytest -q test_process_native_client.py
The fake worker is this same file with --fake-worker <private-socket-fd>.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from types import SimpleNamespace
import ctypes
import importlib.util
import json
import mmap
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

# Third Party
import pytest

# The adapter package imports compiled Bitmap. Load this component directly so
# the process/SHM tests exercise its real code without requiring native builds.
_source = (
    Path(__file__).resolve().parents[3]
    / "lmcache/v1/distributed/l2_adapters/mooncake_process_client.py"
)
sys.path.insert(0, str(_source.parents[4]))
_spec = importlib.util.spec_from_file_location("mooncake_process_client", _source)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
ProcessNativeClient = _module.ProcessNativeClient
run_worker = _module.run_worker

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux process/FD semantics"
)


class FakeNative:
    """Native-shaped asynchronous results, while worker protocol stays real."""

    def __init__(self, bootstrap, child_base, arena):
        self._config = bootstrap["setup_config"]
        if self._config.get("fake_mode") == "startup_hang":
            time.sleep(120)
        self._base, self._arena = child_base, arena
        self._fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lock = threading.Lock()
        self._results = []
        self._next_id = 1

    def event_fd(self):
        return self._fd

    def drain_completions(self):
        try:
            os.eventfd_read(self._fd)
        except BlockingIOError:
            pass
        with self._lock:
            result, self._results = self._results, []
        return result

    def _submit(self, kind, keys, buffers=None):
        task_id = self._next_id
        self._next_id += 1
        mode = self._config.get("fake_mode", "normal")
        if mode == "crash":
            os._exit(91)
        if mode == "hang":
            return task_id
        flags = [True] * len(keys)
        ok = True
        if mode == "false_" + kind:
            ok, flags = False, None
        elif mode == "partial_" + kind:
            flags[-1] = False
        if buffers is not None:
            observations = []
            for index, (key, buffer) in enumerate(zip(keys, buffers, strict=True)):
                address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
                observations.append(
                    {
                        "offset": address - self._base,
                        "length": buffer.nbytes,
                        "hex": bytes(buffer).hex(),
                    }
                )
                if kind == "get" and (flags is None or flags[index]):
                    payload = bytes.fromhex(key.removeprefix("payload:"))
                    buffer[:] = payload
            if kind == "set":
                Path(self._config["set_observations"]).write_text(
                    json.dumps(observations)
                )
        with self._lock:
            self._results.append((task_id, ok, "simulated" if not ok else "", flags))
            os.eventfd_write(self._fd, 1)
        return task_id

    def submit_batch_exists(self, keys):
        return self._submit("exists", keys)

    def submit_batch_get(self, keys, buffers):
        return self._submit("get", keys, buffers)

    def submit_batch_set(self, keys, buffers):
        return self._submit("set", keys, buffers)

    def submit_batch_delete(self, keys):
        return self._submit("delete", keys)


class Notifier:
    def __init__(self):
        self._fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.probe = lambda: None
        self.observations = []

    def fileno(self):
        return self._fd

    def notify(self):
        self.observations.append(self.probe())
        os.eventfd_write(self._fd, 1)

    def consume(self):
        try:
            os.eventfd_read(self._fd)
        except BlockingIOError:
            pass

    def close(self):
        os.close(self._fd)


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("Worker did not reach the expected state before deadline")


def stop_worker(pid):
    os.kill(pid, signal.SIGSTOP)
    status = Path(f"/proc/{pid}/stat")
    wait_until(lambda: status.read_text().split(") ", 1)[1].split()[0] in {"T", "t"})


def completions(client, task_ids):
    results = {}

    def drain():
        for item in client.drain_completions():
            assert item[0] not in results, "A task completed twice"
            results[item[0]] = item
        return task_ids <= results.keys()

    wait_until(drain)
    return results


@pytest.fixture
def arena():
    # Container /tmp may be overlayfs, whose mmap backing inode differs from
    # fstat(open(path)). Exercise the actual shared-memory contract on tmpfs.
    with tempfile.TemporaryDirectory(
        prefix="lmcache-process-test-", dir="/dev/shm"
    ) as root:
        path = Path(root) / "lmcache_l1_pool_protocol_test"
        with path.open("wb") as stream:
            stream.truncate(65536)
        with path.open("r+b") as stream:
            region = mmap.mmap(stream.fileno(), 65536, access=mmap.ACCESS_WRITE)
        base = ctypes.addressof(ctypes.c_ubyte.from_buffer(region))
        try:
            yield SimpleNamespace(path=path, region=region, base=base, size=len(region))
        finally:
            region.close()


@pytest.fixture
def make_client(arena, tmp_path):
    clients = []

    def make(mode="normal", operation_timeout=2, **kwargs):
        notifier = Notifier()
        wait_ready = kwargs.pop("wait_ready", True)
        client = ProcessNativeClient(
            {"fake_mode": mode, "set_observations": str(tmp_path / "set.json")},
            2,
            None,
            arena.base,
            arena.size,
            arena.path.name,
            notifier,
            operation_timeout=operation_timeout,
            startup_timeout=kwargs.pop("startup_timeout", 5),
            kill_timeout=2,
            restart_interval=0.05,
            shm_root=arena.path.parent,
            worker_command=kwargs.pop(
                "worker_command",
                [sys.executable, str(Path(__file__).resolve()), "--fake-worker"],
            ),
            **kwargs,
        )
        clients.append(client)
        if wait_ready:
            wait_until(lambda: client.is_ready)
        return client, notifier

    yield make
    for client in clients:
        client.close()


def test_real_child_uses_offsets_and_writes_parent_shared_arena(
    make_client, arena, tmp_path
):
    client, _ = make_client()
    payload = b"offset-roundtrip"
    arena.region[8192 : 8192 + len(payload)] = payload
    buffer = memoryview(arena.region)[8192 : 8192 + len(payload)]
    task = client.submit_batch_set(["key"], [buffer])
    assert completions(client, {task})[task][1] is True
    observed = json.loads((tmp_path / "set.json").read_text())
    assert observed == [{"offset": 8192, "length": len(payload), "hex": payload.hex()}]
    replacement = b"child-to-parent!"
    assert len(replacement) == len(payload)
    task = client.submit_batch_get(["payload:" + replacement.hex()], [buffer])
    assert completions(client, {task})[task][1] is True
    assert bytes(buffer) == replacement
    buffer.release()


@pytest.mark.parametrize("kind", ["get", "set"])
@pytest.mark.parametrize("failure", ["false", "partial"])
def test_dma_failure_is_not_published_until_child_is_reaped(
    make_client, arena, kind, failure
):
    client, notifier = make_client(failure + "_" + kind)
    pid = client.report_status()["worker_pid"]
    notifier.probe = lambda: not Path(f"/proc/{pid}").exists()
    buffers = [memoryview(arena.region)[0:4], memoryview(arena.region)[4096:4100]]
    keys = ["payload:" + b"data".hex()] * 2
    task = getattr(client, "submit_batch_" + kind)(keys, buffers)
    result = completions(client, {task})[task]
    assert result[1] is False
    assert notifier.observations and all(notifier.observations)
    assert not Path(f"/proc/{pid}").exists()
    assert arena.path.exists()
    for buffer in buffers:
        buffer.release()


@pytest.mark.parametrize("failure", ["hang", "crash", "sigstop"])
def test_timeout_crash_or_stopped_worker_is_reaped_before_failure(
    make_client, arena, failure
):
    mode = failure if failure in {"hang", "crash"} else "normal"
    client, notifier = make_client(mode, operation_timeout=0.2)
    pid = client.report_status()["worker_pid"]
    notifier.probe = lambda: not Path(f"/proc/{pid}").exists()
    if failure == "sigstop":
        stop_worker(pid)
    buffer = memoryview(arena.region)[:4]
    task = client.submit_batch_get(["payload:" + b"data".hex()], [buffer])
    result = completions(client, {task})[task]
    assert result[1] is False
    assert not Path(f"/proc/{pid}").exists()
    assert notifier.observations and all(notifier.observations)
    buffer.release()


def test_stopped_reader_cannot_block_submit_and_pending_is_bounded(make_client):
    client, _ = make_client(operation_timeout=0.5, max_pending=3)
    pid = client.report_status()["worker_pid"]
    stop_worker(pid)
    task_ids = set()
    latencies = []
    for _ in range(3):
        started = time.monotonic()
        task_ids.add(client.submit_batch_exists(["x" * (1 << 20)]))
        latencies.append(time.monotonic() - started)
    with pytest.raises(BufferError, match="Drain"):
        client.submit_batch_exists(["overflow"])
    assert max(latencies) < 0.25
    assert client.report_status()["worker_pending"] <= 3
    results = completions(client, task_ids)
    assert all(not result[1] for result in results.values())
    assert not Path(f"/proc/{pid}").exists()


def test_restart_changes_generation_without_reusing_parent_ids(make_client, arena):
    client, _ = make_client()
    first = client.submit_batch_exists(["key"])
    assert completions(client, {first})[first][1] is True
    pid = client.report_status()["worker_pid"]
    generation = client.report_status()["worker_generation"]
    arena.region[8192:8196] = b"live"
    os.kill(pid, signal.SIGKILL)
    wait_until(
        lambda: client.is_ready
        and client.report_status()["worker_generation"] > generation
    )
    second = client.submit_batch_exists(["key"])
    assert second > first
    assert completions(client, {second})[second][1] is True
    assert not Path(f"/proc/{pid}").exists()
    assert client.report_status()["worker_pid"] != pid
    assert arena.region[8192:8196] == b"live"
    assert arena.path.exists()


def test_close_reaps_worker_before_returning_and_before_failure_notice(
    make_client, arena
):
    client, notifier = make_client("hang")
    pid = client.report_status()["worker_pid"]
    notifier.probe = lambda: not Path(f"/proc/{pid}").exists()
    buffer = memoryview(arena.region)[:4]
    client.submit_batch_get(["payload:" + b"data".hex()], [buffer])
    client.close()
    assert not Path(f"/proc/{pid}").exists()
    assert not client.report_status()["worker_manager_alive"]
    assert notifier.observations and all(notifier.observations)
    assert arena.path.exists()
    client.close()
    buffer.release()


def test_quiesce_reaps_worker_but_keeps_failure_notifier_pollable(make_client, arena):
    client, notifier = make_client("hang")
    pid = client.report_status()["worker_pid"]
    notifier.probe = lambda: not Path(f"/proc/{pid}").exists()
    buffer = memoryview(arena.region)[:4]
    task = client.submit_batch_get(["payload:" + b"data".hex()], [buffer])
    descriptor = client.event_fd()
    client.quiesce()
    assert not Path(f"/proc/{pid}").exists()
    assert not client.report_status()["worker_manager_alive"]
    assert not client.is_ready
    assert select.select([descriptor], [], [], 0)[0] == [descriptor]
    result = completions(client, {task})[task]
    assert result[1] is False
    assert notifier.observations and all(notifier.observations)
    assert select.select([descriptor], [], [], 0)[0] == []
    after = client.submit_batch_exists(["after-quiesce"])
    assert completions(client, {after})[after][1] is False
    os.fstat(descriptor)
    client.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)
    with pytest.raises(RuntimeError, match="closed"):
        client.submit_batch_exists(["after-close"])
    buffer.release()


def test_unconfirmed_worker_death_never_publishes_failure_or_releases_pending():
    class FatalRequired(Exception):
        pass

    class UnreapableProcess:
        pid = 123456

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("fake-worker", timeout)

    client = ProcessNativeClient.__new__(ProcessNativeClient)
    client._lock = threading.RLock()
    client._ready = True
    client._process = UnreapableProcess()
    client._connection = None
    client._kill_timeout = 0.01
    client._pending = {1: ("get", time.monotonic())}
    client._outbound = queue.Queue()
    client._completed = []
    client._aborted = 0

    def fatal(_reason, _pid):
        raise FatalRequired()

    client._fatal = fatal
    with pytest.raises(FatalRequired):
        client._abort("simulated inability to reap")
    assert client._pending.keys() == {1}
    assert client._completed == []


def test_rejects_same_size_file_that_does_not_back_parent_mapping(arena, tmp_path):
    wrong = tmp_path / "wrong_arena"
    wrong.write_bytes(bytes(arena.size))
    notifier = Notifier()
    try:
        with pytest.raises(ValueError, match="supplied shared file"):
            ProcessNativeClient(
                {},
                1,
                None,
                arena.base,
                arena.size,
                wrong.name,
                notifier,
                shm_root=tmp_path,
            )
    finally:
        notifier.close()


def test_rejects_private_mapping_even_when_file_identity_matches(arena):
    with arena.path.open("r+b") as stream:
        private = mmap.mmap(stream.fileno(), arena.size, access=mmap.ACCESS_COPY)
    base = ctypes.addressof(ctypes.c_ubyte.from_buffer(private))
    notifier = Notifier()
    try:
        with pytest.raises(ValueError, match="supplied shared file"):
            ProcessNativeClient(
                {},
                1,
                None,
                base,
                arena.size,
                arena.path.name,
                notifier,
                shm_root=arena.path.parent,
            )
    finally:
        notifier.close()
        private.close()


def test_restart_retains_original_backing_after_path_replacement(make_client, arena):
    client, _ = make_client()
    generation = client.report_status()["worker_generation"]
    pid = client.report_status()["worker_pid"]
    arena.path.unlink()
    arena.path.write_bytes(bytes(arena.size))
    os.kill(pid, signal.SIGKILL)
    wait_until(
        lambda: client.is_ready
        and client.report_status()["worker_generation"] > generation
    )
    with memoryview(arena.region)[:4] as buffer:
        task = client.submit_batch_get(["payload:" + b"kept".hex()], [buffer])
        assert completions(client, {task})[task][1] is True
        assert bytes(buffer) == b"kept"
    assert arena.path.read_bytes() == bytes(arena.size)


def test_undrained_completions_use_admission_budget(make_client):
    client, _ = make_client(max_pending=2)
    client.quiesce()
    first = client.submit_batch_exists(["a"])
    second = client.submit_batch_exists(["b"])
    with pytest.raises(BufferError, match="Drain"):
        client.submit_batch_exists(["c"])
    assert client.report_status()["worker_undrained"] == 2
    assert len(completions(client, {first, second})) == 2
    third = client.submit_batch_exists(["c"])
    assert completions(client, {third})[third][1] is False


def test_rejects_buffers_outside_arena_before_submission(make_client):
    client, _ = make_client()
    with pytest.raises(ValueError, match="outside"):
        client.submit_batch_get(["k"], [bytearray(4)])
    assert client.report_status()["worker_pending"] == 0


def test_startup_failure_serves_failures_and_restarts_without_blocking(make_client):
    client, _ = make_client("startup_hang", wait_ready=False, startup_timeout=0.2)
    wait_until(lambda: client.report_status()["worker_generation"] >= 1)
    pid = client.report_status()["worker_pid"]
    task = client.submit_batch_exists(["missing"])
    assert completions(client, {task})[task][1] is False
    wait_until(lambda: client.report_status()["worker_generation"] > 1)
    assert not Path(f"/proc/{pid}").exists()
    assert not client.is_ready


def test_child_imports_parent_source_when_launched_outside_checkout(
    make_client, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    script = (
        "import lmcache, runpy, sys; "
        f"sys.argv = [{str(Path(__file__).resolve())!r}, "
        "'--fake-worker', sys.argv[1]]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    client, _ = make_client(worker_command=[sys.executable, "-c", script])
    task = client.submit_batch_exists(["k"])
    assert completions(client, {task})[task][1] is True


if __name__ == "__main__" and sys.argv[1:2] == ["--fake-worker"]:
    run_worker(socket.socket(fileno=int(sys.argv[2])), native_factory=FakeNative)
