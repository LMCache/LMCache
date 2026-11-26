# SPDX-License-Identifier: Apache-2.0
"""Admission and shutdown contracts without Mooncake, CUDA, or RDMA."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypeVar
import importlib.util
import json
import socket
import subprocess
import sys
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.platform.event_notifier import EventNotifier

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.l2_adapters.mooncake_resilience import (
        GuardConfig,
        GuardedNativeClient,
        NativeCompletion,
        RecoveryProbe,
        SubprocessDataProbe,
        TerminalNativeClient,
    )
else:
    # The adapter package initializer imports native extensions; this component
    # itself is independent of them and is loaded unchanged from its real file.
    component_path = (
        Path(__file__).resolve().parents[3]
        / "lmcache/v1/distributed/l2_adapters/mooncake_resilience.py"
    )
    specification = importlib.util.spec_from_file_location(
        "lmcache_test_mooncake_resilience", component_path
    )
    assert specification is not None and specification.loader is not None
    component = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = component
    specification.loader.exec_module(component)
    GuardConfig = component.GuardConfig
    GuardedNativeClient = component.GuardedNativeClient
    NativeCompletion = component.NativeCompletion
    RecoveryProbe = component.RecoveryProbe
    SubprocessDataProbe = component.SubprocessDataProbe
    TerminalNativeClient = component.TerminalNativeClient

# Reuse the process suite's arena and real child protocol without importing the
# distributed test package, whose initializer requires the native extension.
process_test_path = (
    Path(__file__).resolve().with_name("test_mooncake_process_client.py")
)
process_test_spec = importlib.util.spec_from_file_location(
    "mooncake_guard_process_test_helpers", process_test_path
)
assert process_test_spec is not None and process_test_spec.loader is not None
process_tests = importlib.util.module_from_spec(process_test_spec)
sys.modules[process_test_spec.name] = process_tests
process_test_spec.loader.exec_module(process_tests)
arena = process_tests.arena
make_client = process_tests.make_client

T = TypeVar("T")
POLICY = GuardConfig(
    lookup_timeout=0.1,
    slow_io_threshold=0.1,
    failure_threshold=1,
    cooldown=0.1,
    probe_timeout=0.5,
    probe_interval=0.01,
    recovery_successes=2,
    max_pending=8,
    poll_interval=0.005,
)


def wait_for(predicate: Callable[[], T], timeout: float = 5.0) -> T:
    """Wait for an observable state, failing within a bounded test deadline."""
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("Expected public state did not arrive before the deadline")


def ready(client: GuardedNativeClient) -> None:
    """Wait for recovery probes to admit native operations."""
    wait_for(lambda: client.report_status()["remote_available"])


def completed(client: GuardedNativeClient, task_id: int) -> NativeCompletion:
    """Drain available results until the requested operation completes."""
    results: dict[int, NativeCompletion] = {}

    def drain() -> NativeCompletion | None:
        for result in client.drain_completions():
            assert result[0] not in results, "An operation completed twice"
            results[result[0]] = result
        return results.get(task_id)

    result = wait_for(drain)
    assert result is not None
    return result


class SocketNotifier(EventNotifier):
    """Use actual selectable descriptors for the completion protocol."""

    def __init__(self) -> None:
        self.reader, self.writer = socket.socketpair()
        self.reader.setblocking(False)
        self.writer.setblocking(False)

    def fileno(self) -> int:
        return self.reader.fileno()

    def notify_fileno(self) -> int:
        return self.writer.fileno()

    def notify(self) -> None:
        try:
            self.writer.send(b"x")
        except BlockingIOError:
            pass

    def consume(self) -> None:
        while True:
            try:
                if not self.reader.recv(65536):
                    break
            except BlockingIOError:
                break

    def close(self) -> None:
        self.reader.close()
        self.writer.close()


class Native:
    """Controllable terminal client; quiesce settles all outstanding results."""

    def __init__(self) -> None:
        self.signal = SocketNotifier()
        self.calls: list[tuple[int, str, list[memoryview] | None]] = []
        self.pending: set[int] = set()
        self.results: list[NativeCompletion] = []
        self.lock = threading.Lock()
        self.is_ready = True
        self.closed = False
        self.quiesce_entered = threading.Event()
        self.release_quiesce = threading.Event()
        self.release_quiesce.set()

    def event_fd(self) -> int:
        return self.signal.fileno()

    def submit_batch_exists(self, keys: list[str]) -> int:
        return self.submit("exists")

    def submit_batch_get(self, keys: list[str], buffers: list[memoryview]) -> int:
        return self.submit("get", buffers)

    def submit_batch_set(self, keys: list[str], buffers: list[memoryview]) -> int:
        return self.submit("set", buffers)

    def submit_batch_delete(self, keys: list[str]) -> int:
        return self.submit("delete")

    def submit(self, kind: str, buffers: list[memoryview] | None = None) -> int:
        with self.lock:
            task_id = len(self.calls)
            self.calls.append((task_id, kind, buffers))
            self.pending.add(task_id)
            return task_id

    def finish(
        self, task_id: int, ok: bool = True, flags: list[bool] | None = None
    ) -> None:
        with self.lock:
            self.pending.remove(task_id)
            self.results.append((task_id, ok, "" if ok else "native failure", flags))
            self.signal.notify()

    def drain_completions(self) -> list[NativeCompletion]:
        with self.lock:
            self.signal.consume()
            results, self.results = self.results, []
            return results

    def quiesce(self) -> None:
        self.quiesce_entered.set()
        assert self.release_quiesce.wait(5), "Test did not release quiesce"
        with self.lock:
            pending = list(self.pending)
        for task_id in pending:
            self.finish(task_id, ok=False)

    def close(self) -> None:
        if not self.closed:
            self.quiesce()
            self.signal.close()
            self.closed = True


class CallbackProbe:
    """Provide a closable independent probe with a controllable result."""

    def __init__(self, action: Callable[[], int] = lambda: 0) -> None:
        self.action = action
        self.closed = False

    def __call__(self) -> int:
        return self.action()

    def close(self) -> None:
        self.closed = True


class MakeGuard(Protocol):
    def __call__(
        self,
        config: GuardConfig = POLICY,
        probe: RecoveryProbe | None = None,
        factory: Callable[[], TerminalNativeClient] | None = None,
        native: Native | None = None,
    ) -> tuple[GuardedNativeClient, Native]: ...


class InspectableProcessClient(TerminalNativeClient, Protocol):
    def report_status(self) -> dict[str, object]: ...


class MakeProcessClient(Protocol):
    def __call__(
        self, mode: str = "normal", operation_timeout: float = 2.0
    ) -> tuple[InspectableProcessClient, object]: ...


@pytest.fixture
def make_guard() -> Iterator[MakeGuard]:
    clients: list[tuple[GuardedNativeClient, Native]] = []

    def make(
        config: GuardConfig = POLICY,
        probe: RecoveryProbe | None = None,
        factory: Callable[[], TerminalNativeClient] | None = None,
        native: Native | None = None,
    ) -> tuple[GuardedNativeClient, Native]:
        native = native or Native()
        client = GuardedNativeClient(
            factory or (lambda: native),
            probe or CallbackProbe(),
            SocketNotifier(),
            config,
        )
        clients.append((client, native))
        return client, native

    yield make
    for client, native in clients:
        native.release_quiesce.set()
        client.close()
        native.close()


def test_unavailable_startup_bypasses_without_touching_native(
    make_guard: MakeGuard,
) -> None:
    def unavailable() -> TerminalNativeClient:
        raise ConnectionError("test backend unavailable")

    client, native = make_guard(factory=unavailable)
    target = memoryview(bytearray(b"unchanged"))
    task_id = client.submit_batch_get(["key"], [target])
    assert completed(client, task_id)[1:] == (False, "L2 circuit open", [False])
    assert native.calls == []
    assert bytes(target) == b"unchanged"


def test_normal_lookup_misses_do_not_open_circuit(make_guard: MakeGuard) -> None:
    client, native = make_guard()
    ready(client)
    for _ in range(3):
        task_id = client.submit_batch_exists(["missing"])
        native.finish(task_id, flags=[False])
        assert completed(client, task_id)[1:] == (True, "", [False])
    assert client.report_status()["remote_available"] is True


def test_lookup_deadline_discards_the_late_result(make_guard: MakeGuard) -> None:
    client, native = make_guard(replace(POLICY, cooldown=5))
    ready(client)
    task_id = client.submit_batch_exists(["key"])
    assert completed(client, task_id)[1:] == (False, "lookup deadline", [False])
    assert client.report_status()["expired_lookup_pending"] == 1
    native.finish(task_id, flags=[True])
    wait_for(lambda: client.report_status()["expired_lookup_pending"] == 0)
    assert client.drain_completions() == []


@pytest.mark.parametrize("kind", ["get", "set"])
def test_slow_dma_retains_original_completion(make_guard: MakeGuard, kind: str) -> None:
    client, native = make_guard(replace(POLICY, cooldown=5))
    ready(client)
    target = memoryview(bytearray(b"old"))
    submit = client.submit_batch_get if kind == "get" else client.submit_batch_set
    task_id = submit(["key"], [target])
    wait_for(lambda: client.report_status()["slow_dma_operations"] == 1)
    assert client.drain_completions() == []
    assert client.report_status()["native_dma_pending"] == 1
    assert native.calls == [(task_id, kind, [target])]
    native.finish(task_id, flags=[True])
    assert completed(client, task_id)[1:] == (True, "", [True])


def test_partial_data_failure_opens_circuit(make_guard: MakeGuard) -> None:
    client, native = make_guard(replace(POLICY, cooldown=5))
    ready(client)
    task_id = client.submit_batch_get(["a", "b"], [memoryview(bytearray(4))] * 2)
    native.finish(task_id, flags=[True, False])
    assert completed(client, task_id)[3] == [True, False]
    bypass = client.submit_batch_set(["key"], [memoryview(bytearray(4))])
    assert completed(client, bypass)[1] is False
    assert len(native.calls) == 1


def test_recovery_waits_for_pending_dma_and_native_readiness(
    make_guard: MakeGuard,
) -> None:
    client, native = make_guard()
    ready(client)
    task_id = client.submit_batch_get(["key"], [memoryview(bytearray(4))])
    wait_for(lambda: client.report_status()["slow_dma_operations"] == 1)
    wait_for(
        lambda: client.report_status()["probe_result"]
        == "healthy_but_native_operations_pending"
    )
    native.is_ready = False
    native.finish(task_id, ok=False)
    completed(client, task_id)
    wait_for(
        lambda: client.report_status()["reason"]
        == "waiting for native worker readiness"
    )
    assert client.report_status()["remote_available"] is False
    native.is_ready = True
    ready(client)


def test_expired_lookup_still_counts_against_native_capacity(
    make_guard: MakeGuard,
) -> None:
    client, native = make_guard(replace(POLICY, max_pending=1, failure_threshold=10))
    ready(client)
    task_id = client.submit_batch_exists(["one"])
    completed(client, task_id)
    bypass = client.submit_batch_exists(["two"])
    assert completed(client, bypass)[1] is False
    assert len(native.calls) == 1
    assert client.report_status()["expired_lookup_pending"] == 1


def test_native_failure_during_recovery_does_not_stall_half_open(
    make_guard: MakeGuard,
) -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def probe() -> int:
        calls.append(1)
        if len(calls) == POLICY.recovery_successes + 1:
            entered.set()
            assert release.wait(5)
        return 0

    client, native = make_guard(
        replace(POLICY, failure_threshold=3), probe=CallbackProbe(probe)
    )
    ready(client)
    task_id = client.submit_batch_get(["key"], [memoryview(bytearray(4))])
    try:
        assert entered.wait(2)
        native.finish(task_id, ok=False)
        completed(client, task_id)
    finally:
        release.set()
    ready(client)


def test_undrained_bypass_results_apply_backpressure(make_guard: MakeGuard) -> None:
    client, native = make_guard(
        replace(POLICY, max_pending=2), probe=CallbackProbe(lambda: -1)
    )
    tasks = [client.submit_batch_exists(["key"]) for _ in range(2)]
    with pytest.raises(BufferError, match="Drain"):
        client.submit_batch_exists(["key"])
    assert {result[0] for result in client.drain_completions()} == set(tasks)
    next_id = client.submit_batch_exists(["key"])
    assert completed(client, next_id)[1] is False
    assert native.calls == []


def test_native_backpressure_becomes_a_safe_bypass(make_guard: MakeGuard) -> None:
    class FullNative(Native):
        def submit_batch_exists(self, keys: list[str]) -> int:
            raise BufferError("not admitted")

    client, native = make_guard(native=FullNative())
    ready(client)
    task_id = client.submit_batch_exists(["key"])
    assert completed(client, task_id)[1] is False
    assert native.calls == []


def test_get_limit_rejects_new_dma_without_completing_existing_work(
    make_guard: MakeGuard,
) -> None:
    client, native = make_guard(replace(POLICY, max_pending_get=1, cooldown=5))
    ready(client)
    buffers = [memoryview(bytearray(4))]
    original = client.submit_batch_get(["key"], buffers)
    skipped = client.submit_batch_get(["key"], buffers)
    assert completed(client, skipped)[3] == [False]
    assert client.report_status()["native_dma_pending"] == 1
    assert native.calls == [(original, "get", buffers)]


def test_periodic_data_probe_detects_errors_hidden_by_misses(
    make_guard: MakeGuard,
) -> None:
    healthy = [True]
    probe = CallbackProbe(lambda: 0 if healthy[0] else -1)
    client, native = make_guard(
        replace(POLICY, healthy_probe_interval=0.03), probe=probe
    )
    ready(client)
    task_id = client.submit_batch_exists(["missing"])
    native.finish(task_id, flags=[False])
    assert completed(client, task_id)[1] is True
    healthy[0] = False
    wait_for(lambda: client.report_status()["remote_available"] is False)
    healthy[0] = True
    ready(client)


def test_timed_out_probe_is_retained_until_it_finishes(make_guard: MakeGuard) -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def blocked() -> int:
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return 0

    client, _ = make_guard(
        replace(POLICY, probe_timeout=0.05), probe=CallbackProbe(blocked)
    )
    try:
        assert entered.wait(2)
        wait_for(
            lambda: client.report_status()["probe_result"] == "timeout_worker_retained"
        )
        for _ in range(10):
            assert completed(client, client.submit_batch_exists(["key"]))[1] is False
        assert calls == [1]
    finally:
        release.set()


def test_quiesce_bypasses_new_work_and_relays_fenced_results(
    make_guard: MakeGuard,
) -> None:
    client, native = make_guard()
    ready(client)
    task_id = client.submit_batch_get(["key"], [memoryview(bytearray(4))])
    native.release_quiesce.clear()
    stopping = threading.Thread(target=client.quiesce)
    stopping.start()
    try:
        assert native.quiesce_entered.wait(2)
        assert client.drain_completions() == []
        bypass = client.submit_batch_get(["new"], [memoryview(bytearray(4))])
        assert completed(client, bypass)[1] is False
        assert len(native.calls) == 1
    finally:
        native.release_quiesce.set()
        stopping.join(3)
    assert not stopping.is_alive()
    assert completed(client, task_id)[1] is False
    assert client.report_status()["native_pending"] == 0
    assert client.event_fd() >= 0
    client.quiesce()
    client.close()
    assert client.event_fd() == -1
    with pytest.raises(RuntimeError, match="closed"):
        client.submit_batch_exists(["closed"])


def test_close_joins_initialization_before_closing_client(
    make_guard: MakeGuard,
) -> None:
    entered, release = threading.Event(), threading.Event()
    native = Native()

    def initialize() -> TerminalNativeClient:
        entered.set()
        assert release.wait(5)
        return native

    client, _ = make_guard(factory=initialize, native=native)
    assert entered.wait(2)
    closing = threading.Thread(target=client.close)
    closing.start()
    try:
        wait_for(lambda: client.report_status()["reason"] == "client quiescing")
        assert closing.is_alive()
        assert not native.closed
    finally:
        release.set()
        closing.join(3)
    assert not closing.is_alive()
    assert native.closed


@pytest.mark.parametrize("stop", ["deadline", "close"])
def test_probe_child_is_reaped(monkeypatch: pytest.MonkeyPatch, stop: str) -> None:
    original = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    started = threading.Event()

    def sleeping_child(
        command: list[str], *, stdin: int, stdout: int, stderr: int, env: dict[str, str]
    ) -> subprocess.Popen[bytes]:
        process = original(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        children.append(process)
        started.set()
        return process

    monkeypatch.setattr(subprocess, "Popen", sleeping_child)
    probe = SubprocessDataProbe({}, timeout=0.1 if stop == "deadline" else 10)
    results: list[int] = []
    running = threading.Thread(target=lambda: results.append(probe()))
    running.start()
    try:
        assert started.wait(2)
        if stop == "close":
            probe.close()
        running.join(3)
        assert not running.is_alive()
        assert results == [-1]
        assert children[0].poll() is not None
        probe.close()
        assert probe() == -1
    finally:
        probe.close()
        running.join(3)


@pytest.mark.parametrize("corrupt", [False, True])
def test_probe_verifies_data_and_removes_only_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    package = tmp_path / "mooncake"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "store.py").write_text(
        """\
import json
import os
from pathlib import Path

class MooncakeDistributedStore:
    def setup(self, *args, **kwargs):
        assert args[2:4] == (0, 1 << 20)
        self.path = Path(args[0])
        return 0
    def is_exist(self, key):
        return 0
    def put(self, key, value):
        self.key, self.value = key, value
        return 0
    def get(self, key):
        assert key == self.key
        return b'wrong' if os.environ['TEST_CORRUPT'] == '1' else self.value
    def remove(self, key, force=False):
        assert key == self.key and force
        self.path.write_text(json.dumps({'key': key, 'removed': True}))
        return 0
"""
    )
    observation = tmp_path / "observed.json"
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("TEST_CORRUPT", "1" if corrupt else "0")
    probe = SubprocessDataProbe(
        {
            "local_hostname": str(observation),
            "metadata_server": "test-metadata",
            "master_server_addr": "test-master",
        },
        timeout=5,
    )
    try:
        assert probe() == (-1 if corrupt else 0)
        result = json.loads(observation.read_text())
        assert result["key"].startswith("__lmcache_circuit_probe__")
        assert result["removed"] is True
    finally:
        probe.close()


def test_unreapable_probe_retains_its_only_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreapableProcess:
        stdin = None

        def __init__(self) -> None:
            self.allow_reap = False

        def poll(self) -> int | None:
            return 0 if self.allow_reap else None

        def communicate(self, data: bytes, timeout: float) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("probe", timeout)

        def kill(self) -> None:
            pass

        def wait(self, timeout: float) -> int:
            if not self.allow_reap:
                raise subprocess.TimeoutExpired("probe", timeout)
            return 0

    child = UnreapableProcess()
    created: list[UnreapableProcess] = []

    def create(*args: object, **kwargs: object) -> UnreapableProcess:
        created.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", create)
    probe = SubprocessDataProbe({}, timeout=0.01, kill_timeout=0.01)
    with pytest.raises(LMCacheTimeoutError, match="reaped"):
        probe()
    assert probe() == -1
    assert created == [child]
    with pytest.raises(LMCacheTimeoutError, match="reaped"):
        probe.close()
    child.allow_reap = True
    probe.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process/FD semantics")
@pytest.mark.parametrize("stop", ["timeout", "quiesce"])
def test_guard_relays_process_fence_before_releasing_shared_buffers(
    make_client: MakeProcessClient,
    arena: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    """A hung worker cannot turn a guard timeout into early buffer reuse."""
    process, _ = make_client("hang", operation_timeout=0.4 if stop == "timeout" else 10)
    pid = process.report_status()["worker_pid"]
    notifier = SocketNotifier()
    published_after_reap: list[bool] = []
    notify = notifier.notify

    def observe() -> None:
        published_after_reap.append(not Path(f"/proc/{pid}").exists())
        notify()

    monkeypatch.setattr(notifier, "notify", observe)
    client = GuardedNativeClient(
        lambda: process,
        CallbackProbe(),
        notifier,
        replace(POLICY, slow_io_threshold=0.05, cooldown=5),
    )
    buffer = memoryview(arena.region)[:4]
    arena.region[:4] = b"held"
    arena.region[8192:8196] = b"live"
    try:
        ready(client)
        task_id = client.submit_batch_get(["key"], [buffer])
        wait_for(lambda: client.report_status()["slow_dma_operations"] == 1)
        assert Path(f"/proc/{pid}").exists()
        assert client.drain_completions() == []
        assert published_after_reap == []
        if stop == "quiesce":
            client.quiesce()
        assert completed(client, task_id)[1] is False
        assert published_after_reap == [True]
        assert not Path(f"/proc/{pid}").exists()
        assert client.report_status()["remote_available"] is False
        assert client.report_status()["native_dma_pending"] == 0
        assert arena.path.exists()
        assert arena.region[8192:8196] == b"live"
        buffer[:] = b"safe"
        assert arena.region[:4] == b"safe"
    finally:
        client.close()
        buffer.release()
