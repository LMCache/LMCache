# SPDX-License-Identifier: Apache-2.0
"""Admission control for a terminal-completion-safe Mooncake client.

This component does not register an adapter or protect L1 allocations from
eviction. The caller must retain every submitted buffer until its terminal
completion. In particular, an SDK timeout is not necessarily a terminal DMA
completion; an unmodified in-process Mooncake client does not meet this contract.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol
import json
import math
import os
import select
import subprocess
import sys
import threading
import time
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.event_notifier import EventNotifier

logger = init_logger(__name__)

NativeCompletion = tuple[int, bool, str, list[bool] | None]
OperationKind = Literal["exists", "get", "set", "delete"]

# Executed in a separate interpreter: the parent must never import the SDK.
_PROBE_SCRIPT = """\
import json
import sys
import uuid
from mooncake.store import MooncakeDistributedStore

request = json.load(sys.stdin)
config, key = request["config"], request["key"]
payload = (uuid.uuid4().hex * 4).encode("ascii")
store = MooncakeDistributedStore()
result = store.setup(
    config["local_hostname"], config["metadata_server"], 0, 1 << 20,
    config.get("protocol", "rdma"), config.get("rdma_devices", ""),
    config["master_server_addr"], tenant_id=config.get("tenant_id", "default"),
)
if result != 0:
    raise SystemExit(1)
exists = int(store.is_exist(key))
if exists < 0 or (exists == 1 and int(store.remove(key, force=True)) != 0):
    raise SystemExit(1)
if int(store.put(key, payload)) != 0:
    raise SystemExit(1)
try:
    data = store.get(key)
    valid = data is not None and bytes(data) == payload
finally:
    removed = int(store.remove(key, force=True)) == 0
raise SystemExit(0 if valid and removed else 1)
"""


class TerminalNativeClient(Protocol):
    """Native completion protocol with explicit buffer-lifetime guarantees.

    Submission must be nonblocking and return a unique nonnegative ID. BufferError
    means the operation was not admitted. Other submission exceptions do not
    establish buffer safety: the caller must fence before releasing those
    buffers. Every GET/SET result, including a failure, must mean that the backend
    can no longer access its buffers.
    The client must not close its event notifier during ``quiesce``.
    """

    @property
    def is_ready(self) -> bool:
        """Return whether the client is ready to accept remote operations."""
        ...

    def event_fd(self) -> int:
        """Return the pollable completion descriptor."""
        ...

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Submit a lookup without retaining or accessing caller buffers."""
        ...

    def submit_batch_get(self, keys: list[str], buffers: list[memoryview]) -> int:
        """Submit reads into caller-owned buffers and return the operation ID."""
        ...

    def submit_batch_set(self, keys: list[str], buffers: list[memoryview]) -> int:
        """Submit writes from caller-owned buffers and return the operation ID."""
        ...

    def submit_batch_delete(self, keys: list[str]) -> int:
        """Submit deletion and return the operation ID."""
        ...

    def drain_completions(self) -> list[NativeCompletion]:
        """Return and remove available terminal results without blocking."""
        ...

    def quiesce(self) -> None:
        """Fence all buffer users and make every pending result drainable.

        A failure to fence must raise or stop the process; it must never return
        success. Completion resources remain usable until ``close``.
        """
        ...

    def close(self) -> None:
        """Fence all buffer users and release resources, idempotently."""
        ...


class RecoveryProbe(Protocol):
    """An independent health probe that never receives caller buffers."""

    def __call__(self) -> int:
        """Return zero for a verified data round trip, or nonzero on failure."""
        ...

    def close(self) -> None:
        """Stop active probing and prevent subsequent probes, idempotently."""
        ...


@dataclass(frozen=True)
class GuardConfig:
    """Admission and probe policy; all time values are positive seconds.

    ``lookup_timeout`` bounds EXISTS results. ``slow_io_threshold`` only stops
    new submissions: it never completes a submitted GET/SET. ``max_pending``
    includes expired lookups whose native operations have not finished, and
    ``max_pending_get`` separately bounds submitted GET batches. Pending public
    operations plus undrained results are also bounded by ``max_pending``;
    callers must drain results or submission raises BufferError before admission.
    Recovery needs ``recovery_successes`` consecutive timely probes, separated by
    ``probe_interval`` after ``cooldown``. Healthy probes run every
    ``healthy_probe_interval``. ``poll_interval`` bounds completion polling.
    """

    lookup_timeout: float = 2.0
    slow_io_threshold: float = 10.0
    failure_threshold: int = 3
    cooldown: float = 15.0
    probe_timeout: float = 2.0
    probe_interval: float = 2.0
    recovery_successes: int = 2
    max_pending: int = 16
    max_pending_get: int = 4
    healthy_probe_interval: float = 15.0
    poll_interval: float = 0.02

    def __post_init__(self) -> None:
        """Reject non-finite timings and non-integer admission counts."""
        integer_fields = {
            "failure_threshold",
            "recovery_successes",
            "max_pending",
            "max_pending_get",
        }
        for name, value in vars(self).items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                or (name in integer_fields and not isinstance(value, int))
            ):
                raise ValueError(f"{name} must be a positive finite number")


@dataclass
class _Operation:
    kind: OperationKind
    size: int
    started: float
    slow_reported: bool = False


class GuardedNativeClient:
    """Bypass unhealthy L2 operations while retaining submitted buffer ownership.

    This class is thread-safe. It owns the supplied notifier, probe, and factory
    result. The factory must return promptly or have its own bounded startup
    mechanism; arbitrary blocked Python initializers cannot be canceled safely.
    At most one initialization and one probe can run at a time.
    """

    def __init__(
        self,
        native_factory: Callable[[], TerminalNativeClient],
        probe: RecoveryProbe,
        notifier: EventNotifier,
        config: GuardConfig | None = None,
    ) -> None:
        """Start asynchronous initialization and health monitoring.

        Args:
            native_factory: Constructs a client satisfying TerminalNativeClient.
            probe: Independent, closable data probe with no access to caller L1.
            notifier: Owned event notifier, retained until final close.
            config: Admission/probe policy, or the default policy.
        """
        self._factory = native_factory
        self._probe = probe
        self._notifier = notifier
        self._config = config or GuardConfig()
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._closed = False
        self._quiesced = False
        self._stop = threading.Event()
        self._init_thread: threading.Thread | None = None
        self._probe_thread: threading.Thread | None = None
        self._native: TerminalNativeClient | None = None
        self._pending: dict[int, _Operation] = {}
        self._expired_lookups: set[int] = set()
        self._completed: deque[NativeCompletion] = deque()
        self._next_local_id = -1
        self._state = "OPEN"
        self._reason = "initializing"
        self._failures = 0
        self._recoveries = 0
        self._bypassed = 0
        self._lookup_timeouts = 0
        self._slow_dma_operations = 0
        self._init_busy = False
        self._probe_busy = False
        self._probe_started = 0.0
        self._probe_result = "not_run"
        self._probe_mode = "recovery"
        self._probe_timeout_reported = False
        self._next_healthy_probe = 0.0
        self._fault_version = 0
        self._probe_version = 0
        self._next_attempt = 0.0
        self._worker = threading.Thread(
            target=self._run, name="mooncake-circuit", daemon=True
        )
        self._worker.start()

    def event_fd(self) -> int:
        """Return the completion descriptor, valid until final close."""
        return self._notifier.fileno()

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Submit a lookup; a deadline may produce a miss before native completion.

        Args:
            keys: Remote keys to look up.

        Returns:
            The result ID, including for immediate bypass results.

        Raises:
            RuntimeError: If the client has been closed.
            BufferError: If pending operations and undrained results fill capacity.
        """
        return self._submit("exists", keys)

    def submit_batch_get(self, keys: list[str], buffers: list[memoryview]) -> int:
        """Submit GET, retaining caller ownership until its terminal result.

        Args:
            keys: Remote keys to read.
            buffers: Writable destinations owned by the caller until completion.

        Returns:
            The result ID. Bypassed operations never reach the native client.

        Raises:
            RuntimeError: If closed, or the native client violates its ID contract.
            BufferError: If pending operations and undrained results fill capacity.
        """
        return self._submit("get", keys, buffers)

    def submit_batch_set(self, keys: list[str], buffers: list[memoryview]) -> int:
        """Submit SET, retaining caller ownership until its terminal result.

        Args:
            keys: Remote keys to write.
            buffers: Sources kept immutable and allocated until completion.

        Returns:
            The result ID. Bypassed operations never reach the native client.

        Raises:
            RuntimeError: If closed, or the native client violates its ID contract.
            BufferError: If pending operations and undrained results fill capacity.
        """
        return self._submit("set", keys, buffers)

    def submit_batch_delete(self, keys: list[str]) -> int:
        """Submit deletion of keys and return its ID; bypass reports failure.

        Args:
            keys: Remote keys to remove.

        Raises:
            RuntimeError: If the client has been closed.
            BufferError: If pending operations and undrained results fill capacity.
        """
        return self._submit("delete", keys)

    def drain_completions(self) -> list[NativeCompletion]:
        """Remove available results and reset their notification, without waiting.

        Returns:
            Tuples of (ID, success, error, optional per-key success flags).
            Every submitted ID produces at most one result.
        """
        with self._lock:
            self._notifier.consume()
            results = list(self._completed)
            self._completed.clear()
            return results

    def report_status(self) -> dict[str, object]:
        """Return a synchronized snapshot of circuit state and bounded work.

        Returns:
            ``circuit_state`` (open/half_open/closed), ``remote_available`` and
            ``reason``; operation/failure counters; pending-operation counts;
            initialization/probe state; monitor liveness; and ``guard_config``.
        """
        with self._lock:
            return {
                "circuit_state": self._state.lower(),
                "remote_available": self._state == "CLOSED" and not self._stop.is_set(),
                "reason": self._reason,
                "bypassed_operations": self._bypassed,
                "failures": self._failures,
                "lookup_timeouts": self._lookup_timeouts,
                "slow_dma_operations": self._slow_dma_operations,
                "native_pending": len(self._pending),
                "expired_lookup_pending": len(self._expired_lookups),
                "native_dma_pending": sum(
                    op.kind in ("get", "set") for op in self._pending.values()
                ),
                "initialization_in_flight": self._init_busy,
                "probe_in_flight": self._probe_busy,
                "probe_result": self._probe_result,
                "guard_thread_alive": self._worker.is_alive(),
                "guard_config": vars(self._config).copy(),
            }

    def quiesce(self) -> None:
        """Fence native work and relay its results, retaining completion resources.

        During and after quiesce, new submissions receive immediate failure
        results without native access. A concurrent close serializes with this
        method. The caller must retain all allocations until this method returns
        successfully or their individual terminal results are received.

        Raises:
            RuntimeError: If the native client returns from quiesce without
                making all pending results available.
            LMCacheTimeoutError: If the independent probe cannot be reaped.
        """
        with self._close_lock:
            if self._quiesced or self._closed:
                return
            with self._lock:
                self._stop.set()
                self._open_locked("client quiescing")
            self._worker.join()
            if self._init_thread is not None:
                self._init_thread.join()
            with self._lock:
                native = self._native
            if native is not None:
                native.quiesce()
                self._accept_completions(native.drain_completions())
            self._probe.close()
            if self._probe_thread is not None:
                self._probe_thread.join()
            with self._lock:
                if self._pending or self._expired_lookups:
                    raise RuntimeError(
                        "Native quiesce did not settle pending operations"
                    )
                self._quiesced = True

    def close(self) -> None:
        """Quiesce and release owned resources, idempotently.

        An unbounded custom initializer/probe can block shutdown. Production
        callers must supply bounded implementations, such as the process client
        and SubprocessDataProbe. A quiesce error propagates without closing the
        notifier or reporting unconfirmed native operations as completed.
        """
        with self._close_lock:
            if self._closed:
                return
            self.quiesce()
            with self._lock:
                native = self._native
            if native is not None:
                native.close()
            with self._lock:
                self._closed = True
                self._notifier.close()

    def _open_locked(self, reason: str) -> None:
        self._fault_version += 1
        if self._state != "OPEN":
            self._next_attempt = time.monotonic() + self._config.cooldown
        self._state = "OPEN"
        self._reason = reason
        self._recoveries = 0

    def _failure_locked(self, reason: str) -> None:
        self._fault_version += 1
        self._failures += 1
        if self._failures >= self._config.failure_threshold:
            self._open_locked(reason)

    def _publish_locked(self, result: NativeCompletion) -> None:
        self._completed.append(result)
        self._notifier.notify()

    def _skip_locked(self, count: int) -> int:
        task_id = self._next_local_id
        self._next_local_id -= 1
        self._bypassed += 1
        self._publish_locked((task_id, False, "L2 circuit open", [False] * count))
        return task_id

    def _submit(
        self,
        kind: OperationKind,
        keys: list[str],
        buffers: list[memoryview] | None = None,
    ) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("Mooncake client is closed")
            if len(self._pending) + len(self._completed) >= self._config.max_pending:
                self._open_locked("undrained result limit reached")
                raise BufferError(
                    "Drain Mooncake completions before submitting more work"
                )
            native = self._native
            if self._stop.is_set() or self._state != "CLOSED" or native is None:
                return self._skip_locked(len(keys))
            if (
                kind == "get"
                and sum(op.kind == "get" for op in self._pending.values())
                >= self._config.max_pending_get
            ):
                self._open_locked("GET admission limit reached")
                return self._skip_locked(len(keys))
            if (
                len(self._pending) + len(self._expired_lookups)
                >= self._config.max_pending
            ):
                self._open_locked("native operation limit reached")
                return self._skip_locked(len(keys))
            try:
                if kind == "get":
                    assert buffers is not None
                    task_id = native.submit_batch_get(keys, buffers)
                elif kind == "set":
                    assert buffers is not None
                    task_id = native.submit_batch_set(keys, buffers)
                elif kind == "exists":
                    task_id = native.submit_batch_exists(keys)
                else:
                    task_id = native.submit_batch_delete(keys)
            except BufferError:
                self._open_locked("native admission limit reached")
                return self._skip_locked(len(keys))
            except Exception:
                self._open_locked(f"native {kind} submission raised")
                raise
            if (
                task_id < 0
                or task_id in self._pending
                or task_id in self._expired_lookups
            ):
                raise RuntimeError("Native client returned an invalid future id")
            self._pending[task_id] = _Operation(kind, len(keys), time.monotonic())
            return task_id

    def _initialize(self) -> None:
        """Publish one client; shutdown joins this thread before fencing it."""
        try:
            client = self._factory()
            with self._lock:
                self._native = client
                if not self._stop.is_set():
                    self._reason = "waiting for recovery probe"
                    self._next_attempt = 0.0
        except Exception as error:
            logger.exception("Mooncake client initialization failed")
            with self._lock:
                self._reason = f"native setup failed ({type(error).__name__})"
                self._next_attempt = time.monotonic() + self._config.cooldown
        finally:
            with self._lock:
                self._init_busy = False

    def _check_probe(self) -> None:
        """Apply a probe only if no newer native failure invalidated it."""
        try:
            code = self._probe()
            good = isinstance(code, int) and not isinstance(code, bool) and code == 0
        except Exception:
            logger.exception("Mooncake recovery probe failed")
            good = False
        with self._lock:
            elapsed = time.monotonic() - self._probe_started
            self._probe_busy = False
            if self._stop.is_set():
                return
            if not good or elapsed > self._config.probe_timeout:
                self._probe_result = "failed" if not good else "late"
                if self._probe_mode == "healthy":
                    if not self._probe_timeout_reported:
                        self._failure_locked("periodic data probe failed")
                    self._next_healthy_probe = (
                        time.monotonic() + self._config.healthy_probe_interval
                    )
                    return
                self._state = "HALF_OPEN"
                self._open_locked("recovery probe failed or exceeded deadline")
                return
            if self._fault_version != self._probe_version:
                self._probe_result = "discarded_after_new_failure"
                if self._probe_mode == "recovery" and self._state == "HALF_OPEN":
                    self._open_locked("recovery probe invalidated by native failure")
                return
            if self._probe_mode == "healthy":
                self._probe_result = "success"
                self._failures = 0
                self._next_healthy_probe = (
                    time.monotonic() + self._config.healthy_probe_interval
                )
                return
            if self._pending or self._expired_lookups:
                self._probe_result = "healthy_but_native_operations_pending"
                self._state = "HALF_OPEN"
                self._open_locked("waiting for previous native operations to finish")
                return
            if self._native is None or not self._native.is_ready:
                self._state = "HALF_OPEN"
                self._open_locked("waiting for native worker readiness")
                return
            self._recoveries += 1
            self._probe_result = "success"
            if self._recoveries >= self._config.recovery_successes:
                self._state = "CLOSED"
                self._reason = "independent Mooncake data probe succeeded"
                self._failures = 0
                self._recoveries = 0
                self._next_healthy_probe = (
                    time.monotonic() + self._config.healthy_probe_interval
                )
            else:
                self._state = "OPEN"
                self._next_attempt = time.monotonic() + self._config.probe_interval

    def _accept_completions(self, results: list[NativeCompletion]) -> None:
        with self._lock:
            for task_id, ok, error, flags in results:
                if task_id in self._expired_lookups:
                    self._expired_lookups.remove(task_id)
                    continue
                operation = self._pending.pop(task_id, None)
                if operation is None:
                    self._open_locked("completion for an unknown native operation")
                    continue
                failed_data = (
                    operation.kind in ("get", "set") and flags and not all(flags)
                )
                if not ok or failed_data:
                    self._failure_locked(f"native {operation.kind} failed")
                elif self._state == "CLOSED" and operation.kind in ("get", "set"):
                    # EXISTS can collapse SDK errors into misses.
                    self._failures = 0
                self._publish_locked((task_id, ok, error, flags))

    def _tick(self) -> tuple[bool, bool]:
        """Expire buffer-free lookups and schedule at most one probe/setup."""
        now = time.monotonic()
        with self._lock:
            for task_id, operation in list(self._pending.items()):
                age = now - operation.started
                if operation.kind == "exists" and age >= self._config.lookup_timeout:
                    del self._pending[task_id]
                    self._expired_lookups.add(task_id)
                    self._lookup_timeouts += 1
                    self._failure_locked("lookup exceeded total deadline")
                    self._publish_locked(
                        (task_id, False, "lookup deadline", [False] * operation.size)
                    )
                elif (
                    operation.kind in ("get", "set")
                    and age >= self._config.slow_io_threshold
                    and not operation.slow_reported
                ):
                    operation.slow_reported = True
                    self._slow_dma_operations += 1
                    self._open_locked("slow data operation; terminal result pending")
            if (
                self._probe_busy
                and now - self._probe_started > self._config.probe_timeout
                and not self._probe_timeout_reported
            ):
                self._probe_timeout_reported = True
                if self._probe_mode == "healthy":
                    self._failure_locked("periodic data probe exceeded deadline")
                else:
                    self._open_locked(
                        "recovery probe exceeded deadline; worker retained"
                    )
                self._probe_result = "timeout_worker_retained"
            initialize = (
                self._native is None
                and not self._init_busy
                and now >= self._next_attempt
            )
            probe = (
                self._native is not None
                and not self._probe_busy
                and (
                    (self._state == "OPEN" and now >= self._next_attempt)
                    or (self._state == "CLOSED" and now >= self._next_healthy_probe)
                )
            )
            if initialize:
                self._init_busy = True
            if probe:
                self._probe_busy = True
                self._probe_started = now
                self._probe_result = "running"
                self._probe_timeout_reported = False
                self._probe_mode = "healthy" if self._state == "CLOSED" else "recovery"
                self._probe_version = self._fault_version
                if self._probe_mode == "recovery":
                    self._state = "HALF_OPEN"
            return initialize, probe

    def _run(self) -> None:
        """Monitor completions and admission without blocking on SDK setup/probes."""
        while not self._stop.is_set():
            wait = self._config.poll_interval
            try:
                with self._lock:
                    native = self._native
                if native is not None:
                    ready, _, _ = select.select(
                        [native.event_fd()], [], [], self._config.poll_interval
                    )
                    wait = 0
                    if ready:
                        self._accept_completions(native.drain_completions())
                initialize, probe = self._tick()
                if initialize:
                    self._init_thread = threading.Thread(
                        target=self._initialize, name="mooncake-init", daemon=True
                    )
                    self._init_thread.start()
                if probe:
                    self._probe_thread = threading.Thread(
                        target=self._check_probe, name="mooncake-probe", daemon=True
                    )
                    self._probe_thread.start()
            except Exception as error:
                logger.exception("Mooncake circuit monitor failed")
                with self._lock:
                    self._open_locked(f"guard monitor error ({type(error).__name__})")
            self._stop.wait(wait)


class SubprocessDataProbe:
    """Verify remote bytes using a child SDK process with no caller L1 mapping.

    One child is allowed at a time. An unreaped child prevents further probes;
    it is never replaced by an unbounded succession of workers. Termination
    failure is reported to the caller instead of blocking indefinitely.
    """

    def __init__(
        self, setup_config: dict[str, str], timeout: float, kill_timeout: float = 5.0
    ) -> None:
        """Configure an independent PUT/GET/verify/remove probe.

        Args:
            setup_config: SDK connection fields: local_hostname, metadata_server,
                master_server_addr; optional protocol, rdma_devices, tenant_id.
            timeout: Deadline for the complete child operation, including startup.
            kill_timeout: Maximum additional wait for a killed child to exit.

        Raises:
            ValueError: If either deadline is not positive and finite.
        """
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in (timeout, kill_timeout)
        ):
            raise ValueError("Probe deadlines must be positive finite numbers")
        self._input = json.dumps(
            {
                "config": setup_config,
                "key": "__lmcache_circuit_probe__" + uuid.uuid4().hex,
            }
        ).encode("utf-8")
        self._timeout = timeout
        self._kill_timeout = kill_timeout
        self._lock = threading.Lock()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None

    def __call__(self) -> int:
        """Return zero after a verified round trip, otherwise minus one.

        Concurrent calls and calls after close return minus one without starting
        a child. The process has no caller buffers, so probe failure cannot
        release or invalidate L1 allocations.

        Raises:
            LMCacheTimeoutError: If a killed child cannot be reaped in time.
            OSError: If the process could not be started or terminated.
        """
        with self._lock:
            if self._closed or self._process is not None:
                return -1
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            process = subprocess.Popen(
                [sys.executable, "-c", _PROBE_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            self._process = process
        try:
            try:
                process.communicate(self._input, timeout=self._timeout)
            except subprocess.TimeoutExpired:
                return -1
            return 0 if process.returncode == 0 else -1
        finally:
            self._reap(process)
            if process.stdin is not None:
                process.stdin.close()
            with self._lock:
                self._process = None

    def close(self) -> None:
        """Prevent further probes and kill/reap the active child, idempotently.

        Raises:
            LMCacheTimeoutError: If the child remains alive after kill_timeout;
                its handle is retained so a later close can retry termination.
        """
        with self._lock:
            self._closed = True
            process = self._process
        if process is not None:
            self._reap(process)
            with self._lock:
                if self._process is process:
                    self._process = None

    def _reap(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=self._kill_timeout)
        except subprocess.TimeoutExpired as error:
            raise LMCacheTimeoutError("Mooncake probe could not be reaped") from error
