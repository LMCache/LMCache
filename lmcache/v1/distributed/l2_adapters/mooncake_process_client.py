# SPDX-License-Identifier: Apache-2.0
"""Experimental process transport for native Mooncake operations.

This component is deliberately not registered as an L2 adapter. Callers must own
every submitted buffer until its terminal completion or quiesce, independently
of expiring L1 locks. See the mirrored design document for integration gates.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, Protocol
import ctypes
import importlib.util
import json
import math
import mmap
import os
import queue
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

logger = init_logger(__name__)
MAX_FRAME = 16 << 20
MAX_KEYS = 16384


class ProcessNotifier(Protocol):
    """Pollable completion notifier owned by the process client."""

    def fileno(self) -> int:
        """Return the pollable file descriptor."""
        ...

    def notify(self) -> None:
        """Signal that completions are available."""
        ...

    def consume(self) -> None:
        """Consume accumulated wakeups without blocking."""
        ...

    def close(self) -> None:
        """Release the notifier."""
        ...


def _open_shared_arena(path: Path, base: int, size: int) -> int:
    """Retain the exact file backing the parent's writable shared mapping."""
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_size != size:
            raise ValueError("SHM file size does not match the L1 descriptor")
        cursor = base
        for line in Path("/proc/self/maps").read_text().splitlines():
            address, permissions, offset, device, inode, *_ = line.split()
            start, end = (int(part, 16) for part in address.split("-"))
            if not start <= cursor < end:
                continue
            major, minor = (int(part, 16) for part in device.split(":"))
            if (
                permissions != "rw-s"
                or int(inode) != info.st_ino
                or os.makedev(major, minor) != info.st_dev
                or int(offset, 16) + cursor - start != cursor - base
            ):
                raise ValueError("L1 pointer does not map the supplied shared file")
            cursor = min(end, base + size)
            if cursor == base + size:
                return descriptor
        raise ValueError("L1 pointer is not covered by the supplied shared file")
    except BaseException:
        os.close(descriptor)
        raise


def _frame(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_FRAME:
        raise ValueError("Native command exceeds the frame limit")
    return struct.pack("!I", len(data)) + data


def _read_message(connection: socket.socket) -> dict[str, Any]:
    def exact(size: int) -> bytes:
        pieces = bytearray()
        while len(pieces) < size:
            data = connection.recv(size - len(pieces))
            if not data:
                raise EOFError("Native worker socket closed")
            pieces.extend(data)
        return bytes(pieces)

    length = struct.unpack("!I", exact(4))[0]
    if length > MAX_FRAME:
        raise ValueError("Native frame exceeds the limit")
    value = json.loads(exact(length))
    if not isinstance(value, dict):
        raise TypeError("Native message must be an object")
    return value


def run_worker(
    connection: socket.socket, native_factory: Callable[..., Any] | None = None
) -> None:
    """Attach the inherited shared arena and dispatch the private protocol.

    Args:
        connection: Private parent/child socket, starting with the bootstrap frame.
        native_factory: Test implementation of the SDK protocol. When omitted,
            import the native Mooncake extension only in this child.

    Raises:
        ValueError: The shared descriptor or protocol bounds are invalid.
        ImportError: Native extension loading fails.

    This function runs only in a dedicated child. Fatal dispatch/completion
    failures exit that process so the parent can fence outstanding operations.
    """
    bootstrap = _read_message(connection)
    descriptor = bootstrap["shm_fd"]
    info = os.fstat(descriptor)
    if (info.st_dev, info.st_ino, info.st_size) != (
        bootstrap["shm_dev"],
        bootstrap["shm_ino"],
        bootstrap["l1_size"],
    ):
        raise ValueError("Shared L1 identity changed before worker attachment")
    arena = mmap.mmap(descriptor, bootstrap["l1_size"], access=mmap.ACCESS_WRITE)
    os.close(descriptor)
    child_base = ctypes.addressof(ctypes.c_ubyte.from_buffer(arena))
    if native_factory is None:
        # Third Party
        import torch  # noqa: F401 - Resolve native extension libtorch dependencies.

        extensions = list(
            Path(__file__).resolve().parents[3].glob("lmcache_mooncake*.so")
        )
        if len(extensions) != 1:
            raise RuntimeError("Expected one native Mooncake extension")
        specification = importlib.util.spec_from_file_location(
            "lmcache_mooncake", extensions[0]
        )
        if specification is None or specification.loader is None:
            raise ImportError("Cannot load the native Mooncake extension")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        registration = module.L1RegistrationConfig()
        registration.enabled, registration.base, registration.size = (
            True,
            child_base,
            len(arena),
        )
        native = module.LMCacheMooncakeClient(
            config=bootstrap["setup_config"],
            num_workers=bootstrap["num_workers"],
            l1_registration=registration,
            per_op_workers=bootstrap["per_op_workers"],
        )
    else:
        native = native_factory(bootstrap, child_base, arena)
    state_lock, send_lock = threading.Lock(), threading.Lock()
    pending: dict[int, tuple[int, str, list[memoryview]]] = {}

    def send(message: dict[str, Any]) -> None:
        with send_lock:
            connection.sendall(_frame(message))

    def commands() -> None:
        try:
            while True:
                message = _read_message(connection)
                kind, keys = message["kind"], message["keys"]
                if (
                    kind not in ("exists", "get", "set", "delete")
                    or len(keys) > MAX_KEYS
                ):
                    raise ValueError("Invalid native command")
                views = []
                for offset, size in message["buffers"]:
                    if (
                        not isinstance(offset, int)
                        or not isinstance(size, int)
                        or offset < 0
                        or size <= 0
                        or offset + size > len(arena)
                    ):
                        raise ValueError("Native shared-buffer bounds violation")
                    views.append(memoryview(arena)[offset : offset + size])
                if kind in ("get", "set") and len(views) != len(keys):
                    raise ValueError("Native key/buffer count mismatch")
                with state_lock:
                    method = getattr(native, f"submit_batch_{kind}")
                    future = (
                        method(keys, views) if kind in ("get", "set") else method(keys)
                    )
                    pending[int(future)] = (message["id"], kind, views)
        except BaseException:  # noqa: BLE001 - Fatal worker protocol boundary.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            os._exit(71)

    threading.Thread(target=commands, name="native-commands", daemon=True).start()
    send({"ready": True})
    while True:
        readable, _, _ = select.select([native.event_fd()], [], [], 0.05)
        if not readable:
            continue
        results = native.drain_completions()
        with state_lock:
            for future, ok, _error, flags in results:
                if int(future) not in pending:
                    os._exit(72)
                task_id, kind, _views = pending.pop(int(future))
                if kind in ("get", "set") and (
                    not ok or (flags is not None and not all(flags))
                ):
                    # Do not release the parent's L1 locks before QP destruction.
                    os._exit(73)
                send({"id": task_id, "ok": bool(ok), "flags": flags})


class ProcessNativeClient:
    """Provide nonblocking native operations backed by a private SDK worker.

    Only the manager thread writes to the private socket. Submission never waits
    for socket I/O. This Linux-only component passes offsets, not KV payloads.

    The caller must retain the arena and exclusive GET destinations / immutable
    SET sources until completion. Timeouts do not replace allocator ownership.
    Failed GET/SET completions are published only after the worker is reaped.
    An unconfirmed worker exit terminates the parent instead of releasing DMA
    buffers. A fresh arena is required after a parent crash.
    """

    def __init__(
        self,
        setup_config: dict[str, str],
        num_workers: int,
        per_op_workers: dict[str, int] | None,
        l1_base: int,
        l1_size: int,
        shm_name: str,
        notifier: ProcessNotifier,
        operation_timeout: float = 8.0,
        startup_timeout: float = 15.0,
        kill_timeout: float = 10.0,
        restart_interval: float = 2.0,
        max_pending: int = 16,
        shm_root: str | Path = "/dev/shm",
        worker_command: list[str] | None = None,
        fatal_marker: str | Path | None = None,
        fatal_exit: Callable[[int], NoReturn] = os._exit,
    ) -> None:
        """Start the worker manager after validating shared mapping identity.

        Args:
            setup_config: SDK configuration passed unchanged to the child.
            num_workers: Native shared thread count.
            per_op_workers: Optional dedicated native thread counts.
            l1_base: Address of the parent's mapping, starting at file offset 0.
            l1_size: Full mapping and backing file size in bytes.
            shm_name: Basename of the file backing that mapping.
            notifier: Nonblocking completion notifier; ownership transfers here.
            operation_timeout: Submitted operation deadline, in seconds.
            startup_timeout: Native initialization deadline, in seconds.
            kill_timeout: Maximum wait for a killed worker before fatal exit.
            restart_interval: Delay before starting a replacement worker.
            max_pending: Maximum accepted operations with undrained results.
            shm_root: Directory containing the shared file.
            worker_command: Alternate worker executable for protocol tests.
            fatal_marker: Optional diagnostic file for unreaped workers.
            fatal_exit: Fatal process exit function, injectable for tests.

        Raises:
            ValueError: Invalid timings, sizes, names or mapping identity.
            RuntimeError: Platform is not Linux.
            OSError: Shared mapping metadata cannot be read.
        """
        if sys.platform != "linux":
            raise RuntimeError("Process Mooncake transport requires Linux")
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", shm_name):
            raise ValueError("SHM name must be a canonical basename")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (l1_base, l1_size, max_pending)
        ):
            raise ValueError("L1 descriptor and pending limit must be positive")
        for value in (
            operation_timeout,
            startup_timeout,
            kill_timeout,
            restart_interval,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "Native process timings must be positive finite numbers"
                )
        self._base, self._size = l1_base, l1_size
        self._notifier = notifier
        self._timeout, self._startup_timeout = operation_timeout, startup_timeout
        self._kill_timeout, self._restart_interval = kill_timeout, restart_interval
        self._limit = max_pending
        self._command = worker_command or [
            sys.executable,
            str(Path(__file__).resolve()),
            "--native-worker",
        ]
        self._marker = Path(fatal_marker) if fatal_marker is not None else None
        self._fatal_exit = fatal_exit
        shm_path = Path(shm_root) / shm_name
        self._arena_fd = _open_shared_arena(shm_path, l1_base, l1_size)
        info = os.fstat(self._arena_fd)
        self._bootstrap = {
            "setup_config": setup_config,
            "num_workers": num_workers,
            "per_op_workers": per_op_workers,
            "shm_fd": self._arena_fd,
            "l1_size": l1_size,
            "shm_dev": info.st_dev,
            "shm_ino": info.st_ino,
        }
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._ready = False
        self._process: subprocess.Popen | None = None
        self._connection: socket.socket | None = None
        self._outbound: queue.Queue[tuple[int, bytes]] = queue.Queue(max_pending)
        self._pending: dict[int, tuple[str, float, int]] = {}
        self._completed: list[tuple[int, bool, str, list[bool] | None]] = []
        self._next_id = 1
        self._generation = 0
        self._reason = "starting worker"
        self._worker_pid: int | None = None
        self._aborted = 0
        self._manager = threading.Thread(
            target=self._manage, name="mooncake-native-process", daemon=True
        )
        try:
            self._manager.start()
        except BaseException:
            os.close(self._arena_fd)
            raise

    @property
    def is_ready(self) -> bool:
        """Whether the current SDK worker has registered its shared L1 map."""
        with self._lock:
            return self._ready and not self._stop.is_set()

    def event_fd(self) -> int:
        """Return the completion descriptor, valid until close."""
        return int(self._notifier.fileno())

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Submit a lookup; return its unique completion ID."""
        return self._submit("exists", keys)

    def submit_batch_get(self, keys: list[str], buffers: list[Any]) -> int:
        """Submit GETs into caller-owned shared buffers; return a completion ID."""
        return self._submit("get", keys, buffers)

    def submit_batch_set(self, keys: list[str], buffers: list[Any]) -> int:
        """Submit SETs from immutable shared buffers; return a completion ID."""
        return self._submit("set", keys, buffers)

    def submit_batch_delete(self, keys: list[str]) -> int:
        """Submit deletes; return their unique completion ID."""
        return self._submit("delete", keys)

    def drain_completions(self) -> list[tuple[int, bool, str, list[bool] | None]]:
        """Consume wakeups and return each terminal result exactly once."""
        with self._lock:
            self._notifier.consume()
            results = self._completed
            self._completed = []
            return results

    def report_status(self) -> dict[str, Any]:
        """Return worker readiness, generation, queue usage and fault counters."""
        with self._lock:
            return {
                "worker_ready": self._ready,
                "worker_pid": self._worker_pid,
                "worker_generation": self._generation,
                "worker_reason": self._reason,
                "worker_pending": len(self._pending),
                "worker_undrained": len(self._completed),
                "worker_aborts": self._aborted,
                "worker_operation_timeout": self._timeout,
                "worker_manager_alive": self._manager.is_alive(),
                "native_process_config": {
                    "operation_timeout": self._timeout,
                    "startup_timeout": self._startup_timeout,
                    "kill_timeout": self._kill_timeout,
                    "restart_interval": self._restart_interval,
                    "max_pending": self._limit,
                },
            }

    def close(self) -> None:
        """Wait for worker process death before allowing shared L1 teardown."""
        with self._close_lock:
            if self._closed:
                return
            self.quiesce()
            with self._lock:
                self._closed = True
                self._notifier.close()
                os.close(self._arena_fd)

    def quiesce(self) -> None:
        """Reap the worker and publish terminal failures, retaining the notifier.

        Further submissions safely fail without reaching a worker. Callers can
        drain results before close releases notification resources.
        """
        with self._close_lock:
            self._stop.set()
            self._manager.join()

    def _publish_locked(
        self, task_id: int, ok: bool, reason: str, flags: list[bool] | None
    ) -> None:
        self._completed.append((task_id, ok, reason, flags))
        self._notifier.notify()

    def _buffers(self, buffers: list[Any]) -> list[list[int]]:
        locations = []
        for buffer in buffers:
            view = memoryview(buffer)
            if not view.c_contiguous or view.nbytes == 0:
                raise ValueError("Native buffers must be nonempty contiguous views")
            address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
            offset = address - self._base
            if offset < 0 or offset + view.nbytes > self._size:
                raise ValueError("Native buffer lies outside the shared L1 arena")
            locations.append([offset, view.nbytes])
        return locations

    def _submit(
        self, kind: str, keys: list[str], buffers: list[Any] | None = None
    ) -> int:
        if len(keys) > MAX_KEYS or any(not isinstance(key, str) for key in keys):
            raise ValueError("Native keys exceed the supported batch layout")
        locations = self._buffers(buffers) if buffers is not None else []
        if buffers is not None and len(keys) != len(locations):
            raise ValueError("Native key/buffer lengths differ")
        with self._lock:
            if self._closed:
                raise RuntimeError("Native process client is closed")
            if len(self._pending) + len(self._completed) >= self._limit:
                raise BufferError(
                    "Drain native completions before submitting more work"
                )
            task_id = self._next_id
            self._next_id += 1
            if self._stop.is_set() or not self._ready:
                self._publish_locked(
                    task_id, False, "native worker unavailable", [False] * len(keys)
                )
                return task_id
            message = _frame(
                {"id": task_id, "kind": kind, "keys": keys, "buffers": locations}
            )
            self._pending[task_id] = (kind, time.monotonic(), len(keys))
            try:
                self._outbound.put_nowait((task_id, message))
            except queue.Full:
                del self._pending[task_id]
                self._publish_locked(
                    task_id, False, "native queue full", [False] * len(keys)
                )
            return task_id

    def _start(self) -> None:
        parent, child = socket.socketpair()
        # Keep the child on the same package source as its parent, including
        # source-checkout validation where lmcache is not installed as a wheel.
        package_root = str(Path(__file__).resolve().parents[4])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (package_root, environment.get("PYTHONPATH", "")))
        )
        try:
            process = subprocess.Popen(
                self._command + [str(child.fileno())],
                pass_fds=(child.fileno(), self._arena_fd),
                env=environment,
            )
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        parent.settimeout(0.05)
        with self._lock:
            self._process, self._connection = process, parent
            self._worker_pid = process.pid
            self._generation += 1
        parent.sendall(_frame(self._bootstrap))

    def _fatal(self, reason: str, worker_pid: int | None) -> None:
        record = {
            "pid": os.getpid(),
            "time": time.time(),
            "reason": reason,
            "native_worker_pid": worker_pid,
        }
        if self._marker is not None:
            try:
                self._marker.parent.mkdir(parents=True, exist_ok=True)
                with self._marker.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            except OSError as error:
                print(
                    "Native fatal marker could not be written "
                    f"({type(error).__name__})",
                    file=sys.stderr,
                    flush=True,
                )
        self._fatal_exit(78)
        raise RuntimeError("Fatal exit handler unexpectedly returned")

    def _abort(self, reason: str) -> None:
        with self._lock:
            self._ready = False
            self._reason = reason
            process, connection = self._process, self._connection
        if connection is not None:
            connection.close()
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=self._kill_timeout)
            except subprocess.TimeoutExpired:
                self._fatal("native worker could not be reaped after kill", process.pid)
            except OSError:
                self._fatal(
                    "native worker termination could not be confirmed", process.pid
                )
        # Supported Linux providers tear down worker-owned verbs resources on exit.
        with self._lock:
            self._process = self._connection = None
            self._aborted += 1
            for task_id in self._pending:
                self._publish_locked(task_id, False, reason, None)
            self._pending.clear()
            while True:
                try:
                    self._outbound.get_nowait()
                except queue.Empty:
                    break

    def _message(self, message: dict[str, Any]) -> None:
        if message.get("ready") is True:
            with self._lock:
                self._ready = True
                self._reason = "worker ready"
            return
        task_id, ok, flags = message["id"], message["ok"], message.get("flags")
        if not isinstance(task_id, int) or isinstance(task_id, bool):
            raise TypeError("Native completion id must be an integer")
        with self._lock:
            if task_id not in self._pending:
                raise ValueError("Native completion id is not pending")
            kind, _started, expected = self._pending[task_id]
            if not isinstance(ok, bool) or (
                flags is not None
                and (
                    not isinstance(flags, list)
                    or len(flags) != expected
                    or any(not isinstance(flag, bool) for flag in flags)
                )
            ):
                raise ValueError("Malformed native completion")
            if kind in ("get", "set") and (
                not ok or (flags is not None and not all(flags))
            ):
                raise RuntimeError("Unsafe native DMA failure received")
            del self._pending[task_id]
            self._publish_locked(
                task_id, bool(ok), "" if ok else "native operation failed", flags
            )

    def _serve(self) -> None:
        started = time.monotonic()
        incoming = bytearray()
        while not self._stop.is_set():
            with self._lock:
                connection, process = self._connection, self._process
                ready = self._ready
                oldest = min(
                    (entry[1] for entry in self._pending.values()), default=None
                )
            if process is None or connection is None or process.poll() is not None:
                raise EOFError("Native worker exited")
            now = time.monotonic()
            if not ready and now - started >= self._startup_timeout:
                raise LMCacheTimeoutError("Native worker startup deadline")
            if oldest is not None and now - oldest >= self._timeout:
                raise LMCacheTimeoutError("Native operation deadline")
            for _ in range(self._limit):
                try:
                    task_id, data = self._outbound.get_nowait()
                except queue.Empty:
                    break
                with self._lock:
                    pending = task_id in self._pending
                if pending:
                    connection.sendall(data)
            readable, _, _ = select.select([connection], [], [], 0.01)
            if not readable:
                continue
            data = connection.recv(262144)
            if not data:
                raise EOFError("Native worker closed its channel")
            incoming.extend(data)
            while len(incoming) >= 4:
                length = struct.unpack("!I", incoming[:4])[0]
                if length > MAX_FRAME:
                    raise ValueError("Native worker frame exceeds the limit")
                if len(incoming) < length + 4:
                    break
                self._message(json.loads(incoming[4 : 4 + length]))
                del incoming[: 4 + length]

    def _manage(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._start()
                    self._serve()
                    reason = "native client stopping"
                except Exception as error:  # noqa: BLE001 - Isolate SDK faults.
                    logger.exception("Native worker failed; fencing pending operations")
                    reason = f"native worker fault ({type(error).__name__})"
                self._abort(reason)
                self._stop.wait(self._restart_interval)
        finally:
            self._abort("native client stopped")


if __name__ == "__main__" and sys.argv[1:2] == ["--native-worker"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        run_worker(socket.socket(fileno=int(sys.argv[2])))
    except BaseException:  # noqa: BLE001 - Fatal worker initialization boundary.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(74)
