# SPDX-License-Identifier: Apache-2.0
"""Lightweight store-path timing utility for multiprocess transfer contexts."""

# Standard
import logging
import threading
import time

# First Party
from lmcache.utils import init_logger

logger = init_logger(__name__)


class StoreTimer:
    """Thread-safe timing utility supporting multiple named groups.

    Only records timestamps when debug logging is enabled — zero overhead in
    production.  All public methods (mark, emit) early-return immediately
    when not at DEBUG level, avoiding any lock acquisition, dict lookup, or
    time.perf_counter() call.

    A single timer instance can track multiple independent *names* (e.g.
    different store paths or sub-operations), each with its own ordered
    sequence of (step, time) entries.  Safe for concurrent use from multiple
    threads.

    The log output shows **elapsed time between adjacent steps** (delta),
    not absolute time from t0, making it easy to identify which step is the
    bottleneck.

    Usage::

        timer = StoreTimer(prefix="req-42")

        # Thread A — GPU IPC path
        timer.mark("gpu_ipc", "copy_start")
        timer.mark("gpu_ipc", "copy_done")
        timer.mark("gpu_ipc", "kv_releasable")
        timer.emit("gpu_ipc")
        # [STORE-TIMING] prefix=req-42 name=gpu_ipc
        #   copy_start -> copy_done=3.089ms copy_done ->
        #   kv_releasable=2.228ms total=5.317ms

        # Thread B — SHM path (can run concurrently)
        timer.mark("shm", "serialize_start")
        timer.mark("shm", "serialize_done")
        timer.mark("shm", "write_complete")
        timer.emit("shm")

    Args:
        prefix: Optional prefix prepended to all log lines for easier
            grep / filtering (e.g. request id).
    """

    __slots__ = ("_enabled", "_prefix", "_groups", "_lock")

    def __init__(self, prefix: str = "") -> None:
        """Initialize the timer.

        Args:
            prefix: Optional prefix for log output (e.g. request id).
        """
        self._enabled = logger.isEnabledFor(logging.DEBUG)
        if not self._enabled:
            return
        self._prefix = prefix
        self._groups: dict[str, list[tuple[str, float]]] = {}
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        """Whether the timer is enabled."""
        return self._enabled

    def mark(self, name: str, step: str) -> None:
        """Record a step under a named group.

        Early-returns when debug logging is disabled — no perf_counter call,
        no lock, no dict access.  Thread-safe.  The same (name, step) pair
        can be recorded multiple times (reentrant) — each occurrence is kept.

        Args:
            name: Group name identifying the operation or path (e.g.
                ``"gpu_ipc"``, ``"shm"``, ``"pickle"``).
            step: Step name describing what just completed (e.g.
                ``"copy_start"``, ``"copy_done"``).
        """
        if not self._enabled:
            return
        t = time.perf_counter()
        with self._lock:
            self._groups.setdefault(name, []).append((step, t))

    def emit(self, name: str) -> None:
        """Emit a ``[STORE-TIMING]`` debug log line for a specific name group.

        The output shows the **delta** between each pair of adjacent steps,
        plus the total time from first to last step.

        Early-returns when debug logging is disabled.  If *name* has not been
        recorded or has fewer than 2 steps, this is a no-op.

        Args:
            name: The group name to emit timing for.
        """
        if not self._enabled:
            return
        with self._lock:
            entries = self._groups.pop(name, None)
            if entries is None or len(entries) < 2:
                return
            # snapshot under lock
            entries = list(entries)

        # Build delta pairs: step_a -> step_b=<delta>ms
        parts = []
        for i in range(1, len(entries)):
            prev_step, prev_t = entries[i - 1]
            curr_step, curr_t = entries[i]
            delta_ms = (curr_t - prev_t) * 1000
            parts.append(f"{prev_step} -> {curr_step}={delta_ms:.3f}ms")

        total_ms = (entries[-1][1] - entries[0][1]) * 1000
        parts.append(f"total={total_ms:.3f}ms")

        logger.debug(
            "[STORE-TIMING] %sname=%s %s",
            f"prefix={self._prefix} " if self._prefix else "",
            name,
            " ".join(parts),
        )

    def emit_all(self) -> None:
        """Emit timing lines for all recorded name groups.

        Early-returns when debug logging is disabled.
        """
        if not self._enabled:
            return
        with self._lock:
            names = list(self._groups.keys())
        for name in names:
            self.emit(name)
