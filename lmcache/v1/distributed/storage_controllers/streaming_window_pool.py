# SPDX-License-Identifier: Apache-2.0
"""Bounded staging windows for streaming reads into device memory."""

# Standard
from collections import deque
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError


@dataclass(frozen=True, eq=False)
class StreamingWindow:
    """One outstanding reservation from a :class:`StreamingWindowPool`."""

    id: int
    memory_obj: MemoryObj


class StreamingWindowPool:
    """Allocate transient windows from a bounded, dedicated memory manager.

    Window reservations are independent of the L1 object index and its lock
    TTL. A caller must retain a window until the device has stopped reading
    it. Waiters are served in submission order, including across layouts.

    Args:
        arena: A flat, contiguous byte tensor reserved from DRAM L1. The
            caller keeps it allocated while windows are active.
    """

    def __init__(self, arena: torch.Tensor) -> None:
        self._capacity_bytes = arena.numel()
        self._allocator = TensorMemoryAllocator(arena, align_bytes=4096)
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active: dict[int, StreamingWindow] = {}
        self._next_id = 0
        self._closed = False

    def acquire(
        self, layout: MemoryLayoutDesc, timeout: float | None = None
    ) -> StreamingWindow:
        """Wait for and allocate one window with the requested layout.

        Args:
            layout: Layout of the L2 object to be transferred.
            timeout: Maximum wait in seconds, or None for no deadline.

        Returns:
            An owned window. Release it after its device completion event.

        Raises:
            ValueError: The requested object cannot fit in the pool.
            LMCacheTimeoutError: The deadline expired while waiting.
            RuntimeError: The pool is closed.
        """
        size = get_size_bytes(layout.shapes, layout.dtypes)
        if not 0 < size <= self._capacity_bytes:
            raise ValueError("window size must fit in the staging pool")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        waiter = object()
        with self._condition:
            if self._closed:
                raise RuntimeError("staging window pool is closed")
            self._waiters.append(waiter)
            try:
                while True:
                    if self._closed:
                        raise RuntimeError("staging window pool is closed")
                    if self._waiters[0] is waiter:
                        objects = self._allocator.batched_allocate(
                            layout.shapes, layout.dtypes, 1
                        )
                        if objects is not None:
                            window = StreamingWindow(self._next_id, objects[0])
                            self._next_id += 1
                            self._active[window.id] = window
                            self._waiters.popleft()
                            self._condition.notify_all()
                            return window
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise LMCacheTimeoutError(
                            "timed out waiting for a staging window"
                        )
                    self._condition.wait(remaining)
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    def release(self, window: StreamingWindow) -> None:
        """Return an acquired window after its last reader has finished.

        Args:
            window: An active reservation from this pool.

        Raises:
            ValueError: The window is not active in this pool.
        """
        with self._condition:
            if self._active.get(window.id) is not window:
                raise ValueError("staging window is not active in this pool")
            self._release([window])

    def release_completed(self, window_ids: list[int]) -> None:
        """Return windows named by the GPU stream completion callback.

        Args:
            window_ids: IDs of the active windows used by the completed copy.

        Raises:
            KeyError: A window ID is not active.
        """
        with self._condition:
            self._release([self._active[window_id] for window_id in window_ids])

    def get_memory_usage(self) -> tuple[int, int]:
        """Return allocator used and total bytes."""
        with self._condition:
            return self._allocator.total_allocated_size, self._capacity_bytes

    def shutdown(self) -> None:
        """Stop admitting windows and wake waiters; active windows may return."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until all acquired windows are released.

        Args:
            timeout: Maximum wait in seconds, or None for no deadline.

        Returns:
            True when the pool has no active windows, False on timeout.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        """Close the pool after all windows are returned and wake waiters.

        Raises:
            RuntimeError: One or more windows are still owned by callers.
        """
        with self._condition:
            if self._active:
                raise RuntimeError("cannot close a pool with active windows")
            self._closed = True
            self._condition.notify_all()

    def _release(self, windows: list[StreamingWindow]) -> None:
        self._allocator.batched_free([window.memory_obj for window in windows])
        for window in windows:
            del self._active[window.id]
        self._condition.notify_all()
