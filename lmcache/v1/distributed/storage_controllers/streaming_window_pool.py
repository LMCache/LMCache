# SPDX-License-Identifier: Apache-2.0
"""Bounded staging windows for streaming reads into device memory."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Protocol
import threading
import time

# First Party
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import L1MemoryManagerProtocol
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError


class CompletionFence(Protocol):
    """Device work completion contract for a staging window."""

    def is_complete(self) -> bool:
        """Return whether device work using the window has completed."""
        ...


@dataclass(frozen=True, eq=False)
class StreamingWindow:
    """One outstanding reservation from a :class:`StreamingWindowPool`."""

    memory_obj: MemoryObj


class StreamingWindowPool:
    """Allocate transient windows from a bounded, dedicated memory manager.

    Window reservations are independent of the L1 object index and its lock
    TTL. A caller must retain a window until the device has stopped reading
    it. Waiters are served in submission order, including across layouts.

    Args:
        allocator: Dedicated memory manager; its lifecycle remains with the
            caller. No other consumer may allocate from it while the pool is
            active, because their releases cannot wake this pool's waiters.
        capacity_bytes: Configured usable size of the allocator.

    Raises:
        ValueError: Capacity is invalid or the allocator cannot expose an
            aligned memory arena.
    """

    def __init__(self, allocator: L1MemoryManagerProtocol, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        desc = allocator.get_l1_memory_desc()
        if desc is None or desc.align_bytes <= 0 or capacity_bytes != desc.size:
            raise ValueError("allocator must expose an aligned arena of this capacity")
        self._allocator = allocator
        self._capacity_bytes = capacity_bytes
        self._alignment_bytes = desc.align_bytes
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active: set[StreamingWindow] = set()
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
            RuntimeError: The pool is closed or allocation failed unexpectedly.
        """
        size = get_size_bytes(layout.shapes, layout.dtypes)
        aligned_size = (
            (size + self._alignment_bytes - 1) // self._alignment_bytes
        ) * self._alignment_bytes
        if size <= 0 or aligned_size > self._capacity_bytes:
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
                        error, objects = self._allocator.allocate(layout, 1)
                        if error == L1Error.SUCCESS:
                            window = StreamingWindow(objects[0])
                            self._active.add(window)
                            self._waiters.popleft()
                            self._condition.notify_all()
                            return window
                        if error != L1Error.OUT_OF_MEMORY:
                            raise RuntimeError(f"staging allocation failed: {error}")
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

    def release(
        self, window: StreamingWindow, completion: CompletionFence | None = None
    ) -> bool:
        """Return a window when its device transfer has completed.

        Args:
            window: A currently owned window from this pool.
            completion: Completion fence recorded after the device transfer;
                None is permitted only if no device work was submitted.

        Returns:
            False if the fence is not complete, leaving the window owned;
            otherwise True after returning its memory to the pool.

        Raises:
            ValueError: The window is not currently owned by this pool.
            RuntimeError: The allocator failed to release the window.
        """
        with self._condition:
            if window not in self._active:
                raise ValueError("staging window is not active in this pool")
            if completion is not None and not completion.is_complete():
                return False
            error = self._allocator.free([window.memory_obj])
            if error != L1Error.SUCCESS:
                raise RuntimeError(f"staging release failed: {error}")
            self._active.remove(window)
            self._condition.notify_all()
            return True

    def get_memory_usage(self) -> tuple[int, int]:
        """Return allocator used and total bytes."""
        with self._condition:
            return self._allocator.get_memory_usage()

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
