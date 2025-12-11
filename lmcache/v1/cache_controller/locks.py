# SPDX-License-Identifier: Apache-2.0
"""
Lock utilities for thread-safe operations.

This module provides thread synchronization primitives with timeout support.
"""

# Standard
from contextlib import contextmanager
from typing import Optional
import threading
import time


class RWLockTimeoutError(Exception):
    """Exception raised when a lock acquisition times out."""

    pass


class RWLockWithTimeout:
    """
    A simple read-write lock with timeout support.
    Multiple readers can hold the lock simultaneously, but only one writer.
    """

    def __init__(self):
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._condition = threading.Condition(threading.Lock())

    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """Acquire a read lock with optional timeout."""
        deadline = time.time() + timeout if timeout is not None else None

        with self._condition:
            while self._writer_active or self._writers_waiting > 0:
                if deadline is not None and time.time() >= deadline:
                    return False
                remaining = deadline - time.time() if deadline else None
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            self._readers += 1
            return True

    def release_read(self):
        """Release a read lock."""
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """Acquire a write lock with optional timeout."""
        deadline = time.time() + timeout if timeout is not None else None

        with self._condition:
            self._writers_waiting += 1
            try:
                while self._readers > 0 or self._writer_active:
                    if deadline is not None and time.time() >= deadline:
                        return False
                    remaining = deadline - time.time() if deadline else None
                    if remaining is not None and remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
                self._writer_active = True
                return True
            finally:
                self._writers_waiting -= 1

    def release_write(self):
        """Release a write lock."""
        with self._condition:
            self._writer_active = False
            self._condition.notify_all()

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Context manager for read lock with timeout."""
        # TODO(baoloongmao): Move timeout values to configuration
        effective_timeout = 0.05 if timeout is None else timeout  # 50ms default
        if not self.acquire_read(effective_timeout):
            raise RWLockTimeoutError("Failed to acquire read lock within timeout")
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Context manager for write lock with timeout."""
        # TODO(baoloongmao): Move timeout values to configuration
        effective_timeout = 0.1 if timeout is None else timeout  # 100ms default
        if not self.acquire_write(effective_timeout):
            raise RWLockTimeoutError("Failed to acquire write lock within timeout")
        try:
            yield
        finally:
            self.release_write()


class FastLockWithTimeout:
    """
    A fast lock with timeout support for WorkerNode.
    Optimized for high frequency operations on small critical sections.
    Uses non-blocking fast path for better performance.
    """

    __slots__ = ("_lock",)

    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the lock with optional timeout."""
        if timeout is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self):
        """Release the lock."""
        self._lock.release()

    def __enter__(self):
        # Fast path: try non-blocking acquire first (no syscall overhead)
        if self._lock.acquire(blocking=False):
            return self
        # Slow path: reduced timeout for faster failure detection
        if not self._lock.acquire(timeout=10):  # 10s timeout
            # TODO(baoloongmao): Mark as operation failed for metrics
            #  and schedule full sync
            raise RWLockTimeoutError("Failed to acquire WorkerNode lock within 10s")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
