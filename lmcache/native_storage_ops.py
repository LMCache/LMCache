# SPDX-License-Identifier: Apache-2.0
"""Pure-Python fallback implementations for ``lmcache.native_storage_ops``.

The C++ extension (``native_storage_ops.cpython-*.so``) is the production path
and is preferred whenever it is available. This Python module exists so that
CPU-only / non-CUDA environments and CI runners that lack the compiled
extension can still ``import lmcache.native_storage_ops`` without raising
``ImportError``.

Public API mirrors ``lmcache/native_storage_ops.pyi`` exactly:

* :class:`TTLLock`
* :class:`Bitmap`
* :class:`ParallelPatternMatcher`
* :class:`PeriodicEventNotifier`
* :class:`RangePatternMatcher`

Performance notice
------------------
These implementations prioritize correctness and portability over speed. They
are NOT a replacement for the C++ implementation on performance-critical paths
(e.g. :class:`TTLLock` is backed by ``std::atomic`` in C++ but uses
``threading.Lock`` here). Use them only when the native extension is
unavailable.

"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any, Optional, Set
import errno
import os
import struct
import threading
import time

__all__ = [
    "TTLLock",
    "Bitmap",
    "ParallelPatternMatcher",
    "PeriodicEventNotifier",
    "RangePatternMatcher",
]


# ---------------------------------------------------------------------------
# TTLLock
# ---------------------------------------------------------------------------
class TTLLock:
    """Thread-safe lock with TTL (Time-To-Live) semantics.

    Mirrors the C++ ``TTLLock``. The lock maintains a counter that can be
    incremented by :meth:`lock` and decremented by :meth:`unlock`. If the TTL
    expires the lock is considered released regardless of the counter.

    Notes
    -----
    The C++ implementation uses ``std::atomic`` for lock-free operation. This
    Python fallback uses :class:`threading.Lock`, which is correct under the
    GIL but has different performance characteristics. Use :func:`time.monotonic`
    so that wall-clock adjustments cannot break TTL semantics.
    """

    def __init__(self, ttl_second: int = 300) -> None:
        self._ttl_ms: int = int(ttl_second * 1000)
        self._mu: threading.Lock = threading.Lock()
        self._count: int = 0
        self._expire_at_ms: int = 0

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def _expired(self) -> bool:
        return self._expire_at_ms <= self._now_ms()

    def lock(self) -> None:
        """Increment the counter and refresh the TTL.

        If the previous TTL has expired, the counter is reset before the
        increment, so a stale (expired) state always becomes count == 1.
        """
        with self._mu:
            if self._expired():
                self._count = 0
            self._count += 1
            self._expire_at_ms = self._now_ms() + self._ttl_ms

    def unlock(self) -> None:
        """Decrement the counter (clamped at zero).

        Intentionally does NOT clear the counter on TTL expiration; that is the
        responsibility of :meth:`is_locked` / :meth:`lock`. Clearing here would
        break the counting-semaphore contract under concurrent holders.
        """
        with self._mu:
            if self._count > 0:
                self._count -= 1

    def is_locked(self) -> bool:
        """Return True iff the lock is currently held and not expired."""
        with self._mu:
            if self._expired():
                self._count = 0
                return False
            return self._count > 0

    def reset(self) -> None:
        """Reset the lock to the initial state (counter = 0, TTL expired)."""
        with self._mu:
            self._count = 0
            self._expire_at_ms = 0


# ---------------------------------------------------------------------------
# Bitmap
# ---------------------------------------------------------------------------
class Bitmap:
    """Bitmap for tracking the success/failure of L2 storage operations.

    Each bit represents the state of a single key. Out-of-range single-bit
    operations are silently ignored to match the C++ contract (which makes
    callers' lives easier when sizes change).
    """

    __slots__ = ("_data", "_size")

    def __init__(self, size: int, prefix_bits: Optional[int] = None) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")

        self._size = size
        if size == 0:
            self._data = bytearray()
        else:
            num_bytes = (size + 7) // 8
            self._data = bytearray(num_bytes)

        if prefix_bits is not None:
            prefix_bits = int(prefix_bits)
            if prefix_bits > 0:
                if prefix_bits > size:
                    prefix_bits = size
                for i in range(prefix_bits):
                    self.set(i)

    # -- helpers ------------------------------------------------------------
    def _byte_index(self, bit_index: int) -> int:
        return bit_index // 8

    def _bit_offset(self, bit_index: int) -> int:
        return bit_index % 8

    # -- single-bit ops -----------------------------------------------------
    def set(self, index: int) -> None:
        """Set the bit at the specified index to 1.

        Out-of-range indices are silently ignored.

        Args:
            index: The bit position to set.
        """
        if 0 <= index < self._size:
            byte_idx = self._byte_index(index)
            bit_offset = self._bit_offset(index)
            self._data[byte_idx] |= 1 << bit_offset

    def batched_set(self, indices: Sequence[int]) -> None:
        """Set multiple bits at once.

        Out-of-range indices are silently ignored.

        Args:
            indices: Sequence of bit positions to set.
        """
        for i in indices:
            self.set(i)

    def clear(self, index: int) -> None:
        """Clear the bit at the specified index (set to 0).

        Out-of-range indices are silently ignored.

        Args:
            index: The bit position to clear.
        """
        if 0 <= index < self._size:
            byte_idx = self._byte_index(index)
            bit_offset = self._bit_offset(index)
            self._data[byte_idx] &= ~(1 << bit_offset)

    def test(self, index: int) -> bool:
        """Return True if the bit at the specified index is set.

        Returns False for out-of-range indices.

        Args:
            index: The bit position to test.

        Returns:
            True if the bit is set, False otherwise.
        """
        if 0 <= index < self._size:
            byte_idx = self._byte_index(index)
            bit_offset = self._bit_offset(index)
            return bool((self._data[byte_idx] >> bit_offset) & 1)
        return False

    # -- aggregates ---------------------------------------------------------
    def popcount(self) -> int:
        """Return the number of set bits (population count).

        Returns:
            The count of bits that are set to 1.
        """
        if not self._data:
            return 0
        count = 0
        num_full_bytes = self._size // 8
        for i in range(num_full_bytes):
            count += bin(self._data[i]).count("1")
        remaining_bits = self._size % 8
        if remaining_bits > 0:
            last_byte = self._data[-1]
            mask = (1 << remaining_bits) - 1
            count += bin(last_byte & mask).count("1")
        return count

    def count_leading_zeros(self) -> int:
        """Return the number of consecutive zero bits starting from index 0.

        Returns:
            The count of leading zero bits. Returns 0 if bit 0 is set,
            and ``size`` if all bits are zero.
        """
        if self._size == 0:
            return 0
        count = 0
        num_full_bytes = self._size // 8
        for i in range(num_full_bytes):
            b = self._data[i]
            if b == 0:
                count += 8
            else:
                count += (b & -b).bit_length() - 1
                return count
        remaining_bits = self._size % 8
        if remaining_bits > 0:
            last_byte = self._data[-1]
            mask = (1 << remaining_bits) - 1
            last_byte &= mask
            if last_byte == 0:
                count += remaining_bits
            else:
                count += (last_byte & -last_byte).bit_length() - 1
        return count

    def count_leading_ones(self) -> int:
        """Return the number of consecutive one bits starting from index 0.

        Returns:
            The count of leading one bits. Returns 0 if bit 0 is clear,
            and ``size`` if all bits are set.
        """
        inverted = ~self
        return inverted.count_leading_zeros()

    # -- bitwise ops --------------------------------------------------------
    def __and__(self, other: "Bitmap") -> "Bitmap":
        result_size = min(self._size, other._size)
        result = Bitmap(result_size)
        for i in range(min(len(self._data), len(other._data))):
            result._data[i] = self._data[i] & other._data[i]
        return result

    def __or__(self, other: "Bitmap") -> "Bitmap":
        result_size = min(self._size, other._size)
        result = Bitmap(result_size)
        for i in range(min(len(self._data), len(other._data))):
            result._data[i] = self._data[i] | other._data[i]
        # Clear bits beyond size in the last byte to maintain the invariant.
        remaining = result_size % 8
        if remaining and result._data:
            result._data[-1] &= (1 << remaining) - 1
        return result

    def __invert__(self) -> "Bitmap":
        result = Bitmap(self._size)
        for i in range(len(self._data)):
            result._data[i] = ~self._data[i] & 0xFF
        remaining_bits = self._size % 8
        if remaining_bits > 0 and self._data:
            mask = (1 << remaining_bits) - 1
            result._data[-1] &= mask
        return result

    # -- introspection ------------------------------------------------------
    def get_indices_list(self) -> list[int]:
        """Return a sorted list of indices of all set bits.

        Returns:
            A list of bit positions (in ascending order) that are set to 1.
        """
        indices: list[int] = []
        for i in range(len(self._data)):
            byte = self._data[i]
            bit_base = i * 8
            bit = 0
            while byte and bit < 8:
                if byte & 1:
                    idx = bit_base + bit
                    if idx < self._size:
                        indices.append(idx)
                byte >>= 1
                bit += 1
        return indices

    def get_indices_set(self) -> Set[int]:
        """Return a set of indices of all set bits.

        Returns:
            A set of bit positions that are set to 1.
        """
        return set(self.get_indices_list())

    def gather(self, items: Sequence[Any]) -> list[Any]:
        """Collect items at positions where the corresponding bit is set.

        Args:
            items: A sequence to index into. Items beyond ``len(items)`` are
                silently skipped.

        Returns:
            A list of elements from ``items`` at the set bit positions.
        """
        return [items[i] for i in self.get_indices_list() if i < len(items)]

    def __repr__(self) -> str:
        if self._size == 0:
            return ""
        chars = []
        for i in range(self._size):
            chars.append("1" if self.test(i) else "0")
        return "".join(chars)


# ---------------------------------------------------------------------------
# ParallelPatternMatcher
# ---------------------------------------------------------------------------
class ParallelPatternMatcher:
    """Find every starting position where ``pattern`` occurs in ``data``.

    Naive O(n*m) sliding window. The C++ implementation parallelizes this; the
    Python fallback prioritizes correctness only.
    """

    __slots__ = ("_pattern",)

    def __init__(self, pattern: list[int]) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        self._pattern: list[int] = list(pattern)

    def match(self, data: list[int]) -> list[int]:
        """Find all starting positions where the pattern occurs in data.

        Args:
            data: The integer sequence to search in.

        Returns:
            A sorted list of starting positions where the pattern is found.
        """
        pat = self._pattern
        m = len(pat)
        if m == 0 or len(data) < m:
            return []
        return [i for i in range(len(data) - m + 1) if data[i : i + m] == pat]


# ---------------------------------------------------------------------------
# PeriodicEventNotifier
# ---------------------------------------------------------------------------
class PeriodicEventNotifier:
    """Singleton that periodically signals registered file descriptors.

    A daemon thread writes to every registered fd at a configurable interval.
    The thread sleeps when no fds are registered and wakes automatically when
    the first fd is added.

    Differences from the C++ implementation
    ---------------------------------------
    * Uses :class:`threading.Condition` for synchronization, similar to C++'s
      std::condition_variable but with Python semantics.
    * Tolerates ``OSError`` from ``os.write`` (the user may have closed the fd
      externally).
    * The background thread is a daemon so it never blocks interpreter exit,
      but callers should still call :meth:`shutdown` for clean teardown.
    """

    _instance: Optional["PeriodicEventNotifier"] = None
    _class_lock: threading.Lock = threading.Lock()

    # Created via :meth:`create`; do not call this constructor directly.
    def __init__(self, interval_ms: int, use_eventfd: bool) -> None:
        # Clamped to >= 1 ms to mirror the C++ behaviour.
        self._interval_s: float = max(int(interval_ms), 1) / 1000.0
        self._use_eventfd: bool = bool(use_eventfd)

        self._fds: Set[int] = set()
        self._lock: threading.Lock = threading.Lock()
        self._cv: threading.Condition = threading.Condition(self._lock)
        self._stop: bool = False

        self._thread: threading.Thread = threading.Thread(
            target=self._run,
            name="PeriodicEventNotifier",
            daemon=True,
        )

    # -- lifecycle (static API) --------------------------------------------
    @staticmethod
    def create(interval_ms: int, use_eventfd: bool) -> None:
        """Create the singleton. Idempotent: a second call is a no-op."""
        with PeriodicEventNotifier._class_lock:
            if PeriodicEventNotifier._instance is not None:
                return
            inst = PeriodicEventNotifier(interval_ms, use_eventfd)
            PeriodicEventNotifier._instance = inst
            inst._thread.start()

    @staticmethod
    def get() -> Optional["PeriodicEventNotifier"]:
        """Return the singleton instance, or None if not yet created."""
        return PeriodicEventNotifier._instance

    @staticmethod
    def shutdown() -> None:
        """Shut the singleton down and join its thread. Idempotent."""
        with PeriodicEventNotifier._class_lock:
            inst = PeriodicEventNotifier._instance
            if inst is None:
                return
            with inst._cv:
                inst._stop = True
                inst._cv.notify()
            PeriodicEventNotifier._instance = None
        # Join outside the class lock to avoid deadlocking with anything the
        # background thread may try to acquire on its way out.
        inst._thread.join()

    # -- instance API -------------------------------------------------------
    def register_fd(self, fd: int) -> None:
        """Register a file descriptor for periodic signalling."""
        with self._cv:
            self._fds.add(int(fd))
            self._cv.notify()

    def unregister_fd(self, fd: int) -> None:
        """Unregister a previously-registered fd. No-op if not registered."""
        with self._cv:
            self._fds.discard(int(fd))

    def set_interval_ms(self, interval_ms: int) -> None:
        """Change the notification interval (clamped to >= 1 ms)."""
        with self._cv:
            self._interval_s = max(int(interval_ms), 1) / 1000.0
            self._cv.notify()  # let the running loop pick up the new interval

    # -- background loop ----------------------------------------------------
    def _run(self) -> None:
        # eventfd write semantics: 8-byte uint64 (native byte order).
        eventfd_payload = struct.pack("@Q", 1)
        # Pipe write semantics: a single byte is sufficient to wake a reader.
        pipe_payload = b"\x00"

        while True:
            with self._cv:
                # Wait until there are fds to signal, or shutdown was requested.
                self._cv.wait_for(lambda: self._fds or self._stop)
                if self._stop:
                    break
                fds_snapshot = list(self._fds)
                payload_use_eventfd = self._use_eventfd
                interval = self._interval_s

            payload = eventfd_payload if payload_use_eventfd else pipe_payload
            for fd in fds_snapshot:
                try:
                    os.write(fd, payload)
                except OSError as e:
                    # The fd may have been closed externally.
                    # Automatically unregister it on EBADF/EPIPE to avoid
                    # repeated write attempts on every tick (CPU overhead).
                    if e.errno in (errno.EBADF, errno.EPIPE):
                        self.unregister_fd(fd)
                    # Other errors are silently ignored.

            # Wait for next tick or early shutdown. Using wait_for(predicate)
            # avoids the lost-wakeup race where shutdown() fires while we're
            # between releasing the lock above and re-acquiring it here.
            with self._cv:
                current_interval = self._interval_s

                def _should_wake(ci: float = current_interval) -> bool:
                    return self._stop or not self._fds or self._interval_s != ci

                self._cv.wait_for(_should_wake, timeout=interval)
                if self._stop:
                    break


# ---------------------------------------------------------------------------
# RangePatternMatcher
# ---------------------------------------------------------------------------
class RangePatternMatcher:
    """Find ``(start, end)`` ranges in ``data`` delimited by two patterns.

    Returns minimal ranges: when multiple end-pattern occurrences follow a
    start-pattern occurrence, the FIRST end is matched.
    """

    __slots__ = ("_start", "_end")

    _MAX_PATTERN_LEN: int = 5

    def __init__(
        self,
        start_pattern: list[int],
        end_pattern: list[int],
    ) -> None:
        if not start_pattern or not end_pattern:
            raise ValueError("patterns must not be empty")
        if (
            len(start_pattern) > self._MAX_PATTERN_LEN
            or len(end_pattern) > self._MAX_PATTERN_LEN
        ):
            raise ValueError(
                f"patterns must not have more than {self._MAX_PATTERN_LEN} elements"
            )
        self._start: list[int] = list(start_pattern)
        self._end: list[int] = list(end_pattern)

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        """Find all (start, end) ranges delimited by the start/end patterns.

        Produces minimal non-overlapping ranges: for each start-pattern
        occurrence, the first subsequent end-pattern occurrence is matched.

        Args:
            data: The integer sequence to search in.

        Returns:
            A list of ``(start_pos, end_pos)`` tuples where ``start_pos`` is
            the index of the first element of the start pattern and
            ``end_pos`` is the index one past the last element of the end
            pattern.
        """
        ranges: list[tuple[int, int]] = []
        start_len = len(self._start)
        end_len = len(self._end)
        n = len(data)
        if n < start_len + end_len:
            return ranges

        i = 0
        while i <= n - start_len:
            if data[i : i + start_len] == self._start:
                # Found start; scan forward for the first end pattern.
                search_idx = i + start_len
                matched = False
                while search_idx <= n - end_len:
                    if data[search_idx : search_idx + end_len] == self._end:
                        end_idx = search_idx + end_len
                        ranges.append((i, end_idx))
                        # Skip past the matched range to avoid producing
                        # nested / overlapping matches.
                        i = end_idx
                        matched = True
                        break
                    search_idx += 1
                if not matched:
                    i += 1
            else:
                i += 1
        return ranges
