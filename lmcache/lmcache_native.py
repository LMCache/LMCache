# SPDX-License-Identifier: Apache-2.0
"""Pure-Python fallback for ``lmcache.lmcache_native``.

This keeps source-only and ``NO_NATIVE_EXT=1`` installs importable when the
compiled extension is unavailable. When the extension is built, Python imports
the extension module instead of this file.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from enum import IntEnum
from threading import Event, Lock, RLock, Thread
import builtins
import os
import time


def _bit_string(size: int, bits: set[int]) -> str:
    """Return the bitmap as a left-to-right ``0``/``1`` string."""
    return "".join("1" if i in bits else "0" for i in range(size))


class EngineKVFormat(IntEnum):
    """Enumeration of different engine KV cache memory layouts."""

    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5
    NL_X_TWO_NB_NH_BS_HS = 6
    NL_X_NB_TWO_NH_BS_HS = 7
    NB_NL_TWO_NH_BS_HS = 8
    TWO_X_NL_X_NB_BS_NH_HS = 9
    NL_X_NB_NH_BS_TWO_HS = 10
    NL_X_NB_BS_NH_TWO_HS = 11
    NL_X_NB_NH_BS_CS = 12
    NL_X_NB_BS_NH_CS = 13
    NL_X_NB_BSV_BSS = 14


GPUKVFormat = EngineKVFormat


class TransferDirection(IntEnum):
    """Specifies the direction of a memory transfer."""

    H2D = 0
    D2H = 1


def is_cross_layer(engine_kv_format: EngineKVFormat) -> bool:
    """Return ``True`` if all layers are fused into one tensor."""
    return engine_kv_format in (
        EngineKVFormat.NB_NL_TWO_BS_NH_HS,
        EngineKVFormat.NB_NL_TWO_NH_BS_HS,
    )


def is_kv_list(engine_kv_format: EngineKVFormat) -> bool:
    """Return ``True`` if keys and values are separate top-level lists."""
    return engine_kv_format in (
        EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
    )


def is_layer_list(engine_kv_format: EngineKVFormat) -> bool:
    """Return ``True`` if there is one list entry per layer."""
    return engine_kv_format in (
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
        EngineKVFormat.NL_X_NB_NH_BS_CS,
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    )


def is_mla(engine_kv_format: EngineKVFormat) -> bool:
    """Return ``True`` for MLA formats."""
    return engine_kv_format in (
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    )


class TTLLock:
    """Thread-safe TTL lock fallback."""

    def __init__(self, ttl_second: int = 300) -> None:
        self._ttl_second = max(int(ttl_second), 0)
        self._count = 0
        self._deadline = 0.0
        self._lock = Lock()

    def lock(self) -> None:
        """Increment the lock counter and refresh the TTL."""
        with self._lock:
            now = time.monotonic()
            if self._deadline <= now:
                self._count = 0
            self._count += 1
            self._deadline = now + self._ttl_second

    def unlock(self) -> None:
        """Decrement the lock counter without going below zero."""
        with self._lock:
            self._refresh_expiry_locked()
            if self._count > 0:
                self._count -= 1

    def is_locked(self) -> bool:
        """Return whether the lock is currently held."""
        with self._lock:
            self._refresh_expiry_locked()
            return self._count > 0

    def reset(self) -> None:
        """Reset the lock to the unlocked state."""
        with self._lock:
            self._count = 0
            self._deadline = 0.0

    def _refresh_expiry_locked(self) -> None:
        if self._deadline <= time.monotonic():
            self._count = 0
            self._deadline = 0.0


class Bitmap:
    """Simple Python bitmap fallback."""

    def __init__(self, size: int, prefix_bits: int = 0) -> None:
        self._size = max(int(size), 0)
        self._bits = set(range(min(max(int(prefix_bits), 0), self._size)))

    def set(self, index: int) -> None:
        """Set one bit."""
        normalized = self._normalize_index(index)
        self._bits.add(normalized)

    def batched_set(self, indices: Sequence[int]) -> None:
        """Set all valid indices in ``indices``."""
        for index in indices:
            if 0 <= int(index) < self._size:
                self._bits.add(int(index))

    def set_range(self, start: int, end: int) -> None:
        """Set every bit in ``[start, end)``."""
        clamped_start = max(int(start), 0)
        clamped_end = min(max(int(end), 0), self._size)
        if clamped_start >= clamped_end:
            return
        self._bits.update(range(clamped_start, clamped_end))

    def clear(self, index: int) -> None:
        """Clear one bit."""
        normalized = self._normalize_index(index)
        self._bits.discard(normalized)

    def test(self, index: int) -> bool:
        """Return whether one bit is set."""
        normalized = self._normalize_index(index)
        return normalized in self._bits

    def popcount(self) -> int:
        """Return the number of set bits."""
        return len(self._bits)

    def count_leading_zeros(self) -> int:
        """Return the number of leading zero bits."""
        count = 0
        while count < self._size and count not in self._bits:
            count += 1
        return count

    def count_leading_ones(self) -> int:
        """Return the number of leading one bits."""
        count = 0
        while count < self._size and count in self._bits:
            count += 1
        return count

    def highest_set_bit(self) -> int:
        """Return the highest set bit index, or ``-1`` if empty."""
        return max(self._bits, default=-1)

    def __and__(self, other: "Bitmap") -> "Bitmap":
        """Return the bitwise intersection."""
        size = min(self._size, other._size)
        result = Bitmap(size)
        result._bits = {i for i in self._bits & other._bits if i < size}
        return result

    def __iand__(self, other: "Bitmap") -> "Bitmap":
        """In-place bitwise intersection."""
        self._bits &= other._bits
        self._bits = {i for i in self._bits if i < self._size}
        return self

    def __invert__(self) -> "Bitmap":
        """Return the bitwise complement within the bitmap size."""
        result = Bitmap(self._size)
        result._bits = set(range(self._size)) - self._bits
        return result

    def __or__(self, other: "Bitmap") -> "Bitmap":
        """Return the bitwise union."""
        size = max(self._size, other._size)
        result = Bitmap(size)
        result._bits = set(self._bits | other._bits)
        return result

    def __ior__(self, other: "Bitmap") -> "Bitmap":
        """In-place bitwise union."""
        self._size = max(self._size, other._size)
        self._bits |= other._bits
        return self

    def get_indices_list(self) -> list[int]:
        """Return set-bit indices in ascending order."""
        return sorted(self._bits)

    def get_indices_set(self) -> builtins.set[int]:
        """Return set-bit indices."""
        return set(self._bits)

    def gather(self, items: Sequence[object]) -> list[object]:
        """Return items selected by the set bits."""
        return [items[index] for index in self.get_indices_list()]

    def __repr__(self) -> str:
        """Return the bitmap as a ``0``/``1`` string."""
        return _bit_string(self._size, self._bits)

    def _normalize_index(self, index: int) -> int:
        normalized = int(index)
        if normalized < 0 or normalized >= self._size:
            raise IndexError(index)
        return normalized


class PeriodicEventNotifier:
    """Best-effort Python fallback for the native periodic notifier."""

    _instance: "PeriodicEventNotifier | None" = None
    _instance_lock = Lock()

    def __init__(self, interval_ms: int = 10, use_eventfd: bool = True) -> None:
        self._interval_s = max(int(interval_ms), 1) / 1000.0
        self._use_eventfd = use_eventfd
        self._fds: set[int] = set()
        self._lock = RLock()
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    @classmethod
    def create(cls, interval_ms: int = 10, use_eventfd: bool = True) -> None:
        """Create the singleton notifier if needed."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(interval_ms=interval_ms, use_eventfd=use_eventfd)
            else:
                cls._instance.set_interval_ms(interval_ms)

    @classmethod
    def get(cls) -> "PeriodicEventNotifier | None":
        """Return the singleton notifier, if any."""
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Stop and clear the singleton notifier."""
        with cls._instance_lock:
            instance = cls._instance
            cls._instance = None
        if instance is not None:
            instance._stop()

    def register_fd(self, fd: int) -> None:
        """Register one file descriptor for periodic wakeups."""
        with self._lock:
            self._fds.add(int(fd))

    def unregister_fd(self, fd: int) -> None:
        """Unregister one file descriptor."""
        with self._lock:
            self._fds.discard(int(fd))

    def set_interval_ms(self, interval_ms: int) -> None:
        """Update the wakeup interval."""
        with self._lock:
            self._interval_s = max(int(interval_ms), 1) / 1000.0

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            with self._lock:
                fds = tuple(self._fds)
            for fd in fds:
                self._notify_fd(fd)

    def _notify_fd(self, fd: int) -> None:
        try:
            if self._use_eventfd and hasattr(os, "eventfd_write"):
                os.eventfd_write(fd, 1)
            else:
                os.write(fd, b"\x01")
        except OSError:
            pass

    def _stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


__all__ = [
    "Bitmap",
    "EngineKVFormat",
    "GPUKVFormat",
    "PeriodicEventNotifier",
    "TTLLock",
    "TransferDirection",
    "is_cross_layer",
    "is_kv_list",
    "is_layer_list",
    "is_mla",
]
