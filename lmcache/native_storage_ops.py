# SPDX-License-Identifier: Apache-2.0
"""Pure-Python fallbacks for native storage ops.

This module is only used when the compiled extension ``lmcache.native_storage_ops``
is not available (e.g., CPU-only environments or when CUDA extensions are
skipped). The real C++ implementations provide atomic operations; these
fallbacks are correct but slower and should be used for testing only.
"""

# Standard
from __future__ import annotations

import threading
import time
from typing import Any, Iterable, List, Set


class TTLLock:
    """Thread-safe lock with TTL semantics (Python fallback)."""

    def __init__(self, ttl_second: int = 300) -> None:
        self._ttl_ms = int(ttl_second * 1000)
        self._lock = threading.Lock()
        self._count = 0
        self._expire_at_ms = 0

    def _now_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def _expired(self) -> bool:
        return self._expire_at_ms <= self._now_ms()

    def lock(self) -> None:
        with self._lock:
            if self._expired():
                self._count = 0
            self._count += 1
            self._expire_at_ms = self._now_ms() + self._ttl_ms

    def unlock(self) -> None:
        with self._lock:
            if self._expired():
                self._count = 0
                return
            if self._count > 0:
                self._count -= 1

    def is_locked(self) -> bool:
        with self._lock:
            if self._expired():
                self._count = 0
                return False
            return self._count > 0

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._expire_at_ms = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"TTLLock(count={self._count}, expires={self._expire_at_ms})"


class Bitmap:
    """Simple bitmap implementation for testing purposes."""

    def __init__(self, size: int, prefix_bits: int | None = None) -> None:
        self._bits: List[int] = [0] * size
        if prefix_bits:
            for i in range(min(prefix_bits, size)):
                self._bits[i] = 1

    def set(self, index: int) -> None:
        self._bits[index] = 1

    def clear(self, index: int) -> None:
        self._bits[index] = 0

    def test(self, index: int) -> bool:
        return bool(self._bits[index])

    def popcount(self) -> int:
        return sum(self._bits)

    def count_leading_zeros(self) -> int:
        count = 0
        for bit in self._bits:
            if bit == 0:
                count += 1
            else:
                break
        return count

    def count_leading_ones(self) -> int:
        count = 0
        for bit in self._bits:
            if bit == 1:
                count += 1
            else:
                break
        return count

    def __and__(self, other: "Bitmap") -> "Bitmap":
        size = min(len(self._bits), len(other._bits))
        out = Bitmap(size)
        out._bits = [a & b for a, b in zip(self._bits[:size], other._bits[:size])]
        return out

    def __or__(self, other: "Bitmap") -> "Bitmap":
        size = min(len(self._bits), len(other._bits))
        out = Bitmap(size)
        out._bits = [a | b for a, b in zip(self._bits[:size], other._bits[:size])]
        return out

    def __invert__(self) -> "Bitmap":
        out = Bitmap(len(self._bits))
        out._bits = [0 if b else 1 for b in self._bits]
        return out

    def get_indices_list(self) -> list[int]:
        return [i for i, b in enumerate(self._bits) if b]

    def get_indices_set(self) -> Set[int]:
        return set(self.get_indices_list())

    def gather(self, items: list[Any]) -> list[Any]:
        return [item for item, bit in zip(items, self._bits) if bit]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "".join("1" if b else "0" for b in self._bits)


class ParallelPatternMatcher:
    """Naive pattern matcher for integer sequences (testing fallback)."""

    def __init__(self, pattern: list[int]) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        self.pattern = pattern

    def match(self, data: list[int]) -> list[int]:
        pat = self.pattern
        m = len(pat)
        return [i for i in range(len(data) - m + 1) if data[i : i + m] == pat]


class RangePatternMatcher:
    """Find minimal ranges between start and end patterns (testing fallback)."""

    def __init__(self, start_pattern: list[int], end_pattern: list[int]) -> None:
        if not start_pattern or not end_pattern:
            raise ValueError("patterns must not be empty")
        self._start = start_pattern
        self._end = end_pattern

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        start_matches = ParallelPatternMatcher(self._start).match(data)
        ranges: list[tuple[int, int]] = []
        start_len = len(self._start)
        end_len = len(self._end)

        for start_idx in start_matches:
            search_idx = start_idx + start_len
            while search_idx <= len(data) - end_len:
                if data[search_idx : search_idx + end_len] == self._end:
                    ranges.append((start_idx, search_idx + end_len))
                    break
                search_idx += 1
        return ranges


__all__ = ["TTLLock", "Bitmap", "ParallelPatternMatcher", "RangePatternMatcher"]
