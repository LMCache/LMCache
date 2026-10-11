# SPDX-License-Identifier: Apache-2.0
"""Keep a disk-backed L2 adapter from filling the filesystem it lives on.

``max_capacity_gb`` bounds what the adapter itself writes. It knows nothing
about the rest of the filesystem: on a disk shared with model weights, a tier
that stays inside its own cap can still push the disk past the point where
something else starts deleting files to make room.

``DiskGuard`` turns two filesystem limits into a capacity for the adapter:

- ``high_watermark``: the used share of the filesystem (as ``df`` reports
  it) that the adapter must not push the disk past.
- ``min_free_bytes``: the free space the adapter must leave.

The capacity is what the adapter holds now plus the room left before either
limit. It shrinks when other data fills the disk and grows back when space is
freed, so the ordinary usage-based eviction keeps the disk under the limits.
"""

# Standard
from typing import Callable, Optional
import os
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class DiskGuard:
    """Filesystem limits for an adapter storing under ``path``.

    Args:
        path: Directory the adapter writes to.
        high_watermark: Used share of the filesystem in (0, 1] the adapter
            must stay under; ``0`` disables the limit.
        min_free_bytes: Free bytes the adapter must leave; ``0`` disables
            the limit.
        statvfs: Replaceable for tests.
        ttl: Seconds a filesystem reading is reused.

    Raises:
        ValueError: If a limit is out of range or none is set.
    """

    def __init__(
        self,
        path: str,
        high_watermark: float = 0.0,
        min_free_bytes: int = 0,
        statvfs: Callable[[str], os.statvfs_result] = os.statvfs,
        ttl: float = 1.0,
    ) -> None:
        if not 0.0 <= high_watermark <= 1.0:
            raise ValueError("high_watermark must be in [0, 1]")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        if high_watermark == 0.0 and min_free_bytes == 0:
            raise ValueError("DiskGuard needs high_watermark or min_free_bytes")
        self._path = path
        self._high_watermark = high_watermark
        self._min_free_bytes = int(min_free_bytes)
        self._statvfs = statvfs
        self._ttl = ttl
        self._lock = threading.Lock()
        self._read_at = float("-inf")
        self._headroom: Optional[int] = None
        self._limited_logged_at = float("-inf")

    @property
    def high_watermark(self) -> float:
        return self._high_watermark

    @property
    def min_free_bytes(self) -> int:
        return self._min_free_bytes

    def headroom_bytes(self) -> Optional[int]:
        """Bytes that can still be written before a limit is crossed.

        Negative when the filesystem is already past a limit. ``None`` when
        the filesystem cannot be read, in which case the guard does not limit.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._read_at < self._ttl:
                return self._headroom
            self._read_at = now
            try:
                st = self._statvfs(self._path)
            except OSError as e:
                logger.warning("DiskGuard cannot read %s: %s", self._path, e)
                self._headroom = None
                return None
            # Same arithmetic as df: Use% = used / (used + available).
            used = (st.f_blocks - st.f_bfree) * st.f_frsize
            available = st.f_bavail * st.f_frsize
            rooms = []
            if self._high_watermark > 0.0:
                rooms.append(int(self._high_watermark * (used + available)) - used)
            if self._min_free_bytes > 0:
                rooms.append(available - self._min_free_bytes)
            self._headroom = min(rooms)
            return self._headroom

    def effective_capacity(self, used_bytes: int, max_capacity_bytes: int) -> int:
        """Capacity the adapter should report, given what it holds now.

        Never above ``max_capacity_bytes`` and never below 1, so an adapter
        holding data on a disk that is past its limits reads as over capacity
        instead of as having no capacity to measure against.
        """
        headroom = self.headroom_bytes()
        if headroom is None:
            return max_capacity_bytes
        capacity = max(1, min(max_capacity_bytes, used_bytes + headroom))
        if capacity < max_capacity_bytes:
            now = time.monotonic()
            if now - self._limited_logged_at >= 60.0:
                self._limited_logged_at = now
                logger.info(
                    "DiskGuard: %s limits the tier to %.0f GB "
                    "(configured %.0f GB, holding %.0f GB)",
                    self._path,
                    capacity / (1 << 30),
                    max_capacity_bytes / (1 << 30),
                    used_bytes / (1 << 30),
                )
        return capacity

    def status(self) -> dict:
        headroom = self.headroom_bytes()
        return {
            "high_watermark": self._high_watermark,
            "min_free_gb": self._min_free_bytes / (1 << 30),
            "headroom_gb": None if headroom is None else headroom / (1 << 30),
        }
