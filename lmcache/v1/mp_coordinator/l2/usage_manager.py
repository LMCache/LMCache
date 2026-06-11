# SPDX-License-Identifier: Apache-2.0
"""Per-``cache_salt`` L2 usage manager for the MP coordinator.

Maintains running byte totals per tenant, updated by store events
reported by MP servers. Eviction (byte subtraction) is driven by
the coordinator itself, not by MP servers.
Thread-safe and dependency-free.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class L2UsageManager:
    """Thread-safe in-memory manager of L2 byte usage per ``cache_salt``.

    MP servers report ``store`` events. The coordinator calls
    ``record_evicted`` when it decides to evict data. Byte counters
    are clamped at zero on underflow.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bytes_by_salt: dict[str, int] = {}
        self._total_bytes: int = 0

    def record_stored(self, cache_salt: str, num_bytes: int) -> None:
        """Record that ``num_bytes`` were stored under ``cache_salt``.

        Args:
            cache_salt: The tenant identifier.
            num_bytes: Bytes stored (must be non-negative).

        Raises:
            ValueError: If ``num_bytes`` is negative.
        """
        if num_bytes < 0:
            raise ValueError(f"num_bytes must be non-negative (got {num_bytes})")
        if num_bytes == 0:
            return
        with self._lock:
            self._bytes_by_salt[cache_salt] = (
                self._bytes_by_salt.get(cache_salt, 0) + num_bytes
            )
            self._total_bytes += num_bytes

    def record_evicted(self, cache_salt: str, num_bytes: int) -> None:
        """Record that the coordinator evicted ``num_bytes`` under ``cache_salt``.

        Clamps per-salt and total counters at zero if a subtraction
        would underflow (logs a warning).

        Args:
            cache_salt: The tenant identifier.
            num_bytes: Bytes evicted (must be non-negative).

        Raises:
            ValueError: If ``num_bytes`` is negative.
        """
        if num_bytes < 0:
            raise ValueError(f"num_bytes must be non-negative (got {num_bytes})")
        if num_bytes == 0:
            return
        with self._lock:
            current = self._bytes_by_salt.get(cache_salt, 0)
            new_val = current - num_bytes
            if new_val < 0:
                logger.warning(
                    "Usage underflow for cache_salt=%r: %d - %d = %d, clamping to 0",
                    cache_salt,
                    current,
                    num_bytes,
                    new_val,
                )
                new_val = 0
            if new_val == 0:
                self._bytes_by_salt.pop(cache_salt, None)
            else:
                self._bytes_by_salt[cache_salt] = new_val

            self._total_bytes -= num_bytes
            if self._total_bytes < 0:
                self._total_bytes = 0

    def get(self, cache_salt: str) -> int:
        """Return the current byte usage for ``cache_salt``.

        Args:
            cache_salt: The tenant identifier.

        Returns:
            Bytes currently tracked, or 0 if no usage recorded.
        """
        with self._lock:
            return self._bytes_by_salt.get(cache_salt, 0)

    def get_all(self) -> dict[str, int]:
        """Return a snapshot of per-salt byte usage.

        Returns:
            A copy of the internal mapping (salt -> bytes).
        """
        with self._lock:
            return dict(self._bytes_by_salt)

    def get_total(self) -> int:
        """Return the total bytes tracked across all salts.

        Returns:
            Total byte usage.
        """
        with self._lock:
            return self._total_bytes
