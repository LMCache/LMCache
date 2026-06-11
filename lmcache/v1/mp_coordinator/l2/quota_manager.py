# SPDX-License-Identifier: Apache-2.0
"""Lightweight in-memory quota registry for the MP coordinator.

Holds per-``cache_salt`` byte limits. The coordinator is the single
source of truth for quotas; MP servers query the coordinator to obtain
their limits. This class is intentionally free of heavy dependencies
(no ``torch``, no ``distributed`` layer imports) so the coordinator
process stays lightweight.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class QuotaEntry:
    """Snapshot of a single quota registration.

    Attributes:
        cache_salt: The tenant identifier.
        limit_bytes: The byte budget for this tenant.
    """

    cache_salt: str
    limit_bytes: int


class CoordinatorQuotaManager:
    """Thread-safe in-memory registry of byte quotas keyed by ``cache_salt``.

    All public methods acquire an internal lock so the store stays
    consistent under concurrent access from FastAPI endpoint handlers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limits: dict[str, int] = {}

    def set(self, cache_salt: str, limit_bytes: int) -> None:
        """Create or update the quota for ``cache_salt``.

        Args:
            cache_salt: The tenant identifier.
            limit_bytes: The byte budget (must be non-negative).

        Raises:
            ValueError: If ``limit_bytes`` is negative.
        """
        if limit_bytes < 0:
            raise ValueError(f"limit_bytes must be non-negative (got {limit_bytes})")
        with self._lock:
            self._limits[cache_salt] = limit_bytes

    def get(self, cache_salt: str) -> int | None:
        """Return the limit for ``cache_salt``, or ``None`` if unregistered.

        Args:
            cache_salt: The tenant identifier.

        Returns:
            The byte limit, or ``None`` if no quota is registered.
        """
        with self._lock:
            return self._limits.get(cache_salt)

    def delete(self, cache_salt: str) -> bool:
        """Remove the quota entry for ``cache_salt``.

        Args:
            cache_salt: The tenant identifier.

        Returns:
            ``True`` if an entry was removed, ``False`` if none existed.
        """
        with self._lock:
            return self._limits.pop(cache_salt, None) is not None

    def list_all(self) -> list[QuotaEntry]:
        """Return a snapshot of all registered quotas.

        Returns:
            A detached list of all quota entries.
        """
        with self._lock:
            return [
                QuotaEntry(cache_salt=salt, limit_bytes=limit)
                for salt, limit in self._limits.items()
            ]
