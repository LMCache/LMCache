# SPDX-License-Identifier: Apache-2.0
"""
Per-cache_salt quota registry.

Holds the authoritative map from ``cache_salt`` to a byte budget.
Consumed by the L2 eviction controller each cycle to decide which
users are over their quota, and by the HTTP API for runtime
administration (CRUD endpoints at ``/api/quota/...``).

Allowlist semantics: a ``cache_salt`` with no entry has an effective
limit of ``0`` bytes — stores are still permitted on the hot path,
but any bytes accumulated under that salt will be evicted at the next
eviction cycle. Only salts with an explicit quota retain cached data.
"""

# Future
from __future__ import annotations

# Standard
import threading

# First Party
from lmcache.v1.distributed.internal_api import QuotaEntry


class QuotaManager:
    """Thread-safe registry of byte quotas keyed by ``cache_salt``.

    Quotas are dynamic — CRUD operations (``set_quota``, ``delete_quota``)
    are cheap, and any change takes effect on the very next eviction
    cycle (no restart needed).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # cache_salt -> limit in bytes
        self._limits: dict[str, int] = {}

    def set_quota(self, cache_salt: str, limit_bytes: int) -> None:
        """Create or update the quota for ``cache_salt``.

        Raises ``ValueError`` for a negative limit. A zero limit is
        accepted and behaves identically to having no entry at all
        (keys under this salt will be evicted next cycle) — the
        difference is purely bookkeeping: the entry shows up in
        ``list_quotas()`` until removed.
        """
        if limit_bytes < 0:
            raise ValueError(f"limit_bytes must be non-negative (got {limit_bytes})")
        with self._lock:
            self._limits[cache_salt] = limit_bytes

    def delete_quota(self, cache_salt: str) -> bool:
        """Remove the quota entry for ``cache_salt``.

        Returns ``True`` if an entry was removed, ``False`` if the salt
        had no registration. After deletion the effective limit drops
        to ``0``, so any existing bytes under that salt will be evicted
        at the next eviction cycle.
        """
        with self._lock:
            return self._limits.pop(cache_salt, None) is not None

    def get_limit_bytes(self, cache_salt: str) -> int:
        """Return the effective limit for ``cache_salt``.

        Unregistered salts resolve to ``0`` (allowlist semantics) —
        callers use this value to drive eviction decisions, so the
        zero default deliberately triggers eviction of any bytes
        accumulated under an unknown salt.
        """
        with self._lock:
            return self._limits.get(cache_salt, 0)

    def has_quota(self, cache_salt: str) -> bool:
        """Whether ``cache_salt`` has an explicit registration.

        Useful for distinguishing "zero limit by default" from
        "explicitly registered with limit=0" in status reports.
        """
        with self._lock:
            return cache_salt in self._limits

    def list_quotas(self) -> list[QuotaEntry]:
        """Return a snapshot of all registered quotas.

        The returned list is a detached copy; mutating it does not
        affect the registry.
        """
        with self._lock:
            return [
                QuotaEntry(cache_salt=salt, limit_bytes=limit)
                for salt, limit in self._limits.items()
            ]
