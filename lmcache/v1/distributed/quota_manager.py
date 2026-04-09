# SPDX-License-Identifier: Apache-2.0
"""
Thread-safe per-user storage quota registry.

Queried by the eviction controller each cycle and updated at runtime
via the HTTP API. Users not in the registry have an effective quota
of 0 bytes (their data is evicted at the next cycle).
"""

# Standard
import threading


class QuotaManager:
    """Thread-safe registry of per-user storage quotas.

    Queried by the eviction controller each cycle. Updated at runtime
    via the HTTP API. Users not in the registry have an effective quota
    of 0 bytes (their data is evicted at the next cycle).

    Enforces a capacity invariant: sum of all quotas <= total adapter
    capacity. ``set_quota()`` rejects requests that would violate this.
    """

    def __init__(self, total_capacity_bytes: int):
        """Initialize the quota manager.

        Args:
            total_capacity_bytes: Maximum total capacity in bytes.
                The sum of all per-user quotas must not exceed this.
        """
        self._lock = threading.Lock()
        self._quotas: dict[str, int] = {}  # user_id -> limit in bytes
        self._total_capacity_bytes = total_capacity_bytes

    def get_limit_bytes(self, user_id: str) -> int:
        """Return the quota for a user in bytes.

        Args:
            user_id: The user identifier.

        Returns:
            The quota in bytes, or 0 if the user is not registered.
        """
        with self._lock:
            return self._quotas.get(user_id, 0)

    def set_quota(self, user_id: str, limit_gb: float) -> None:
        """Set or update quota for a user.

        Args:
            user_id: The user identifier.
            limit_gb: Quota limit in gigabytes.

        Raises:
            ValueError: If adding/updating this quota would cause the
                sum of all quotas to exceed total adapter capacity.
        """
        new_limit = int(limit_gb * (1024**3))
        with self._lock:
            old_limit = self._quotas.get(user_id, 0)
            current_total = sum(self._quotas.values())
            new_total = current_total - old_limit + new_limit
            if new_total > self._total_capacity_bytes:
                raise ValueError(
                    f"Cannot set quota for {user_id}: sum of quotas "
                    f"({new_total / (1024**3):.2f} GB) would exceed "
                    f"adapter capacity "
                    f"({self._total_capacity_bytes / (1024**3):.2f} GB)"
                )
            self._quotas[user_id] = new_limit

    def remove_quota(self, user_id: str) -> None:
        """Remove quota for a user.

        The user's cached data will be evicted at the next eviction
        cycle (effective limit becomes 0).

        Args:
            user_id: The user identifier.
        """
        with self._lock:
            self._quotas.pop(user_id, None)

    def get_all_quotas(self) -> dict[str, int]:
        """Return a snapshot of all quotas.

        Returns:
            A dict mapping user_id to quota limit in bytes.
        """
        with self._lock:
            return dict(self._quotas)
