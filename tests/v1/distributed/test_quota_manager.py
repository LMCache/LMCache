# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for QuotaManager.

These tests verify the per-user quota registry:
1. Setting and getting quotas
2. Capacity overflow rejection
3. Quota removal
4. Snapshot of all quotas
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.quota_manager import QuotaManager

# =============================================================================
# Constants
# =============================================================================

_1_GB = 1024**3
_10_GB = 10 * _1_GB


# =============================================================================
# Basic Quota Operations
# =============================================================================


class TestQuotaManagerBasic:
    """Tests for basic set/get/remove operations."""

    def test_set_and_get_quota(self):
        """Setting a quota and retrieving it returns the correct value."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=2.0)
        assert mgr.get_limit_bytes("alice") == 2 * _1_GB

    def test_unregistered_user_returns_zero(self):
        """Getting the quota for an unknown user returns 0."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        assert mgr.get_limit_bytes("unknown") == 0

    def test_remove_quota(self):
        """Removing a quota causes get_limit_bytes to return 0."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=2.0)
        mgr.remove_quota("alice")
        assert mgr.get_limit_bytes("alice") == 0

    def test_remove_nonexistent_user(self):
        """Removing a user that does not exist should not raise."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.remove_quota("ghost")  # should be a no-op


# =============================================================================
# Capacity Invariant Tests
# =============================================================================


class TestQuotaManagerCapacity:
    """Tests for the capacity invariant."""

    def test_capacity_overflow_rejected(self):
        """Setting quotas that exceed total capacity raises ValueError."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=6.0)

        with pytest.raises(ValueError):
            mgr.set_quota("bob", limit_gb=5.0)  # 6 + 5 = 11 > 10

    def test_update_existing_quota(self):
        """Updating a quota accounts for the old value being replaced."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=6.0)

        # Increasing alice from 6 to 8 should be fine (8 <= 10)
        mgr.set_quota("alice", limit_gb=8.0)
        assert mgr.get_limit_bytes("alice") == 8 * _1_GB

    def test_update_existing_quota_overflow(self):
        """Updating a quota can still trigger overflow."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=5.0)
        mgr.set_quota("bob", limit_gb=4.0)

        # Alice tries to go from 5 to 7 → total would be 7 + 4 = 11 > 10
        with pytest.raises(ValueError):
            mgr.set_quota("alice", limit_gb=7.0)

    def test_exact_capacity_is_allowed(self):
        """Quotas that exactly fill capacity should be accepted."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=5.0)
        mgr.set_quota("bob", limit_gb=5.0)  # 5 + 5 = 10 == 10
        assert mgr.get_limit_bytes("alice") == 5 * _1_GB
        assert mgr.get_limit_bytes("bob") == 5 * _1_GB


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestQuotaManagerSnapshot:
    """Tests for get_all_quotas."""

    def test_get_all_quotas(self):
        """get_all_quotas returns a snapshot of all registered quotas."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=2.0)
        mgr.set_quota("bob", limit_gb=3.0)

        quotas = mgr.get_all_quotas()
        assert quotas == {
            "alice": 2 * _1_GB,
            "bob": 3 * _1_GB,
        }

    def test_get_all_quotas_is_copy(self):
        """get_all_quotas returns a copy, not the internal dict."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        mgr.set_quota("alice", limit_gb=2.0)

        snapshot = mgr.get_all_quotas()
        snapshot["alice"] = 0  # mutate the copy

        # Internal state should be unchanged
        assert mgr.get_limit_bytes("alice") == 2 * _1_GB

    def test_get_all_quotas_empty(self):
        """get_all_quotas on a fresh manager returns an empty dict."""
        mgr = QuotaManager(total_capacity_bytes=_10_GB)
        assert mgr.get_all_quotas() == {}
