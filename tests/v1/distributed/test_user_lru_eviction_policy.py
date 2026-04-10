# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for UserLRUEvictionPolicy.

These tests verify the per-user LRU eviction policy:
1. Keys are tracked in per-user buckets
2. LRU ordering is maintained independently per user
3. Eviction can be scoped to a single user or global
4. Eviction destinations are respected
"""

# Third Party

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.user_lru import (
    UserLRUEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import (
    EvictionDestination,
)

# =============================================================================
# Helper Functions
# =============================================================================


def make_key(
    chunk_hash: int,
    model: str = "test",
    kv_rank: int = 0,
    user_id: str = "",
) -> ObjectKey:
    """Create an ObjectKey for testing."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_hash),
        model_name=model,
        kv_rank=kv_rank,
        user_id=user_id,
    )


# =============================================================================
# Per-User Key Tracking Tests
# =============================================================================


class TestUserLRUPerUserTracking:
    """Tests for per-user key tracking."""

    def test_on_keys_created_per_user(self):
        """Keys for different users are tracked in separate buckets."""
        policy = UserLRUEvictionPolicy()

        alice_keys = [make_key(1, user_id="alice"), make_key(2, user_id="alice")]
        bob_keys = [make_key(3, user_id="bob"), make_key(4, user_id="bob")]

        policy.on_keys_created(alice_keys)
        policy.on_keys_created(bob_keys)

        assert policy.get_num_tracked_keys() == 4
        assert policy.get_num_tracked_keys_for_user("alice") == 2
        assert policy.get_num_tracked_keys_for_user("bob") == 2

    def test_on_keys_touched_updates_lru_order(self):
        """Touching a key moves it to the end of that user's LRU list."""
        policy = UserLRUEvictionPolicy()

        # Create alice's keys one at a time for deterministic order: 1, 2, 3
        for i in range(1, 4):
            policy.on_keys_created([make_key(i, user_id="alice")])

        # Touch key 1 — it should now be the most recently used
        policy.on_keys_touched([make_key(1, user_id="alice")])

        # LRU order is now: 2 (oldest), 3, 1 (newest)
        # Evict 1 key from alice — should be key 2 (oldest after touch)
        actions = policy.get_eviction_actions(0.34, user_id="alice")
        assert len(actions) == 1
        evicted_hashes = [
            ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys
        ]
        assert evicted_hashes == [2]

    def test_on_keys_removed_per_user(self):
        """Removing keys from one user does not affect another user."""
        policy = UserLRUEvictionPolicy()

        alice_keys = [make_key(1, user_id="alice"), make_key(2, user_id="alice")]
        bob_keys = [make_key(3, user_id="bob")]

        policy.on_keys_created(alice_keys)
        policy.on_keys_created(bob_keys)

        # Remove alice's key 1
        policy.on_keys_removed([make_key(1, user_id="alice")])

        assert policy.get_num_tracked_keys_for_user("alice") == 1
        assert policy.get_num_tracked_keys_for_user("bob") == 1
        assert policy.get_num_tracked_keys() == 2


# =============================================================================
# Scoped Eviction Tests
# =============================================================================


class TestUserLRUScopedEviction:
    """Tests for user-scoped and global eviction."""

    def test_get_eviction_actions_scoped_to_user(self):
        """Eviction scoped to a user only returns that user's keys."""
        policy = UserLRUEvictionPolicy()

        alice_keys = [make_key(i, user_id="alice") for i in range(1, 5)]
        bob_keys = [make_key(i, user_id="bob") for i in range(5, 9)]

        policy.on_keys_created(alice_keys)
        policy.on_keys_created(bob_keys)

        # Evict 50% of alice's keys
        actions = policy.get_eviction_actions(0.5, user_id="alice")
        assert len(actions) == 1
        evicted = actions[0].keys
        assert len(evicted) == 2

        # All evicted keys belong to alice
        for key in evicted:
            assert key.user_id == "alice"

        # Bob's keys are untouched — policy still tracks them
        assert policy.get_num_tracked_keys_for_user("bob") == 4

    def test_get_eviction_actions_global(self):
        """Eviction with user_id=None returns keys from all users."""
        policy = UserLRUEvictionPolicy()

        policy.on_keys_created([make_key(i, user_id="alice") for i in range(1, 4)])
        policy.on_keys_created([make_key(i, user_id="bob") for i in range(4, 7)])

        # Evict 100% globally — should get all 6 keys
        actions = policy.get_eviction_actions(1.0, user_id=None)
        assert len(actions) == 1
        assert len(actions[0].keys) == 6

    def test_unknown_user_eviction(self):
        """Eviction for an unknown user returns an empty list."""
        policy = UserLRUEvictionPolicy()

        policy.on_keys_created([make_key(1, user_id="alice")])

        actions = policy.get_eviction_actions(1.0, user_id="nonexistent")
        assert actions == []


# =============================================================================
# Eviction Ordering Tests
# =============================================================================


class TestUserLRUEvictionOrder:
    """Tests for LRU ordering within a user."""

    def test_eviction_order_lru(self):
        """Oldest keys within a user are evicted first."""
        policy = UserLRUEvictionPolicy()

        # Create keys one at a time for deterministic LRU order: 1, 2, 3, 4, 5
        for i in range(1, 6):
            policy.on_keys_created([make_key(i, user_id="alice")])

        # Evict 3 of 5 (60%)
        actions = policy.get_eviction_actions(0.6, user_id="alice")
        assert len(actions) == 1
        evicted_hashes = [
            ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys
        ]
        # Oldest keys (1, 2, 3) should be evicted first
        assert evicted_hashes == [1, 2, 3]

    def test_single_key_eviction(self):
        """With ratio > 0 and only 1 key, at least 1 key is evicted."""
        policy = UserLRUEvictionPolicy()

        policy.on_keys_created([make_key(1, user_id="alice")])

        actions = policy.get_eviction_actions(0.01, user_id="alice")
        assert len(actions) == 1
        assert len(actions[0].keys) == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestUserLRUEdgeCases:
    """Tests for edge cases."""

    def test_empty_user_id(self):
        """Keys with user_id='' are tracked in a '' bucket."""
        policy = UserLRUEvictionPolicy()

        policy.on_keys_created([make_key(1, user_id=""), make_key(2, user_id="")])

        assert policy.get_num_tracked_keys_for_user("") == 2

        actions = policy.get_eviction_actions(1.0, user_id="")
        assert len(actions) == 1
        assert len(actions[0].keys) == 2

    def test_empty_policy_returns_no_actions(self):
        """An empty policy returns no eviction actions."""
        policy = UserLRUEvictionPolicy()
        assert policy.get_eviction_actions(1.0) == []
        assert policy.get_eviction_actions(1.0, user_id="alice") == []

    def test_ratio_zero_returns_no_actions(self):
        """A ratio of 0 returns no eviction actions."""
        policy = UserLRUEvictionPolicy()
        policy.on_keys_created([make_key(1, user_id="alice")])
        assert policy.get_eviction_actions(0.0, user_id="alice") == []


# =============================================================================
# Eviction Destination Tests
# =============================================================================


class TestUserLRUEvictionDestination:
    """Tests for eviction destination behavior."""

    def test_default_destination_is_discard(self):
        """Default destination should be DISCARD."""
        policy = UserLRUEvictionPolicy()
        policy.on_keys_created([make_key(1, user_id="alice")])
        actions = policy.get_eviction_actions(1.0, user_id="alice")
        assert actions[0].destination == EvictionDestination.DISCARD

    def test_register_eviction_destination(self):
        """Registered destination is used in returned EvictionActions."""
        policy = UserLRUEvictionPolicy(default_destination=EvictionDestination.DISCARD)
        policy.register_eviction_destination(EvictionDestination.L2_CACHE)
        policy.on_keys_created([make_key(1, user_id="alice")])

        actions = policy.get_eviction_actions(1.0, user_id="alice")
        assert actions[0].destination == EvictionDestination.L2_CACHE
