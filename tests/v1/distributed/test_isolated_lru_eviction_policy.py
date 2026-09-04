# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :class:`IsolatedLRUEvictionPolicy`.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import EvictionDestination
from lmcache.v1.mp_coordinator.utils.encoding import encode_key


def _key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


class TestSupportIsolation:
    def test_support_isolation_is_true(self):
        """``IsolatedLRUEvictionPolicy`` reports isolation support so
        the controller routes to the by-cache_salt eviction branch."""
        assert IsolatedLRUEvictionPolicy().support_isolation is True


class TestPerSaltTracking:
    def test_keys_bucketed_by_salt(self):
        """Each ``cache_salt`` lives in its own LRU list; buckets do
        not bleed into each other."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice"), _key(2, "bob"), _key(3, "alice")])
        assert p.get_num_tracked_keys("alice") == 2
        assert p.get_num_tracked_keys("bob") == 1
        assert p.get_num_tracked_keys("charlie") == 0
        assert sorted(p.get_tracked_salts()) == ["alice", "bob"]

    def test_empty_bucket_dropped_after_removal(self):
        """When a bucket empties the policy forgets the salt entirely
        — keeps status / list snapshots compact."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        p.on_keys_removed([_key(1, "alice")])
        assert "alice" not in p.get_tracked_salts()

    def test_touch_updates_order(self):
        """Touching a key moves it to MRU. After ``on_keys_created``
        the reversed-insertion rule leaves ``k2`` as LRU (first out).
        Touching ``k2`` must promote it so ``k1`` becomes the LRU
        victim — this specifically exercises ``move_to_end`` on an
        existing entry (a no-op touch on the already-MRU key would
        pass the test trivially and hide bugs in the touch path)."""
        p = IsolatedLRUEvictionPolicy()
        k1 = _key(1, "alice")
        k2 = _key(2, "alice")
        p.on_keys_created([k1, k2])
        # Default LRU order now: [k2 (oldest), k1 (newest)].
        p.on_keys_touched([k2])
        # Expected after touch: [k1 (oldest), k2 (newest)].
        actions = p.get_eviction_actions(expected_ratio=0.5, cache_salt="alice")
        evicted = actions[0].keys
        assert evicted == [k1], (
            f"k1 should be LRU after touching k2; got {[k.chunk_hash for k in evicted]}"
        )


class TestScopedEviction:
    def test_scoped_eviction_stays_in_bucket(self):
        """Eviction scoped to a salt only returns keys from that
        bucket — other salts' data is untouched."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice"), _key(2, "bob"), _key(3, "alice")])
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="alice")
        assert len(actions) == 1
        evicted_salts = {k.cache_salt for k in actions[0].keys}
        assert evicted_salts == {"alice"}
        assert len(actions[0].keys) == 2

    def test_scoped_eviction_empty_bucket_returns_empty(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        # No bob bucket yet.
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="bob")
        assert actions == []

    def test_missing_cache_salt_raises(self):
        """``IsolatedLRU`` is per-bucket only — calling without an
        explicit ``cache_salt`` raises ``ValueError`` rather than
        silently falling back to a global pool."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        with pytest.raises(ValueError, match="cache_salt"):
            p.get_eviction_actions(expected_ratio=1.0, cache_salt=None)


class TestEvictionAmount:
    def test_equal_sizes_preserve_count_based_rounding(self):
        """Uniform objects keep the original floor-based victim count."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(3)]
        p.on_keys_created_with_sizes(keys, [4] * len(keys))

        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert len(actions[0].keys) == 1

    def test_variable_sizes_control_eviction_amount_per_salt(self):
        """Each salt evicts the minimal LRU prefix reaching its byte target."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(4)]
        for key, size in zip(keys, [1, 1, 8, 8], strict=True):
            p.on_keys_created_with_sizes([key], [size])

        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert actions[0].keys == keys[:3]

    def test_failed_oversized_victim_does_not_block_bucket_progress(self):
        """An attempted victim is deferred within its own salt bucket."""
        p = IsolatedLRUEvictionPolicy()
        oversized = _key(0, "alice")
        alternatives = [_key(i, "alice") for i in range(1, 101)]
        for key, size in zip(
            [oversized, *alternatives],
            [320, *([1] * 100)],
            strict=True,
        ):
            p.on_keys_created_with_sizes([key], [size])

        first = p.get_eviction_actions(0.2, cache_salt="alice")[0].keys
        p.on_eviction_attempted(first)
        second = p.get_eviction_actions(0.2, cache_salt="alice")[0].keys

        assert first == [oversized]
        assert second == alternatives[:84]

    def test_other_salts_do_not_change_byte_target(self):
        """A large object in another salt must not affect this bucket."""
        p = IsolatedLRUEvictionPolicy()
        alice = [_key(i, "alice") for i in range(4)]
        for key, size in zip(alice, [1, 1, 8, 8], strict=True):
            p.on_keys_created_with_sizes([key], [size])
        p.on_keys_created_with_sizes([_key(100, "bob")], [1024])

        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert actions[0].keys == alice[:3]

    def test_size_refresh_updates_bucket_accounting(self):
        """Re-storing a key replaces its old size in the correct bucket."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(2)]
        p.on_keys_created_with_sizes(keys, [1, 9])

        p.on_keys_created_with_sizes([keys[1]], [1])
        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert actions[0].keys == [keys[0]]

    def test_unsized_refresh_preserves_known_size(self):
        """An unsized notification must not replace a bucket's byte weight."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(2)]
        p.on_keys_created_with_sizes([keys[0]], [1])
        p.on_keys_created_with_sizes([keys[1]], [9])

        p.on_keys_created([keys[1]])
        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert actions[0].keys == keys

    def test_byte_mode_starts_after_last_unknown_size_arrives(self):
        """A bucket switches from count to byte mode only when complete."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(2)]
        p.on_keys_created(keys)
        p.on_keys_created_with_sizes([keys[0]], [1])

        partial = p.get_eviction_actions(0.5, cache_salt="alice")
        p.on_keys_created_with_sizes([keys[1]], [9])
        complete = p.get_eviction_actions(0.5, cache_salt="alice")

        assert partial[0].keys == [keys[1]]
        assert complete[0].keys == keys
        assert p.capture()["sizes"]["alice"] == [1, 9]

    def test_removal_updates_bucket_accounting(self):
        """Removed objects no longer contribute to a bucket's byte target."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(3)]
        for key, size in zip(keys, [1, 9, 2], strict=True):
            p.on_keys_created_with_sizes([key], [size])

        p.on_keys_removed([keys[1]])
        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert actions[0].keys == [keys[0]]

    def test_uniform_rounding_is_restored_after_removal(self):
        """Removing the odd-sized object restores count-based rounding."""
        p = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(4)]
        p.on_keys_created_with_sizes(keys, [4, 8, 4, 4])

        p.on_keys_removed([keys[1]])
        actions = p.get_eviction_actions(0.5, cache_salt="alice")

        assert len(actions[0].keys) == 1

    def test_at_least_one_when_ratio_positive(self):
        """Matches ``LRUEvictionPolicy`` — a positive ratio that rounds
        to zero still yields one eviction so the caller makes forward
        progress."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(i, "alice") for i in range(3)])
        actions = p.get_eviction_actions(expected_ratio=0.01, cache_salt="alice")
        assert len(actions[0].keys) == 1

    def test_zero_ratio_evicts_nothing(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(i, "alice") for i in range(3)])
        assert p.get_eviction_actions(expected_ratio=0.0, cache_salt="alice") == []

    def test_eviction_destination_default_is_discard(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="alice")
        assert actions[0].destination == EvictionDestination.DISCARD

    def test_key_eligible_filter_skips_locked_keys(self):
        """The filter is useful for skipping pinned / locked keys. Any
        key for which the filter returns False should be bypassed when
        choosing victims."""
        p = IsolatedLRUEvictionPolicy()
        k1 = _key(1, "alice")
        k2 = _key(2, "alice")
        p.on_keys_created([k1, k2])
        # Only allow k2 to be evicted.
        actions = p.get_eviction_actions(
            expected_ratio=1.0,
            cache_salt="alice",
            key_eligible_filter=lambda k: k == k2,
        )
        assert actions[0].keys == [k2]


class TestDurableState:
    def test_the_restored_order_decides_who_is_evicted_first(self):
        """A restore that reorders a bucket silently changes the victim,
        which no other state would reveal."""
        policy = IsolatedLRUEvictionPolicy()
        # Created in reverse, so 3 sits at the LRU head; touching it sends
        # it to the other end and leaves an order creation alone cannot.
        policy.on_keys_created([_key(i) for i in (1, 2, 3)])
        policy.on_keys_touched([_key(3)])

        restored = IsolatedLRUEvictionPolicy()
        restored.restore(policy.capture())

        assert restored.capture()["buckets"][""] == [
            encode_key(_key(2)),
            encode_key(_key(1)),
            encode_key(_key(3)),
        ]
        victims = restored.get_eviction_actions(expected_ratio=0.4, cache_salt="")
        assert [key.chunk_hash for key in victims[0].keys] == [
            ObjectKey.IntHash2Bytes(2)
        ]

    def test_restore_preserves_heterogeneous_byte_weights(self):
        """A checkpoint round trip must retain byte-aware victim selection."""
        policy = IsolatedLRUEvictionPolicy()
        keys = [_key(i, "alice") for i in range(4)]
        for key, size in zip(keys, [1, 1, 8, 8], strict=True):
            policy.on_keys_created_with_sizes([key], [size])

        restored = IsolatedLRUEvictionPolicy()
        restored.restore(policy.capture())

        assert restored.capture() == policy.capture()
        actions = restored.get_eviction_actions(0.5, cache_salt="alice")
        assert actions[0].keys == keys[:3]

    def test_partial_sizes_stay_in_count_mode_across_restore(self):
        """Unknown weights are not serialized as byte sizes."""
        keys = [_key(i, "alice") for i in range(101)]
        legacy_state = {
            "buckets": {"alice": [encode_key(key) for key in keys]},
        }
        policy = IsolatedLRUEvictionPolicy()
        policy.restore(legacy_state)
        policy.on_keys_created_with_sizes([keys[-1]], [512])

        actions = policy.get_eviction_actions(0.2, cache_salt="alice")
        captured = policy.capture()

        assert len(actions[0].keys) == 20
        assert "alice" not in captured["sizes"]

        restored = IsolatedLRUEvictionPolicy()
        restored.restore(captured)
        restored_actions = restored.get_eviction_actions(0.2, cache_salt="alice")
        assert len(restored_actions[0].keys) == 20
