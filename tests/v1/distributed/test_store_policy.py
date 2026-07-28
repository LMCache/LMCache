# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for store policy interface and DefaultStorePolicy.

Tests are written against the StorePolicy contract defined in store_policy.py.
"""

# Third Party

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


# =============================================================================
# DefaultStorePolicy Tests
# =============================================================================


class TestDefaultStorePolicyTargets:
    """Test DefaultStorePolicy.select_store_targets behavior."""

    def test_single_adapter_all_keys(self):
        """All keys should be sent to the single adapter."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert 0 in result
        assert result[0] == keys

    def test_multiple_adapters_all_keys_to_each(self):
        """All keys should be sent to every adapter."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert len(result) == 2
        assert result[0] == keys
        assert result[1] == keys

    def test_empty_adapters_returns_empty(self):
        """No adapters means no store targets."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(0)]

        result = policy.select_store_targets(keys, [])

        assert result == {}

    def test_empty_keys_returns_empty_lists(self):
        """Empty keys list should produce empty lists for each adapter."""
        policy = DefaultStorePolicy()
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets([], adapters)

        assert 0 in result
        assert result[0] == []

    def test_returns_copies_not_references(self):
        """Returned lists should be independent copies of the input."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(0)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        # Mutating the result should not affect the input
        result[0].append(make_object_key(99))
        assert len(keys) == 1


class TestDefaultStorePolicyDeletions:
    """Test DefaultStorePolicy.select_l1_deletions behavior."""

    def test_never_deletes(self):
        """DefaultStorePolicy should never delete from L1."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(5)]

        result = policy.select_l1_deletions(keys)

        assert result == []

    def test_empty_keys_returns_empty(self):
        """Empty input should return empty output."""
        policy = DefaultStorePolicy()

        result = policy.select_l1_deletions([])

        assert result == []


# =============================================================================
# StripedStorePolicy Tests
# =============================================================================


class TestStripedStorePolicyAdapterIndex:
    """Test StripedStorePolicy._adapter_index_for_key properties."""

    def test_non_negative_result(self):
        """Index must be in [0, num_adapters)."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        keys = [make_object_key(i) for i in range(100)]
        for num in range(1, 8):
            for key in keys:
                idx = StripedStorePolicy._adapter_index_for_key(key, num)
                assert 0 <= idx < num, f"index {idx} out of range [0, {num})"

    def test_deterministic_same_key_same_index(self):
        """Same key must always map to the same index within a process."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        key = make_object_key(42)
        num = 4
        idx1 = StripedStorePolicy._adapter_index_for_key(key, num)
        idx2 = StripedStorePolicy._adapter_index_for_key(key, num)
        idx3 = StripedStorePolicy._adapter_index_for_key(key, num)
        assert idx1 == idx2 == idx3

    def test_different_keys_can_map_differently(self):
        """Not all keys should collapse to a single adapter."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        keys = [make_object_key(i) for i in range(100)]
        num = 4
        indices = {
            StripedStorePolicy._adapter_index_for_key(k, num) for k in keys
        }
        # With 100 keys and 4 adapters, all 4 indices should appear
        assert indices == {0, 1, 2, 3}, f"only got {indices}"

    def test_cross_process_stable_blake3(self):
        """BLAKE3 is deterministic across processes (no PYTHONHASHSEED).

        We verify by recomputing the BLAKE3 independently and comparing.
        """
        import blake3

        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        key = make_object_key(7)
        num = 3
        idx = StripedStorePolicy._adapter_index_for_key(key, num)
        expected = int.from_bytes(
            blake3.blake3(str(key).encode()).digest()[:8], "big"
        ) % num
        assert idx == expected


class TestStripedStorePolicyDistribution:
    """Test StripedStorePolicy.select_store_targets distribution uniformity."""

    def test_uniform_distribution(self):
        """Keys should be spread roughly evenly across adapters."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        keys = [make_object_key(i) for i in range(4000)]
        adapters = [make_descriptor(i) for i in range(4)]

        result = policy.select_store_targets(keys, adapters)

        counts = {i: len(result[i]) for i in range(4)}
        # Each adapter should get ~1000 keys; allow 10% deviation
        for i, count in counts.items():
            assert 850 < count < 1150, f"adapter {i} got {count}, expected ~1000"

    def test_each_key_in_exactly_one_adapter(self):
        """No key should appear in more than one adapter's list."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        keys = [make_object_key(i) for i in range(50)]
        adapters = [make_descriptor(i) for i in range(3)]

        result = policy.select_store_targets(keys, adapters)

        # Count occurrences of each key across all adapter lists
        key_counts: dict[int, int] = {}
        for adapter_keys in result.values():
            for k in adapter_keys:
                # Use chunk_hash bytes as identity
                kbytes = bytes(k.chunk_hash)
                key_counts[kbytes] = key_counts.get(kbytes, 0) + 1
        for kbytes, count in key_counts.items():
            assert count == 1, f"key {kbytes} appeared {count} times"

    def test_empty_adapters_returns_empty(self):
        """No adapters means no store targets."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        keys = [make_object_key(0)]

        result = policy.select_store_targets(keys, [])

        assert result == {}

    def test_empty_keys_returns_empty_lists(self):
        """Empty keys list should produce empty lists for each adapter."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets([], adapters)

        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

    def test_single_adapter_all_keys(self):
        """With one adapter, all keys go to it."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        keys = [make_object_key(i) for i in range(5)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert result[0] == keys


class TestStripedStorePolicyDeletions:
    """Test StripedStorePolicy.select_l1_deletions behavior."""

    def test_never_deletes(self):
        """StripedStorePolicy should never delete from L1 (like Default)."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()
        keys = [make_object_key(i) for i in range(5)]

        result = policy.select_l1_deletions(keys)

        assert result == []

    def test_empty_keys_returns_empty(self):
        """Empty input should return empty output."""
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            StripedStorePolicy,
        )

        policy = StripedStorePolicy()

        result = policy.select_l1_deletions([])

        assert result == []
