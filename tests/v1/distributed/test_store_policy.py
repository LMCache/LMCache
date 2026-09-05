# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for store policy interface and DefaultStorePolicy.

Tests are written against the StorePolicy contract defined in store_policy.py.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
    StripedStorePolicy,
    rendezvous_adapter_index_for_key,
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


def make_stable_descriptor(index: int, placement_id: str) -> AdapterDescriptor:
    """Create a descriptor with a persistent placement identifier."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    config.placement_id = placement_id
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


class TestStripedStorePolicy:
    """Stable rendezvous placement contract."""

    def test_each_key_has_one_owner(self) -> None:
        policy = StripedStorePolicy()
        keys = [make_object_key(i) for i in range(1000)]
        adapters = [make_stable_descriptor(i, f"disk-{i}") for i in range(4)]

        targets = policy.select_store_targets(keys, adapters)
        routed_keys = [key for values in targets.values() for key in values]

        assert set(routed_keys) == set(keys)
        assert len(routed_keys) == len(keys)

    def test_runtime_order_does_not_change_owner(self) -> None:
        keys = [make_object_key(i) for i in range(100)]
        original = [make_stable_descriptor(i, f"disk-{i}") for i in range(4)]
        reordered = [original[2], original[0], original[3], original[1]]
        original_by_index = {adapter.index: adapter for adapter in original}
        reordered_by_index = {adapter.index: adapter for adapter in reordered}

        original_owners = [
            original_by_index[
                rendezvous_adapter_index_for_key(key, original)
            ].placement_id
            for key in keys
        ]
        reordered_owners = [
            reordered_by_index[
                rendezvous_adapter_index_for_key(key, reordered)
            ].placement_id
            for key in keys
        ]

        assert original_owners == reordered_owners

    def test_adding_disk_only_remaps_approximately_ideal_share(self) -> None:
        keys = [make_object_key(i) for i in range(10_000)]
        before = [make_stable_descriptor(i, f"disk-{i}") for i in range(8)]
        after = before + [make_stable_descriptor(8, "disk-8")]
        before_by_index = {adapter.index: adapter for adapter in before}
        after_by_index = {adapter.index: adapter for adapter in after}

        remapped = 0
        for key in keys:
            before_owner = before_by_index[
                rendezvous_adapter_index_for_key(key, before)
            ].placement_id
            after_owner = after_by_index[
                rendezvous_adapter_index_for_key(key, after)
            ].placement_id
            remapped += before_owner != after_owner

        ratio = remapped / len(keys)
        assert 0.09 < ratio < 0.14

    def test_removing_disk_preserves_other_owners(self) -> None:
        keys = [make_object_key(i) for i in range(2000)]
        before = [make_stable_descriptor(i, f"disk-{i}") for i in range(8)]
        after = before[:-1]
        before_by_index = {adapter.index: adapter for adapter in before}
        after_by_index = {adapter.index: adapter for adapter in after}

        for key in keys:
            before_owner = before_by_index[
                rendezvous_adapter_index_for_key(key, before)
            ].placement_id
            after_owner = after_by_index[
                rendezvous_adapter_index_for_key(key, after)
            ].placement_id
            if before_owner != "disk-7":
                assert after_owner == before_owner

    def test_missing_or_duplicate_placement_id_is_rejected(self) -> None:
        key = make_object_key(0)
        with pytest.raises(ValueError, match="stable placement_id"):
            rendezvous_adapter_index_for_key(key, [make_descriptor(0)])

        duplicate = [
            make_stable_descriptor(0, "same-disk"),
            make_stable_descriptor(1, "same-disk"),
        ]
        with pytest.raises(ValueError, match="unique"):
            rendezvous_adapter_index_for_key(key, duplicate)
