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
    OPTIONAL_WINDOW_ATTRIBUTE,
    DefaultStorePolicy,
    WindowInL1StorePolicy,
    create_store_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor

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


def make_descriptor(index: int) -> L2AdapterDescriptor:
    """Create an L2AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return L2AdapterDescriptor(index=index, config=config)


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
# WindowInL1StorePolicy Tests
# =============================================================================


def make_window_key(chunk_id: int) -> ObjectKey:
    """Create a key flagged as an optional sliding window."""
    key = make_object_key(chunk_id)
    key.extra_attributes[OPTIONAL_WINDOW_ATTRIBUTE] = True
    return key


class TestWindowInL1StorePolicy:
    """Windows stay in L1 only; every other key moves to L2."""

    def test_registered_as_swa_l1(self):
        assert isinstance(create_store_policy("swa_l1"), WindowInL1StorePolicy)

    def test_windows_never_go_to_l2(self):
        policy = WindowInL1StorePolicy()
        global_kv = [make_object_key(0), make_object_key(1)]
        windows = [make_window_key(2)]

        result = policy.select_store_targets(
            global_kv + windows, [make_descriptor(0), make_descriptor(1)]
        )

        assert result == {0: global_kv, 1: global_kv}

    def test_l2_keys_are_deleted_from_l1(self):
        policy = WindowInL1StorePolicy()
        keys = [make_object_key(i) for i in range(3)]

        assert policy.select_l1_deletions(keys) == keys
