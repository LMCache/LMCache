# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for store policy interface, DefaultStorePolicy and P2PStorePolicy.

Tests are written against the StorePolicy contract defined in store_policy.py.
"""

# Standard
from typing import cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_policy import (
    DefaultStorePolicy,
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
        policy = DefaultStorePolicy([])
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert 0 in result
        assert result[0] == keys

    def test_multiple_adapters_all_keys_to_each(self):
        """All keys should be sent to every adapter."""
        policy = DefaultStorePolicy([])
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert len(result) == 2
        assert result[0] == keys
        assert result[1] == keys

    def test_empty_adapters_returns_empty(self):
        """No adapters means no store targets."""
        policy = DefaultStorePolicy([])
        keys = [make_object_key(0)]

        result = policy.select_store_targets(keys, [])

        assert result == {}

    def test_empty_keys_returns_empty_lists(self):
        """Empty keys list should produce empty lists for each adapter."""
        policy = DefaultStorePolicy([])
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets([], adapters)

        assert 0 in result
        assert result[0] == []

    def test_returns_copies_not_references(self):
        """Returned lists should be independent copies of the input."""
        policy = DefaultStorePolicy([])
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
        policy = DefaultStorePolicy([])
        keys = [make_object_key(i) for i in range(5)]

        result = policy.select_l1_deletions(keys, make_descriptor(0))

        assert result == []

    def test_empty_keys_returns_empty(self):
        """Empty input should return empty output."""
        policy = DefaultStorePolicy([])

        result = policy.select_l1_deletions([], make_descriptor(0))

        assert result == []


# =============================================================================
# P2PStorePolicy Tests
# =============================================================================

_LOCAL_DRAM = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
_LOCAL_CXL = MemoryTags(MemorySegment.CXL, MemoryResidency.LOCAL)


class _StubL1:
    """The part of an L1 manager a store policy reads."""

    def __init__(self, tags: MemoryTags, used_bytes: int, total_bytes: int = 1000):
        self._tags = tags
        self._usage = (used_bytes, total_bytes)

    def get_memory_tags(self) -> MemoryTags:
        return self._tags

    def get_memory_usage(self) -> tuple[int, int]:
        return self._usage


def make_peer_descriptor(
    index: int, write_enabled: bool = True, threshold: float = 0.7
) -> L2AdapterDescriptor:
    """Create the descriptor of a P2P adapter for one peer."""
    config = P2PL2AdapterConfig(
        peer_mq_server_url=f"tcp://peer{index}:5555",
        peer_transfer_channel_server_url=f"peer{index}:7600",
        write_enabled=write_enabled,
        write_l1_usage_threshold=threshold,
    )
    return L2AdapterDescriptor(index=index, config=config)


def make_p2p_policy(used_bytes: int, tags: MemoryTags = _LOCAL_DRAM):
    """Create the registered ``p2p`` policy over one L1 of 1000 bytes."""
    l1_manager = cast(L1ManagerInterface, _StubL1(tags, used_bytes))
    return create_store_policy("p2p", [l1_manager])


class TestP2PStorePolicyTargets:
    """Test where P2PStorePolicy sends newly written keys."""

    @pytest.mark.parametrize(
        ("used_bytes", "threshold", "stored_to_peer"),
        [
            pytest.param(699, 0.7, False, id="below-the-threshold"),
            pytest.param(700, 0.7, True, id="at-the-threshold"),
            pytest.param(950, 0.7, True, id="above-the-threshold"),
            pytest.param(600, 0.5, True, id="the-adapters-own-threshold-counts"),
        ],
    )
    def test_peer_gets_keys_only_from_the_usage_threshold_on(
        self, used_bytes: int, threshold: float, stored_to_peer: bool
    ):
        """Objects stay local while the DRAM L1 has room; other adapters get
        every key either way."""
        policy = make_p2p_policy(used_bytes)
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_peer_descriptor(1, threshold=threshold)]

        targets = policy.select_store_targets(keys, adapters)

        assert targets[0] == keys
        assert (targets.get(1) == keys) is stored_to_peer
        assert set(targets) == ({0, 1} if stored_to_peer else {0})

    def test_write_disabled_peer_never_gets_keys(self):
        """A peer this server only reads from gets no store task at all."""
        policy = make_p2p_policy(used_bytes=1000)
        keys = [make_object_key(0)]

        targets = policy.select_store_targets(
            keys, [make_peer_descriptor(0, write_enabled=False)]
        )

        assert targets == {}

    def test_write_enabled_peers_take_turns(self):
        """Each store task goes to exactly one peer, rotating over the peers
        that accept writes."""
        policy = make_p2p_policy(used_bytes=900)
        adapters = [
            make_peer_descriptor(0),
            make_peer_descriptor(1, write_enabled=False),
            make_peer_descriptor(2),
        ]

        chosen = [
            sorted(policy.select_store_targets([make_object_key(i)], adapters))
            for i in range(4)
        ]

        assert chosen == [[0], [2], [0], [2]]

    def test_without_a_local_dram_l1_no_peer_is_chosen(self):
        """Only the local DRAM L1 is transferable to a peer."""
        policy = make_p2p_policy(used_bytes=1000, tags=_LOCAL_CXL)

        targets = policy.select_store_targets(
            [make_object_key(0)], [make_peer_descriptor(0)]
        )

        assert targets == {}


class TestP2PStorePolicyDeletions:
    """Test which L1 copies P2PStorePolicy drops after a store."""

    @pytest.mark.parametrize(
        ("adapter", "deleted"),
        [
            pytest.param(make_peer_descriptor(0), True, id="write-enabled-peer"),
            pytest.param(
                make_peer_descriptor(0, write_enabled=False),
                False,
                id="write-disabled-peer",
            ),
            pytest.param(make_descriptor(0), False, id="other-adapter"),
        ],
    )
    def test_local_copy_is_dropped_only_after_a_store_into_a_peer(
        self, adapter: L2AdapterDescriptor, deleted: bool
    ):
        policy = make_p2p_policy(used_bytes=900)
        keys = [make_object_key(i) for i in range(2)]

        result = policy.select_l1_deletions(keys, adapter)

        assert result == (keys if deleted else [])
