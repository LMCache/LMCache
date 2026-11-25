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
    L1_ONLY_ATTRIBUTE,
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
    StoreAction,
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


def make_l1_only_key(chunk_id: int) -> ObjectKey:
    """Create a key marked L1-only, as the server marks an unfull-chunk tail."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        extra_attributes={L1_ONLY_ATTRIBUTE: True},
    )


class TestDefaultStorePolicy:
    """Test DefaultStorePolicy.get_store_actions behavior."""

    def test_every_key_to_every_adapter_and_kept_in_l1(self):
        policy = DefaultStorePolicy([])
        keys = [make_object_key(i) for i in range(3)]

        actions = policy.get_store_actions(
            keys, [make_descriptor(0), make_descriptor(1)]
        )

        assert actions == [
            StoreAction(keys=keys, destination=0, l1_delete=False),
            StoreAction(keys=keys, destination=1, l1_delete=False),
        ]

    def test_no_adapters_means_no_actions(self):
        policy = DefaultStorePolicy([])

        assert policy.get_store_actions([make_object_key(0)], []) == []

    def test_returns_copies_not_references(self):
        """Returned key lists are independent copies of the input."""
        policy = DefaultStorePolicy([])
        keys = [make_object_key(0)]

        (action,) = policy.get_store_actions(keys, [make_descriptor(0)])
        action.keys.append(make_object_key(99))

        assert len(keys) == 1


class TestBufferOnlyStorePolicy:
    """Test the ``skip_l1`` policy, meant for noop eviction."""

    def test_every_key_to_every_adapter_then_out_of_l1(self):
        policy = create_store_policy("skip_l1", [])
        keys = [make_object_key(i) for i in range(2)]

        actions = policy.get_store_actions(
            keys, [make_descriptor(0), make_descriptor(1)]
        )

        assert actions == [
            StoreAction(keys=keys, destination=0, l1_delete=True),
            StoreAction(keys=keys, destination=1, l1_delete=True),
        ]

    @pytest.mark.parametrize(
        "adapters",
        [pytest.param([], id="no-adapters"), pytest.param([make_descriptor(0)])],
    )
    def test_l1_only_keys_leave_l1_right_away(self, adapters):
        """Nothing else would free an L1-only key under noop eviction."""
        policy = create_store_policy("skip_l1", [])
        keys = [make_l1_only_key(i) for i in range(2)]

        actions = policy.get_store_actions(keys, adapters)

        assert actions == [StoreAction(keys=keys, destination=None, l1_delete=True)]


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


def peer_destinations(actions: list[StoreAction]) -> list[int | None]:
    """Destinations of the actions that drop their keys from L1."""
    return [a.destination for a in actions if a.l1_delete]


class TestP2PStorePolicy:
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
        every key either way and keep the L1 copy. A peer store drops it."""
        policy = make_p2p_policy(used_bytes)
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_peer_descriptor(1, threshold=threshold)]

        actions = policy.get_store_actions(keys, adapters)

        expected = [StoreAction(keys=keys, destination=0, l1_delete=False)]
        if stored_to_peer:
            expected.append(StoreAction(keys=keys, destination=1, l1_delete=True))
        assert actions == expected

    def test_write_disabled_peer_never_gets_keys(self):
        """A peer this server only reads from gets no store task at all."""
        policy = make_p2p_policy(used_bytes=1000)

        actions = policy.get_store_actions(
            [make_object_key(0)], [make_peer_descriptor(0, write_enabled=False)]
        )

        assert actions == []

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
            peer_destinations(policy.get_store_actions([make_object_key(i)], adapters))
            for i in range(4)
        ]

        assert chosen == [[0], [2], [0], [2]]

    def test_without_a_local_dram_l1_no_peer_is_chosen(self):
        """Only the local DRAM L1 is transferable to a peer."""
        policy = make_p2p_policy(used_bytes=1000, tags=_LOCAL_CXL)

        actions = policy.get_store_actions(
            [make_object_key(0)], [make_peer_descriptor(0)]
        )

        assert actions == []
