# SPDX-License-Identifier: Apache-2.0
"""Regression tests for chunk-coherent distributed LRU eviction."""

# Standard
from collections.abc import Iterable

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy import (
    IsolatedLRUEvictionPolicy,
    LRUEvictionPolicy,
)


def _key(
    chunk_id: int,
    kv_rank: int,
    object_group_id: int,
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="hybrid-model",
        kv_rank=kv_rank,
        object_group_id=object_group_id,
        cache_salt=cache_salt,
    )


def _family(
    chunk_id: int,
    cache_salt: str = "",
) -> set[ObjectKey]:
    return {
        _key(chunk_id, kv_rank, object_group_id, cache_salt)
        for kv_rank in range(2)
        for object_group_id in range(2)
    }


def _register_rank_group_batches(
    policy: LRUEvictionPolicy | IsolatedLRUEvictionPolicy,
    chunk_ids: Iterable[int],
    cache_salt: str = "",
) -> None:
    """Mimic asynchronously completed per-rank/object-group store batches."""
    for kv_rank in range(2):
        for object_group_id in range(2):
            policy.on_keys_created(
                [
                    _key(chunk_id, kv_rank, object_group_id, cache_salt)
                    for chunk_id in chunk_ids
                ]
            )


def test_global_lru_does_not_split_rank_or_object_group_family() -> None:
    policy = LRUEvictionPolicy()
    _register_rank_group_batches(policy, range(3))

    actions = policy.get_eviction_actions(0.25)

    assert len(actions) == 1
    selected = set(actions[0].keys)
    assert len(selected) == 4
    assert selected in (_family(0), _family(1), _family(2))


def test_locked_family_is_skipped_as_a_unit() -> None:
    policy = LRUEvictionPolicy()
    _register_rank_group_batches(policy, range(2))
    locked = _key(1, kv_rank=0, object_group_id=0)

    actions = policy.get_eviction_actions(
        1.0,
        key_eligible_filter=lambda key: key != locked,
    )

    assert len(actions) == 1
    selected = set(actions[0].keys)
    assert selected == _family(0)
    assert selected.isdisjoint(_family(1))


def test_removed_family_members_are_not_returned_as_stale_siblings() -> None:
    policy = LRUEvictionPolicy()
    _register_rank_group_batches(policy, range(2))
    removed = _family(0)
    policy.on_keys_removed(list(removed))

    actions = policy.get_eviction_actions(1.0)

    assert len(actions) == 1
    assert set(actions[0].keys) == _family(1)


def test_incomplete_family_is_skipped_until_late_sibling_arrives() -> None:
    policy = LRUEvictionPolicy()
    family = _family(0)
    policy.on_keys_created(list(family))
    late_sibling = _key(0, kv_rank=1, object_group_id=1)
    policy.on_keys_removed([late_sibling])

    assert policy.get_eviction_actions(1.0) == []

    policy.on_keys_created([late_sibling])
    actions = policy.get_eviction_actions(1.0)
    assert len(actions) == 1
    assert set(actions[0].keys) == family


def test_family_topology_is_scoped_to_cache_model_namespace() -> None:
    policy = LRUEvictionPolicy()
    hybrid_family = _family(0)
    policy.on_keys_created(list(hybrid_family))
    policy.on_keys_removed(list(hybrid_family))
    ordinary_key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(1),
        model_name="ordinary-model",
        kv_rank=0,
    )
    policy.on_keys_created([ordinary_key])

    actions = policy.get_eviction_actions(1.0)

    assert len(actions) == 1
    assert actions[0].keys == [ordinary_key]


def test_packed_rank_topology_skips_family_before_other_rank_completes() -> None:
    policy = LRUEvictionPolicy()
    ranks = [
        ObjectKey.ComputeKVRank(
            world_size=2,
            global_rank=rank,
            local_world_size=2,
            local_rank=rank,
        )
        for rank in range(2)
    ]
    rank_zero_batch = {
        _key(0, kv_rank=ranks[0], object_group_id=object_group_id)
        for object_group_id in range(2)
    }
    rank_one_batch = {
        _key(0, kv_rank=ranks[1], object_group_id=object_group_id)
        for object_group_id in range(2)
    }

    policy.on_keys_created(list(rank_zero_batch))
    assert policy.get_eviction_actions(1.0) == []

    policy.on_keys_created(list(rank_one_batch))
    actions = policy.get_eviction_actions(1.0)
    assert len(actions) == 1
    assert set(actions[0].keys) == rank_zero_batch | rank_one_batch


def test_parallel_topologies_do_not_require_each_others_ranks() -> None:
    policy = LRUEvictionPolicy()
    keys = []
    for world_size in (2, 4):
        for rank in range(world_size):
            keys.append(
                _key(
                    chunk_id=world_size,
                    kv_rank=ObjectKey.ComputeKVRank(
                        world_size=world_size,
                        global_rank=rank,
                        local_world_size=world_size,
                        local_rank=rank,
                    ),
                    object_group_id=0,
                )
            )
    policy.on_keys_created(keys)

    actions = policy.get_eviction_actions(0.1)

    assert len(actions) == 1
    selected = actions[0].keys
    selected_topologies = {
        ((key.kv_rank >> 24) & 0xFF, key.chunk_hash) for key in selected
    }
    assert len(selected_topologies) == 1
    assert len(selected) in (2, 4)


@pytest.mark.parametrize("cache_salt", ["tenant-a", "tenant-b"])
def test_isolated_lru_keeps_each_salt_chunk_coherent(cache_salt: str) -> None:
    policy = IsolatedLRUEvictionPolicy()
    _register_rank_group_batches(policy, range(3), cache_salt)

    actions = policy.get_eviction_actions(0.25, cache_salt=cache_salt)

    assert len(actions) == 1
    selected = set(actions[0].keys)
    assert len(selected) == 4
    assert selected in (
        _family(0, cache_salt),
        _family(1, cache_salt),
        _family(2, cache_salt),
    )


def test_isolated_lru_never_crosses_salt_boundaries() -> None:
    policy = IsolatedLRUEvictionPolicy()
    _register_rank_group_batches(policy, range(2), "tenant-a")
    _register_rank_group_batches(policy, range(2), "tenant-b")

    actions = policy.get_eviction_actions(1.0, cache_salt="tenant-a")

    assert len(actions) == 1
    assert set(actions[0].keys) == _family(0, "tenant-a") | _family(1, "tenant-a")
