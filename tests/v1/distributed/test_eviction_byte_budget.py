# SPDX-License-Identifier: Apache-2.0
"""Byte budgets preserve replacement order and legacy count callers."""

# Standard
from typing import cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionByteBudget, EvictionPolicy
from lmcache.v1.distributed.eviction_policy.arc import ARCEvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.noop import NoOpEvictionPolicy

pytestmark = pytest.mark.no_shared_allocator


def make_keys(policy: EvictionPolicy, count: int = 8) -> list[ObjectKey]:
    """Admit keys individually, returning them in oldest-first order."""
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(i), "model", 0, cache_salt="alice")
        for i in range(count)
    ]
    for key in keys:
        policy.on_keys_created([key])
    return keys


@pytest.fixture(
    params=[LRUEvictionPolicy, ARCEvictionPolicy, IsolatedLRUEvictionPolicy]
)
def policy(request: pytest.FixtureRequest) -> EvictionPolicy:
    """Return each replacement policy that supports victim selection."""
    return cast(type[EvictionPolicy], request.param)()


def test_mixed_objects_reach_byte_target(policy: EvictionPolicy) -> None:
    """A 16 KiB budget needs four old 4 KiB objects, not two keys."""
    keys = make_keys(policy)
    sizes = dict(zip(keys, [4096] * 7 + [36864], strict=True))
    actions = policy.get_eviction_actions(
        0.25,
        cache_salt="alice",
        byte_budget=EvictionByteBudget(16384, sizes.__getitem__),
    )
    assert actions[0].keys == keys[:4]


def test_large_oldest_object_rounds_to_whole_object(policy: EvictionPolicy) -> None:
    """The oldest object can cover a budget without evicting newer ones."""
    keys = make_keys(policy)
    sizes = dict(zip(keys, [36864] + [4096] * 7, strict=True))
    actions = policy.get_eviction_actions(
        0.25,
        cache_salt="alice",
        byte_budget=EvictionByteBudget(16384, sizes.__getitem__),
    )
    assert actions[0].keys == keys[:1]


def test_ineligible_objects_do_not_consume_budget(policy: EvictionPolicy) -> None:
    """Zero-size keys and the caller's additional filter are both respected."""
    keys = make_keys(policy)
    sizes = dict.fromkeys(keys, 4096)
    sizes[keys[0]] = 0
    actions = policy.get_eviction_actions(
        0.25,
        cache_salt="alice",
        key_eligible_filter=lambda key: key != keys[1],
        byte_budget=EvictionByteBudget(8192, sizes.__getitem__),
    )
    assert actions[0].keys == keys[2:4]


def test_insufficient_eligible_bytes_return_available_prefix(
    policy: EvictionPolicy,
) -> None:
    """A budget is best effort when only two objects are reclaimable."""
    keys = make_keys(policy)
    sizes = dict.fromkeys(keys, 0)
    sizes.update(dict.fromkeys(keys[:2], 4096))
    actions = policy.get_eviction_actions(
        0.25,
        cache_salt="alice",
        byte_budget=EvictionByteBudget(16384, sizes.__getitem__),
    )
    assert actions[0].keys == keys[:2]


@pytest.mark.parametrize("target", [0, -1])
def test_nonpositive_budget_selects_nothing(
    policy: EvictionPolicy, target: int
) -> None:
    """A byte budget overrides even a positive key-count ratio."""
    make_keys(policy)
    assert (
        policy.get_eviction_actions(
            1.0,
            cache_salt="alice",
            byte_budget=EvictionByteBudget(target, lambda key: 4096),
        )
        == []
    )


def test_no_eligible_bytes_select_nothing(policy: EvictionPolicy) -> None:
    """Missing, locked or already removed keys cannot satisfy a byte budget."""
    make_keys(policy)
    assert (
        policy.get_eviction_actions(
            0.25,
            cache_salt="alice",
            byte_budget=EvictionByteBudget(16384, lambda key: 0),
        )
        == []
    )


def test_sizing_stops_at_selected_prefix(policy: EvictionPolicy) -> None:
    """Victim sizing does not inspect the whole cache to sum its weight."""
    keys = make_keys(policy)
    inspected: list[ObjectKey] = []

    def size(key: ObjectKey) -> int:
        inspected.append(key)
        return 4096

    actions = policy.get_eviction_actions(
        1.0, cache_salt="alice", byte_budget=EvictionByteBudget(8192, size)
    )
    assert actions[0].keys == keys[:2]
    assert inspected == keys[:2]


def test_callers_without_budget_keep_count_rounding(policy: EvictionPolicy) -> None:
    """Existing L2 and coordinator callers retain floor-based key counts."""
    keys = make_keys(policy)
    assert policy.get_eviction_actions(0.3, cache_salt="alice")[0].keys == keys[:2]


def test_fractional_uniform_byte_target_rounds_up(policy: EvictionPolicy) -> None:
    """A 6 KiB byte target requires two whole 4 KiB objects."""
    keys = make_keys(policy, 3)
    actions = policy.get_eviction_actions(
        0.5, cache_salt="alice", byte_budget=EvictionByteBudget(6144, lambda key: 4096)
    )
    assert actions[0].keys == keys[:2]


def test_arc_retires_only_byte_selected_victims() -> None:
    """An unrelated removal must not enter ARC's policy-eviction history."""
    policy = ARCEvictionPolicy()
    keys = make_keys(policy)
    actions = policy.get_eviction_actions(
        1.0, byte_budget=EvictionByteBudget(8192, lambda key: 4096)
    )
    policy.on_keys_removed(actions[0].keys)
    policy.on_keys_removed(keys[2:3])
    assert policy.get_debug_state()["b1"] == keys[:2]


def test_isolated_budget_does_not_borrow_another_salts_bytes() -> None:
    """A large object in another salt cannot satisfy this salt's budget."""
    policy = IsolatedLRUEvictionPolicy()
    alice = make_keys(policy, 2)
    bob = ObjectKey(ObjectKey.IntHash2Bytes(100), "model", 0, cache_salt="bob")
    policy.on_keys_created([bob])
    sizes = {alice[0]: 4096, alice[1]: 4096, bob: 36864}
    actions = policy.get_eviction_actions(
        0.25,
        cache_salt="alice",
        byte_budget=EvictionByteBudget(16384, sizes.__getitem__),
    )
    assert actions[0].keys == alice


def test_noop_keeps_eviction_disabled() -> None:
    """A byte target must not turn a write-buffer policy into a cache policy."""
    policy = NoOpEvictionPolicy()
    make_keys(policy)
    assert (
        policy.get_eviction_actions(
            1.0, byte_budget=EvictionByteBudget(16384, lambda key: 4096)
        )
        == []
    )
