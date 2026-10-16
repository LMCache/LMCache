# SPDX-License-Identifier: Apache-2.0
"""Contract tests shared by every cache eviction policy."""

# Standard
from collections import OrderedDict

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cache_policy import POLICY_MAPPING, get_cache_policy

ALL_POLICIES = sorted(POLICY_MAPPING)


class _Entry:
    """Minimal stand-in for a cached object exposing ``can_evict``."""

    def __init__(self, can_evict: bool = True) -> None:
        self.can_evict = can_evict


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_get_evict_candidates_selects_evictable_key(policy_name):
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=True)})

    assert policy.get_evict_candidates(cache, num_candidates=1) == ["k"]


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_get_evict_candidates_skips_non_evictable_key(policy_name):
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=False)})

    assert policy.get_evict_candidates(cache, num_candidates=1) == []


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_get_evict_candidates_is_side_effect_free(policy_name):
    """A selected but not evicted key must still accept hits (issue #5444)."""
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=True)})

    assert policy.get_evict_candidates(cache, num_candidates=1) == ["k"]

    policy.update_on_hit("k", cache)

    assert policy.get_evict_candidates(cache, num_candidates=1) == ["k"]


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_force_evict_then_key_is_no_longer_a_candidate(policy_name):
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=True)})

    del cache["k"]
    policy.update_on_force_evict("k")

    assert policy.get_evict_candidates(cache, num_candidates=1) == []


def test_lfu_evicts_least_frequently_used_first():
    policy = get_cache_policy("LFU")
    for key in ("a", "b", "c"):
        policy.update_on_put(key)
    cache = OrderedDict({"a": _Entry(), "b": _Entry(), "c": _Entry()})

    # b -> 3, c -> 2, a -> 1
    policy.update_on_hit("b", cache)
    policy.update_on_hit("b", cache)
    policy.update_on_hit("c", cache)

    assert policy.get_evict_candidates(cache, num_candidates=3) == ["a", "c", "b"]
