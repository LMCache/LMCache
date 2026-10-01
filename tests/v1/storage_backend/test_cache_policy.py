# SPDX-License-Identifier: Apache-2.0
"""Tests for the local-CPU cache eviction policies.

These exercise the public policy contract (``get_cache_policy`` plus the
``update_on_*`` / ``get_evict_candidates`` interface) rather than any policy's
internals, so they hold for every policy regardless of how it tracks state.
"""

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
    """Selecting a candidate must not evict it from the policy's bookkeeping.

    The caller may legitimately leave a returned candidate resident -- e.g.
    ``LocalCPUBackend.batched_allocate`` skips a candidate whose layer is
    pinned. The key is then still in the cache, so a later hit on it (and a
    later selection) must behave normally. Regression test for an LFU crash
    where ``get_evict_candidates`` dropped the candidate from its frequency
    bookkeeping, so the next ``update_on_hit`` raised ``KeyError``.
    """
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=True)})

    assert policy.get_evict_candidates(cache, num_candidates=1) == ["k"]

    # The key was selected but not evicted; it is still resident.
    policy.update_on_hit("k", cache)

    # It remains a valid eviction candidate afterwards.
    assert policy.get_evict_candidates(cache, num_candidates=1) == ["k"]


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_force_evict_then_key_is_no_longer_a_candidate(policy_name):
    policy = get_cache_policy(policy_name)
    policy.update_on_put("k")
    cache = OrderedDict({"k": _Entry(can_evict=True)})

    # Mirror the backend: drop the key from the cache and notify the policy.
    del cache["k"]
    policy.update_on_force_evict("k")

    assert policy.get_evict_candidates(cache, num_candidates=1) == []


def test_lfu_evicts_least_frequently_used_first():
    policy = get_cache_policy("LFU")
    for key in ("a", "b", "c"):
        policy.update_on_put(key)
    cache = OrderedDict({"a": _Entry(), "b": _Entry(), "c": _Entry()})

    # Raise frequencies: b -> 3, c -> 2, a stays at 1.
    policy.update_on_hit("b", cache)
    policy.update_on_hit("b", cache)
    policy.update_on_hit("c", cache)

    assert policy.get_evict_candidates(cache, num_candidates=3) == ["a", "c", "b"]
