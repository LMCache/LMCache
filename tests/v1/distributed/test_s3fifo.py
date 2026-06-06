# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for S3FIFO eviction policy.

These tests verify the basic functionality of the S3FIFOEvictionPolicy:

1. Key tracking (create, delete)
2. Safe handling of duplicate/nonexistent keys
3. Eviction actions behavior with different ratios
4. Eviction candidates API correctness
5. Eviction destination logic
6. Key eligibility filtering support
"""


# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.s3fifo import S3FIFOEvictionPolicy
from lmcache.v1.distributed.internal_api import EvictionDestination


# =============================================================================
# Helper
# =============================================================================

def make_key(chunk_hash: int, model: str = "model", kv_rank: int = 0) -> ObjectKey:
    """Create an ObjectKey for testing."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model, kv_rank=kv_rank)


# =============================================================================
# Basic Functionality
# =============================================================================

def test_empty_policy_has_no_keys():
    policy = S3FIFOEvictionPolicy()
    assert policy.get_num_tracked_keys() == 0


def test_create_keys_increases_count():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(1), make_key(2), make_key(3)])

    assert policy.get_num_tracked_keys() == 3


def test_delete_keys_decreases_count():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(1), make_key(2), make_key(3)])
    policy.on_keys_removed([make_key(2)])

    assert policy.get_num_tracked_keys() == 2


def test_delete_nonexistent_key_is_safe():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_removed([make_key(999)])

    assert policy.get_num_tracked_keys() == 0


def test_duplicate_key_does_not_duplicate_count():
    policy = S3FIFOEvictionPolicy()

    key = make_key(1)
    policy.on_keys_created([key])
    policy.on_keys_created([key])

    assert policy.get_num_tracked_keys() == 1


# =============================================================================
# Eviction Actions (Ratio)
# =============================================================================

def test_ratio_zero_returns_no_actions():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(10)])

    actions = policy.get_eviction_actions(0.0)
    assert actions == []


def test_ratio_one_returns_some_evictions():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(10)])

    actions = policy.get_eviction_actions(1.0)

    assert len(actions) >= 1
    assert len(actions[0].keys) == 10


def test_ratio_half_returns_partial_eviction():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(10)])

    actions = policy.get_eviction_actions(0.5)

    assert len(actions) >= 1
    assert len(actions[0].keys) == 5


def test_ratio_clamped_behavior():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(10)])

    assert policy.get_eviction_actions(-1.0) == []
    assert len(policy.get_eviction_actions(2.0)[0].keys) == 10


# =============================================================================
# Eviction Candidates
# =============================================================================

def test_get_eviction_candidates_respects_count():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(10)])

    candidates = policy.get_eviction_candidates(3)

    assert len(candidates) == 3


def test_get_eviction_candidates_does_not_crash_with_large_count():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(1), make_key(2)])

    candidates = policy.get_eviction_candidates(100)

    assert len(candidates) == 2


# =============================================================================
# Eviction Destination
# =============================================================================

def test_default_destination_is_discard():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(1)])

    actions = policy.get_eviction_actions(1.0)

    assert actions[0].destination == EvictionDestination.DISCARD


def test_custom_destination_is_used():
    policy = S3FIFOEvictionPolicy(default_destination=EvictionDestination.L2_CACHE)

    policy.on_keys_created([make_key(1)])

    actions = policy.get_eviction_actions(1.0)

    assert actions[0].destination == EvictionDestination.L2_CACHE


def test_registered_destination_overrides_default():
    policy = S3FIFOEvictionPolicy(default_destination=EvictionDestination.DISCARD)

    policy.register_eviction_destination(EvictionDestination.L2_CACHE)

    policy.on_keys_created([make_key(1)])

    actions = policy.get_eviction_actions(1.0)

    assert actions[0].destination == EvictionDestination.L2_CACHE


# =============================================================================
# Key Eligibility Filter
# =============================================================================

def test_key_eligible_filter_skips_keys():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(5)])

    def filter_fn(key: ObjectKey) -> bool:
        return ObjectKey.Bytes2IntHash(key.chunk_hash) % 2 == 0

    actions = policy.get_eviction_actions(1.0, key_eligible_filter=filter_fn)

    evicted = {
        ObjectKey.Bytes2IntHash(k.chunk_hash)
        for k in actions[0].keys
    }

    assert evicted.issubset({0, 2, 4})


def test_key_eligible_filter_none_allows_all():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(5)])

    actions = policy.get_eviction_actions(1.0, key_eligible_filter=None)

    assert len(actions[0].keys) == 5


def test_key_eligible_filter_all_rejected():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(5)])

    actions = policy.get_eviction_actions(1.0, key_eligible_filter=lambda _: False)

    assert actions == []


def test_key_eligible_filter_respects_limit():
    policy = S3FIFOEvictionPolicy()

    policy.on_keys_created([make_key(i) for i in range(20)])

    actions = policy.get_eviction_actions(
        0.2,
        key_eligible_filter=lambda k: True
    )

    assert len(actions[0].keys) == 4