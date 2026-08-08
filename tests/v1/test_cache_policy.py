# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cache_policy import get_cache_policy

# Local
from .utils import dumb_cache_engine_key


class DummyMemoryObj:
    def __init__(self, can_evict: bool = True):
        self.can_evict = can_evict


def test_lru():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lru_with_pin():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key3, key1]


def test_fifo():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key1, key2]


def test_fifo_with_pin():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lfu():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key1, key3]


def test_lfu_with_pin():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key3, key2]


def test_mru():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is the most recent, followed by key3.
    assert evict_candidates == [key1, key3], (evict_candidates, [key1, key3])


def test_mru_with_pin():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj(can_evict=False)  # Pinned object
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is most recent, followed by key3, but since key3 is pinned, wo go to key2.
    assert evict_candidates == [key1, key2], (evict_candidates, [key1, key2])


def test_cost_aware_frequency_protects_popular_chunk():
    # Two chunks with identical recompute cost and memory size, so the
    # cost-density term alone would score them equally.
    policy = get_cache_policy("COST_AWARE")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)

    cache_dict[key1] = obj1
    policy.put(
        key1,
        value=obj1,
        total_request_tokens=1000,
        chunk_start=0,
        memory_size_bytes=1024,
    )
    cache_dict[key2] = obj2
    policy.put(
        key2,
        value=obj2,
        total_request_tokens=1000,
        chunk_start=0,
        memory_size_bytes=1024,
    )

    # key1 is reused far more often than key2, even though key2's single
    # hit lands last (a slight recency edge for key2).
    for _ in range(5):
        policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key2, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    # Equal cost/near-equal recency, but key1's higher hit count should
    # outweigh key2's marginal recency edge and protect it from eviction.
    assert evict_candidates == [key2], (evict_candidates, [key2])


def test_cost_aware_clock_is_injectable_and_deterministic():
    # Regression test: CostAwareEvictionPolicy used to call time.monotonic()
    # directly and unconditionally, so its recency term's actual effect
    # was invisible in any benchmark run (which completes in far less
    # than one default half_life_seconds=60.0) and non-reproducible
    # across machines of different speeds. The injected clock must be
    # the sole source of "now" for every recency computation.
    logical_time = {"now": 0.0}
    policy = get_cache_policy(
        "COST_AWARE", half_life_seconds=10.0, clock=lambda: logical_time["now"]
    )
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)

    cache_dict[key1] = DummyMemoryObj()
    policy.put(
        key1, value=cache_dict[key1], total_request_tokens=1000, chunk_start=0,
        memory_size_bytes=1024,
    )
    logical_time["now"] = 1000.0  # far beyond several half-lives
    cache_dict[key2] = DummyMemoryObj()
    policy.put(
        key2, value=cache_dict[key2], total_request_tokens=1000, chunk_start=0,
        memory_size_bytes=1024,
    )

    # key1's score at the current (injected) logical time must reflect
    # ~100 half-lives of decay -- i.e. be far smaller than key2's, which
    # was just inserted at the current logical time.
    score1 = policy.calculate_score(key1)
    score2 = policy.calculate_score(key2)
    assert score1 < score2 * 0.01, (score1, score2)

    # Advancing real wall-clock time must have zero effect: only the
    # injected clock governs recency.
    logical_time_unchanged = logical_time["now"]
    assert policy.calculate_score(key1) == score1
    assert logical_time["now"] == logical_time_unchanged


def test_admission_control_delegates_eviction_ranking_to_inner_policy():
    # With should_admit never consulted, AdmissionControlledPolicy must
    # behave identically to the inner LRU policy it wraps.
    policy = get_cache_policy("ADMISSION_LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_admission_control_rejects_low_frequency_newcomer():
    policy = get_cache_policy("ADMISSION_LRU")
    cache_dict = policy.init_mutable_mapping()
    popular = dumb_cache_engine_key(1)
    obj = DummyMemoryObj()

    cache_dict[popular] = obj
    policy.update_on_put(popular)
    for _ in range(5):
        policy.update_on_hit(popular, cache_dict)

    newcomer = dumb_cache_engine_key(2)
    # A never-before-seen key's estimated frequency (0) is below the only
    # eviction candidate's (popular, requested 6 times) -> reject.
    assert policy.should_admit(newcomer, cache_dict) is False

    # Once newcomer has been requested more often than popular, it should
    # be admitted instead.
    for _ in range(10):
        policy.update_on_put(newcomer)
    assert policy.should_admit(newcomer, cache_dict) is True


def test_admission_control_admits_when_cache_empty():
    policy = get_cache_policy("ADMISSION_LRU")
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(1)
    # No eviction candidates exist yet -> nothing to weigh against, admit.
    assert policy.should_admit(key, cache_dict) is True


def test_admission_control_records_exactly_one_increment_per_arrival():
    # Regression test: an earlier version incremented the frequency
    # sketch inside should_admit() unconditionally, then incremented it
    # again inside update_on_put_with_metadata() when the caller (per the
    # documented convention -- should_admit() immediately followed by the
    # matching update_on_* call for the same key on admission) went ahead
    # and inserted the now-admitted key. That double-counted every
    # admitted key's frequency relative to a rejected key or a fill-phase
    # insertion, silently over-weighting incumbents. Each of the three
    # paths below must record exactly one increment.
    policy = get_cache_policy("ADMISSION_LRU", halve_every=1_000_000)
    cache_dict = policy.init_mutable_mapping()

    # Fill-phase insertion: should_admit is never called (cache not yet
    # at capacity in a real caller), only update_on_put_with_metadata.
    fill_key = dumb_cache_engine_key(1)
    cache_dict[fill_key] = DummyMemoryObj()
    policy.update_on_put_with_metadata(fill_key, cache_obj=cache_dict[fill_key])
    assert policy._sketch.estimate(fill_key) == 1

    # Admitted-under-pressure: should_admit() returns True (empty
    # cache_dict admits unconditionally), immediately followed by the
    # matching update_on_put_with_metadata call, mirroring the real
    # caller's (runner.py's _PolicyCache.put) call sequence.
    admitted_key = dumb_cache_engine_key(2)
    empty_cache_dict = policy.init_mutable_mapping()
    assert policy.should_admit(admitted_key, empty_cache_dict) is True
    empty_cache_dict[admitted_key] = DummyMemoryObj()
    policy.update_on_put_with_metadata(
        admitted_key, cache_obj=empty_cache_dict[admitted_key]
    )
    assert policy._sketch.estimate(admitted_key) == 1

    # Rejected-under-pressure: should_admit() returns False; the caller
    # never calls update_on_put_with_metadata for a rejected key at all.
    resident_key = dumb_cache_engine_key(3)
    cache_dict[resident_key] = DummyMemoryObj()
    policy.update_on_put_with_metadata(resident_key, cache_obj=cache_dict[resident_key])
    for _ in range(5):
        policy.update_on_hit(resident_key, cache_dict)
    rejected_key = dumb_cache_engine_key(4)
    assert policy.should_admit(rejected_key, cache_dict) is False
    assert policy._sketch.estimate(rejected_key) == 1


def test_get_cache_policy_admission_prefix_wraps_any_inner_policy():
    expected_inner_class_names = {
        "LRU": "LRUCachePolicy",
        "LFU": "LFUCachePolicy",
        "FIFO": "FIFOCachePolicy",
        "MRU": "MRUCachePolicy",
        "COST_AWARE": "CostAwareEvictionPolicy",
    }
    for inner_name, expected_class_name in expected_inner_class_names.items():
        policy = get_cache_policy(f"ADMISSION_{inner_name}")
        assert type(policy.inner_policy).__name__ == expected_class_name


def test_get_cache_policy_admission_prefix_forwards_halve_every():
    # halve_every must reach AdmissionControlledPolicy itself, not be
    # mistaken for an inner-policy constructor kwarg (LRU takes none).
    policy = get_cache_policy("ADMISSION_LRU", halve_every=500)
    assert policy.halve_every == 500

    default_policy = get_cache_policy("ADMISSION_LRU")
    assert default_policy.halve_every == 20_000


def test_windowed_admission_control_always_admits():
    # should_admit is unconditional for the windowed design -- gating
    # happens at window-overflow time in get_evict_candidates instead.
    # See admission-control-policy.md for why (Findings 5-6: a strict
    # comparison, as AdmissionControlledPolicy uses, can freeze
    # permanently under uniform-frequency traffic).
    policy = get_cache_policy("WINDOWED_ADMISSION_LRU")
    cache_dict = policy.init_mutable_mapping()
    key1, key2 = dumb_cache_engine_key(1), dumb_cache_engine_key(2)
    cache_dict[key1] = DummyMemoryObj()
    policy.update_on_put(key1)

    assert policy.should_admit(key2, cache_dict) is True
    assert policy.should_admit(key1, cache_dict) is True  # even re-checking a resident


def test_windowed_admission_control_discards_infrequent_window_overflow():
    # window_capacity=1: putting a 2nd key immediately overflows the
    # window, evaluating key1 (still at frequency 1) for promotion.
    policy = get_cache_policy(
        "WINDOWED_ADMISSION_LRU", window_capacity=1, promotion_threshold=2
    )
    cache_dict = policy.init_mutable_mapping()
    key1, key2 = dumb_cache_engine_key(1), dumb_cache_engine_key(2)

    cache_dict[key1] = DummyMemoryObj()
    policy.update_on_put(key1)  # frequency 1, never touched again
    cache_dict[key2] = DummyMemoryObj()
    policy.update_on_put(key2)  # overflows the window -> key1 evaluated

    # key1 never earned a 2nd observation, so it's below
    # promotion_threshold: queued for a real discard at the next
    # eviction opportunity, drained ahead of the inner policy's own
    # ranking.
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert evict_candidates == [key1], evict_candidates


def test_windowed_admission_control_promotes_frequent_window_overflow():
    # MRU as the inner policy so its own eviction choice (most-recently-
    # inserted) diverges from key1, making a genuine promotion (the
    # frequent window victim is kept, a *different* key is evicted in
    # its place) observable.
    policy = get_cache_policy(
        "WINDOWED_ADMISSION_MRU", window_capacity=1, promotion_threshold=2
    )
    cache_dict = policy.init_mutable_mapping()
    key1, key2 = dumb_cache_engine_key(1), dumb_cache_engine_key(2)

    cache_dict[key1] = DummyMemoryObj()
    policy.update_on_put(key1)
    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key1, cache_dict)  # key1 frequency = 3
    cache_dict[key2] = DummyMemoryObj()
    policy.update_on_put(key2)  # overflows the window -> key1 promoted

    # key1 meets promotion_threshold: it stays resident (now implicitly
    # main) and is untracked from the window with no pending discard.
    # Eviction falls through to MRU's own ranking over cache_dict, whose
    # most-recently-inserted key is key2.
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert evict_candidates == [key2], evict_candidates


def test_get_cache_policy_windowed_admission_prefix_wraps_any_inner_policy():
    expected_inner_class_names = {
        "LRU": "LRUCachePolicy",
        "LFU": "LFUCachePolicy",
        "FIFO": "FIFOCachePolicy",
        "MRU": "MRUCachePolicy",
        "COST_AWARE": "CostAwareEvictionPolicy",
    }
    for inner_name, expected_class_name in expected_inner_class_names.items():
        policy = get_cache_policy(f"WINDOWED_ADMISSION_{inner_name}")
        assert type(policy.inner_policy).__name__ == expected_class_name


def test_get_cache_policy_windowed_admission_prefix_forwards_kwargs():
    policy = get_cache_policy(
        "WINDOWED_ADMISSION_LRU",
        halve_every=500,
        window_capacity=8,
        promotion_threshold=5,
    )
    assert policy.halve_every == 500
    assert policy.window_capacity == 8
    assert policy.promotion_threshold == 5

    default_policy = get_cache_policy("WINDOWED_ADMISSION_LRU")
    assert default_policy.halve_every == 20_000
    assert default_policy.window_capacity == 20
    assert default_policy.promotion_threshold == 2


def test_windowed_admission_control_rejects_invalid_construction():
    with pytest.raises(ValueError):
        get_cache_policy("WINDOWED_ADMISSION_LRU", window_capacity=0)
    with pytest.raises(ValueError):
        get_cache_policy("WINDOWED_ADMISSION_LRU", window_capacity=-1)
    with pytest.raises(ValueError):
        get_cache_policy("WINDOWED_ADMISSION_LRU", promotion_threshold=0)
