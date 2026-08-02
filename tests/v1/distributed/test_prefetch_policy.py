# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for prefetch policy interface and DefaultPrefetchPolicy.

Tests are written against the PrefetchPolicy contract defined in
prefetch_policy.py.
"""

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
    RetainPrefetchPolicy,
    StripedPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StripedStorePolicy,
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


def make_bitmap(size: int, set_bits: list[int]) -> Bitmap:
    """Create a Bitmap with specific bits set."""
    bitmap = Bitmap(size)
    for i in set_bits:
        bitmap.set(i)
    return bitmap


def plan_to_indices(plan: dict[int, Bitmap]) -> dict[int, list[int]]:
    """Convert a Bitmap-based load plan to index lists for easy assertion."""
    return {
        adapter_idx: bitmap.get_indices_list() for adapter_idx, bitmap in plan.items()
    }


# =============================================================================
# DefaultPrefetchPolicy Tests
# =============================================================================


class TestDefaultPrefetchPolicy:
    """Test DefaultPrefetchPolicy.select_load_plan behavior."""

    def test_single_adapter_all_keys_found(self):
        """All found keys assigned to the single adapter."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0)]
        lookup_results = {0: make_bitmap(3, [0, 1, 2])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1, 2]}

    def test_single_adapter_partial_hits(self):
        """Only found keys are in the plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(4)]
        adapters = [make_descriptor(0)]
        # Only keys 0 and 2 found
        lookup_results = {0: make_bitmap(4, [0, 2])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 2]}

    def test_multi_adapter_overlap_first_wins(self):
        """When key is in multiple adapters, lowest-index adapter gets it."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        # Both adapters have key 1
        lookup_results = {
            0: make_bitmap(3, [0, 1]),
            1: make_bitmap(3, [1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # key 0 → adapter 0, key 1 → adapter 0 (first wins), key 2 → adapter 1
        assert plan_to_indices(result) == {0: [0, 1], 1: [2]}

    def test_multi_adapter_disjoint(self):
        """Each adapter gets its unique keys."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(4)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(4, [0, 1]),
            1: make_bitmap(4, [2, 3]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1], 1: [2, 3]}

    def test_no_hits_returns_empty(self):
        """Empty bitmaps → empty plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(3, []),
            1: make_bitmap(3, []),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {}

    def test_empty_adapters_returns_empty(self):
        """No adapters means no plan."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]

        result = policy.select_load_plan(keys, {}, [])

        assert plan_to_indices(result) == {}

    def test_empty_keys_returns_empty(self):
        """Empty keys list → empty plan."""
        policy = DefaultPrefetchPolicy()
        adapters = [make_descriptor(0)]
        lookup_results = {0: make_bitmap(0, [])}

        result = policy.select_load_plan([], lookup_results, adapters)

        assert plan_to_indices(result) == {}

    def test_adapter_order_matters(self):
        """Adapter with lower index always has priority, regardless of order
        in the adapters list."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]
        # Adapter 1 listed first, but adapter 0 should win
        adapters = [make_descriptor(1), make_descriptor(0)]
        lookup_results = {
            0: make_bitmap(1, [0]),
            1: make_bitmap(1, [0]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # Adapter 0 gets the key (lower index wins)
        assert plan_to_indices(result) == {0: [0]}

    def test_three_adapters_with_overlap(self):
        """Three adapters with partial overlaps."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]
        lookup_results = {
            0: make_bitmap(5, [0, 3]),  # has keys 0, 3
            1: make_bitmap(5, [1, 2, 3]),  # has keys 1, 2, 3
            2: make_bitmap(5, [2, 3, 4]),  # has keys 2, 3, 4
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # key 0 → adapter 0
        # key 1 → adapter 1
        # key 2 → adapter 1 (lower than adapter 2)
        # key 3 → adapter 0 (lowest that has it)
        # key 4 → adapter 2
        assert plan_to_indices(result) == {0: [0, 3], 1: [1, 2], 2: [4]}

    def test_missing_lookup_result_for_adapter(self):
        """Adapter with no lookup result is skipped."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(2)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        # Only adapter 1 has results
        lookup_results = {1: make_bitmap(2, [0, 1])}

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {1: [0, 1]}

    def test_each_key_appears_at_most_once(self):
        """No key should be assigned to multiple adapters."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]
        # All adapters have all keys
        lookup_results = {
            0: make_bitmap(3, [0, 1, 2]),
            1: make_bitmap(3, [0, 1, 2]),
            2: make_bitmap(3, [0, 1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # All keys should go to adapter 0 only
        assert plan_to_indices(result) == {0: [0, 1, 2]}
        assert 1 not in result
        assert 2 not in result


# =============================================================================
# DefaultPrefetchPolicy.select_l1_retentions Tests
# =============================================================================


class TestDefaultPrefetchPolicyRetentions:
    """Test DefaultPrefetchPolicy.select_l1_retentions."""

    def test_returns_all_false(self):
        """Default policy marks all keys as temporary."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        result = policy.select_l1_retentions(keys)
        assert result == [False] * 5

    def test_empty_keys(self):
        """Empty keys list returns empty list."""
        policy = DefaultPrefetchPolicy()
        result = policy.select_l1_retentions([])
        assert result == []

    def test_single_key(self):
        """Single key returns single False."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(0)]
        result = policy.select_l1_retentions(keys)
        assert result == [False]

    def test_length_matches_input(self):
        """Output length always matches input length."""
        policy = DefaultPrefetchPolicy()
        for n in [0, 1, 10, 100]:
            keys = [make_object_key(i) for i in range(n)]
            result = policy.select_l1_retentions(keys)
            assert len(result) == n


# =============================================================================
# RetainPrefetchPolicy Tests
# =============================================================================


class TestRetainPrefetchPolicyRetentions:
    """Test RetainPrefetchPolicy.select_l1_retentions."""

    def test_returns_all_true(self):
        """Retain policy marks all keys as permanent."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        result = policy.select_l1_retentions(keys)
        assert result == [True] * 5

    def test_empty_keys(self):
        """Empty keys list returns empty list."""
        policy = RetainPrefetchPolicy()
        result = policy.select_l1_retentions([])
        assert result == []

    def test_single_key(self):
        """Single key returns single True."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(0)]
        result = policy.select_l1_retentions(keys)
        assert result == [True]

    def test_length_matches_input(self):
        """Output length always matches input length."""
        policy = RetainPrefetchPolicy()
        for n in [0, 1, 10, 100]:
            keys = [make_object_key(i) for i in range(n)]
            result = policy.select_l1_retentions(keys)
            assert len(result) == n


class TestRetainPrefetchPolicyLoadPlan:
    """RetainPrefetchPolicy inherits load plan from Default."""

    def test_inherits_load_plan(self):
        """Load plan should behave identically to Default."""
        policy = RetainPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(3, [0, 1]),
            1: make_bitmap(3, [1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1], 1: [2]}


# =============================================================================
# DefaultPrefetchPolicy.select_lookup_targets Tests
# =============================================================================


class TestDefaultPrefetchPolicyLookupTargets:
    """DefaultPrefetchPolicy.select_lookup_targets returns None (all-to-all)."""

    def test_returns_none(self):
        """Default policy returns None (query all adapters for all keys)."""
        policy = DefaultPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        result = policy.select_lookup_targets(keys, adapters)
        assert result is None

    def test_returns_none_empty_keys(self):
        """None even with empty keys."""
        policy = DefaultPrefetchPolicy()
        result = policy.select_lookup_targets([], [make_descriptor(0)])
        assert result is None

    def test_returns_none_empty_adapters(self):
        """None even with empty adapters."""
        policy = DefaultPrefetchPolicy()
        result = policy.select_lookup_targets([make_object_key(0)], [])
        assert result is None


# =============================================================================
# StripedPrefetchPolicy.select_lookup_targets Tests
# =============================================================================


class TestStripedPrefetchPolicyLookupTargets:
    """Test StripedPrefetchPolicy.select_lookup_targets routing."""

    def test_returns_none_empty_adapters(self):
        """No adapters → None (controller falls back to all-to-all no-op)."""
        policy = StripedPrefetchPolicy()
        result = policy.select_lookup_targets([make_object_key(0)], [])
        assert result is None

    def test_each_key_in_exactly_one_adapter(self):
        """Every key index appears in exactly one adapter's list."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(100)]
        adapters = [make_descriptor(i) for i in range(4)]

        result = policy.select_lookup_targets(keys, adapters)
        assert result is not None

        # Collect all indices
        all_indices: list[int] = []
        for indices in result.values():
            all_indices.extend(indices)

        # Every index 0..99 appears exactly once
        assert sorted(all_indices) == list(range(100))

    def test_all_adapters_present(self):
        """Every adapter gets an entry, even if empty."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]

        result = policy.select_lookup_targets(keys, adapters)
        assert result is not None
        assert set(result.keys()) == {0, 1, 2}

    def test_single_adapter_all_keys(self):
        """With one adapter, all keys go to it."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(5)]
        adapters = [make_descriptor(0)]

        result = policy.select_lookup_targets(keys, adapters)
        assert result is not None
        assert result == {0: [0, 1, 2, 3, 4]}

    def test_empty_keys(self):
        """Empty keys → all adapters present with empty lists."""
        policy = StripedPrefetchPolicy()
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_lookup_targets([], adapters)
        assert result is not None
        assert result == {0: [], 1: []}

    def test_deterministic_routing(self):
        """Same keys + adapters → same routing every time."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(20)]
        adapters = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]

        result1 = policy.select_lookup_targets(keys, adapters)
        result2 = policy.select_lookup_targets(keys, adapters)
        assert result1 == result2

    def test_cross_process_stable(self):
        """Routing must be stable across processes (BLAKE3, not Python hash)."""
        import multiprocessing

        def compute_routing(result_dict, keys_args, adapters_args):
            keys = [
                ObjectKey(
                    chunk_hash=ObjectKey.IntHash2Bytes(i),
                    model_name="test_model",
                    kv_rank=0,
                )
                for i in keys_args
            ]
            adapters = [
                AdapterDescriptor(
                    index=i,
                    config=MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0),
                )
                for i in adapters_args
            ]
            policy = StripedPrefetchPolicy()
            result = policy.select_lookup_targets(keys, adapters)
            result_dict["result"] = result

        keys_ids = list(range(10))
        adapters_ids = [0, 1, 2, 3]

        # Compute in main process
        keys = [make_object_key(i) for i in keys_ids]
        adapters = [make_descriptor(i) for i in adapters_ids]
        policy = StripedPrefetchPolicy()
        main_result = policy.select_lookup_targets(keys, adapters)

        # Compute in child process
        mgr = multiprocessing.Manager()
        result_dict = mgr.dict()
        proc = multiprocessing.Process(
            target=compute_routing, args=(result_dict, keys_ids, adapters_ids)
        )
        proc.start()
        proc.join()
        child_result = result_dict.get("result")

        assert child_result is not None
        assert child_result == main_result

    def test_consistent_with_store_policy(self):
        """Lookup routing must match store routing: each key's lookup
        adapter == its store adapter."""
        policy_store = StripedStorePolicy()
        policy_prefetch = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(50)]
        adapters = [make_descriptor(i) for i in range(4)]

        # Where does each key get stored?
        store_targets = policy_store.select_store_targets(keys, adapters)

        # Where does each key get looked up?
        lookup_targets = policy_prefetch.select_lookup_targets(keys, adapters)
        assert lookup_targets is not None

        # For each key, the store adapter must match the lookup adapter
        for adapter_id, store_keys in store_targets.items():
            lookup_indices = lookup_targets.get(adapter_id, [])
            for key in store_keys:
                key_idx = keys.index(key)
                assert key_idx in lookup_indices, (
                    f"Key {key_idx} stored on adapter {adapter_id} "
                    f"but not in its lookup list"
                )

    def test_adapter_order_doesnt_matter(self):
        """Routing result is the same regardless of adapter list order."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(10)]
        adapters_sorted = [make_descriptor(0), make_descriptor(1), make_descriptor(2)]
        adapters_shuffled = [make_descriptor(2), make_descriptor(0), make_descriptor(1)]

        result1 = policy.select_lookup_targets(keys, adapters_sorted)
        result2 = policy.select_lookup_targets(keys, adapters_shuffled)
        assert result1 == result2

    def test_uniform_distribution(self):
        """Keys are distributed roughly evenly across adapters."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(1000)]
        adapters = [make_descriptor(i) for i in range(4)]

        result = policy.select_lookup_targets(keys, adapters)
        assert result is not None

        counts = {ad: len(indices) for ad, indices in result.items()}
        # With 1000 keys and 4 adapters, expect ~250 each ± 20%
        for ad, count in counts.items():
            assert 180 < count < 320, (
                f"Adapter {ad} got {count} keys, expected ~250"
            )


# =============================================================================
# StripedPrefetchPolicy.select_load_plan Tests (inherited from Default)
# =============================================================================


class TestStripedPrefetchPolicyLoadPlan:
    """StripedPrefetchPolicy inherits load plan from DefaultPrefetchPolicy."""

    def test_inherits_load_plan(self):
        """Load plan should behave identically to Default."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]
        lookup_results = {
            0: make_bitmap(3, [0, 1]),
            1: make_bitmap(3, [1, 2]),
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        assert plan_to_indices(result) == {0: [0, 1], 1: [2]}

    def test_striped_load_plan_with_targeted_lookup(self):
        """Under striped storage, each key is on exactly one adapter.
        The inherited load plan should assign each found key to its
        sole adapter."""
        policy = StripedPrefetchPolicy()
        keys = [make_object_key(i) for i in range(6)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        # Simulate targeted lookup results: each key only on one adapter
        lookup_results = {
            0: make_bitmap(6, [0, 2, 4]),  # adapter 0 has keys 0, 2, 4
            1: make_bitmap(6, [1, 3, 5]),  # adapter 1 has keys 1, 3, 5
        }

        result = policy.select_load_plan(keys, lookup_results, adapters)

        # Each key goes to its only adapter (no overlap to resolve)
        assert plan_to_indices(result) == {0: [0, 2, 4], 1: [1, 3, 5]}


# =============================================================================
# Registry Tests
# =============================================================================


class TestPrefetchPolicyRegistry:
    """Test that all policies are registered correctly."""

    def test_striped_registered(self):
        """StripedPrefetchPolicy is registered under 'striped'."""
        from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
            create_prefetch_policy,
            get_registered_prefetch_policies,
        )

        registered = get_registered_prefetch_policies()
        assert "striped" in registered

    def test_create_striped_policy(self):
        """create_prefetch_policy('striped') returns StripedPrefetchPolicy."""
        from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
            create_prefetch_policy,
        )

        policy = create_prefetch_policy("striped")
        assert isinstance(policy, StripedPrefetchPolicy)

    def test_striped_is_default_subclass(self):
        """StripedPrefetchPolicy inherits from DefaultPrefetchPolicy."""
        assert issubclass(StripedPrefetchPolicy, DefaultPrefetchPolicy)


# =============================================================================
# Targeted Lookup Integration Tests (simulate controller flow)
# =============================================================================


class TestTargetedLookupIntegration:
    """Simulate the full targeted-lookup flow that PrefetchController performs:
    select_lookup_targets → subset lookup → bitmap remap → select_load_plan.

    These tests verify the contract between StripedPrefetchPolicy and the
    controller's remapping logic without requiring a running controller
    (which needs CUDA / native_storage_ops).
    """

    def _simulate_targeted_lookup(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
        found_per_adapter: dict[int, set[int]],
    ) -> dict[int, Bitmap]:
        """Simulate the controller's targeted-lookup + remap flow.

        Args:
            keys: Full key list.
            adapters: Adapter descriptors.
            found_per_adapter: For each adapter index, the set of *global*
                key indices that are found on it (subset of what was routed
                to it).

        Returns:
            Global lookup_results dict (adapter_idx -> Bitmap over full
            key list), as the controller would produce after remapping.
        """
        policy = StripedPrefetchPolicy()
        routing = policy.select_lookup_targets(keys, adapters)
        assert routing is not None

        num_keys = len(keys)
        lookup_results: dict[int, Bitmap] = {}

        for adapter_id, global_indices in routing.items():
            if not global_indices:
                continue

            # Simulate adapter returning a subset bitmap (relative to the
            # subset_keys it received).  A bit j is set if the j-th key
            # in the subset was found.
            subset_size = len(global_indices)
            subset_bitmap = Bitmap(subset_size)
            found_set = found_per_adapter.get(adapter_id, set())
            for j, global_idx in enumerate(global_indices):
                if global_idx in found_set:
                    subset_bitmap.set(j)

            # Remap subset bitmap → global bitmap (controller logic)
            global_result = Bitmap(num_keys)
            for j in subset_bitmap.get_indices_list():
                global_result.set(global_indices[j])
            lookup_results[adapter_id] = global_result

        return lookup_results

    def test_all_found(self):
        """All keys found on their respective adapters."""
        keys = [make_object_key(i) for i in range(8)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        policy = StripedPrefetchPolicy()
        routing = policy.select_lookup_targets(keys, adapters)
        assert routing is not None

        # All routed keys are "found"
        found_per_adapter = {
            ad: set(indices) for ad, indices in routing.items()
        }

        lookup_results = self._simulate_targeted_lookup(
            keys, adapters, found_per_adapter
        )

        # Feed into load plan
        load_plan = policy.select_load_plan(keys, lookup_results, adapters)
        plan_indices = plan_to_indices(load_plan)

        # All 8 keys should be in the plan, each on its one adapter
        all_loaded = sorted(
            idx for indices in plan_indices.values() for idx in indices
        )
        assert all_loaded == list(range(8))

    def test_partial_found(self):
        """Some keys not found on their adapter (evicted / never stored)."""
        keys = [make_object_key(i) for i in range(8)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        policy = StripedPrefetchPolicy()
        routing = policy.select_lookup_targets(keys, adapters)
        assert routing is not None

        # Find only the first key on each adapter (total = number of
        # adapters with at least one routed key)
        found_per_adapter: dict[int, set[int]] = {}
        expected_found: set[int] = set()
        for ad, indices in routing.items():
            if indices:
                found_per_adapter[ad] = {indices[0]}
                expected_found.add(indices[0])

        lookup_results = self._simulate_targeted_lookup(
            keys, adapters, found_per_adapter
        )

        load_plan = policy.select_load_plan(keys, lookup_results, adapters)
        plan_indices = plan_to_indices(load_plan)

        all_loaded = sorted(
            idx for indices in plan_indices.values() for idx in indices
        )
        # Exactly the found keys should be loaded
        assert set(all_loaded) == expected_found

    def test_none_found(self):
        """No keys found → empty load plan."""
        keys = [make_object_key(i) for i in range(4)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        lookup_results = self._simulate_targeted_lookup(
            keys, adapters, {}
        )

        load_plan = StripedPrefetchPolicy().select_load_plan(
            keys, lookup_results, adapters
        )
        assert plan_to_indices(load_plan) == {}

    def test_remapping_preserves_global_indices(self):
        """The remapped global bitmap has bits at the correct global
        positions, not at subset-local positions."""
        keys = [make_object_key(i) for i in range(6)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        policy = StripedPrefetchPolicy()
        routing = policy.select_lookup_targets(keys, adapters)
        assert routing is not None

        # Find only the first routed key on each adapter
        found_per_adapter: dict[int, set[int]] = {}
        for ad, indices in routing.items():
            if indices:
                found_per_adapter[ad] = {indices[0]}

        lookup_results = self._simulate_targeted_lookup(
            keys, adapters, found_per_adapter
        )

        # The found keys should be at their global positions
        expected_found = set()
        for ad, indices in routing.items():
            if indices:
                expected_found.add(indices[0])

        merged = Bitmap(len(keys))
        for bm in lookup_results.values():
            merged |= bm

        assert set(merged.get_indices_list()) == expected_found

    def test_single_adapter_all_keys(self):
        """Single adapter gets all keys."""
        keys = [make_object_key(i) for i in range(5)]
        adapters = [make_descriptor(0)]

        found_per_adapter = {0: {0, 1, 2, 3, 4}}

        lookup_results = self._simulate_targeted_lookup(
            keys, adapters, found_per_adapter
        )

        load_plan = StripedPrefetchPolicy().select_load_plan(
            keys, lookup_results, adapters
        )
        assert plan_to_indices(load_plan) == {0: [0, 1, 2, 3, 4]}

    def test_empty_keys_targeted(self):
        """Empty key list → empty routing, empty results."""
        keys: list[ObjectKey] = []
        adapters = [make_descriptor(0), make_descriptor(1)]

        policy = StripedPrefetchPolicy()
        routing = policy.select_lookup_targets(keys, adapters)
        assert routing is not None
        assert all(not v for v in routing.values())

        load_plan = policy.select_load_plan(keys, {}, adapters)
        assert plan_to_indices(load_plan) == {}
