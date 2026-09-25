# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the grouped prefetch policies.

A request is a grid: one row per key group, one column per chunk. The tests
build L1 and L2 lookup results as ``MapState`` maps, run the policy, and check
the plan only through the public map interface: which indices appear, which
cells each index holds, and the guarantees the docstrings make (every planned
cell is locked in its tier, no cell is planned twice, invalid input yields an
empty plan, and the policy never mutates its inputs).
"""

# Standard
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    FetchingPolicy,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.prefetch_policy_v2 import (
    DefaultPrefetchPolicy,
    PrefetchPlan,
    PrefetchPolicy,
    RetainPrefetchPolicy,
    create_prefetch_policy,
    get_registered_prefetch_policies,
    register_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    L2AdapterDescriptor,
    MapState,
)

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2, 3])], dtypes=[torch.float16])
FULL = FULL_ATTENTION_WINDOW_CHUNKS


# =============================================================================
# Builders
# =============================================================================


def _group(num_keys: int, gid: int = 0, window: int = FULL) -> GroupedObjectKeys:
    """Build one key group of ``num_keys`` chunk-ordered keys."""
    keys = [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="m",
            kv_rank=0,
            object_group_id=gid,
        )
        for i in range(num_keys)
    ]
    return GroupedObjectKeys(
        keys=keys,
        object_group_id=gid,
        layout_desc=LAYOUT,
        sliding_window_size=window,
    )


def _row(size: int, *indices: int) -> Bitmap:
    """Build a bitmap of ``size`` bits with ``indices`` set."""
    bitmap = Bitmap(size)
    for index in indices:
        bitmap.set(index)
    return bitmap


def _bits(bitmap: Bitmap) -> list[int]:
    """Return the set bit indices of ``bitmap``."""
    return bitmap.get_indices_list()


def _rows(state: MapState, index: int) -> list[list[int]]:
    """Return the set bit indices of every row stored under ``index``."""
    return [_bits(bitmap) for bitmap in state[index]]


def _l1_descs(*indices: int) -> dict[int, L1ManagerDescriptor]:
    """Build L1 manager descriptors for ``indices``."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False)
    )
    return {i: L1ManagerDescriptor(index=i, config=config) for i in indices}


def _l2_descs(*indices: int) -> dict[int, L2AdapterDescriptor]:
    """Build L2 adapter descriptors for ``indices``."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return {i: L2AdapterDescriptor(index=i, config=config) for i in indices}


def _state(entries: dict[int, list[Bitmap]]) -> MapState:
    """Build a map from index to rows."""
    state = MapState()
    for index, rows in entries.items():
        state[index] = Bitmap2D(rows)
    return state


def _snapshot(state: MapState, indices: list[int]) -> dict[int, list[list[int]]]:
    """Capture the bits stored under each present index for later comparison."""
    return {i: _rows(state, i) for i in indices if i in state}


def _plan(
    policy: PrefetchPolicy,
    key_groups: list[GroupedObjectKeys],
    l1: MapState,
    l2: MapState,
    fetching_policy: FetchingPolicy = "prefix",
    l1_indices: tuple[int, ...] = (0,),
    l2_indices: tuple[int, ...] = (0,),
) -> PrefetchPlan:
    """Run ``plan_load`` with windows taken from the key groups."""
    return policy.plan_load(
        key_groups,
        l1,
        l2,
        _l1_descs(*l1_indices),
        _l2_descs(*l2_indices),
        fetching_policy,
    )


def _is_empty(plan: PrefetchPlan) -> bool:
    """Return whether the plan has no index in either map."""
    return plan.l1_planned_keys.merge().size() == (
        0,
        0,
    ) and plan.l2_planned_keys.merge().size() == (0, 0)


def _assert_plan_contract(
    plan: PrefetchPlan,
    l1: MapState,
    l2: MapState,
    l1_indices: tuple[int, ...],
    l2_indices: tuple[int, ...],
) -> None:
    """Check the two guarantees every plan makes.

    Every planned cell is locked in the tier and index it is planned from,
    and no cell is planned under more than one index across both tiers.
    """
    seen: list[set[tuple[int, int]]] = []
    for planned, locked, indices in (
        (plan.l1_planned_keys, l1, l1_indices),
        (plan.l2_planned_keys, l2, l2_indices),
    ):
        for index in indices:
            if index not in planned:
                continue
            assert index in locked, f"index {index} planned but never locked"
            cells: set[tuple[int, int]] = set()
            for row_id, (p_row, l_row) in enumerate(
                zip(planned[index], locked[index], strict=True)
            ):
                assert set(_bits(p_row)) <= set(_bits(l_row)), (
                    f"index {index} row {row_id} plans unlocked cells"
                )
                cells |= {(row_id, col) for col in _bits(p_row)}
            for other in seen:
                assert not (cells & other), "a cell is planned twice"
            seen.append(cells)


# =============================================================================
# Default policy: prefix fetching
# =============================================================================


class TestDefaultPrefix:
    def test_l1_hit_is_served_from_l1_not_l2(self) -> None:
        """A cell held by both tiers is planned from L1 only."""
        groups = [_group(4)]
        l1 = _state({0: [_row(4, 0, 1)]})
        l2 = _state({0: [_row(4, 0, 1, 2, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l1_planned_keys, 0) == [[0, 1]]
        assert _rows(plan.l2_planned_keys, 0) == [[2, 3]]

    def test_lower_l2_index_wins(self) -> None:
        """Within L2 a cell held by two adapters goes to the lower index."""
        groups = [_group(4)]
        l1 = MapState()
        l2 = _state({0: [_row(4, 0, 1)], 1: [_row(4, 0, 1, 2, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, l2_indices=(0, 1))

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1]]
        assert _rows(plan.l2_planned_keys, 1) == [[2, 3]]

    def test_lower_l1_index_wins(self) -> None:
        """Within L1 a cell held by two managers goes to the lower index."""
        groups = [_group(3)]
        l1 = _state({0: [_row(3, 0)], 1: [_row(3, 0, 1, 2)]})
        l2 = MapState()

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, l1_indices=(0, 1))

        assert _rows(plan.l1_planned_keys, 0) == [[0]]
        assert _rows(plan.l1_planned_keys, 1) == [[1, 2]]

    def test_index_order_not_insertion_order(self) -> None:
        """Priority follows the numeric index even when a higher index was
        stored first."""
        groups = [_group(2)]
        l1 = MapState()
        l2 = _state({3: [_row(2, 0, 1)], 1: [_row(2, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, l2_indices=(1, 3))

        assert _rows(plan.l2_planned_keys, 1) == [[0, 1]]
        assert 3 not in plan.l2_planned_keys

    def test_prefix_stops_at_first_gap(self) -> None:
        """A found chunk past a gap is dropped even though it is locked."""
        groups = [_group(5)]
        l1 = MapState()
        l2 = _state({0: [_row(5, 0, 1, 3, 4)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1]]

    def test_prefix_is_bounded_by_the_shortest_row(self) -> None:
        """Every full-attention row must have a chunk for it to be in the prefix."""
        groups = [_group(4, gid=0), _group(4, gid=1)]
        l1 = MapState()
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1], [0, 1]]

    def test_prefix_across_tiers(self) -> None:
        """The prefix is computed over the union of L1 and L2 hits."""
        groups = [_group(4)]
        l1 = _state({0: [_row(4, 1)]})
        l2 = _state({0: [_row(4, 0, 2)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l1_planned_keys, 0) == [[1]]
        assert _rows(plan.l2_planned_keys, 0) == [[0, 2]]

    @pytest.mark.parametrize(
        ("window", "expected_windowed_row"),
        [(1, [3]), (2, [2, 3]), (4, [0, 1, 2, 3])],
    )
    def test_windowed_row_keeps_only_trailing_window(
        self, window: int, expected_windowed_row: list[int]
    ) -> None:
        """A sliding-window row keeps the last ``window`` chunks of the prefix;
        a full-attention row keeps the whole prefix."""
        groups = [_group(4, gid=0), _group(4, gid=1, window=window)]
        l1 = MapState()
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 0, 1, 2, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1, 2, 3], expected_windowed_row]

    def test_windowed_row_only_needs_its_window_present(self) -> None:
        """A windowed row missing chunks before its window still serves the
        full prefix."""
        groups = [_group(4, gid=0), _group(4, gid=1, window=1)]
        l1 = MapState()
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1, 2, 3], [3]]

    def test_windowed_row_missing_its_window_shortens_prefix(self) -> None:
        """A windowed row without the chunk at the end of the prefix bounds
        the prefix to the last chunk it does have."""
        groups = [_group(4, gid=0), _group(4, gid=1, window=1)]
        l1 = MapState()
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1], [1]]

    def test_dropped_cells_appear_nowhere(self) -> None:
        """A locked cell outside the prefix is not planned under any index."""
        groups = [_group(4)]
        l1 = _state({0: [_row(4, 3)]})
        l2 = _state({0: [_row(4, 0, 3)], 1: [_row(4, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, l2_indices=(0, 1))

        assert 0 not in plan.l1_planned_keys
        assert _rows(plan.l2_planned_keys, 0) == [[0]]
        assert 1 not in plan.l2_planned_keys

    def test_nothing_found_gives_empty_plan(self) -> None:
        """With no hits in either tier the plan has no index at all."""
        groups = [_group(3)]
        l1 = _state({0: [_row(3)]})
        l2 = _state({0: [_row(3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)

        assert _is_empty(plan)

    def test_no_l1_managers(self) -> None:
        """Without any L1 manager everything needed is loaded from L2."""
        groups = [_group(3)]
        l2 = _state({0: [_row(3, 0, 1, 2)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, MapState(), l2, l1_indices=())

        assert _is_empty(PrefetchPlan(plan.l1_planned_keys, MapState()))
        assert _rows(plan.l2_planned_keys, 0) == [[0, 1, 2]]

    def test_no_l2_adapters(self) -> None:
        """Without any L2 adapter the plan is the L1 prefix alone."""
        groups = [_group(3)]
        l1 = _state({0: [_row(3, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, MapState(), l2_indices=())

        assert _rows(plan.l1_planned_keys, 0) == [[0, 1]]
        assert plan.l2_planned_keys.merge().size() == (0, 0)


# =============================================================================
# Default policy: full fetching
# =============================================================================


class TestDefaultFull:
    def test_full_keeps_gaps(self) -> None:
        """Every found cell is planned, gaps included."""
        groups = [_group(6)]
        l1 = MapState()
        l2 = _state({0: [_row(6, 0, 2, 5)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, "full")

        assert _rows(plan.l2_planned_keys, 0) == [[0, 2, 5]]

    def test_full_rows_are_independent(self) -> None:
        """One row's misses do not remove another row's hits."""
        groups = [_group(3, gid=0), _group(3, gid=1)]
        l1 = MapState()
        l2 = _state({0: [_row(3, 0, 1, 2), _row(3, 2)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, "full")

        assert _rows(plan.l2_planned_keys, 0) == [[0, 1, 2], [2]]

    def test_full_l1_first_then_lowest_l2(self) -> None:
        """Tier and index priority apply under full fetching too."""
        groups = [_group(4)]
        l1 = _state({0: [_row(4, 1)]})
        l2 = _state({0: [_row(4, 1, 3)], 1: [_row(4, 0, 1, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, "full", l2_indices=(0, 1))

        assert _rows(plan.l1_planned_keys, 0) == [[1]]
        assert _rows(plan.l2_planned_keys, 0) == [[3]]
        assert _rows(plan.l2_planned_keys, 1) == [[0]]

    def test_full_with_sliding_window_row_gives_empty_plan(self) -> None:
        """Full fetching refuses a request with any sliding-window row."""
        groups = [_group(4, gid=0), _group(4, gid=1, window=2)]
        l1 = _state({0: [_row(4, 0), _row(4, 0)]})
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 0, 1, 2, 3)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2, "full")

        assert _is_empty(plan)


# =============================================================================
# Default policy: contract and invalid input
# =============================================================================


class TestDefaultContract:
    @pytest.mark.parametrize("fetching_policy", ["prefix", "full"])
    def test_plan_obeys_locking_and_uniqueness(
        self, fetching_policy: FetchingPolicy
    ) -> None:
        """Every planned cell is locked where it is planned and no cell is
        planned twice, on an overlapping multi-tier layout."""
        groups = [_group(6, gid=0), _group(6, gid=1)]
        l1 = _state(
            {
                0: [_row(6, 0, 1), _row(6, 0)],
                1: [_row(6, 1, 2), _row(6, 0, 1)],
            }
        )
        l2 = _state(
            {
                0: [_row(6, 0, 1, 2, 3, 5), _row(6, 1, 3)],
                1: [_row(6, 2, 3, 4, 5), _row(6, 2, 3, 4, 5)],
            }
        )

        plan = _plan(
            DefaultPrefetchPolicy(),
            groups,
            l1,
            l2,
            fetching_policy,
            l1_indices=(0, 1),
            l2_indices=(0, 1),
        )

        _assert_plan_contract(plan, l1, l2, (0, 1), (0, 1))

    def test_inputs_are_not_mutated(self) -> None:
        """Planning leaves both locked maps exactly as they were."""
        groups = [_group(4, gid=0), _group(4, gid=1)]
        l1 = _state({0: [_row(4, 0, 1), _row(4, 0)]})
        l2 = _state(
            {0: [_row(4, 0, 1, 2, 3), _row(4, 1, 2)], 1: [_row(4, 3), _row(4, 3)]}
        )
        before_l1 = _snapshot(l1, [0])
        before_l2 = _snapshot(l2, [0, 1])

        _plan(DefaultPrefetchPolicy(), groups, l1, l2, l2_indices=(0, 1))

        assert _snapshot(l1, [0]) == before_l1
        assert _snapshot(l2, [0, 1]) == before_l2

    def test_plan_does_not_alias_inputs(self) -> None:
        """Changing a planned row afterwards does not change the locked map."""
        groups = [_group(3)]
        l1 = MapState()
        l2 = _state({0: [_row(3, 0, 1, 2)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)
        plan.l2_planned_keys[0][0].clear(0)

        assert _rows(l2, 0) == [[0, 1, 2]]

    def test_empty_key_groups_give_empty_plan(self) -> None:
        """A request with no rows plans nothing."""
        plan = DefaultPrefetchPolicy().plan_load(
            [], MapState(), MapState(), _l1_descs(), _l2_descs(), "prefix"
        )

        assert _is_empty(plan)

    def test_row_width_mismatch_gives_empty_plan(self) -> None:
        """Locked rows narrower than the key groups are invalid input."""
        groups = [_group(6)]
        l2 = _state({0: [_row(4, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, MapState(), l2)

        assert _is_empty(plan)

    def test_row_count_mismatch_gives_empty_plan(self) -> None:
        """Locked maps with a different number of rows than the key groups are
        invalid input."""
        groups = [_group(4, gid=0), _group(4, gid=1)]
        l2 = _state({0: [_row(4, 0, 1)]})

        plan = _plan(DefaultPrefetchPolicy(), groups, MapState(), l2)

        assert _is_empty(plan)


# =============================================================================
# Retention
# =============================================================================


class TestRetention:
    def test_default_retains_nothing(self) -> None:
        """The default policy marks every loading key temporary."""
        keys = _group(4).keys

        retained = DefaultPrefetchPolicy().plan_l1_retention(
            keys, _l1_descs(0)[0], _l2_descs(0)[0]
        )

        assert retained == [False] * 4

    def test_retain_keeps_every_loading_key(self) -> None:
        """The retain policy marks every loading key retained."""
        keys = _group(4).keys

        retained = RetainPrefetchPolicy().plan_l1_retention(
            keys, _l1_descs(0)[0], _l2_descs(0)[0]
        )

        assert retained == [True] * 4

    @pytest.mark.parametrize(
        "policy_cls", [DefaultPrefetchPolicy, RetainPrefetchPolicy]
    )
    def test_result_is_parallel_to_keys(self, policy_cls: type[PrefetchPolicy]) -> None:
        """One flag comes back per key, including for keys spanning groups."""
        keys = _group(3, gid=0).keys + _group(2, gid=1).keys

        retained = policy_cls().plan_l1_retention(
            keys, _l1_descs(0)[0], _l2_descs(0)[0]
        )

        assert len(retained) == len(keys)
        assert all(isinstance(flag, bool) for flag in retained)

    @pytest.mark.parametrize(
        "policy_cls", [DefaultPrefetchPolicy, RetainPrefetchPolicy]
    )
    def test_no_keys_gives_no_flags(self, policy_cls: type[PrefetchPolicy]) -> None:
        """An empty key list yields an empty flag list without raising."""
        assert (
            policy_cls().plan_l1_retention([], _l1_descs(0)[0], _l2_descs(0)[0]) == []
        )

    @pytest.mark.parametrize(
        "policy_cls", [DefaultPrefetchPolicy, RetainPrefetchPolicy]
    )
    def test_decision_does_not_depend_on_the_pair(
        self, policy_cls: type[PrefetchPolicy]
    ) -> None:
        """Both built-in policies give the same answer for any manager/adapter pair."""
        keys = _group(3).keys
        policy = policy_cls()

        first = policy.plan_l1_retention(keys, _l1_descs(0)[0], _l2_descs(0)[0])
        second = policy.plan_l1_retention(keys, _l1_descs(2)[2], _l2_descs(5)[5])

        assert first == second

    def test_retain_plans_loads_like_default(self) -> None:
        """Retain differs from default only in retention, not in planning."""
        groups = [_group(4, gid=0), _group(4, gid=1, window=1)]
        l1 = _state({0: [_row(4, 0), _row(4, 0)]})
        l2 = _state({0: [_row(4, 0, 1, 2, 3), _row(4, 0, 1, 2, 3)]})

        default_plan = _plan(DefaultPrefetchPolicy(), groups, l1, l2)
        retain_plan = _plan(RetainPrefetchPolicy(), groups, l1, l2)

        assert _snapshot(retain_plan.l1_planned_keys, [0]) == _snapshot(
            default_plan.l1_planned_keys, [0]
        )
        assert _snapshot(retain_plan.l2_planned_keys, [0]) == _snapshot(
            default_plan.l2_planned_keys, [0]
        )


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    def test_builtin_policies_are_registered(self) -> None:
        """Both shipped policies can be found by name."""
        names = get_registered_prefetch_policies()

        assert "default" in names
        assert "retain" in names

    def test_create_returns_matching_policy(self) -> None:
        """Creating by name yields an instance of the registered class."""
        assert isinstance(create_prefetch_policy("default"), DefaultPrefetchPolicy)
        assert isinstance(create_prefetch_policy("retain"), RetainPrefetchPolicy)
        assert isinstance(create_prefetch_policy("retain"), PrefetchPolicy)

    def test_create_returns_new_instances(self) -> None:
        """Each creation is a fresh object."""
        assert create_prefetch_policy("default") is not create_prefetch_policy(
            "default"
        )

    def test_create_unknown_name_raises(self) -> None:
        """An unregistered name is rejected."""
        with pytest.raises(ValueError):
            create_prefetch_policy(f"missing-{uuid.uuid4().hex}")

    def test_register_and_create_custom_policy(self) -> None:
        """A newly registered policy is listed and creatable."""
        name = f"custom-{uuid.uuid4().hex}"

        class CustomPolicy(RetainPrefetchPolicy):
            pass

        register_prefetch_policy(name, CustomPolicy)

        assert name in get_registered_prefetch_policies()
        assert isinstance(create_prefetch_policy(name), CustomPolicy)

    def test_register_duplicate_name_raises(self) -> None:
        """Registering the same name twice is rejected."""
        name = f"dup-{uuid.uuid4().hex}"
        register_prefetch_policy(name, DefaultPrefetchPolicy)

        with pytest.raises(ValueError):
            register_prefetch_policy(name, RetainPrefetchPolicy)
