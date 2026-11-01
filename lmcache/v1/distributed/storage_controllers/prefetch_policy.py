# SPDX-License-Identifier: Apache-2.0
"""
Prefetch policy interface for grouped, multi-tier load planning.

A prefetch request is a grid of object keys: one row per key group (an object
group on one kv rank) and one column per chunk. After the lookup phase, the
controller knows which cells each L1 manager holds read-locked and which cells
each L2 adapter holds read-locked. The policy turns those two views into a
plan: which cells to serve from which L1 manager, and which cells to load from
which L2 adapter. Every locked cell the plan leaves out is released.

The policy also decides which of the cells loaded from L2 stay resident in L1
after the reader that requested them has finished.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    FetchingPolicy,
    GroupedObjectKeys,
    ObjectKey,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    L2AdapterDescriptor,
    MapState,
)

logger = init_logger(__name__)

# Helper functions


def _grid_matches_groups(grid: Bitmap2D, key_groups: list[GroupedObjectKeys]) -> bool:
    """Return whether ``grid`` has one row per key group, each as wide as
    that group's key list."""
    if not key_groups:
        return grid.size() == (0, 0)
    return grid.size() == (len(key_groups), len(key_groups[0].keys))


def _merged_or_zeros(state: MapState, key_groups: list[GroupedObjectKeys]) -> Bitmap2D:
    """Union a map across its indices, or return an all-zero grid for an empty map.

    Args:
        state: The map to merge.
        key_groups: The rows the result is shaped after when ``state`` is empty.

    Returns:
        A grid with one row per key group.
    """
    merged = state.merge()
    if len(merged) > 0:
        return merged
    return Bitmap2D.zeros(len(key_groups), len(key_groups[0].keys))


def _plan_tier(
    needed: Bitmap2D,
    locked_keys: MapState,
    indices: list[int],
) -> tuple[MapState, Bitmap2D]:
    """Assign still-needed cells to the given indices, first index wins.

    Args:
        needed: The cells not yet planned.
        locked_keys: Index -> grid of the cells locked there.
        indices: The indices to consider, in priority order.

    Returns:
        The per-index plan and the cells that remain unplanned afterwards.
        An index that contributes no cell is left out of the plan.
    """
    plan = MapState()
    remaining = needed
    for index in indices:
        if index not in locked_keys:
            continue
        local = remaining & locked_keys[index]
        if local.popcount() == 0:
            continue
        plan[index] = local
        remaining = remaining - local
    return plan, remaining


@dataclass(frozen=True)
class PrefetchPlan:
    """The outcome of planning one prefetch request.

    Both maps share the layout of the lookup results they were derived from:
    row ``i`` is key group ``i`` and column ``j`` is chunk ``j``.

    Note:
        A cell is planned **at most once** across every L1 manager and every L2
        adapter in the plan.
    """

    l1_planned_keys: MapState
    """L1 manager index -> per-row bitmaps of the cells served from that
    manager's resident objects."""

    l2_planned_keys: MapState
    """L2 adapter index -> per-row bitmaps of the cells loaded from that
    adapter into L1."""


class PrefetchPolicy(ABC):
    """Abstract interface for multi-tier prefetch planning for grouped object
    keys.

    The policy sees every tier's lookup result at once and decides, per cell,
    whether it is served from L1, loaded from L2, or not needed.

    The policy will also take the sliding window size and the prefetch mode into
    account.

    The functions should be **pure** without any side-effects.
    """

    @abstractmethod
    def plan_load(
        self,
        key_groups: list[GroupedObjectKeys],
        l1_locked_keys: MapState,
        l2_locked_keys: MapState,
        l1_manager_descs: dict[int, L1ManagerDescriptor],
        l2_adapter_descs: dict[int, L2AdapterDescriptor],
        fetching_policy: FetchingPolicy,
    ) -> PrefetchPlan:
        """Decide which cells are served from L1 and which are loaded from L2.

        Args:
            key_groups: The original object keys in the request, grouped by
                different (object_group_id, kv_rank) tuples.
            l1_locked_keys: L1 manager index -> per-group bitmaps indicating
                which keys are read-locked in L1 managers.
            l2_locked_keys: L2 adapter index -> per-row bitmaps indicating
                which keys are locked in the specific l2 adapter.
            l1_manager_descs: L1 manager index -> descriptor, for every L1
                manager that may appear in ``l1_locked_keys``.
            l2_adapter_descs: L2 adapter index -> descriptor, for every L2
                adapter that may appear in ``l2_locked_keys``.
            fetching_policy: ``"prefix"`` or ``"full"``.

        Returns:
            The plan consists of the planned L1 and L2 keys.
            No cell is planned from more than one tier or more than
            one manager or adapter.

        Note:
            This function does not raise any exceptions. Upon un-recoverable
            errors (e.g., invalid input argument), it will return an empty
            plan.
        """
        raise NotImplementedError

    def plan_l1_retention(
        self,
        keys: list[ObjectKey],
        l1_manager_desc: L1ManagerDescriptor,
        l2_adapter_desc: L2AdapterDescriptor,
    ) -> list[bool]:
        """Decide which keys loaded from L2 stay resident in L1 after L1-L0
        retrieve.

        A key that is not retained is temporary and is deleted from L1 once
        the L1-L0 transfer finishes on that key.

        Args:
            keys: The keys about to be written into the L1 manager from the
                L2 adapter, in the order they will be written.
            l1_manager_desc: Descriptor of the L1 manager receiving the keys.
            l2_adapter_desc: Descriptor of the L2 adapter the keys are loaded
                from.

        Returns:
            One flag per key, parallel to ``keys``. ``True`` retains the key
            in L1; ``False`` marks it temporary.

        Note:
            This function should not raise.
        """
        raise NotImplementedError


class DefaultPrefetchPolicy(PrefetchPolicy):
    """Serve every needed cell from the lowest-indexed tier that holds it.

    L1 managers are considered before L2 adapters, and within a tier lower
    indices win. Nothing loaded from L2 is retained in L1.
    """

    def plan_load(
        self,
        key_groups: list[GroupedObjectKeys],
        l1_locked_keys: MapState,
        l2_locked_keys: MapState,
        l1_manager_descs: dict[int, L1ManagerDescriptor],
        l2_adapter_descs: dict[int, L2AdapterDescriptor],
        fetching_policy: FetchingPolicy,
    ) -> PrefetchPlan:
        """Plan the needed cells, L1 first, then L2, lowest index first.

        For ``prefix``, it will consider the sliding window. For ``full``,
        it will require all sliding windows being negative (i.e., no sliding
        window), otherwise it will return an empty plan.

        Args:
            key_groups: The original object keys in the request, grouped by
                different (object_group_id, kv_rank) tuples.
            l1_locked_keys: L1 manager index -> per-group bitmaps indicating
                which keys are read-locked in L1 managers.
            l2_locked_keys: L2 adapter index -> per-row bitmaps indicating
                which keys are locked in the specific l2 adapter.
            l1_manager_descs: L1 manager index -> descriptor, for every L1
                manager that may appear in ``l1_locked_keys``.
            l2_adapter_descs: L2 adapter index -> descriptor, for every L2
                adapter that may appear in ``l2_locked_keys``.
            fetching_policy: ``"prefix"`` or ``"full"``.

        Returns:
            The plan. Every cell in it is locked in the tier it is planned
            from, and no cell appears twice.

        Note:
            This function does not raise. Invalid input yields an empty plan
            and an error log line.
        """
        empty = PrefetchPlan(MapState(), MapState())
        if not key_groups:
            logger.error("plan_load: request has no key groups")
            return empty
        sliding_windows = [group.sliding_window_size for group in key_groups]

        l1_found = _merged_or_zeros(l1_locked_keys, key_groups)
        l2_found = _merged_or_zeros(l2_locked_keys, key_groups)
        if not _grid_matches_groups(l1_found, key_groups) or not _grid_matches_groups(
            l2_found, key_groups
        ):
            logger.error("plan_load: locked-key layout does not match key groups")
            return empty
        found = l1_found + l2_found

        if fetching_policy == "full":
            if any(w > FULL_ATTENTION_WINDOW_CHUNKS for w in sliding_windows):
                logger.error(
                    "plan_load: 'full' fetching does not support sliding-window "
                    "key groups (windows=%s)",
                    sliding_windows,
                )
                return empty
            needed = found
        else:
            try:
                _hit_length, retain = fold_unfold_grouped(
                    found.to_list(), sliding_windows
                )
            except ValueError:
                logger.exception("plan_load: prefix fold failed")
                return empty
            needed = Bitmap2D(retain) & found

        l1_plan, needed = _plan_tier(needed, l1_locked_keys, sorted(l1_manager_descs))
        l2_plan, _needed = _plan_tier(needed, l2_locked_keys, sorted(l2_adapter_descs))
        return PrefetchPlan(l1_planned_keys=l1_plan, l2_planned_keys=l2_plan)

    def plan_l1_retention(
        self,
        keys: list[ObjectKey],
        l1_manager_desc: L1ManagerDescriptor,
        l2_adapter_desc: L2AdapterDescriptor,
    ) -> list[bool]:
        """Retain nothing: every loaded key is temporary.

        Args:
            keys: The keys about to be written into the L1 manager from the
                L2 adapter.
            l1_manager_desc: Descriptor of the L1 manager receiving the keys.
            l2_adapter_desc: Descriptor of the L2 adapter the keys are loaded
                from.

        Returns:
            ``False`` for every key.

        Note:
            This function does not raise.
        """
        return [False] * len(keys)


class RetainPrefetchPolicy(DefaultPrefetchPolicy):
    """Plan loads like the default policy but keep every loaded key in L1.

    Use this when prefetched data is likely to be reused by subsequent
    requests (e.g. shared system-prompt chunks).
    """

    def plan_l1_retention(
        self,
        keys: list[ObjectKey],
        l1_manager_desc: L1ManagerDescriptor,
        l2_adapter_desc: L2AdapterDescriptor,
    ) -> list[bool]:
        """Retain every loaded key.

        Args:
            keys: The keys about to be written into the L1 manager from the
                L2 adapter.
            l1_manager_desc: Descriptor of the L1 manager receiving the keys.
            l2_adapter_desc: Descriptor of the L2 adapter the keys are loaded
                from.

        Returns:
            ``True`` for every key.

        Note:
            This function does not raise.
        """
        return [True] * len(keys)


# -----------------------------------------------------------------------------
# Registry: prefetch policy name -> policy class
# -----------------------------------------------------------------------------

_PREFETCH_POLICY_REGISTRY: dict[str, type[PrefetchPolicy]] = {}


def register_prefetch_policy(
    name: str,
    policy_cls: type[PrefetchPolicy],
) -> None:
    """
    Register a prefetch policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete PrefetchPolicy subclass.

    Raises:
        ValueError: If a policy is already registered under ``name``.
    """
    if name in _PREFETCH_POLICY_REGISTRY:
        raise ValueError(f"Prefetch policy already registered: {name!r}")
    _PREFETCH_POLICY_REGISTRY[name] = policy_cls


def get_registered_prefetch_policies() -> list[str]:
    """Return the list of registered prefetch policy names."""
    return list(_PREFETCH_POLICY_REGISTRY)


def create_prefetch_policy(name: str) -> PrefetchPolicy:
    """
    Create a prefetch policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new PrefetchPolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _PREFETCH_POLICY_REGISTRY:
        known = ", ".join(sorted(_PREFETCH_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown prefetch policy {name!r}. Known: {known}")
    return _PREFETCH_POLICY_REGISTRY[name]()


register_prefetch_policy("default", DefaultPrefetchPolicy)
register_prefetch_policy("retain", RetainPrefetchPolicy)
