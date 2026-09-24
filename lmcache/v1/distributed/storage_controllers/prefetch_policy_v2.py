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
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    FetchingPolicy,
    GroupedObjectKeys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
    MapState,
)

logger = init_logger(__name__)

# Helper functions


def _intersect_rows(left: list[Bitmap], right: list[Bitmap]) -> list[Bitmap]:
    """Return the row-wise intersection of two bitmap lists.

    Args:
        left: One bitmap per row.
        right: One bitmap per row, parallel to ``left`` with equal widths.

    Returns:
        New bitmaps, one per row, with a bit set where both inputs set it.

    Raises:
        ValueError: If the two lists differ in length.
    """
    return [a & b for a, b in zip(left, right, strict=True)]


def _subtract_rows(left: list[Bitmap], right: list[Bitmap]) -> list[Bitmap]:
    """Return the row-wise difference of two bitmap lists.

    Args:
        left: One bitmap per row.
        right: One bitmap per row, parallel to ``left`` with equal widths.

    Returns:
        New bitmaps, one per row, with the bits of ``left`` that are not set
        in ``right``.

    Raises:
        ValueError: If the two lists differ in length.
    """
    return [a & ~b for a, b in zip(left, right, strict=True)]


def _union_rows(left: list[Bitmap], right: list[Bitmap]) -> list[Bitmap]:
    """Return the row-wise union of two bitmap lists.

    Args:
        left: One bitmap per row.
        right: One bitmap per row, parallel to ``left`` with equal widths.

    Returns:
        New bitmaps, one per row, with a bit set where either input sets it.

    Raises:
        ValueError: If the two lists differ in length.
    """
    return [a | b for a, b in zip(left, right, strict=True)]


def _empty_rows(rows: list[Bitmap]) -> list[Bitmap]:
    """Return all-zero bitmaps with the same widths as ``rows``."""
    return [Bitmap(bitmap.size()) for bitmap in rows]


def _has_any_bit(rows: list[Bitmap]) -> bool:
    """Return whether any bitmap in ``rows`` has a set bit."""
    return any(bitmap.popcount() > 0 for bitmap in rows)


def _rows_match_groups(rows: list[Bitmap], key_groups: list[GroupedObjectKeys]) -> bool:
    """Return whether ``rows`` has one bitmap per key group, each as wide as
    that group's key list."""
    if len(rows) != len(key_groups):
        return False
    return all(
        bitmap.size() == len(group.keys)
        for bitmap, group in zip(rows, key_groups, strict=True)
    )


def _merged_or_empty(
    state: MapState, key_groups: list[GroupedObjectKeys]
) -> list[Bitmap]:
    """Union a map across its indices, or return all-zero rows for an empty map.

    Args:
        state: The map to merge.
        key_groups: The rows the result is shaped after when ``state`` is empty.

    Returns:
        One bitmap per key group.
    """
    merged = state.merge()
    if merged:
        return merged
    return [Bitmap(len(group.keys)) for group in key_groups]


def _plan_tier(
    needed: list[Bitmap],
    locked_keys: MapState,
    indices: list[int],
) -> tuple[MapState, list[Bitmap]]:
    """Assign still-needed cells to the given indices, first index wins.

    Args:
        needed: One bitmap per row of the cells not yet planned.
        locked_keys: Index -> per-row bitmaps of the cells locked there.
        indices: The adapter/manager indices to consider, in priority order.

    Returns:
        The per-index plan and the cells that remain unplanned afterwards.
        An index that contributes no cell is left out of the plan.
    """
    plan = MapState()
    remaining = needed
    for index in indices:
        if index not in locked_keys:
            continue
        local = _intersect_rows(remaining, locked_keys[index])
        if not _has_any_bit(local):
            continue
        plan[index] = local
        remaining = _subtract_rows(remaining, local)
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
        sliding_windows: list[int],
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
            sliding_windows: sliding window lengths for each key group. Has
                the same length as the ``key_groups`` list.
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
        key_groups: list[GroupedObjectKeys],
        loading_keys: list[Bitmap],
    ) -> list[Bitmap]:
        """Decide which keys loaded from L2 stay resident in L1 after L1-L0
        retrieve.

        A key that is not retained is temporary and is deleted from L1 once
        the L1-L0 transfer finishes on that key.

        Args:
            key_groups: The original object keys in the request, grouped by
                different (object_group_id, kv_rank) tuples.
            loading_keys: One (global) bitmap per group marking the keys about
                to be written into L1 from L2. Should have the same length
                as ``key_groups``.

        Returns:
            Global bitmaps indicating which keys to retain in L1 (for each group).
            The length of the returned list is the same as ``loading_keys``.

            The returned bitmap is guaranteed to be a subset of the input bitmap.

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
        sliding_windows: list[int],
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
            sliding_windows: sliding window lengths for each key group. Has
                the same length as the ``key_groups`` list.
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
        if len(sliding_windows) != len(key_groups):
            logger.error(
                "plan_load: %d sliding windows for %d key groups",
                len(sliding_windows),
                len(key_groups),
            )
            return empty

        l1_found = _merged_or_empty(l1_locked_keys, key_groups)
        l2_found = _merged_or_empty(l2_locked_keys, key_groups)
        if not _rows_match_groups(l1_found, key_groups) or not _rows_match_groups(
            l2_found, key_groups
        ):
            logger.error("plan_load: locked-key layout does not match key groups")
            return empty
        found = _union_rows(l1_found, l2_found)

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
                _hit_length, retain = fold_unfold_grouped(found, sliding_windows)
            except ValueError:
                logger.exception("plan_load: prefix fold failed")
                return empty
            needed = _intersect_rows(retain, found)

        l1_plan, needed = _plan_tier(needed, l1_locked_keys, sorted(l1_manager_descs))
        l2_plan, _needed = _plan_tier(needed, l2_locked_keys, sorted(l2_adapter_descs))
        return PrefetchPlan(l1_planned_keys=l1_plan, l2_planned_keys=l2_plan)

    def plan_l1_retention(
        self,
        key_groups: list[GroupedObjectKeys],
        loading_keys: list[Bitmap],
    ) -> list[Bitmap]:
        """Retain nothing: every loaded key is temporary.

        Args:
            key_groups: The original object keys in the request, grouped by
                different (object_group_id, kv_rank) tuples.
            loading_keys: One (global) bitmap per group marking the keys about
                to be written into L1 from L2.

        Returns:
            All-zero bitmaps with the same layout as ``loading_keys``.

        Note:
            This function does not raise.
        """
        return _empty_rows(loading_keys)


class RetainPrefetchPolicy(DefaultPrefetchPolicy):
    """Plan loads like the default policy but keep every loaded key in L1.

    Use this when prefetched data is likely to be reused by subsequent
    requests (e.g. shared system-prompt chunks).
    """

    def plan_l1_retention(
        self,
        key_groups: list[GroupedObjectKeys],
        loading_keys: list[Bitmap],
    ) -> list[Bitmap]:
        """Retain every loaded key.

        Args:
            key_groups: The original object keys in the request, grouped by
                different (object_group_id, kv_rank) tuples.
            loading_keys: One (global) bitmap per group marking the keys about
                to be written into L1 from L2.

        Returns:
            Copies of ``loading_keys``.

        Note:
            This function does not raise. A layout that does not match
            ``key_groups`` is logged and retains nothing.
        """
        if not _rows_match_groups(loading_keys, key_groups):
            logger.error(
                "plan_l1_retention: loading-key layout does not match key groups"
            )
            return _empty_rows(loading_keys)
        return [b.copy() for b in loading_keys]


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
