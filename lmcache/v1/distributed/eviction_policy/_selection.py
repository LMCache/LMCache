# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for selecting coherent distributed-cache eviction victims."""

# Standard
from collections.abc import Callable, Collection, Iterable

# First Party
from lmcache.v1.distributed.api import ObjectKey

KVTopology = tuple[int, int] | None
TopologyNamespace = tuple[str, KVTopology]


def _kv_topology(kv_rank: int) -> KVTopology:
    """Decode the stable world/local-world topology from a packed KV rank."""
    world_size = (kv_rank >> 24) & 0xFF
    global_rank = (kv_rank >> 16) & 0xFF
    local_world_size = (kv_rank >> 8) & 0xFF
    local_rank = kv_rank & 0xFF
    if (
        world_size > 0
        and local_world_size > 0
        and global_rank < world_size
        and local_rank < local_world_size
        and global_rank % local_world_size == local_rank
    ):
        return world_size, local_world_size
    # Tests and legacy callers may use an unpacked integer rank. Preserve
    # their observed-coordinate behavior rather than guessing a topology.
    return None


def _topology_namespace(key: ObjectKey) -> TopologyNamespace:
    """Return the cache-model and parallel-layout namespace for ``key``."""
    return key.model_name, _kv_topology(key.kv_rank)


def _logical_chunk_family(
    key: ObjectKey,
) -> tuple[bytes, str, str, KVTopology]:
    """Return the identity shared by every rank/group object for one chunk."""
    return (
        key.chunk_hash,
        key.model_name,
        key.cache_salt,
        _kv_topology(key.kv_rank),
    )


class ChunkFamilyTopology:
    """Track the observed rank/group coordinates of logical chunk families.

    Eviction policies already serialize key lifecycle notifications and victim
    selection under their own lock. Recording only the small topology keeps
    selection proportional to the number of victim families without either
    rebuilding or retaining a second full-cache map.
    """

    def __init__(self) -> None:
        self._coordinates: dict[TopologyNamespace, set[tuple[int, int]]] = {}
        self._ordered_coordinates: dict[
            TopologyNamespace,
            tuple[tuple[int, int], ...],
        ] = {}

    def observe(self, keys: Iterable[ObjectKey]) -> None:
        """Record rank/group coordinates present for each cache namespace."""
        changed_namespaces = set()
        for key in keys:
            namespace = _topology_namespace(key)
            coordinates = self._coordinates.setdefault(namespace, set())
            previous_count = len(coordinates)
            coordinates.add((key.kv_rank, key.object_group_id))
            if len(coordinates) != previous_count:
                changed_namespaces.add(namespace)
        for namespace in changed_namespaces:
            observed = self._coordinates[namespace]
            topology = namespace[1]
            if topology is None:
                expected = observed
            else:
                world_size, local_world_size = topology
                group_ids = {object_group_id for _, object_group_id in observed}
                expected = {
                    (
                        ObjectKey.ComputeKVRank(
                            world_size=world_size,
                            global_rank=global_rank,
                            local_world_size=local_world_size,
                            local_rank=global_rank % local_world_size,
                        ),
                        object_group_id,
                    )
                    for global_rank in range(world_size)
                    for object_group_id in group_ids
                }
            self._ordered_coordinates[namespace] = tuple(sorted(expected))

    def members(
        self,
        key: ObjectKey,
        tracked_keys: Collection[ObjectKey],
    ) -> tuple[ObjectKey, ...]:
        """Return a complete family, or empty while any sibling is absent."""
        coordinates = self._ordered_coordinates.get(_topology_namespace(key), ())
        if coordinates == ((key.kv_rank, key.object_group_id),):
            return (key,)

        members = []
        for kv_rank, object_group_id in coordinates:
            candidate = ObjectKey(
                chunk_hash=key.chunk_hash,
                model_name=key.model_name,
                kv_rank=kv_rank,
                object_group_id=object_group_id,
                cache_salt=key.cache_salt,
            )
            if candidate not in tracked_keys:
                # Stores complete asynchronously across rank/group batches.
                # Evicting the currently visible subset lets late siblings
                # publish a permanently incomplete family after this pass.
                return ()
            members.append(candidate)
        return tuple(members)


def select_chunk_coherent_victims(
    ordered_keys: Collection[ObjectKey],
    target_count: int,
    family_topology: ChunkFamilyTopology,
    key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
) -> list[ObjectKey]:
    """Select LRU victims without splitting a logical chunk family.

    A distributed or hybrid chunk is persisted as multiple ``ObjectKey``
    instances that differ only by ``kv_rank`` and ``object_group_id``. Deleting
    an arbitrary subset makes a later lookup appear promising while the load
    cannot reconstruct any tokens. Once the LRU boundary selects one member,
    return every tracked member of that logical family.

    When an eligibility filter is present, a family is selected only if every
    member is eligible. This prevents eviction from splitting a family around
    a read/write-locked object. The returned count may exceed ``target_count``
    by at most one family; eviction ratios are approximate by contract.

    Args:
        ordered_keys: Keys in least-to-most-recently-used order.
        target_count: Approximate number of object keys to select.
        family_topology: Rank/group coordinates observed by the calling
            eviction policy.
        key_eligible_filter: Optional per-key eligibility predicate.

    Returns:
        Selected keys in family/LRU order, or an empty list when no complete
        eligible family can be selected. A family remains ineligible while an
        expected rank/group sibling has not completed its store.
    """
    if target_count <= 0:
        return []

    selected: list[ObjectKey] = []
    visited: set[tuple[bytes, str, str, KVTopology]] = set()
    for key in ordered_keys:
        family_id = _logical_chunk_family(key)
        if family_id in visited:
            continue
        visited.add(family_id)
        members = family_topology.members(key, ordered_keys)
        if not members:
            continue
        if key_eligible_filter is not None and any(
            not key_eligible_filter(member) for member in members
        ):
            continue
        selected.extend(members)
        if len(selected) >= target_count:
            break
    return selected
