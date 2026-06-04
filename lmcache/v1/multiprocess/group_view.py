# SPDX-License-Identifier: Apache-2.0
"""LMCache's engine-neutral view of a serving engine's KV cache groups.

An *engine group* is one distinct paged-block address space exposed by the
serving engine (e.g. one of vLLM's hybrid KV cache groups): block IDs are only
meaningful within a single group, and layers from different groups must never be
merged into one LMCache KV group. Engine group ids are assumed dense and
consecutive starting from 0.

LMCache's neutral KV cache spec is simply a ``list[LMCacheGroupView]`` (passed as
a ``Sequence[LMCacheGroupView]`` where only order matters). The group order is
the protocol-visible LMCache group order used by store/retrieve block IDs. An
empty list means a single non-hybrid group (the default for engines that do not
report KV cache group metadata). Engine-specific conversion belongs in the
corresponding ``lmcache.integration.<engine>`` package, not here.
"""

# Standard
from collections.abc import Mapping, Sequence

# Third Party
import msgspec


class LMCacheGroupView(msgspec.Struct, frozen=True):
    """One LMCache KV group: layers of one engine group that share a copy kernel.

    Carries the layer indices and which engine group they belong to. Several
    ``LMCacheGroupView`` instances may share the same ``engine_group_id`` when
    one engine group is split by physical transfer identity (e.g. differing
    hidden dims). A ``list[LMCacheGroupView]`` is carried verbatim in the
    ``REGISTER_KV_CACHE`` IPC payload; the message queue handles
    encoding/decoding.
    """

    engine_group_id: int
    """Engine group this view's layers live in (one distinct paged-block address
    space). Selects which request block-id list applies. Dense from 0."""

    layer_indices: tuple[int, ...] = ()
    """Registered KV tensor indices assigned to this group."""


def num_engine_groups(groups: Sequence[LMCacheGroupView]) -> int:
    """Return the number of engine groups (block-id lists per transfer request).

    Engine group ids are assumed dense and consecutive from 0.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``max(engine_group_id) + 1``, or ``1`` for an empty ``groups`` (single
        non-hybrid group).
    """
    if not groups:
        return 1
    return max(group.engine_group_id for group in groups) + 1


def num_group_views(groups: Sequence[LMCacheGroupView]) -> int:
    """Return the number of LMCache KV groups visible to transfer requests.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``len(groups)``, or ``1`` for an empty ``groups`` (single non-hybrid
        group).
    """
    if not groups:
        return 1
    return len(groups)


def _engine_group_id_per_view(
    groups: Sequence[LMCacheGroupView],
) -> tuple[int, ...]:
    """Return, per LMCache group, the engine group it draws block IDs from.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        A tuple whose length equals the number of LMCache groups (i.e.
        :func:`num_group_views`); element ``i`` is the engine group id
        that LMCache group ``i`` reads block IDs from. ``(0,)`` for an empty
        ``groups`` (single non-hybrid group).
    """
    if not groups:
        return (0,)
    return tuple(group.engine_group_id for group in groups)


def expand_block_ids_to_views(
    groups: Sequence[LMCacheGroupView],
    engine_side_block_ids: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Re-index engine-side block IDs to one list per LMCache group.

    The serving engine reports block IDs per engine group. LMCache transfer
    requests are indexed by LMCache KV group, so each LMCache group reuses the
    block IDs from its source engine group.

    Args:
        groups: The LMCache KV groups, in protocol order.
        engine_side_block_ids: Block IDs indexed by engine group id, i.e. one
            inner ``list[int]`` per engine group (element ``g`` is engine group
            ``g``'s block list).

    Returns:
        Block IDs re-indexed by LMCache group order: one inner list per LMCache
        group, copied from that group's source engine group.
    """
    return [
        list(engine_side_block_ids[engine_group_id])
        for engine_group_id in _engine_group_id_per_view(groups)
    ]


def slice_block_ids_per_group(
    allocated_block_ids: Mapping[int, Sequence[int]],
    group_block_sizes: Sequence[int],
    canonical_block_size: int,
    start: int,
    end: int,
) -> list[list[int]]:
    """Slice each engine group's block IDs for a canonical block range.

    Block accounting (``start``/``end``, hit counts, chunk sizes) is expressed
    in the *canonical* block size -- the GCD of all groups' block sizes (the
    granularity at which the engine hashes blocks). Each group's block IDs,
    however, are counted in that group's own block size, which can be a larger
    multiple of the canonical size for hybrid models (e.g.
    ``google/gemma-4-E4B-it``: sliding-window groups block_size=32,
    full-attention groups block_size=16, canonical=16). A canonical block index
    maps to a per-group index by dividing by
    ``k_g = group_block_sizes[g] // canonical_block_size``; the full group has
    twice as many block IDs per chunk as the sliding group.

    Args:
        allocated_block_ids: Block IDs keyed by engine group id; a missing group
            is treated as an empty list.
        group_block_sizes: Per-engine-group block size, in engine-group order;
            the result has one inner list per group. Each must be a positive
            multiple of ``canonical_block_size`` (callers validate this once).
        canonical_block_size: The GCD block size in which ``start``/``end`` are
            expressed.
        start: Start **canonical block** index (inclusive), not a token index.
        end: End **canonical block** index (exclusive), not a token index.

    Returns:
        One ``list[int]`` of block IDs per engine group. Group ``g`` is sliced
        to ``[start // k_g, end // k_g)`` so it spans the same token range
        expressed in that group's own block size.

    Raises:
        ValueError: If ``start``/``end`` do not align to a group's per-group
            block boundary.
    """
    sliced: list[list[int]] = []
    for engine_group_idx, block_size in enumerate(group_block_sizes):
        k = block_size // canonical_block_size
        if start % k != 0 or end % k != 0:
            raise ValueError(
                f"canonical block range [{start}, {end}) does not align to "
                f"group {engine_group_idx} block factor {k}"
            )
        group_block_ids = allocated_block_ids.get(engine_group_idx, [])
        sliced.append(list(group_block_ids[start // k : end // k]))
    return sliced


def get_engine_group_indices(
    groups: Sequence[LMCacheGroupView],
    num_registered_layers: int,
) -> list[int] | None:
    """Return the engine group index for each registered KV tensor.

    Args:
        groups: The LMCache KV groups, in protocol order.
        num_registered_layers: Number of KV tensors registered with the server,
            i.e. the length of the per-layer mapping to produce.

    Returns:
        A list of length ``num_registered_layers`` mapping each registered
        tensor index to its engine group id, or ``None`` when there is no group
        metadata (empty ``groups`` or zero layers) so callers fall back to
        single-group behavior. Registered tensors not referenced by any group
        are marked with ``EXCLUDED_ENGINE_GROUP`` (cross-layer KV-sharing layers
        whose KV lives in their target owner's blocks); downstream grouping
        skips them.

    Raises:
        ValueError: If a group references a layer index outside
            ``[0, num_registered_layers)``.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    if not groups or num_registered_layers == 0:
        return None

    # Default to "excluded": layers no group references are intentionally left
    # out of grouping (e.g. KV-sharing layers aliasing a target owner's cache).
    per_layer_engine_group_idx = [EXCLUDED_ENGINE_GROUP] * num_registered_layers

    for group in groups:
        for layer_idx in group.layer_indices:
            if layer_idx < 0 or layer_idx >= num_registered_layers:
                raise ValueError(
                    f"Layer index {layer_idx} is outside registered layer "
                    f"range [0, {num_registered_layers})"
                )
            per_layer_engine_group_idx[layer_idx] = group.engine_group_id

    return per_layer_engine_group_idx
