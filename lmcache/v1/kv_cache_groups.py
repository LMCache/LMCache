# SPDX-License-Identifier: Apache-2.0
"""LMCache-owned KV cache group metadata.

Engine integrations convert their native KV cache groups into these neutral
LMCache types before sending metadata over multiprocess IPC. They are
``msgspec.Struct`` types, so the multiprocess message queue encodes and decodes
them directly (no separate JSON step) — see
:mod:`lmcache.v1.multiprocess.mq`.

A *hybrid block group* is one distinct paged-block address space: block IDs are
only meaningful within a single group, and layers from different groups must
never be merged into one LMCache KV group. They correspond to the serving
engine's KV cache groups (e.g. vLLM's hybrid KV cache manager), but that mapping
is resolved at the engine boundary — from LMCache's side these are just neutral
block groups. This file is the intended home for engine-derived fields LMCache
may need to normalize in the future, such as sliding-window windows, Mamba state
groups, or logical vs. physical block-size details.

Do not put vLLM/SGLang-specific object access here. Engine-specific conversion
belongs in the corresponding ``lmcache.integration.<engine>`` package.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable, Sequence

# Third Party
import msgspec


class LMCacheKVGroup(msgspec.Struct, frozen=True):
    """One LMCache hybrid block group: layers sharing one block-id space.

    A ``msgspec.Struct`` so it can be encoded/decoded directly as part of a
    multiprocess IPC payload.
    """

    hybrid_block_group_id: int
    """Hybrid block group ID; selects which request block-id list applies."""

    layer_indices: tuple[int, ...] = ()
    """Registered KV tensor indices assigned to this hybrid block group."""


class LMCacheKVSpec(msgspec.Struct, frozen=True):
    """LMCache's neutral KV cache group spec (a tuple of hybrid block groups).

    A ``msgspec.Struct`` carried verbatim in the ``REGISTER_KV_CACHE`` IPC
    payload; the message queue handles encoding/decoding.
    """

    groups: tuple[LMCacheKVGroup, ...] = ()

    @classmethod
    def from_groups(cls, groups: Iterable[LMCacheKVGroup]) -> "LMCacheKVSpec":
        """Build a validated spec from individual groups.

        Args:
            groups: Iterable of :class:`LMCacheKVGroup`. Order is preserved and
                becomes the protocol-visible LMCache group order.

        Returns:
            An :class:`LMCacheKVSpec` wrapping the given groups as a tuple.

        Raises:
            ValueError: If any group has a negative ``hybrid_block_group_id``.
        """
        groups_tuple = tuple(groups)
        bad_ids = [
            group.hybrid_block_group_id
            for group in groups_tuple
            if group.hybrid_block_group_id < 0
        ]
        if bad_ids:
            raise ValueError(f"hybrid_block_group_id must be non-negative: {bad_ids}")
        return cls(groups_tuple)

    @property
    def num_hybrid_block_groups(self) -> int:
        """Number of hybrid block groups (block-id lists per transfer request).

        Returns:
            ``max(hybrid_block_group_id) + 1`` (ids are dense from 0), or ``1``
            for an empty spec (single-group fallback).
        """
        if not self.groups:
            return 1
        return max(group.hybrid_block_group_id for group in self.groups) + 1

    @property
    def num_lmc_kv_cache_groups(self) -> int:
        """Number of LMCache KV layer groups visible to transfer requests.

        Returns:
            The number of LMCache KV groups, or ``1`` for an empty spec
            (single-group fallback).
        """
        if not self.groups:
            return 1
        return len(self.groups)

    def hybrid_block_group_ids_by_lmc_group(self) -> tuple[int, ...]:
        """Return the hybrid block group ID for each LMCache group.

        Returns:
            A tuple indexed by LMCache group order whose ``i``-th element is the
            hybrid block group id that LMCache group ``i`` draws block IDs from.
            ``(0,)`` for an empty spec (single-group fallback).
        """
        if not self.groups:
            return (0,)
        return tuple(group.hybrid_block_group_id for group in self.groups)

    def expand_block_ids_to_lmc_groups(
        self,
        block_ids_per_hybrid_block_group: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Expand hybrid-block-group block IDs to LMCache-group block IDs.

        Block IDs arrive indexed by hybrid block group (the serving engine
        reports them per KV cache group). LMCache transfer requests are indexed
        by LMCache KV group, so each LMCache group reuses the block IDs from its
        source hybrid block group.

        Args:
            block_ids_per_hybrid_block_group: Block IDs indexed by hybrid block
                group id (element ``g`` is group ``g``'s block list). May be
                empty when nothing is allocated yet.

        Returns:
            Block IDs re-indexed by LMCache group order: one inner list per
            LMCache group, copied from that group's source hybrid block group.
            When the input is empty, returns one empty list per LMCache group.

        Raises:
            ValueError: If a group references a hybrid block group id beyond the
                supplied ``block_ids_per_hybrid_block_group``.
        """
        if not block_ids_per_hybrid_block_group:
            return [[] for _ in self.hybrid_block_group_ids_by_lmc_group()]

        block_ids_per_lmc_group: list[list[int]] = []
        for hybrid_block_group_id in self.hybrid_block_group_ids_by_lmc_group():
            if hybrid_block_group_id >= len(block_ids_per_hybrid_block_group):
                raise ValueError(
                    "Missing block IDs for hybrid block group "
                    f"{hybrid_block_group_id}; got "
                    f"{len(block_ids_per_hybrid_block_group)} groups"
                )
            block_ids_per_lmc_group.append(
                list(block_ids_per_hybrid_block_group[hybrid_block_group_id])
            )
        return block_ids_per_lmc_group

    def get_per_layer_hybrid_block_group_indices(
        self,
        num_registered_layers: int,
    ) -> list[int] | None:
        """Return the hybrid block group index for each registered KV tensor.

        Args:
            num_registered_layers: Number of KV tensors registered with the
                server, i.e. the length of the per-layer mapping to produce.

        Returns:
            A list of length ``num_registered_layers`` mapping each registered
            tensor index to its hybrid block group id, or ``None`` when there is
            no group metadata (no groups, zero layers, or a single non-hybrid
            group) so callers fall back to single-group behavior.

        Raises:
            ValueError: If a group references a layer index outside
                ``[0, num_registered_layers)``, if the groups cover only some
                registered layers, or if no layer mapping is available but
                multiple hybrid block groups exist (ambiguous HMA mapping).
        """
        if not self.groups or num_registered_layers == 0:
            return None

        per_layer_hybrid_block_group_idx = [0] * num_registered_layers
        matched_indices: set[int] = set()

        for group in self.groups:
            for layer_idx in group.layer_indices:
                if layer_idx < 0 or layer_idx >= num_registered_layers:
                    raise ValueError(
                        f"Layer index {layer_idx} is outside registered layer "
                        f"range [0, {num_registered_layers})"
                    )
                per_layer_hybrid_block_group_idx[layer_idx] = (
                    group.hybrid_block_group_id
                )
                matched_indices.add(layer_idx)

        if matched_indices:
            missing_indices = set(range(num_registered_layers)) - matched_indices
            if missing_indices:
                raise ValueError(
                    "Hybrid block groups did not cover registered KV "
                    f"cache layer indices: {sorted(missing_indices)[:8]}"
                )
            return per_layer_hybrid_block_group_idx

        if self.num_hybrid_block_groups > 1:
            raise ValueError(
                "Unable to map registered KV cache tensors to hybrid block "
                "groups for HMA."
            )

        return None
