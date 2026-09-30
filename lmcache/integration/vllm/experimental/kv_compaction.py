# SPDX-License-Identifier: Apache-2.0
"""Plan paged-KV copies for in-place token-dropping compaction."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence


def _physical_kv_slot_id(
    kv_entry_index: int,
    request_block_ids: Sequence[int],
    physical_slots_per_block: int,
) -> int:
    """Return the physical slot for a KV-entry index.

    Args:
        kv_entry_index: Position in the current KV-entry sequence.
        request_block_ids: Request block IDs in sequence order from one
            paged-block address space.
        physical_slots_per_block: Physical KV slots per block in that same
            address space.

    Returns:
        Physical KV slot ID in that address space.
    """
    block_id = request_block_ids[kv_entry_index // physical_slots_per_block]
    return (
        block_id * physical_slots_per_block + kv_entry_index % physical_slots_per_block
    )


def plan_kv_compaction_moves(
    request_block_ids: Sequence[int],
    retained_kv_entry_indices: Sequence[int],
    physical_slots_per_block: int,
) -> list[tuple[int, int]]:
    """Plan the copies that pack a request's retained KV entries at the front.

    For example, with block IDs ``[4, 1]`` and two physical slots per
    block, KV-entry positions ``[0, 1, 2, 3]`` live in physical slots
    ``[8, 9, 2, 3]``. Retaining positions ``[1, 3]`` packs those entries
    into positions ``[0, 1]``, so this returns copies ``[(9, 8), (3, 9)]``.

    The planner sees only the capacity the supplied blocks represent,
    ``len(request_block_ids) * physical_slots_per_block``, and not the
    request's exact current KV-entry length when its final block is partially
    used. Callers must therefore pass indices of existing KV entries only.

    Args:
        request_block_ids: Request block IDs in sequence order from one
            paged-block address space. Blocks must be distinct.
        retained_kv_entry_indices: Strictly increasing positions in the current
            KV-entry sequence, not logical or original token positions.
        physical_slots_per_block: Positive number of physical KV slots per
            block in that same address space.

    Returns:
        ``(source_physical_slot, destination_physical_slot)`` copies, ordered
        so that applying them in sequence never overwrites a slot a later copy
        reads. Entries already in their compacted slot are omitted. The slot
        IDs are meaningful only in the supplied block-ID address space and
        physical-slot geometry.

    Raises:
        ValueError: If retained KV-entry indices are not strictly increasing or
            fall outside the capacity represented by the supplied blocks.
    """
    kv_entry_capacity = len(request_block_ids) * physical_slots_per_block
    physical_slot_moves: list[tuple[int, int]] = []
    previous_retained_kv_entry_index = -1
    for destination_kv_entry_index, retained_kv_entry_index in enumerate(
        retained_kv_entry_indices
    ):
        if not (
            previous_retained_kv_entry_index
            < retained_kv_entry_index
            < kv_entry_capacity
        ):
            raise ValueError(
                "retained_kv_entry_indices must be strictly increasing and "
                f"below {kv_entry_capacity}, got {retained_kv_entry_index} "
                f"after {previous_retained_kv_entry_index}"
            )
        previous_retained_kv_entry_index = retained_kv_entry_index
        source_physical_slot = _physical_kv_slot_id(
            retained_kv_entry_index, request_block_ids, physical_slots_per_block
        )
        destination_physical_slot = _physical_kv_slot_id(
            destination_kv_entry_index, request_block_ids, physical_slots_per_block
        )
        if source_physical_slot != destination_physical_slot:
            physical_slot_moves.append(
                (source_physical_slot, destination_physical_slot)
            )
    return physical_slot_moves
