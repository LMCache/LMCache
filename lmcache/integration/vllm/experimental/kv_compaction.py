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
    """Map a request-local KV-entry index to its physical slot."""
    block_id = request_block_ids[kv_entry_index // physical_slots_per_block]
    return (
        block_id * physical_slots_per_block + kv_entry_index % physical_slots_per_block
    )


def plan_kv_compaction_moves(
    request_block_ids: Sequence[int],
    retained_kv_entry_indices: Sequence[int],
    physical_slots_per_block: int,
) -> list[tuple[int, int]]:
    """Plan the physical copies that move retained KV entries to the front.

    For example::

        Current KV: [A B C D]
        Keep (0-based): [1, 3]
        Result:         [B D _ _]

    The planner knows block capacity, not the request's exact live KV length.
    Callers must pass only indices of existing KV entries.

    Args:
        request_block_ids: Request block IDs in sequence order. IDs must be
            distinct and belong to the same paged-KV address space.
        retained_kv_entry_indices: Strictly increasing positions in the current
            KV sequence, not logical or original token positions.
        physical_slots_per_block: Positive number of physical KV slots per block.

    Returns:
        ``(source_physical_slot, destination_physical_slot)`` copies in safe
        in-place order. Copies already in place are omitted.

    Raises:
        ValueError: If retained indices are not strictly increasing or exceed
            the supplied block capacity.
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
