# SPDX-License-Identifier: Apache-2.0
"""Paged-KV slot moves that pack a request's retained KV entries at the front.

Token dropping decides which entries of a request's KV sequence survive; this
module turns that decision into the physical slot copies a compaction has to
perform inside the blocks the request already owns.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from itertools import pairwise
from typing import Sequence


def _kv_slot_id(
    kv_sequence_index: int,
    request_block_ids: Sequence[int],
    tokens_per_block: int,
) -> int:
    """Return the physical KV slot holding a KV-sequence entry.

    Args:
        kv_sequence_index: Index into the request's current KV sequence.
        request_block_ids: Paged-KV blocks owned by the request, in order.
        tokens_per_block: KV entries one block holds.

    Returns:
        The physical slot id of that entry.
    """
    block_id = request_block_ids[kv_sequence_index // tokens_per_block]
    return block_id * tokens_per_block + kv_sequence_index % tokens_per_block


@dataclass(frozen=True)
class KVCompactionPlan:
    """Slot copies that compact one request's KV sequence.

    Sources and destinations are parallel and ordered by destination; applying
    them in that order never overwrites a slot that a later move reads. Entries
    already sitting in their compacted slot are not listed.
    """

    move_source_kv_slot_ids: tuple[int, ...]
    move_destination_kv_slot_ids: tuple[int, ...]
    compacted_kv_sequence_length: int


def build_kv_compaction_plan(
    request_block_ids: Sequence[int],
    retained_kv_sequence_indices: Sequence[int],
    current_kv_sequence_length: int,
    tokens_per_block: int,
) -> KVCompactionPlan:
    """Plan the slot copies that compact a request's KV sequence.

    Args:
        request_block_ids: Paged-KV blocks owned by the request, in order.
        retained_kv_sequence_indices: Strictly increasing indices into the
            current KV sequence, not original logical token positions.
        current_kv_sequence_length: KV entries the request currently holds.
        tokens_per_block: KV entries one block holds.

    Returns:
        The copies needed to pack the retained entries at the front of the
        request's existing blocks, omitting entries already in place.

    Raises:
        AssertionError: If the caller's arguments break an invariant.
    """
    assert tokens_per_block > 0, "tokens_per_block must be positive."
    assert current_kv_sequence_length > 0, (
        "current_kv_sequence_length must be positive."
    )
    assert len(request_block_ids) * tokens_per_block >= current_kv_sequence_length, (
        "request_block_ids must cover current_kv_sequence_length."
    )
    assert len(retained_kv_sequence_indices) > 0, (
        "retained_kv_sequence_indices must not be empty."
    )
    assert all(
        earlier < later for earlier, later in pairwise(retained_kv_sequence_indices)
    ), "retained_kv_sequence_indices must be strictly increasing."
    assert (
        0 <= retained_kv_sequence_indices[0]
        and retained_kv_sequence_indices[-1] < current_kv_sequence_length
    ), "retained_kv_sequence_indices must lie inside the current KV sequence."

    move_source_kv_slot_ids: list[int] = []
    move_destination_kv_slot_ids: list[int] = []
    for compacted_index, retained_index in enumerate(retained_kv_sequence_indices):
        source_kv_slot_id = _kv_slot_id(
            retained_index, request_block_ids, tokens_per_block
        )
        destination_kv_slot_id = _kv_slot_id(
            compacted_index, request_block_ids, tokens_per_block
        )
        if source_kv_slot_id != destination_kv_slot_id:
            move_source_kv_slot_ids.append(source_kv_slot_id)
            move_destination_kv_slot_ids.append(destination_kv_slot_id)

    return KVCompactionPlan(
        move_source_kv_slot_ids=tuple(move_source_kv_slot_ids),
        move_destination_kv_slot_ids=tuple(move_destination_kv_slot_ids),
        compacted_kv_sequence_length=len(retained_kv_sequence_indices),
    )
