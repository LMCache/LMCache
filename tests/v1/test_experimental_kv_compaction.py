# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the experimental paged-KV compaction planner."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.kv_compaction import (
    plan_kv_compaction_moves,
)


@pytest.mark.parametrize(
    (
        "request_block_ids",
        "retained_kv_entry_indices",
        "physical_slots_per_block",
        "expected_physical_slot_moves",
    ),
    [
        pytest.param([5], [0, 1], 4, [], id="retained_prefix_needs_no_copies"),
        pytest.param([5], [], 4, [], id="nothing_retained"),
        pytest.param([5], [1, 3], 4, [(21, 20), (23, 21)], id="within_one_block"),
        pytest.param([2, 7], [5, 6], 4, [(29, 8), (30, 9)], id="across_blocks"),
        pytest.param(
            [9, 3, 6],
            [1, 3, 5],
            2,
            [(19, 18), (7, 19), (13, 6)],
            id="non_contiguous_blocks",
        ),
        pytest.param([0, 1], [0, 1, 5, 7], 4, [(5, 2), (7, 3)], id="prefix_then_gaps"),
    ],
)
def test_maps_retained_kv_entry_indices_to_physical_copies(
    request_block_ids: list[int],
    retained_kv_entry_indices: list[int],
    physical_slots_per_block: int,
    expected_physical_slot_moves: list[tuple[int, int]],
) -> None:
    assert (
        plan_kv_compaction_moves(
            request_block_ids=request_block_ids,
            retained_kv_entry_indices=retained_kv_entry_indices,
            physical_slots_per_block=physical_slots_per_block,
        )
        == expected_physical_slot_moves
    )


def test_copies_applied_in_order_compact_the_sequence_in_place() -> None:
    """Applying the copies in order packs the retained entries at the front.

    One destination is an earlier copy's source, so reordering the copies
    would read an already-overwritten slot.
    """
    request_block_ids = [7, 2, 5]
    physical_slots_per_block = 4
    # Slot of each KV-entry index under those blocks: 0-3 -> block 7, 4-7 ->
    # block 2, 8-11 -> block 5.
    physical_slot_by_kv_entry_index = [28, 29, 30, 31, 8, 9, 10, 11, 20, 21, 22, 23]
    retained_kv_entry_indices = [1, 4, 7, 9, 11]
    kv_entry_by_physical_slot = {
        slot: f"entry{index}"
        for index, slot in enumerate(physical_slot_by_kv_entry_index)
    }

    for source_physical_slot, destination_physical_slot in plan_kv_compaction_moves(
        request_block_ids=request_block_ids,
        retained_kv_entry_indices=retained_kv_entry_indices,
        physical_slots_per_block=physical_slots_per_block,
    ):
        kv_entry_by_physical_slot[destination_physical_slot] = (
            kv_entry_by_physical_slot[source_physical_slot]
        )

    compacted = [
        kv_entry_by_physical_slot[slot]
        for slot in physical_slot_by_kv_entry_index[: len(retained_kv_entry_indices)]
    ]
    assert compacted == [f"entry{index}" for index in retained_kv_entry_indices]


@pytest.mark.parametrize(
    "retained_kv_entry_indices",
    [
        pytest.param([1, 1], id="duplicate"),
        pytest.param([2, 1], id="decreasing"),
        pytest.param([-1], id="negative"),
        pytest.param([4], id="past_the_supplied_blocks"),
    ],
)
def test_rejects_retained_kv_entry_indices_it_cannot_compact(
    retained_kv_entry_indices: list[int],
) -> None:
    """Indices it cannot pack are a caller bug; fail instead of corrupting KV."""
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_kv_compaction_moves(
            request_block_ids=[5],
            retained_kv_entry_indices=retained_kv_entry_indices,
            physical_slots_per_block=4,
        )
