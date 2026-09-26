# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the token-dropping KV compaction planner.

Every expected plan below is written out by hand from the request's block
ids: slot of KV-sequence index ``i`` is
``request_block_ids[i // tokens_per_block] * tokens_per_block + i % tokens_per_block``.
"""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.token_dropping.compaction_plan import (
    build_kv_compaction_plan,
)


@pytest.mark.parametrize(
    (
        "request_block_ids",
        "retained_kv_sequence_indices",
        "current_kv_sequence_length",
        "tokens_per_block",
        "expected_move_source_kv_slot_ids",
        "expected_move_destination_kv_slot_ids",
        "expected_compacted_kv_sequence_length",
    ),
    [
        pytest.param([5], [0, 1], 4, 4, (), (), 2, id="retained_already_in_place"),
        pytest.param([5], [1, 3], 4, 4, (21, 23), (20, 21), 2, id="sparse_retained"),
        pytest.param([2, 7], [5, 6], 8, 4, (29, 30), (8, 9), 2, id="crosses_blocks"),
        pytest.param(
            [9, 3, 6], [1, 3, 5], 6, 2, (19, 7, 13), (18, 19, 6), 3, id="blocks_apart"
        ),
        pytest.param([1, 4], [2, 5], 6, 4, (6, 17), (4, 5), 2, id="partial_last_block"),
        pytest.param([3, 8], [0, 1, 2, 3], 4, 2, (), (), 4, id="keep_all"),
        pytest.param([0, 1], [0, 1, 5, 7], 8, 4, (5, 7), (2, 3), 4, id="mixed"),
    ],
)
def test_plans_the_moves_that_pack_retained_entries(
    request_block_ids: list[int],
    retained_kv_sequence_indices: list[int],
    current_kv_sequence_length: int,
    tokens_per_block: int,
    expected_move_source_kv_slot_ids: tuple[int, ...],
    expected_move_destination_kv_slot_ids: tuple[int, ...],
    expected_compacted_kv_sequence_length: int,
) -> None:
    """Retained entries move to the front, and entries in place are skipped."""
    plan = build_kv_compaction_plan(
        request_block_ids=request_block_ids,
        retained_kv_sequence_indices=retained_kv_sequence_indices,
        current_kv_sequence_length=current_kv_sequence_length,
        tokens_per_block=tokens_per_block,
    )

    assert plan.move_source_kv_slot_ids == expected_move_source_kv_slot_ids
    assert plan.move_destination_kv_slot_ids == expected_move_destination_kv_slot_ids
    assert plan.compacted_kv_sequence_length == expected_compacted_kv_sequence_length


@pytest.mark.parametrize(
    (
        "request_block_ids",
        "retained_kv_sequence_indices",
        "current_kv_sequence_length",
        "tokens_per_block",
    ),
    [
        pytest.param([5], [], 4, 4, id="empty_retained"),
        pytest.param([5], [2, 1], 4, 4, id="unsorted_retained"),
        pytest.param([5], [1, 1], 4, 4, id="duplicate_retained"),
        pytest.param([5], [0, 4], 4, 4, id="retained_past_sequence_end"),
        pytest.param([5], [-1, 2], 4, 4, id="negative_retained"),
        pytest.param([5], [0], 5, 4, id="blocks_too_small_for_sequence"),
        pytest.param([5], [0], 4, 0, id="non_positive_tokens_per_block"),
        pytest.param([5], [0], 0, 4, id="non_positive_sequence_length"),
    ],
)
def test_rejects_callers_that_break_an_invariant(
    request_block_ids: list[int],
    retained_kv_sequence_indices: list[int],
    current_kv_sequence_length: int,
    tokens_per_block: int,
) -> None:
    """Programmer errors fail loudly instead of being silently repaired."""
    with pytest.raises(AssertionError):
        build_kv_compaction_plan(
            request_block_ids=request_block_ids,
            retained_kv_sequence_indices=retained_kv_sequence_indices,
            current_kv_sequence_length=current_kv_sequence_length,
            tokens_per_block=tokens_per_block,
        )
