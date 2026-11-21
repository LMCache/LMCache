# SPDX-License-Identifier: Apache-2.0
"""Boundary-semantics tests for MP store metadata on hybrid layouts.

``GetStoreMetadata`` bounds the storable prefix by the minimum over engine
groups of ``allocated_blocks * tokens_per_block``, restricted to non-scratch
groups. The scratch exclusion and the recurrent last-token reservation
already have focused tests; this module covers the remaining boundary
behavior: chunk-edge inputs, per-group capacity and scheduled-token
bottlenecks, scratch-position permutations, incremental multi-step stores,
and the guard that non-scratch groups always participate in the minimum.
"""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="store metadata imports vLLM at module top")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)

CHUNK = 32


@dataclass
class _FakeSamplingParams:
    extra_args: dict[str, Any] | None = None


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, prompt_token_ids: list[int]):
        self.request_id = "req-b"
        self.resumable = False
        self.status = RequestStatus.WAITING
        self.cache_salt = ""
        self.prompt_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(list(prompt_token_ids))
        self.mm_features = []
        self.sampling_params = _FakeSamplingParams()
        self.block_hashes: list = []


def _tracker(
    num_tokens: int,
    blocks: dict[int, list[int]],
    num_scheduled: int | None = None,
) -> LMCacheMPRequestTracker:
    tracker = LMCacheMPRequestTracker(_FakeRequest(list(range(num_tokens))))
    tracker.allocated_block_ids = blocks
    tracker.num_scheduled_tokens = (
        num_tokens if num_scheduled is None else num_scheduled
    )
    return tracker


def _store(
    tracker: LMCacheMPRequestTracker,
    group_tokens_per_block: list[int],
) -> LMCacheMPRequestMetadata | None:
    return LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=CHUNK,
        group_tokens_per_block=group_tokens_per_block,
    )


@pytest.mark.parametrize(
    "num_tokens, expected_end",
    [(31, None), (32, 32), (33, 32), (63, 32), (64, 64), (65, 64)],
)
def test_store_ends_on_chunk_boundary(num_tokens: int, expected_end: int):
    """Only full chunks are storable: kC-1/kC/kC+1 floor to
    (k-1)C/kC/kC."""
    tracker = _tracker(num_tokens, {0: list(range(8))})  # 8x16=128, ample
    metadata = _store(tracker, [16])
    if expected_end is None:
        assert metadata is None
    else:
        assert metadata is not None
        assert (metadata.op.start, metadata.op.end) == (0, expected_end)


def test_store_bounded_by_smallest_linear_group_capacity():
    """The storable prefix is capped by the least-covered group, not the
    largest: 6 blocks vs 4 blocks must stop at 64 tokens."""
    tracker = _tracker(200, {0: list(range(6)), 1: list(range(4))})
    metadata = _store(tracker, [16, 16])
    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 64)
    assert metadata.op.block_ids == [list(range(4)), list(range(4))]


@pytest.mark.parametrize("num_scheduled, expected_end", [(63, 32), (64, 64)])
def test_store_bounded_by_scheduled_tokens(num_scheduled: int, expected_end: int):
    """Tokens scheduled in this step bound the store even when allocation
    and prompt length are larger: 63 scheduled tokens store one chunk."""
    tracker = _tracker(128, {0: list(range(8))}, num_scheduled=num_scheduled)
    metadata = _store(tracker, [16])
    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, expected_end)


@pytest.mark.parametrize(
    "group_tokens_per_block, blocks, expected_block_ids",
    [
        (
            [0, 16, 16],
            {0: [9], 1: [0, 1, 2, 3], 2: [4, 5, 6, 7]},
            [[], [0, 1, 2, 3], [4, 5, 6, 7]],
        ),
        (
            [16, 0, 16],
            {0: [0, 1, 2, 3], 1: [9], 2: [4, 5, 6, 7]},
            [[0, 1, 2, 3], [], [4, 5, 6, 7]],
        ),
        (
            [16, 16, 0],
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [9]},
            [[0, 1, 2, 3], [4, 5, 6, 7], []],
        ),
    ],
)
def test_scratch_position_keeps_group_order(
    group_tokens_per_block: list[int],
    blocks: dict[int, list[int]],
    expected_block_ids: list[list[int]],
):
    """A scratch group at any position keeps its (empty) slot: the store op
    lists groups in engine-group order regardless of where the scratch
    group sits."""
    tracker = _tracker(64, blocks)
    metadata = _store(tracker, group_tokens_per_block)
    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 64)
    assert metadata.op.block_ids == expected_block_ids


def test_incremental_store_advances_without_restarting():
    """Decode-step stores continue from the stored frontier: the second
    chunk starts where the first ended, and a step that completes no new
    full chunk stores nothing."""
    tracker = _tracker(96, {0: list(range(6))}, num_scheduled=32)

    first = _store(tracker, [16])
    assert first is not None
    assert (first.op.start, first.op.end) == (0, 32)
    assert tracker.num_stored_tokens == 32

    tracker.increase_num_scheduled_tokens(32)
    second = _store(tracker, [16])
    assert second is not None
    assert (second.op.start, second.op.end) == (32, 64)
    assert tracker.num_stored_tokens == 64

    third = _store(tracker, [16])
    assert third is None
    assert tracker.num_stored_tokens == 64


def test_all_linear_groups_still_take_minimum():
    """A layout with no scratch group (four linear groups with mixed
    capacities, Qwen3.8-27B style) keeps the min-over-groups behavior."""
    tracker = _tracker(
        200,
        {
            0: list(range(6)),
            1: list(range(4)),
            2: list(range(6)),
            3: list(range(6)),
        },
    )
    metadata = _store(tracker, [16, 16, 16, 16])
    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 64)
    assert all(len(group_blocks) == 4 for group_blocks in metadata.op.block_ids)


@pytest.mark.parametrize(
    "small_group_blocks, expected_end",
    [
        # 5 blocks x 8 tokens = 40 tokens: the small group binds, the
        # store floors to one chunk.
        (list(range(5)), 32),
        # 1 block x 8 tokens: the small group caps below one chunk, so
        # nothing is storable.
        ([0], None),
    ],
)
def test_small_non_scratch_group_still_caps_store(
    small_group_blocks: list[int], expected_end: int | None
):
    """The scratch exclusion keys on the zero-span scratch spec, not on
    block counts or small spans: a prefix-cacheable group with a small
    tokens-per-block still bounds the storable prefix."""
    tracker = _tracker(
        128,
        {0: list(range(4)), 1: small_group_blocks},  # 128 vs 40/8
    )
    metadata = _store(tracker, [32, 8])
    if expected_end is None:
        assert metadata is None
    else:
        assert metadata is not None
        assert (metadata.op.start, metadata.op.end) == (0, expected_end)
        assert metadata.op.block_ids == [[0], list(range(4))]
