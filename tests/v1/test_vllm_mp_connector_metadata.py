# SPDX-License-Identifier: Apache-2.0
"""Allocation coverage tests for vLLM MP retrieve metadata."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP metadata imports vLLM at module load")

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

CHUNK_TOKENS = 64


def _tracker(
    *,
    lmcache_hit_tokens: int = CHUNK_TOKENS,
    vllm_hit_tokens: int = 0,
    allocated_block_ids: dict[int, list[int]],
) -> LMCacheMPRequestTracker:
    """Build a tracker ready to retrieve one aligned chunk."""
    token_ids = list(range(CHUNK_TOKENS))
    request = SimpleNamespace(
        request_id="req-1",
        cache_salt="",
        prompt_token_ids=token_ids,
        all_token_ids=token_ids,
        mm_features=[],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_lmcache_hit_tokens = lmcache_hit_tokens
    tracker.num_vllm_hit_tokens = vllm_hit_tokens
    tracker.allocated_block_ids = allocated_block_ids
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    return tracker


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids", "expected_block_ids"),
    [
        ([16], {0: [0, 1, 2, 3]}, [[0, 1, 2, 3]]),
        ([64], {0: [5]}, [[5]]),
        ([16, 32], {0: [0, 1, 2, 3], 1: [10, 11]}, [[0, 1, 2, 3], [10, 11]]),
        ([16], {0: [0, 1, 2, 3, 4]}, [[0, 1, 2, 3]]),
    ],
)
def test_retrieve_emitted_when_every_group_covers_range(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
    expected_block_ids: list[list[int]],
) -> None:
    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        _tracker(allocated_block_ids=allocated_block_ids),
        CHUNK_TOKENS,
        group_tokens_per_block,
    )

    assert metadata is not None
    assert metadata.direction == "RETRIEVE"
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.block_ids == expected_block_ids


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids"),
    [
        ([16], {0: [0, 1, 2]}),
        ([64], {0: []}),
        ([16, 32], {0: [0, 1, 2, 3], 1: [10]}),
        ([16, 16], {0: [0, 1, 2, 3]}),
    ],
)
def test_retrieve_suppressed_when_any_group_is_short(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
) -> None:
    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        _tracker(allocated_block_ids=allocated_block_ids),
        CHUNK_TOKENS,
        group_tokens_per_block,
    )

    assert metadata is None


def test_retrieve_preserves_skip_inside_first_chunk() -> None:
    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        _tracker(
            vllm_hit_tokens=16,
            allocated_block_ids={0: [0, 1, 2, 3]},
        ),
        CHUNK_TOKENS,
        [16],
    )

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.skip_first_n_tokens == 16
