# SPDX-License-Identifier: Apache-2.0
"""Prefix-replay windows are stored only for chunks ending at or past the
prompt end, where every layer's window was computed."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    VLLM_NULL_BLOCK_ID,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)

CHUNK = 64
# Group 0: global KV (64-token blocks). Group 1: replay window (32-token).
GROUP_TOKENS_PER_BLOCK = [64, 32]
REPLAY_GROUPS = (1,)
NUM_PROMPT_TOKENS = 192


def _tracker(all_token_ids: list[int]) -> LMCacheMPRequestTracker:
    request = SimpleNamespace(
        request_id="req-0",
        cache_salt="",
        prompt_token_ids=list(range(NUM_PROMPT_TOKENS)),
        all_token_ids=ConstantList(all_token_ids),
        num_prompt_tokens=NUM_PROMPT_TOKENS,
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.append_block_ids((list(range(1, 6)), list(range(101, 111))))
    return tracker


def _store(tracker: LMCacheMPRequestTracker) -> LMCacheMPRequestMetadata:
    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK,
        GROUP_TOKENS_PER_BLOCK,
        save_decode_cache=True,
        prefix_replay_group_ids=REPLAY_GROUPS,
    )
    assert meta is not None
    return meta


def test_prompt_chunks_null_only_the_replay_window():
    """Chunks ending before the prompt end keep global KV and drop the
    window; the chunk ending exactly at the prompt end keeps both."""
    tokens = list(range(NUM_PROMPT_TOKENS))
    tracker = _tracker(tokens)
    tracker.increase_num_scheduled_tokens(NUM_PROMPT_TOKENS)

    meta = _store(tracker)

    assert (meta.op.start, meta.op.end) == (0, 192)
    assert meta.op.block_ids[0] == [1, 2, 3]
    assert meta.op.block_ids[1] == [VLLM_NULL_BLOCK_ID] * 4 + [105, 106]


def test_decode_chunks_keep_the_window():
    tokens = list(range(NUM_PROMPT_TOKENS))
    tracker = _tracker(tokens)
    tracker.increase_num_scheduled_tokens(NUM_PROMPT_TOKENS)
    _store(tracker)

    tokens.extend(range(1000, 1064))
    tracker.increase_num_scheduled_tokens(64)
    meta = _store(tracker)

    assert (meta.op.start, meta.op.end) == (192, 256)
    assert meta.op.block_ids[1] == [107, 108]


def test_no_replay_groups_store_every_block():
    tracker = _tracker(list(range(NUM_PROMPT_TOKENS)))
    tracker.increase_num_scheduled_tokens(NUM_PROMPT_TOKENS)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK, save_decode_cache=True
    )

    assert meta is not None
    assert meta.op.block_ids[1] == list(range(101, 107))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
