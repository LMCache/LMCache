# SPDX-License-Identifier: Apache-2.0
"""The MP store bound follows vLLM's computed position, not lookup-time hits."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)

CHUNK = 64
# Hybrid under DCP: attention span 16 * dcp 4, Mamba 16; hits land on 16.
GROUP_TOKENS_PER_BLOCK = [64, 16]
NUM_PROMPT_TOKENS = 300


def _tracker() -> LMCacheMPRequestTracker:
    request = SimpleNamespace(
        request_id="req-0",
        cache_salt="",
        prompt_token_ids=list(range(NUM_PROMPT_TOKENS)),
        all_token_ids=ConstantList(list(range(NUM_PROMPT_TOKENS))),
        num_prompt_tokens=NUM_PROMPT_TOKENS,
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.append_block_ids((list(range(1, 6)), list(range(101, 120))))
    return tracker


def _lookup(tracker: LMCacheMPRequestTracker, vllm_hit: int, lmcache_hit: int):
    """Mirror get_num_new_matched_tokens."""
    alignment = 64
    tracker.num_vllm_hit_tokens = vllm_hit // alignment * alignment
    tracker.num_lmcache_hit_tokens = lmcache_hit
    tracker.num_stored_tokens = lmcache_hit


def _store_range(tracker: LMCacheMPRequestTracker) -> tuple[int, int] | None:
    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )
    if meta is None:
        return None
    return meta.op.start, meta.op.end


def test_unaligned_apc_hit_stores_every_computed_chunk():
    """APC hit 112 floors to 64; chunk [192, 256) must still be stored."""
    tracker = _tracker()
    _lookup(tracker, vllm_hit=112, lmcache_hit=64)
    tracker.increase_num_scheduled_tokens(188)
    tracker.update_num_engine_computed_tokens(112, 188)
    assert _store_range(tracker) == (64, 256)


def test_chunked_prefill_follows_engine_position():
    """Chunked prefill stores up to vLLM's computed end."""
    tracker = _tracker()
    _lookup(tracker, vllm_hit=112, lmcache_hit=64)
    tracker.increase_num_scheduled_tokens(100)
    tracker.update_num_engine_computed_tokens(112, 100)
    assert _store_range(tracker) == (64, 192)
    tracker.increase_num_scheduled_tokens(88)
    tracker.update_num_engine_computed_tokens(212, 88)
    assert _store_range(tracker) == (192, 256)


def test_aligned_hits_are_unchanged():
    """Aligned hit: unchanged."""
    tracker = _tracker()
    _lookup(tracker, vllm_hit=128, lmcache_hit=64)
    tracker.increase_num_scheduled_tokens(100)
    tracker.update_num_engine_computed_tokens(128, 100)
    assert _store_range(tracker) == (64, 192)


def test_without_engine_position_falls_back_to_hits():
    """No engine position: old bound."""
    tracker = _tracker()
    _lookup(tracker, vllm_hit=112, lmcache_hit=64)
    tracker.increase_num_scheduled_tokens(188)
    assert _store_range(tracker) == (64, 192)
