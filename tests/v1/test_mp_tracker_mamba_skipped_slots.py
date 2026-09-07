# SPDX-License-Identifier: Apache-2.0
"""The MP tracker nulls the old slot of a block vLLM relocated to the tail."""

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

BLOCK = 100
ATTN, MAMBA = 0, 1


def _tracker() -> LMCacheMPRequestTracker:
    request = SimpleNamespace(
        request_id="req-0",
        cache_salt="",
        prompt_token_ids=list(range(337)),
        all_token_ids=ConstantList(list(range(337))),
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
    )
    return LMCacheMPRequestTracker(request)


def _store_blocks(tracker: LMCacheMPRequestTracker) -> list[list[int]]:
    meta = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, BLOCK, [BLOCK, BLOCK])
    assert meta is not None
    return meta.op.block_ids


def test_relocated_speculative_block_is_nulled():
    """MTP prefill of 337 tokens, block 100, 4 speculative blocks.

    Step 3 spans [200, 337); vLLM nulls Mamba slot 2, moves block 12 to the
    tail and reports [12, 16]. Chunk 2 must not be stored from block 12.
    """
    tracker = _tracker()
    tracker.append_block_ids(([50, 51], [10, 11, 12, 13, 14]))
    tracker.increase_num_scheduled_tokens(100)
    tracker.append_block_ids(([52], [15]))
    tracker.increase_num_scheduled_tokens(100)
    assert _store_blocks(tracker) == [[50, 51], [10, 11]]
    tracker.append_block_ids(([53, 54], [12, 16]))
    tracker.increase_num_scheduled_tokens(137)
    assert tracker.allocated_block_ids[MAMBA] == [10, 11, 0, 13, 14, 15, 12, 16]
    assert tracker.allocated_block_ids[ATTN] == [50, 51, 52, 53, 54]
    assert _store_blocks(tracker) == [[52], [0]]


def test_padded_nulls_are_appended_as_is():
    tracker = _tracker()
    tracker.append_block_ids(([50, 51, 52], [0, 0, 20, 21, 22, 23, 24]))
    tracker.append_block_ids(([53], [0, 25]))
    assert tracker.allocated_block_ids[MAMBA] == [0, 0, 20, 21, 22, 23, 24, 0, 25]
