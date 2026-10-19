# SPDX-License-Identifier: Apache-2.0
"""The MP connector stores, finds and retrieves the unfull chunk."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

CHUNK = 128
PROMPT = 337
# Engine group 0: attention, 16 tokens per block. Engine group 1: Mamba
# state, one block per chunk; its last block holds the state at the end.
GROUP_TOKENS_PER_BLOCK = [16, CHUNK]
MAMBA_BLOCKS = [500, 501, 502]
# Engine group 2: a per-request ring, one block for the request's lifetime.
RING_GROUP = 2
RING_BLOCK = 900


def test_prompt_end_step_stores_full_chunks_and_the_unfull_chunk():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = _store(tracker)

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, PROMPT)
    assert meta.op.include_unfull_chunk
    # 22 = ceil(337 / 16) attention blocks; the Mamba group's third block
    # holds the state at the prompt end.
    assert meta.op.block_ids == [list(range(22)), MAMBA_BLOCKS]
    assert tracker.num_stored_tokens == 2 * CHUNK
    assert _store(tracker) is None


def test_unfull_chunk_waits_for_the_step_computing_the_prompt_end():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(300)

    meta = _store(tracker)
    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, 2 * CHUNK)
    assert not meta.op.include_unfull_chunk
    assert meta.op.block_ids == [list(range(16)), MAMBA_BLOCKS[:2]]

    tracker.increase_num_scheduled_tokens(PROMPT - 300)
    meta = _store(tracker)
    assert meta is not None
    assert (meta.op.start, meta.op.end) == (2 * CHUNK, PROMPT)
    assert meta.op.include_unfull_chunk
    assert meta.op.block_ids == [list(range(16, 22)), MAMBA_BLOCKS[2:]]


def test_unfull_chunk_is_not_stored_after_decode_moved_the_state():
    tracker = _tracker(num_tokens=PROMPT + 1)
    # The prompt end and a decode token computed by the time stores run.
    tracker.increase_num_scheduled_tokens(PROMPT + 1)

    meta = _store(tracker)

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, 2 * CHUNK)


def test_unfull_chunk_is_not_stored_for_a_request_that_keeps_decoding():
    # The next decode step would rewrite the recurrent state while the
    # asynchronous store still copies it.
    tracker = _tracker(max_tokens=16)
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = _store(tracker)

    assert meta is not None
    assert (meta.op.end, meta.op.include_unfull_chunk) == (2 * CHUNK, False)


def test_unfull_chunk_needs_the_option():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    assert (meta.op.end, meta.op.include_unfull_chunk) == (2 * CHUNK, False)


def test_unaligned_hit_retrieves_the_unfull_chunk():
    tracker = _tracker()
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = PROMPT - 1
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, PROMPT - 1)
    assert meta.op.include_unfull_chunk
    assert meta.op.block_ids == [list(range(21)), MAMBA_BLOCKS]


def test_ring_block_rides_with_every_chunk_and_the_unfull_chunk():
    tracker = _tracker()
    tracker.append_block_ids(([], [], [RING_BLOCK]))
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK,
        [*GROUP_TOKENS_PER_BLOCK, 0],
        per_request_state_groups=(RING_GROUP,),
        store_unfull_chunk=True,
    )

    assert meta is not None
    assert meta.op.include_unfull_chunk
    # Two full chunks plus the unfull one each carry the ring block.
    assert meta.op.block_ids == [list(range(22)), MAMBA_BLOCKS, [RING_BLOCK] * 3]


def test_ring_block_is_retrieved_with_every_chunk():
    tracker = _tracker()
    tracker.append_block_ids(([], [], [RING_BLOCK]))
    tracker.num_lmcache_hit_tokens = PROMPT - 1
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        CHUNK,
        [*GROUP_TOKENS_PER_BLOCK, 0],
        per_request_state_groups=(RING_GROUP,),
    )

    assert meta is not None
    assert meta.op.block_ids[RING_GROUP] == [RING_BLOCK] * 3


def test_unregistered_ring_is_not_transferred():
    tracker = _tracker()
    tracker.append_block_ids(([], [], [RING_BLOCK]))
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK, [*GROUP_TOKENS_PER_BLOCK, 0]
    )

    assert meta is not None
    assert meta.op.block_ids[RING_GROUP] == []


def test_lookup_asks_for_the_unfull_chunk_and_keeps_stores_aligned():
    connector, scheduler_adapter = _connector(lookup_hit=PROMPT - 1)

    assert connector.get_num_new_matched_tokens(_request(), 0) == (PROMPT - 1, True)

    lookup_kwargs = scheduler_adapter.maybe_submit_lookup_request.call_args.kwargs
    assert lookup_kwargs["include_unfull_chunk"]
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == PROMPT - 1
    assert tracker.num_stored_tokens == 2 * CHUNK
    assert tracker.unfull_chunk_stored


def test_unfull_hit_overlapping_a_vllm_hit_is_released():
    connector, scheduler_adapter = _connector(lookup_hit=PROMPT - 1)

    # vLLM already holds 288 tokens, past the last chunk boundary (256).
    assert connector.get_num_new_matched_tokens(_request(), 288) == (0, False)

    free_kwargs = scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert (free_kwargs["start"], free_kwargs["end"]) == (2 * CHUNK, PROMPT - 1)
    assert free_kwargs["include_unfull_chunk"]
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == 2 * CHUNK
    assert not tracker.unfull_chunk_stored


def _connector(lookup_hit: int) -> tuple[LMCacheMPConnector, MagicMock]:
    scheduler_adapter = MagicMock()
    scheduler_adapter.lmcache_tokens_per_chunk = CHUNK
    scheduler_adapter.check_lookup_result.return_value = lookup_hit
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = scheduler_adapter
    connector._reserve_last_token_for_lookup = True
    connector._unfull_chunk = True
    connector._hit_alignment_tokens = 16
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False
    return connector, scheduler_adapter


def _request(num_tokens: int = PROMPT, max_tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-0",
        resumable=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        cache_salt="",
        prompt_token_ids=list(range(PROMPT)),
        all_token_ids=ConstantList(list(range(num_tokens))),
        num_prompt_tokens=PROMPT,
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None, max_tokens=max_tokens),
    )


def _tracker(num_tokens: int = PROMPT, max_tokens: int = 1) -> LMCacheMPRequestTracker:
    tracker = LMCacheMPRequestTracker(_request(num_tokens, max_tokens))
    tracker.append_block_ids(
        (
            list(range(-(-num_tokens // 16))),
            MAMBA_BLOCKS[: -(-num_tokens // CHUNK)],
        )
    )
    return tracker


def _store(tracker: LMCacheMPRequestTracker) -> LMCacheMPRequestMetadata | None:
    return LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK,
        GROUP_TOKENS_PER_BLOCK,
        store_unfull_chunk=True,
    )
