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
GRID = 16
# The prompt's last chunk starts at 256; its tail ends on the 16-token grid
# before the prompt's last token.
TAIL_END = (PROMPT - 1) // GRID * GRID
# Engine group 0: attention, 16 tokens per block. Engine group 1: Mamba
# state, one block per chunk.
GROUP_TOKENS_PER_BLOCK = [16, CHUNK]
MAMBA_BLOCKS = [500, 501, 502]
HANDOFF_BLOCK = 900


def test_mamba_tail_ends_where_vllm_handed_over_the_state():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    (meta,) = _store_tails(tracker, {TAIL_END: {1: HANDOFF_BLOCK}})

    assert (meta.op.start, meta.op.end) == (2 * CHUNK, TAIL_END)
    assert meta.op.include_unfull_chunk
    # The Mamba block is the handed-over snapshot, not the running block.
    assert meta.op.block_ids == [list(range(16, 21)), [HANDOFF_BLOCK]]
    assert _store_tails(tracker, {TAIL_END: {1: HANDOFF_BLOCK}}) == []


def test_mamba_tail_needs_a_hand_off_inside_the_last_chunk():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    # A chunk boundary, an earlier chunk, and a missing group all miss.
    hand_offs = {2 * CHUNK: {1: 1}, CHUNK + 16: {1: 2}, TAIL_END: {0: 3}}

    assert _store_tails(tracker, hand_offs) == []


def test_attention_tail_ends_on_the_grid_once_computed():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(300)
    assert _store_tails(tracker, {}, boundary_state_groups=()) == []

    tracker.increase_num_scheduled_tokens(PROMPT - 300)
    (meta,) = _store_tails(tracker, {}, boundary_state_groups=())

    assert (meta.op.start, meta.op.end) == (2 * CHUNK, TAIL_END)
    assert meta.op.block_ids == [list(range(16, 21)), MAMBA_BLOCKS[2:]]


def test_tail_is_stored_while_the_request_keeps_decoding():
    tracker = _tracker(num_tokens=PROMPT + 8)
    tracker.increase_num_scheduled_tokens(PROMPT + 8)

    (meta,) = _store_tails(tracker, {TAIL_END: {1: HANDOFF_BLOCK}})

    assert meta.op.end == TAIL_END


def test_tail_the_lookup_found_is_not_stored_again():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)
    tracker.num_lmcache_hit_tokens = TAIL_END

    assert _store_tails(tracker, {TAIL_END: {1: HANDOFF_BLOCK}}) == []


def test_full_chunk_stores_stay_aligned():
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
    tracker.num_lmcache_hit_tokens = TAIL_END
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, TAIL_END)
    assert meta.op.include_unfull_chunk
    assert meta.op.block_ids == [list(range(21)), MAMBA_BLOCKS]


def test_lookup_asks_for_the_unfull_chunk_on_the_grid():
    connector, scheduler_adapter = _connector(lookup_hit=TAIL_END)

    assert connector.get_num_new_matched_tokens(_request(), 0) == (TAIL_END, True)

    lookup_kwargs = scheduler_adapter.maybe_submit_lookup_request.call_args.kwargs
    assert lookup_kwargs["include_unfull_chunk"]
    assert lookup_kwargs["request_configs"]["lmcache.hint.unfull_chunk_grid"] == GRID
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == TAIL_END
    assert tracker.num_stored_tokens == 2 * CHUNK
    assert tracker.unfull_chunk_ends == {TAIL_END}


def test_unfull_hit_overlapping_a_vllm_hit_is_released():
    connector, scheduler_adapter = _connector(lookup_hit=TAIL_END)

    # vLLM already holds 288 tokens, past the last chunk boundary (256).
    assert connector.get_num_new_matched_tokens(_request(), 288) == (0, False)

    free_kwargs = scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert (free_kwargs["start"], free_kwargs["end"]) == (2 * CHUNK, TAIL_END)
    assert free_kwargs["include_unfull_chunk"]
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == 2 * CHUNK
    assert tracker.unfull_chunk_ends == set()


def test_handed_over_block_is_pinned_until_the_request_is_sent():
    connector, _ = _connector(lookup_hit=0)
    pool = MagicMock()
    pool.blocks = {HANDOFF_BLOCK: "handoff-block"}
    connector._gpu_block_pool = pool
    connector._boundary_state_groups = [1]
    connector._pinned_boundary_state_blocks = {}
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector._unfull_chunk_grid = GRID
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            boundary_state_offloads={"req-0": [(1, HANDOFF_BLOCK, TAIL_END)]}
        )
    )
    metadata = MagicMock()

    connector._add_unfull_chunk_stores(scheduler_output, tracker, metadata)

    (r_meta,), _ = metadata.add_request_metadata.call_args
    assert r_meta.op.end == TAIL_END
    pool.touch.assert_called_once_with(["handoff-block"])
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=None,
            finished_sending={"req-0"},
        )
    )
    pool.free_blocks.assert_called_once_with(["handoff-block"])


def _connector(lookup_hit: int) -> tuple[LMCacheMPConnector, MagicMock]:
    scheduler_adapter = MagicMock()
    scheduler_adapter.lmcache_tokens_per_chunk = CHUNK
    scheduler_adapter.check_lookup_result.return_value = lookup_hit
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = scheduler_adapter
    connector._reserve_last_token_for_lookup = True
    connector._save_unfull_chunk = True
    connector._hit_alignment_tokens = 16
    connector._unfull_chunk_grid = GRID
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


def _store_tails(
    tracker: LMCacheMPRequestTracker,
    boundary_state_blocks: dict[int, dict[int, int]],
    boundary_state_groups: tuple[int, ...] = (1,),
) -> list[LMCacheMPRequestMetadata]:
    return LMCacheMPRequestMetadata.GetUnfullChunkStoreMetadata(
        tracker,
        CHUNK,
        GROUP_TOKENS_PER_BLOCK,
        GRID,
        boundary_state_blocks,
        boundary_state_groups,
    )
