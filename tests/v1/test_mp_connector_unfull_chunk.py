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
# The prompt's last chunk starts at 256; its tail's boundary (a token count)
# is on the 16-token grid below the prompt's last token.
TAIL_BOUNDARY = (PROMPT - 1) // GRID * GRID
# Engine group 0: attention, 16 tokens per block. Engine group 1: Mamba
# state, one block per chunk.
GROUP_TOKENS_PER_BLOCK = [16, CHUNK]
MAMBA_BLOCKS = [500, 501, 502]
BOUNDARY_STATE_BLOCK = 900


def test_mamba_tail_boundary_is_where_vllm_offered_a_boundary_state():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    ((meta, boundary_state_block_ids),) = (
        LMCacheMPRequestMetadata.GetUnfullChunkStoreMetadata(
            tracker,
            CHUNK,
            GROUP_TOKENS_PER_BLOCK,
            GRID,
            {TAIL_BOUNDARY: {1: BOUNDARY_STATE_BLOCK}},
            (1,),
        )
    )

    assert (meta.op.start, meta.op.end) == (2 * CHUNK, TAIL_BOUNDARY)
    # The Mamba block is vLLM's boundary-state snapshot, not the running block.
    assert meta.op.block_ids == [list(range(16, 21)), [BOUNDARY_STATE_BLOCK]]
    assert boundary_state_block_ids == [BOUNDARY_STATE_BLOCK]
    assert _store_tails(tracker, {TAIL_BOUNDARY: {1: BOUNDARY_STATE_BLOCK}}) == []


def test_prompt_ending_request_also_stores_its_exact_tail():
    """A disaggregated prefill ends at its prompt: its block table describes
    exactly P, where the decode resumes."""
    tracker = _tracker(max_tokens=1)
    tracker.increase_num_scheduled_tokens(PROMPT)

    metas = _store_tails(tracker, {TAIL_BOUNDARY: {1: BOUNDARY_STATE_BLOCK}})

    assert [(meta.op.end, meta.op.block_ids[1]) for meta in metas] == [
        (TAIL_BOUNDARY, [BOUNDARY_STATE_BLOCK]),
        # The running Mamba block holds the state at P: nothing runs after.
        (PROMPT, MAMBA_BLOCKS[2:]),
    ]


def test_exact_tail_waits_for_the_whole_prompt():
    tracker = _tracker(max_tokens=1)
    tracker.increase_num_scheduled_tokens(PROMPT - 1)

    assert [meta.op.end for meta in _store_tails(tracker, {})] == []


def test_mamba_tail_needs_a_hand_off_inside_the_last_chunk():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    # A chunk boundary, an earlier chunk, and a missing group all miss.
    hand_offs = {2 * CHUNK: {1: 1}, CHUNK + 16: {1: 2}, TAIL_BOUNDARY: {0: 3}}

    assert _store_tails(tracker, hand_offs) == []


def test_attention_tail_boundary_is_on_the_grid_once_computed():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(300)
    assert _store_tails(tracker, {}, boundary_state_groups=()) == []

    tracker.increase_num_scheduled_tokens(PROMPT - 300)
    (meta,) = _store_tails(tracker, {}, boundary_state_groups=())

    assert (meta.op.start, meta.op.end) == (2 * CHUNK, TAIL_BOUNDARY)
    assert meta.op.block_ids == [list(range(16, 21)), MAMBA_BLOCKS[2:]]


def test_tail_is_stored_while_the_request_keeps_decoding():
    tracker = _tracker(num_tokens=PROMPT + 8)
    tracker.increase_num_scheduled_tokens(PROMPT + 8)

    (meta,) = _store_tails(tracker, {TAIL_BOUNDARY: {1: BOUNDARY_STATE_BLOCK}})

    assert meta.op.end == TAIL_BOUNDARY


def test_tail_the_lookup_found_is_not_stored_again():
    connector, _ = _connector(lookup_hit=TAIL_BOUNDARY)
    connector.get_num_new_matched_tokens(_request(), 0)
    tracker = connector.request_trackers["req-0"]
    tracker.append_block_ids((list(range(22)), MAMBA_BLOCKS))
    tracker.increase_num_scheduled_tokens(PROMPT - TAIL_BOUNDARY)

    assert _store_tails(tracker, {TAIL_BOUNDARY: {1: BOUNDARY_STATE_BLOCK}}) == []


def test_chunk_aligned_prompt_has_no_exact_tail():
    """A prompt of whole chunks ends on a full chunk, which the full-chunk
    store covers; its grid tail is still stored."""
    tracker = _tracker(prompt_tokens=2 * CHUNK, max_tokens=1)
    tracker.increase_num_scheduled_tokens(2 * CHUNK)

    metas = _store_tails(tracker, {}, boundary_state_groups=())

    assert [meta.op.end for meta in metas] == [2 * CHUNK - GRID]


def test_full_chunk_stores_stay_aligned():
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    assert meta.op.end == 2 * CHUNK


def test_unaligned_hit_retrieves_the_unfull_chunk():
    tracker = _tracker()
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = TAIL_BOUNDARY
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, TAIL_BOUNDARY)
    assert meta.op.block_ids == [list(range(21)), MAMBA_BLOCKS]


def test_lookup_asks_for_the_unfull_chunk_on_the_grid():
    connector, scheduler_adapter = _connector(lookup_hit=TAIL_BOUNDARY)

    assert connector.get_num_new_matched_tokens(_request(), 0) == (TAIL_BOUNDARY, True)

    lookup_kwargs = scheduler_adapter.maybe_submit_lookup_request.call_args.kwargs
    assert lookup_kwargs["include_unfull_chunk"]
    assert lookup_kwargs["request_configs"]["lmcache.hint.unfull_chunk_grid"] == GRID
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == TAIL_BOUNDARY
    assert tracker.num_stored_tokens == 2 * CHUNK
    assert tracker.unfull_chunk_boundaries == {TAIL_BOUNDARY}


def test_unfull_hit_overlapping_a_vllm_hit_loads_the_rest_of_the_tail():
    """vLLM already holds 288 tokens, inside the tail [256, 336): the hit
    stays at the tail's boundary and only the tokens past vLLM's hit load."""
    connector, scheduler_adapter = _connector(lookup_hit=TAIL_BOUNDARY)

    assert connector.get_num_new_matched_tokens(_request(), 288) == (
        TAIL_BOUNDARY - 288,
        True,
    )

    scheduler_adapter.free_lookup_locks.assert_not_called()
    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == TAIL_BOUNDARY
    assert tracker.unfull_chunk_boundaries == {TAIL_BOUNDARY}


def test_retrieve_of_a_tail_vllm_partly_holds_skips_vllms_tokens():
    tracker = _tracker()
    tracker.num_vllm_hit_tokens = 288
    tracker.num_lmcache_hit_tokens = TAIL_BOUNDARY
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK, GROUP_TOKENS_PER_BLOCK
    )

    assert meta is not None
    # The range starts at the tail's chunk; vLLM's 32 tokens in it are skipped.
    assert (meta.op.start, meta.op.end, meta.op.skip_first_n_tokens) == (
        2 * CHUNK,
        TAIL_BOUNDARY,
        288 - 2 * CHUNK,
    )


@pytest.mark.parametrize(
    "group_tokens_per_block",
    [
        [GRID],
        # A 32-token group, a scratch group (no tokens) and a 16-token group.
        [2 * GRID, 0, GRID],
    ],
)
@pytest.mark.parametrize(
    "prompt",
    [
        # The tail is the prompt's last token alone (P = 1 mod chunk).
        2 * CHUNK + 1,
        # The last token starts a 16-token block inside the tail.
        2 * CHUNK + GRID + 1,
        # The last token starts a block in every group.
        2 * CHUNK + 2 * GRID + 1,
        # The last token shares a block with the one before.
        2 * CHUNK + 20,
        # The prompt is whole chunks.
        2 * CHUNK,
    ],
)
def test_full_prompt_hit_loads_every_token_into_the_blocks_vllm_allocates(
    prompt: int, group_tokens_per_block: list[int]
):
    """A full-prompt hit reports every prompt token, as vLLM's Offloading and
    Nixl connectors do, so vLLM allocates the whole hit in every group and
    recomputes the last token itself. Holding that token back left its block
    unallocated while the retrieve covered it (#6751)."""
    connector, _ = _connector(lookup_hit=prompt)
    connector._group_tokens_per_block = group_tokens_per_block

    assert connector.get_num_new_matched_tokens(
        _request(prompt_tokens=prompt, max_tokens=1), 0
    ) == (prompt, True)

    tracker = connector.request_trackers["req-0"]
    assert tracker.num_lmcache_hit_tokens == prompt
    assert tracker.num_stored_tokens == 2 * CHUNK
    # LMCache holds the tail, so it is not stored again.
    assert tracker.unfull_chunk_boundaries == ({prompt} if prompt % CHUNK else set())

    # vLLM allocates each group's blocks for the tokens it was told to load.
    blocks = [
        list(range(-(-prompt // span))) if span else []
        for span in group_tokens_per_block
    ]
    tracker.append_block_ids(tuple(blocks))
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK, group_tokens_per_block
    )

    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, prompt)
    assert meta.op.block_ids == blocks


@pytest.mark.parametrize(
    "prompt",
    [
        # vLLM holds the whole first two chunks.
        2 * CHUNK + 1,
        # vLLM holds the first block of the tail's chunk too.
        2 * CHUNK + GRID + 1,
    ],
)
def test_full_prompt_hit_past_a_vllm_hit_of_all_but_the_last_token(prompt: int):
    """vLLM's prefix hit stops at P - 1 and it recomputes the last token, so
    nothing is loaded (as on dev) and every lookup lock, the tail's
    included, is freed once vLLM allocates."""
    connector, scheduler_adapter = _connector(lookup_hit=prompt)
    request = _request(prompt_tokens=prompt, max_tokens=1)

    assert connector.get_num_new_matched_tokens(request, prompt - 1) == (0, False)

    blocks = list(range(-(-prompt // GRID)))
    connector.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: (blocks,)), 0
    )
    tracker = connector.request_trackers["req-0"]
    assert tracker.state == LMCacheMPRequestState.READY
    kwargs = scheduler_adapter.free_lookup_locks.call_args.kwargs
    # An end at the hit boundary also frees the unfull chunk ending there.
    assert (kwargs["start"], kwargs["end"]) == (0, prompt)


WINDOW = 64
NULL_BLOCK = 0


def _sliding_window_blocks(num_tokens: int) -> list[int]:
    """vLLM's ``SlidingWindowManager`` allocation for a hit of ``num_tokens``:
    ``cdiv(num_tokens, GRID)`` blocks, the null block for each block wholly
    below ``get_num_skipped_tokens(num_tokens) = num_tokens - WINDOW + 1``."""
    num_skipped = max(0, num_tokens - WINDOW + 1) // GRID
    return [
        NULL_BLOCK if idx < num_skipped else idx + 1
        for idx in range(-(-num_tokens // GRID))
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        # P = WINDOW - 1 (mod GRID): with P reported, the block holding
        # token P - WINDOW, which recomputing token P - 1 reads, is null.
        207,
        223,
        # Token P - 1 starts a block, so P - 1 leaves it unallocated.
        2 * CHUNK + 1,
        2 * CHUNK + GRID + 1,
        # Neither.
        2 * CHUNK + 20,
        2 * CHUNK,
    ],
)
def test_full_prompt_hit_keeps_the_sliding_window_vllm_recomputes_with(prompt: int):
    """With a sliding-window group the hit is P - 1, as in vLLM's Offloading
    connector, so the window of the first token vLLM computes is allocated;
    when token P - 1 starts a block only the full chunks load."""
    connector, scheduler_adapter = _connector(lookup_hit=prompt)
    connector._full_hit_holds_back_last_token = True
    connector._group_tokens_per_block = [GRID]
    request = _request(prompt_tokens=prompt, max_tokens=1)

    need, async_load = connector.get_num_new_matched_tokens(request, 0)
    assert need is not None

    blocks = _sliding_window_blocks(need)
    # vLLM computes from token min(need, P - 1); its window starts WINDOW - 1
    # tokens earlier.
    window_start = min(need, prompt - 1) - WINDOW + 1
    assert blocks[window_start // GRID] != NULL_BLOCK
    full_chunks = prompt // CHUNK * CHUNK
    trimmed = prompt % CHUNK != 0 and (prompt - 1) % GRID == 0
    assert (need, async_load) == (full_chunks if trimmed else prompt - 1, True)
    connector.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: (blocks,)), need
    )
    if trimmed:
        kwargs = scheduler_adapter.free_lookup_locks.call_args.kwargs
        assert kwargs["extra_ranges"] == ((full_chunks, prompt),)
    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        connector.request_trackers["req-0"], CHUNK, [GRID]
    )
    assert meta is not None
    assert (meta.op.start, meta.op.end) == (0, full_chunks if trimmed else prompt)
    assert meta.op.block_ids == [blocks[: -(-meta.op.end // GRID)]]


def test_boundary_state_block_is_pinned_until_the_request_is_sent():
    connector, _ = _connector(lookup_hit=0)
    pool = MagicMock()
    pool.blocks = {BOUNDARY_STATE_BLOCK: "boundary-state-block"}
    connector._gpu_block_pool = pool
    connector._boundary_state_groups = [1]
    connector._pinned_boundary_state_blocks = {}
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector._unfull_chunk_grid = GRID
    tracker = _tracker()
    tracker.increase_num_scheduled_tokens(PROMPT)
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            boundary_state_offloads={
                "req-0": [(1, BOUNDARY_STATE_BLOCK, TAIL_BOUNDARY)]
            }
        )
    )
    metadata = MagicMock()

    connector._add_unfull_chunk_stores(scheduler_output, tracker, metadata)

    (r_meta,), _ = metadata.add_request_metadata.call_args
    assert r_meta.op.end == TAIL_BOUNDARY
    pool.touch.assert_called_once_with(["boundary-state-block"])
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=None,
            finished_sending={"req-0"},
        )
    )
    pool.free_blocks.assert_called_once_with(["boundary-state-block"])


def test_pinned_boundary_state_block_is_released_after_a_preemption():
    """A request stores its tail, is preempted, then hits the whole tail and
    stores nothing more. It still delays its free, so the block it pinned
    before the preemption is released when it is reported sent."""
    connector, scheduler_adapter = _connector(lookup_hit=2 * CHUNK)
    pool = MagicMock()
    pool.blocks = {BOUNDARY_STATE_BLOCK: "boundary-state-block"}
    connector._gpu_block_pool = pool
    connector._boundary_state_groups = [1]
    connector._pinned_boundary_state_blocks = {}
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector._delay_free_req_ids = set()
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            boundary_state_offloads={
                "req-0": [(1, BOUNDARY_STATE_BLOCK, TAIL_BOUNDARY)]
            }
        )
    )
    request = _request()
    for status, hit in (
        (RequestStatus.WAITING, 2 * CHUNK),
        (RequestStatus.PREEMPTED, TAIL_BOUNDARY),
    ):
        request.status = status
        scheduler_adapter.check_lookup_result.return_value = hit
        connector.get_num_new_matched_tokens(request, 0)
        tracker = connector.request_trackers["req-0"]
        tracker.append_block_ids((list(range(-(-PROMPT // 16))), MAMBA_BLOCKS))
        tracker.increase_num_scheduled_tokens(PROMPT - hit)
        tracker.state = LMCacheMPRequestState.READY
        connector._add_unfull_chunk_stores(scheduler_output, tracker, MagicMock())

    assert not tracker.store_scheduled
    assert connector.request_finished(request, []) == (True, None)
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=None,
            finished_sending={"req-0"},
        )
    )
    pool.free_blocks.assert_called_once_with(["boundary-state-block"])


def _connector(lookup_hit: int) -> tuple[LMCacheMPConnector, MagicMock]:
    scheduler_adapter = MagicMock()
    scheduler_adapter.lmcache_tokens_per_chunk = CHUNK
    scheduler_adapter.check_lookup_result.return_value = lookup_hit
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._writeback_backlog = {}
    connector._writeback_supported = False
    connector._hybrid_coverage = False
    connector._preserved_store_frontiers = {}
    connector.request_trackers = {}
    connector.scheduler_adapter = scheduler_adapter
    connector._reserve_last_token_for_lookup = True
    connector._save_unfull_chunk = True
    connector._hit_alignment_tokens = 16
    connector._unfull_chunk_grid = GRID
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False
    connector._prefix_replay_tokens = 0
    connector._full_hit_holds_back_last_token = False
    return connector, scheduler_adapter


def _request(
    num_tokens: int | None = None,
    max_tokens: int = 16,
    prompt_tokens: int = PROMPT,
) -> SimpleNamespace:
    num_tokens = prompt_tokens if num_tokens is None else num_tokens
    return SimpleNamespace(
        request_id="req-0",
        resumable=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        cache_salt="",
        prompt_token_ids=list(range(prompt_tokens)),
        all_token_ids=ConstantList(list(range(num_tokens))),
        num_prompt_tokens=prompt_tokens,
        mm_features=[],
        lora_request=None,
        sampling_params=SimpleNamespace(extra_args=None, max_tokens=max_tokens),
    )


def _tracker(
    num_tokens: int | None = None,
    max_tokens: int = 16,
    prompt_tokens: int = PROMPT,
) -> LMCacheMPRequestTracker:
    num_tokens = prompt_tokens if num_tokens is None else num_tokens
    tracker = LMCacheMPRequestTracker(_request(num_tokens, max_tokens, prompt_tokens))
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
    stores = LMCacheMPRequestMetadata.GetUnfullChunkStoreMetadata(
        tracker,
        CHUNK,
        GROUP_TOKENS_PER_BLOCK,
        GRID,
        boundary_state_blocks,
        boundary_state_groups,
    )
    return [meta for meta, _block_ids in stores]


def test_hybrid_writeback_targets_only_attention_after_block_adoption():
    tracker = _tracker()
    tracker.writeback.observe(CHUNK, 2 * CHUNK, group_id=0)
    replacement = {gid: list(ids) for gid, ids in tracker.allocated_block_ids.items()}
    replacement[0][8:16] = list(range(1000, 1008))
    tracker.replace_block_ids(tuple(replacement[gid] for gid in range(2)))
    meta, ticket = LMCacheMPRequestMetadata.GetWritebackMetadata(
        tracker,
        (CHUNK, 2 * CHUNK),
        GROUP_TOKENS_PER_BLOCK,
        group_id=0,
    )
    assert meta.op.block_ids == [list(range(1000, 1008)), [0]]
    assert ticket.group_id == 0
    assert tracker.allocated_block_ids[1] == MAMBA_BLOCKS


@pytest.mark.parametrize("state_groups", [1, 3])
@pytest.mark.parametrize("healthy_checkpoint", [False, True])
@pytest.mark.parametrize("repair_policy", [False, True])
def test_hybrid_delivery_reports_attention_hole_and_healthy_checkpoint(
    state_groups, healthy_checkpoint, repair_policy
):
    from lmcache.v1.common.custom_types import RetrieveCoverage, RetrieveGroupCoverage
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    connector, _ = _connector(0)
    connector._hybrid_coverage = True
    connector._repair_policy = repair_policy
    connector._report_request_errors = True
    connector._can_store = True
    connector._delay_free_req_ids = set()
    connector._repair_reports = []
    connector._repair_attestations = []
    connector._repair_represented = set()
    connector._hybrid_attention_groups = {0: 2}
    connector._hybrid_attention_layers = {0: [0, 2]}
    connector._hybrid_recurrent_groups = tuple(range(1, state_groups + 1))
    connector._group_tokens_per_block = [16] + [128] * state_groups
    connector._repair_attempts = {
        "req": LMCacheMPRequestMetadata(
            "req",
            "RETRIEVE",
            LoadStoreOp(
                token_bytes=b"",
                block_ids=[list(range(1, 25))]
                + [[0, 0, 99 + i] for i in range(state_groups)],
                start=0,
                end=384,
            ),
            transfer_generation=7,
            request_configs={"recomputed_generation": "storage-gen"},
        )
    }
    coverage = RetrieveCoverage(
        3,
        [0, 2] if healthy_checkpoint else [0],
        [1] if healthy_checkpoint else [1, 2],
        groups=[
            RetrieveGroupCoverage(0, [0], "attention", 0, [0, 2], [1]),
            RetrieveGroupCoverage(
                1,
                list(range(1, state_groups + 1)),
                "recurrent",
                2,
                [2] if healthy_checkpoint else [],
                [] if healthy_checkpoint else [2],
            ),
        ],
    )
    worker = connector.worker_adapter = MagicMock()
    worker.lmcache_tokens_per_chunk = 128
    worker.get_finished.return_value = (set(), {"req"})
    worker.get_failed_request_ids.return_value = {"req"}
    worker.get_retrieve_completions.return_value = [("req", coverage)]
    worker.get_writeback_results.return_value = ({}, {})
    transfers = connector.get_transfer_results(set())
    metadata = connector.build_connector_worker_meta()
    if healthy_checkpoint:
        assert metadata.preserved_store_frontiers == {"req": ("storage-gen", 384)}
    else:
        assert metadata is None
    if not repair_policy:
        assert transfers.failed_recving == {"req"}
        assert connector.get_load_failure_reports() == []
        assert connector.get_load_attestations() == []
        return
    if not healthy_checkpoint:
        assert transfers.failed_recving == {"req"}
        assert connector.get_load_failure_reports() == []
        assert connector.get_load_attestations() == []
        return
    assert transfers.finished_recving == {"req"} and not transfers.failed_recving
    (report,) = connector.get_load_failure_reports()
    assert report.failed_blocks_by_group == {0: list(range(9, 17))}
    assert report.observed_failed_layers == [0, 2]
    assert report.token_span == (128, 256) and report.rollback_boundary == 0
    assert (
        report.healthy_recurrent_group_ids == tuple(range(1, state_groups + 1))
        and report.transfer_generation == 7
    )
    (attestation,) = connector.get_load_attestations()
    assert (
        attestation.expected_objects == 48 + state_groups
        and attestation.failed_objects == 16
    )
    assert attestation.certifies_content()


@pytest.mark.parametrize("proof", ["current", "stale", "unknown"])
def test_suffix_recompute_keeps_covered_store_frontier_and_only_writes_hole(proof):
    from vllm.v1.outputs import KVConnectorOutput
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata

    connector, adapter = _connector(2 * CHUNK)
    connector._writeback_supported = connector._hybrid_coverage = True
    connector._gpu_block_pool = None
    connector._pending_writebacks = {}
    adapter.world_size = 1
    request = _request()
    connector.get_num_new_matched_tokens(request, 0)
    tracker = connector.request_trackers[request.request_id]
    tracker.state = LMCacheMPRequestState.READY
    generation = tracker.writeback_generation
    receipts = (
        {}
        if proof == "unknown"
        else {
            request.request_id: (generation if proof == "current" else "old", 2 * CHUNK)
        }
    )
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                {},
                failed_group_load_ranges={
                    request.request_id: [(0, 0, CHUNK, generation)]
                },
                preserved_store_frontiers=receipts,
            )
        )
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert tracker.num_stored_tokens == (2 * CHUNK if proof == "current" else 0)
    if proof != "current":
        return
    # The bypass deliberately drops stale delivery destinations. The next
    # ordinary allocation supplies the recompute's physical source blocks.
    assert not tracker.allocated_block_ids
    tracker.replace_block_ids((list(range(2000, 2022)), MAMBA_BLOCKS))
    tracker.increase_num_scheduled_tokens(2 * CHUNK)
    request.num_computed_tokens = 2 * CHUNK
    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            GROUP_TOKENS_PER_BLOCK,
        )
        is None
    )
    metadata, _ = LMCacheMPRequestMetadata.GetWritebackMetadata(
        tracker,
        (0, CHUNK),
        GROUP_TOKENS_PER_BLOCK,
        group_id=0,
    )
    assert (metadata.op.start, metadata.op.end) == (0, CHUNK)
    assert metadata.op.block_ids == [list(range(2000, 2008)), [0]]


@pytest.mark.parametrize("whole_load_failed", [False, True])
def test_attention_only_zero_rewind_keeps_frontier_with_independent_dirty_ranges(
    whole_load_failed,
):
    from vllm.v1.outputs import KVConnectorOutput
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata

    connector, _ = _connector(2 * CHUNK)
    connector._writeback_supported = True
    connector._gpu_block_pool = None
    connector._pending_writebacks = {}
    request = _request()
    connector.get_num_new_matched_tokens(request, 0)
    tracker = connector.request_trackers[request.request_id]
    tracker.state = LMCacheMPRequestState.READY
    failed_end = 2 * CHUNK if whole_load_failed else CHUNK
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                {},
                failed_load_ranges={
                    request.request_id: [(0, failed_end, tracker.writeback_generation)]
                },
            )
        )
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert tracker.num_stored_tokens == 2 * CHUNK
    assert tracker.writeback.ready(2 * CHUNK) == [(0, failed_end)]
