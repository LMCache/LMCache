# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from rkv import R1KV

from lmcache.integration.vllm.experimental.rkv_worker import RKVWorker

BLOCK_SIZE = 16
BUDGET = 32
WINDOW = 8
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 8
LAYER_NAMES = ["layer.0", "layer.1"]


class _FakeAttentionImpl:
    def forward(
        self,
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        *args,
        **kwargs,
    ):
        return None


class _FakeAttention:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.impl = _FakeAttentionImpl()


def _policy():
    return R1KV(
        budget=BUDGET,
        window_size=WINDOW,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )



def _slots(block_ids: list[int], length: int) -> torch.Tensor:
    positions = torch.arange(length, device="cuda")
    blocks = torch.tensor(block_ids, device="cuda")
    return blocks[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE


def _metadata(block_ids: list[int], length: int, query_len: int):
    return SimpleNamespace(
        use_cascade=False,
        query_start_loc=torch.tensor([0, query_len], device="cuda"),
        seq_lens=torch.tensor([length], device="cuda"),
        block_table=torch.tensor([block_ids], device="cuda"),
    )


def _new_cache(num_blocks: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        name: torch.randn(
            num_blocks,
            2,
            BLOCK_SIZE,
            KV_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for name in LAYER_NAMES
    }


def _shared_kept_indices(
    original: dict[str, torch.Tensor],
    slots: torch.Tensor,
    recent_queries: dict[str, torch.Tensor],
) -> torch.Tensor:
    policy = _policy()
    shared_scores = None
    for name in LAYER_NAMES:
        keys = (
            original[name][:, 0][slots // BLOCK_SIZE, slots % BLOCK_SIZE]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        queries = (
            recent_queries[name]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        layer_score = policy.score_kv(keys, queries).mean(dim=1)
        shared_scores = (
            layer_score if shared_scores is None else shared_scores + layer_score
        )

    assert shared_scores is not None
    past_idx = shared_scores.topk(BUDGET - WINDOW, dim=-1).indices
    window_idx = torch.arange(
        slots.numel() - WINDOW,
        slots.numel(),
        device=slots.device,
    ).expand(1, WINDOW)
    return torch.sort(torch.cat([past_idx, window_idx], dim=-1), dim=-1).values[0]


def _seed_query_windows(
    worker: RKVWorker,
    windows: dict[str, dict[str, torch.Tensor]],
) -> None:
    worker._query_slots.clear()
    worker._query_counts.clear()
    worker._free_query_slots.clear()
    worker._query_rings.clear()

    worker._next_query_slot = len(windows)
    worker._query_ring_width = max(16, worker._next_query_slot)
    for slot, (request_id, by_layer) in enumerate(windows.items()):
        worker._query_slots[request_id] = slot
        worker._query_counts[request_id] = WINDOW
        for layer_name, queries in by_layer.items():
            ring = worker._query_rings.get(layer_name)
            if ring is None:
                ring = queries.new_zeros(
                    WINDOW,
                    worker._query_ring_width,
                    queries.shape[1],
                    queries.shape[2],
                )
                worker._query_rings[layer_name] = ring
            ring[:, slot] = queries


def _query_window(
    worker: RKVWorker,
    request_id: str,
    layer_name: str,
) -> torch.Tensor:
    return worker._query_rings[layer_name][:, worker._query_slots[request_id]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepare_forward_builds_physical_view_and_captures_queries():
    caches = _new_cache(8)
    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)

    block_table = torch.tensor([[1, 3, 5], [2, 4, 6]], device="cuda")
    query_start_loc = torch.tensor([0, 1, 2], device="cuda")
    logical_seq_lens = torch.tensor([105, 40], device="cuda")
    old_slot_mapping = torch.tensor([0, 0], device="cuda")

    attn_metadata = {
        name: SimpleNamespace(
            use_cascade=False,
            query_start_loc=query_start_loc,
            seq_lens=logical_seq_lens.clone(),
            max_seq_len=105,
            block_table=block_table.clone(),
            slot_mapping=old_slot_mapping.clone(),
        )
        for name in LAYER_NAMES
    }
    layers = {name: _FakeAttention(name) for name in LAYER_NAMES}
    context = SimpleNamespace(
        attn_metadata=attn_metadata,
        no_compile_layers=layers,
        slot_mapping={name: old_slot_mapping.clone() for name in LAYER_NAMES},
    )

    # Deliberately reverse metadata order. Worker rows must be matched by their
    # authoritative first block, not by scheduler metadata order.
    states = [
        SimpleNamespace(
            request_id="req-b",
            first_block_id=2,
            resident_kv_tokens=40,
            is_genuine_decode=True,
            num_decoded_tokens=0,
            num_new_tokens=1,
        ),
        SimpleNamespace(
            request_id="req-a",
            first_block_id=1,
            resident_kv_tokens=33,
            is_genuine_decode=True,
            num_decoded_tokens=0,
            num_new_tokens=1,
        ),
    ]

    worker.prepare_forward(context, states)

    expected_seq_lens = torch.tensor([33, 40], device="cuda")
    expected_slots = torch.tensor([5 * BLOCK_SIZE, 6 * BLOCK_SIZE + 7], device="cuda")
    expected_blocks = torch.tensor([[1, 3, 5], [2, 4, 6]], device="cuda")

    assert worker._request_ids == ["req-a", "req-b"]
    for name in LAYER_NAMES:
        metadata = context.attn_metadata[name]
        assert torch.equal(metadata.seq_lens, expected_seq_lens)
        assert metadata.max_seq_len == 40
        assert torch.equal(metadata.block_table[:2, :3], expected_blocks)
        assert torch.equal(metadata.slot_mapping, expected_slots)
        assert torch.equal(context.slot_mapping[name], expected_slots)

        query = torch.randn(
            4,
            Q_HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        layers[name].impl.forward(
            layers[name],
            query,
            None,
            None,
            None,
            SimpleNamespace(num_actual_tokens=2),
        )

    for request_id in ("req-a", "req-b"):
        assert worker._query_counts[request_id] == 1
        for name in LAYER_NAMES:
            assert _query_window(worker, request_id, name).shape == (
                WINDOW,
                Q_HEADS,
                HEAD_DIM,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_compacts_all_layers_matches_upstream_update_kv():
    length = BUDGET + WINDOW
    block_ids = [1, 3, 5]
    caches = _new_cache(8)
    original = {name: cache.clone() for name, cache in caches.items()}
    queries = {
        name: torch.randn(
            length,
            Q_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for name in LAYER_NAMES
    }

    slots = _slots(block_ids, length)
    destination_slots = slots[:BUDGET]
    recent_queries = {
        name: queries[name][-WINDOW:].clone() for name in LAYER_NAMES
    }
    kept = _shared_kept_indices(original, slots, recent_queries)
    source_slots = slots[kept]

    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)
    metadata = _metadata(block_ids, length, length)
    worker.begin_step(
        ["req"],
        metadata,
        physical_seq_lens=[length],
        is_genuine_decode=[True],
        num_decoded_tokens=[WINDOW],
        num_new_tokens=[1],
    )
    _seed_query_windows(
        worker,
        {
            "req": {
                name: queries[name][-WINDOW:].clone() for name in LAYER_NAMES
            }
        },
    )

    assert worker.compact() == {"req": BUDGET}

    for name in LAYER_NAMES:
        actual_k = caches[name][:, 0][
            destination_slots // BLOCK_SIZE,
            destination_slots % BLOCK_SIZE,
        ]
        actual_v = caches[name][:, 1][
            destination_slots // BLOCK_SIZE,
            destination_slots % BLOCK_SIZE,
        ]
        expected_k = original[name][:, 0][
            source_slots // BLOCK_SIZE,
            source_slots % BLOCK_SIZE,
        ]
        expected_v = original[name][:, 1][
            source_slots // BLOCK_SIZE,
            source_slots % BLOCK_SIZE,
        ]
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_recompacts_on_decode_buffer_boundaries():
    block_ids = [1, 3, 5]
    caches = _new_cache(8)
    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)

    # Initial prefill never contributes to the observation window or compacts.
    prompt_len = BUDGET + WINDOW
    prompt_metadata = _metadata(block_ids, prompt_len, prompt_len)
    worker.begin_step(
        ["req"],
        prompt_metadata,
        physical_seq_lens=[prompt_len],
        is_genuine_decode=[False],
        num_decoded_tokens=[0],
        num_new_tokens=[prompt_len],
    )
    for name in LAYER_NAMES:
        query = torch.randn(
            prompt_len,
            Q_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        worker.capture_query(name, query)
    assert worker._query_counts.get("req", 0) == 0
    assert worker.compact() == {}

    # First compaction fires after exactly one full buffer of decode tokens.
    for step in range(1, WINDOW + 1):
        metadata = _metadata(block_ids, prompt_len + step, 1)
        worker.begin_step(
            ["req"],
            metadata,
            physical_seq_lens=[prompt_len + step],
            is_genuine_decode=[True],
            num_decoded_tokens=[step],
            num_new_tokens=[1],
        )
        for name in LAYER_NAMES:
            worker.capture_query(
                name,
                torch.randn(
                    1,
                    Q_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            )
        assert worker.compact() == (
            {"req": BUDGET} if step == WINDOW else {}
        )

    # After reclaim, the next buffer boundary compacts again.
    for step in range(1, WINDOW + 1):
        metadata = _metadata(block_ids, BUDGET + step, 1)
        worker.begin_step(
            ["req"],
            metadata,
            physical_seq_lens=[BUDGET + step],
            is_genuine_decode=[True],
            num_decoded_tokens=[WINDOW + step],
            num_new_tokens=[1],
        )
        for name in LAYER_NAMES:
            worker.capture_query(
                name,
                torch.randn(
                    1,
                    Q_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            )
        assert worker.compact() == (
            {"req": BUDGET} if step == WINDOW else {}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_compacts_two_requests_independently():
    length = BUDGET + WINDOW
    request_ids = ["req-a", "req-b"]
    request_blocks = [[1, 3, 5], [2, 4, 6]]
    caches = _new_cache(8)
    original = {name: cache.clone() for name, cache in caches.items()}

    queries_by_layer = {
        name: torch.randn(
            2 * length,
            Q_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for name in LAYER_NAMES
    }
    metadata = SimpleNamespace(
        use_cascade=False,
        query_start_loc=torch.tensor([0, length, 2 * length], device="cuda"),
        seq_lens=torch.tensor([length, length], device="cuda"),
        block_table=torch.tensor(request_blocks, device="cuda"),
    )

    expected_sources = {}
    for req_index, request_id in enumerate(request_ids):
        slots = _slots(request_blocks[req_index], length)
        start = req_index * length
        end = start + length
        recent_queries = {
            name: queries_by_layer[name][start:end][-WINDOW:].clone()
            for name in LAYER_NAMES
        }
        kept = _shared_kept_indices(original, slots, recent_queries)
        expected_sources[request_id] = slots[kept]

    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)
    worker.begin_step(
        request_ids,
        metadata,
        physical_seq_lens=[length, length],
        is_genuine_decode=[True, True],
        num_decoded_tokens=[WINDOW, WINDOW],
        num_new_tokens=[1, 1],
    )
    _seed_query_windows(
        worker,
        {
            request_id: {
                name: queries_by_layer[name][
                    req_index * length : (req_index + 1) * length
                ][-WINDOW:].clone()
                for name in LAYER_NAMES
            }
            for req_index, request_id in enumerate(request_ids)
        },
    )

    assert worker.compact() == {"req-a": BUDGET, "req-b": BUDGET}

    for req_index, request_id in enumerate(request_ids):
        destination_slots = _slots(request_blocks[req_index], length)[:BUDGET]
        source_slots = expected_sources[request_id]
        for name in LAYER_NAMES:
            actual_k = caches[name][:, 0][
                destination_slots // BLOCK_SIZE,
                destination_slots % BLOCK_SIZE,
            ]
            actual_v = caches[name][:, 1][
                destination_slots // BLOCK_SIZE,
                destination_slots % BLOCK_SIZE,
            ]
            expected_k = original[name][:, 0][
                source_slots // BLOCK_SIZE,
                source_slots % BLOCK_SIZE,
            ]
            expected_v = original[name][:, 1][
                source_slots // BLOCK_SIZE,
                source_slots % BLOCK_SIZE,
            ]
            assert torch.equal(actual_k, expected_k)
            assert torch.equal(actual_v, expected_v)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_score_chunking_preserves_kept_set():
    length = BUDGET + WINDOW
    request_ids = ["req-a", "req-b"]
    request_blocks = [[1, 3, 5], [2, 4, 6]]
    caches = _new_cache(8)
    queries = {
        name: torch.randn(
            2,
            WINDOW,
            Q_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for name in LAYER_NAMES
    }
    members = [
        (i, request_id, _slots(request_blocks[i], length))
        for i, request_id in enumerate(request_ids)
    ]
    windows = {
        request_id: {
            name: queries[name][i].clone() for name in LAYER_NAMES
        }
        for i, request_id in enumerate(request_ids)
    }

    def plan(score_chunk_bytes: int):
        worker = RKVWorker(
            BUDGET,
            buffer=WINDOW,
            score_chunk_bytes=score_chunk_bytes,
        )
        worker.register_kv_caches(
            {name: cache.clone() for name, cache in caches.items()}
        )
        _seed_query_windows(worker, windows)
        return worker._compact_group_batched(members)

    default_source, default_destination = plan(512 * 1024 * 1024)
    per_unit = (
        2
        * (2 * caches[LAYER_NAMES[0]].element_size() + 1 + 4)
        * KV_HEADS
        * length
        * length
    )
    chunked_source, chunked_destination = plan(per_unit)

    assert torch.equal(chunked_source, default_source)
    assert torch.equal(chunked_destination, default_destination)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_rejects_scoring_unit_over_memory_cap():
    length = BUDGET + WINDOW
    caches = _new_cache(8)
    per_unit = (
        2
        * (2 * caches[LAYER_NAMES[0]].element_size() + 1 + 4)
        * KV_HEADS
        * length
        * length
    )
    worker = RKVWorker(
        BUDGET,
        buffer=WINDOW,
        score_chunk_bytes=per_unit - 1,
    )
    worker.register_kv_caches(caches)
    _seed_query_windows(
        worker,
        {
            "req": {
                name: torch.randn(
                    WINDOW,
                    Q_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for name in LAYER_NAMES
            }
        },
    )

    with pytest.raises(RuntimeError, match="memory cap"):
        worker._compact_group_batched(
            [(0, "req", _slots([1, 3, 5], length))]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_rejects_non_finite_scores():
    length = BUDGET + WINDOW
    caches = _new_cache(8)
    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)
    _seed_query_windows(
        worker,
        {
            "req": {
                name: torch.randn(
                    WINDOW,
                    Q_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for name in LAYER_NAMES
            }
        },
    )

    def non_finite_score(keys, queries):
        return keys.new_full(
            (keys.shape[0], keys.shape[1], keys.shape[2] - WINDOW),
            float("nan"),
        )

    worker._policy.score_kv = non_finite_score

    with pytest.raises(RuntimeError, match="non-finite"):
        worker._compact_group_batched(
            [(0, "req", _slots([1, 3, 5], length))]
        )


def test_rkv_worker_defaults_match_upstream_vllm_config():
    worker = RKVWorker(BUDGET)
    assert worker._policy.buffer == 128
    assert worker._policy.window_size == 8
    assert worker._policy.kernel_size == 7
    assert worker._policy.mix_lambda == 0.1
    assert worker._policy.retain_ratio == 0.1
    assert worker._policy.retain_direction == "last"
    assert worker.score_chunk_bytes == 512 * 1024 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_records_only_genuine_decode_frontier_query():
    length = BUDGET + WINDOW
    block_ids = [1, 3, 5]
    caches = _new_cache(8)
    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)

    metadata = _metadata(block_ids, length, 4)
    worker.begin_step(
        ["req"],
        metadata,
        physical_seq_lens=[length],
        is_genuine_decode=[False],
        num_decoded_tokens=[0],
        num_new_tokens=[4],
    )
    prefill = torch.randn(
        4, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    for name in LAYER_NAMES:
        worker.capture_query(name, prefill)
    assert worker._query_counts.get("req", 0) == 0
    assert worker.compact() == {}

    worker.begin_step(
        ["req"],
        metadata,
        physical_seq_lens=[length],
        is_genuine_decode=[True],
        num_decoded_tokens=[1],
        num_new_tokens=[1],
    )
    for name in LAYER_NAMES:
        worker.capture_query(name, prefill)
        assert torch.equal(
            _query_window(worker, "req", name)[0],
            prefill[-1],
        )
