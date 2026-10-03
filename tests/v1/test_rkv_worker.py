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
            block_ids=[2, 4, 6],
            resident_kv_tokens=None,
            is_genuine_decode=True,
            should_compress=False,
        ),
        SimpleNamespace(
            request_id="req-a",
            block_ids=[1, 3, 5],
            resident_kv_tokens=33,
            is_genuine_decode=True,
            should_compress=False,
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
            2,
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
            None,
        )

    for request_id in ("req-a", "req-b"):
        for name in LAYER_NAMES:
            assert worker._recent_queries[request_id][name].shape == (
                1,
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
    expected = {}
    for name in LAYER_NAMES:
        keys = (
            original[name][:, 0][slots // BLOCK_SIZE, slots % BLOCK_SIZE]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        values = (
            original[name][:, 1][slots // BLOCK_SIZE, slots % BLOCK_SIZE]
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        recent_queries = (
            queries[name][-WINDOW:].permute(1, 0, 2).unsqueeze(0).contiguous()
        )
        expected[name] = _policy().update_kv(keys, recent_queries, values)

    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)
    metadata = _metadata(block_ids, length, length)
    worker.begin_step(["req"], metadata)
    worker._recent_queries["req"] = {
        name: queries[name][-WINDOW:].clone() for name in LAYER_NAMES
    }

    assert worker.compact() == {"req": BUDGET}

    for name in LAYER_NAMES:
        actual_k = caches[name][:, 0][
            destination_slots // BLOCK_SIZE,
            destination_slots % BLOCK_SIZE,
        ].permute(1, 0, 2).unsqueeze(0)
        actual_v = caches[name][:, 1][
            destination_slots // BLOCK_SIZE,
            destination_slots % BLOCK_SIZE,
        ].permute(1, 0, 2).unsqueeze(0)
        expected_k, expected_v = expected[name]
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
        is_genuine_decode=[False],
        should_compress=[False],
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
    assert worker._recent_queries == {}
    assert worker.compact() == {}

    # First compaction fires after exactly one full buffer of decode tokens.
    for step in range(1, WINDOW + 1):
        metadata = _metadata(block_ids, prompt_len + step, 1)
        worker.begin_step(
            ["req"],
            metadata,
            is_genuine_decode=[True],
            should_compress=[step == WINDOW],
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
            is_genuine_decode=[True],
            should_compress=[step == WINDOW],
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

    expected = {}
    for req_index, request_id in enumerate(request_ids):
        slots = _slots(request_blocks[req_index], length)
        start = req_index * length
        end = start + length
        for name in LAYER_NAMES:
            keys = (
                original[name][:, 0][slots // BLOCK_SIZE, slots % BLOCK_SIZE]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            values = (
                original[name][:, 1][slots // BLOCK_SIZE, slots % BLOCK_SIZE]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            recent_queries = (
                queries_by_layer[name][start:end][-WINDOW:]
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            expected[(request_id, name)] = _policy().update_kv(
                keys, recent_queries, values
            )

    worker = RKVWorker(BUDGET, buffer=WINDOW)
    worker.register_kv_caches(caches)
    worker.begin_step(request_ids, metadata)
    worker._recent_queries = {
        request_id: {
            name: queries_by_layer[name][
                req_index * length : (req_index + 1) * length
            ][-WINDOW:].clone()
            for name in LAYER_NAMES
        }
        for req_index, request_id in enumerate(request_ids)
    }

    assert worker.compact() == {"req-a": BUDGET, "req-b": BUDGET}
    assert worker._n_compactions == 2

    for req_index, request_id in enumerate(request_ids):
        slots = _slots(request_blocks[req_index], length)[:BUDGET]
        for name in LAYER_NAMES:
            actual_k = caches[name][:, 0][
                slots // BLOCK_SIZE, slots % BLOCK_SIZE
            ].permute(1, 0, 2).unsqueeze(0)
            actual_v = caches[name][:, 1][
                slots // BLOCK_SIZE, slots % BLOCK_SIZE
            ].permute(1, 0, 2).unsqueeze(0)
            expected_k, expected_v = expected[(request_id, name)]
            assert torch.equal(actual_k, expected_k)
            assert torch.equal(actual_v, expected_v)



def test_rkv_worker_defaults_match_upstream_vllm_config():
    worker = RKVWorker(BUDGET)
    assert worker.buffer == 128
    assert worker.window_size == 8
    assert worker.kernel_size == 7
    assert worker.mix_lambda == 0.1
    assert worker.retain_ratio == 0.1
    assert worker.retain_direction == "last"
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
        is_genuine_decode=[False],
        should_compress=[True],
    )
    prefill = torch.randn(
        4, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    for name in LAYER_NAMES:
        worker.capture_query(name, prefill)
    assert worker._recent_queries == {}
    assert worker.compact() == {}

    worker.begin_step(
        ["req"],
        metadata,
        is_genuine_decode=[True],
        should_compress=[False],
    )
    for name in LAYER_NAMES:
        worker.capture_query(name, prefill)
        assert torch.equal(worker._recent_queries["req"][name], prefill[-1:])
