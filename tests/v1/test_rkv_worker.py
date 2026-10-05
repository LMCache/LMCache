# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from lmcache.integration.vllm.experimental.rkv import select_rkv_retained_indices
from lmcache.integration.vllm.experimental.rkv_worker import RKVWorker

BLOCK_SIZE = 16
BUDGET = 32
WINDOW = 8
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 8
LAYER_NAMES = ["layer.0", "layer.1"]


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
    worker = RKVWorker(BUDGET)
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
    layers = {name: torch.nn.Identity() for name in LAYER_NAMES}
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
        ),
        SimpleNamespace(
            request_id="req-a",
            block_ids=[1, 3, 5],
            resident_kv_tokens=33,
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
        layers[name](query)

    for request_id in ("req-a", "req-b"):
        for name in LAYER_NAMES:
            assert worker._recent_queries[request_id][name].shape == (
                1,
                Q_HEADS,
                HEAD_DIM,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_compacts_all_layers_to_selected_survivors():
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
    expected_retained = select_rkv_retained_indices(
        [(original[name][:, 0], queries[name][-WINDOW:]) for name in LAYER_NAMES],
        slots,
        BUDGET,
    )
    source_slots = slots[expected_retained]
    destination_slots = slots[:BUDGET]

    worker = RKVWorker(BUDGET)
    worker.register_kv_caches(caches)
    metadata = _metadata(block_ids, length, length)
    worker.begin_step(["req"], metadata)
    for name in LAYER_NAMES:
        worker.capture_query(name, queries[name])

    assert worker.compact() == {"req": BUDGET}

    for name in LAYER_NAMES:
        for kv_index in (0, 1):
            expected = original[name][:, kv_index][
                source_slots // BLOCK_SIZE,
                source_slots % BLOCK_SIZE,
            ]
            actual = caches[name][:, kv_index][
                destination_slots // BLOCK_SIZE,
                destination_slots % BLOCK_SIZE,
            ]
            assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_recompacts_after_one_fresh_query_window():
    block_ids = [1, 3, 5]
    caches = _new_cache(8)
    worker = RKVWorker(BUDGET)
    worker.register_kv_caches(caches)

    # First compaction can use the trailing query window from the prompt.
    prompt_len = BUDGET + WINDOW
    prompt_metadata = _metadata(block_ids, prompt_len, prompt_len)
    worker.begin_step(["req"], prompt_metadata)
    for name in LAYER_NAMES:
        query = torch.randn(
            prompt_len,
            Q_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        worker.capture_query(name, query)
    assert worker.compact() == {"req": BUDGET}

    # After reclaim, eight new one-token decode steps rebuild a fresh window.
    for step in range(1, WINDOW + 1):
        length = BUDGET + step
        metadata = _metadata(block_ids, length, 1)
        worker.begin_step(["req"], metadata)
        for name in LAYER_NAMES:
            query = torch.randn(
                1,
                Q_HEADS,
                HEAD_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            worker.capture_query(name, query)

        updates = worker.compact()
        if step < WINDOW:
            assert updates == {}
        else:
            assert updates == {"req": BUDGET}


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
    block_table = torch.tensor(request_blocks, device="cuda")
    metadata = SimpleNamespace(
        use_cascade=False,
        query_start_loc=torch.tensor([0, length, 2 * length], device="cuda"),
        seq_lens=torch.tensor([length, length], device="cuda"),
        block_table=block_table,
    )

    expected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for req_index, request_id in enumerate(request_ids):
        slots = _slots(request_blocks[req_index], length)
        start = req_index * length
        end = start + length
        retained = select_rkv_retained_indices(
            [
                (
                    original[name][:, 0],
                    queries_by_layer[name][start:end][-WINDOW:],
                )
                for name in LAYER_NAMES
            ],
            slots,
            BUDGET,
        )
        expected[request_id] = (slots[retained], slots[:BUDGET])

    worker = RKVWorker(BUDGET)
    worker.register_kv_caches(caches)
    worker.begin_step(request_ids, metadata)
    for name in LAYER_NAMES:
        worker.capture_query(name, queries_by_layer[name])

    assert worker.compact() == {"req-a": BUDGET, "req-b": BUDGET}

    for req_index, request_id in enumerate(request_ids):
        source_slots, destination_slots = expected[request_id]
        for name in LAYER_NAMES:
            for kv_index in (0, 1):
                expected_values = original[name][:, kv_index][
                    source_slots // BLOCK_SIZE,
                    source_slots % BLOCK_SIZE,
                ]
                actual_values = caches[name][:, kv_index][
                    destination_slots // BLOCK_SIZE,
                    destination_slots % BLOCK_SIZE,
                ]
                assert torch.equal(actual_values, expected_values)
