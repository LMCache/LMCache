# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

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


def _register_rkv(
    worker: RKVWorker,
    request_id: str = "req",
    *,
    budget: int = BUDGET,
    buffer: int = WINDOW,
    config: dict | None = None,
):
    return worker._algorithm_from_state(
        SimpleNamespace(
            request_id=request_id,
            algorithm="rkv",
            config={
                "budget": budget,
                "buffer": buffer,
                **(config or {}),
            },
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepare_forward_skips_vanilla_step_outside_observation_window():
    worker = RKVWorker()
    worker.register_kv_caches(_new_cache(8))
    layers = {name: _FakeAttention(name) for name in LAYER_NAMES}
    worker.install_query_hooks(layers)
    assert worker._query_hooks_installed

    state = SimpleNamespace(
        request_id="req",
        algorithm="rkv",
        config={"budget": BUDGET, "buffer": 16},
        resident_kv_tokens=33,
        has_physical_override=True,
        is_genuine_decode=True,
        num_decoded_tokens=1,
        num_new_tokens=1,
    )

    # The fast path restores vanilla attention before touching its metadata.
    worker.prepare_forward(SimpleNamespace(attn_metadata=None), [state])

    assert worker._seq_lens is None
    assert worker._block_table is None
    assert not worker._query_hooks_installed
    for layer in layers.values():
        assert layer.impl.forward.__func__ is _FakeAttentionImpl.forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_begin_step_only_plans_queries_marked_for_observation():
    worker = RKVWorker()
    _register_rkv(worker, buffer=16)
    worker.register_kv_caches(_new_cache(8))
    metadata = _metadata([1, 3, 5], BUDGET + WINDOW, 1)

    worker.begin_step(
        ["req"],
        metadata,
        physical_seq_lens=[BUDGET + WINDOW],
        observe_query=[False],
        is_genuine_decode=[True],
        num_decoded_tokens=[1],
        num_new_tokens=[1],
    )
    assert worker._query_last_indices is None
    assert worker._query_write_indices is None
    assert "req" not in worker._query_counts

    worker.begin_step(
        ["req"],
        metadata,
        physical_seq_lens=[BUDGET + WINDOW],
        observe_query=[True],
        is_genuine_decode=[True],
        num_decoded_tokens=[9],
        num_new_tokens=[1],
    )
    assert worker._query_last_indices is not None
    assert worker._query_write_indices is not None
    assert worker._query_counts["req"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepare_forward_uses_adaptor_row_order_and_only_captures_queries():
    caches = _new_cache(8)
    worker = RKVWorker()
    worker.register_kv_caches(caches)

    block_table = torch.tensor([[1, 3, 5], [2, 4, 6]], device="cuda")
    query_start_loc = torch.tensor([0, 1, 2], device="cuda")
    physical_seq_lens = torch.tensor([33, 40], device="cuda")
    slot_mapping = torch.tensor([5 * BLOCK_SIZE, 6 * BLOCK_SIZE + 7], device="cuda")

    attn_metadata = {
        name: SimpleNamespace(
            use_cascade=False,
            query_start_loc=query_start_loc,
            seq_lens=physical_seq_lens,
            max_seq_len=105,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )
        for name in LAYER_NAMES
    }
    layers = {name: _FakeAttention(name) for name in LAYER_NAMES}
    context = SimpleNamespace(
        attn_metadata=attn_metadata,
        no_compile_layers=layers,
        slot_mapping={name: slot_mapping for name in LAYER_NAMES},
    )

    # The worker adaptor has already ordered scheduler metadata by
    # GPUModelRunner.input_batch.req_ids and built the physical KV view.
    states = [
        SimpleNamespace(
            request_id="req-a",
            algorithm="rkv",
            config={"budget": BUDGET, "buffer": WINDOW},
            resident_kv_tokens=33,
            has_physical_override=True,
            is_genuine_decode=True,
            num_decoded_tokens=8,
            num_new_tokens=1,
            worker_row=0,
        ),
        SimpleNamespace(
            request_id="req-b",
            algorithm="rkv",
            config={"budget": BUDGET, "buffer": WINDOW},
            resident_kv_tokens=40,
            has_physical_override=True,
            is_genuine_decode=True,
            num_decoded_tokens=8,
            num_new_tokens=1,
            worker_row=1,
        ),
    ]

    worker.prepare_forward(context, states)

    assert worker._request_ids == ["req-a", "req-b"]
    for name in LAYER_NAMES:
        metadata = context.attn_metadata[name]
        assert metadata.seq_lens is physical_seq_lens
        assert metadata.max_seq_len == 105
        assert metadata.block_table is block_table
        assert metadata.slot_mapping is slot_mapping

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
def test_token_drop_worker_applies_kept_positions_and_reports_absolute_p():
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
    kept = torch.cat(
        [
            torch.arange(0, 15, device="cuda"),
            torch.arange(25, 40, device="cuda"),
        ]
    )
    destination_slots = slots[: kept.numel()]
    source_slots = slots[kept]

    worker = RKVWorker()
    algorithm = _register_rkv(worker)

    def select_kept_positions(layer_keys, layer_queries):
        assert len(layer_keys) == len(LAYER_NAMES)
        assert len(layer_queries) == len(LAYER_NAMES)
        assert all(keys.shape == (1, KV_HEADS, length, HEAD_DIM) for keys in layer_keys)
        assert all(
            queries.shape == (1, Q_HEADS, WINDOW, HEAD_DIM)
            for queries in layer_queries
        )
        return kept

    algorithm.select_kept_positions = select_kept_positions
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

    assert worker.compact() == {"req": kept.numel()}

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
    worker = RKVWorker()
    _register_rkv(worker)
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

    kept_by_request = {
        "req-a": torch.cat(
            [
                torch.arange(0, 16, device="cuda"),
                torch.arange(24, 40, device="cuda"),
            ]
        ),
        "req-b": torch.cat(
            [
                torch.arange(0, 8, device="cuda"),
                torch.arange(16, 40, device="cuda"),
            ]
        ),
    }
    expected_sources = {
        request_id: _slots(request_blocks[req_index], length)[
            kept_by_request[request_id]
        ]
        for req_index, request_id in enumerate(request_ids)
    }

    worker = RKVWorker()
    for request_id in request_ids:
        algorithm = _register_rkv(worker, request_id)
        kept = kept_by_request[request_id]
        algorithm.select_kept_positions = lambda *_args, kept=kept: kept
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
def test_token_drop_worker_propagates_algorithm_selection_failure():
    length = BUDGET + WINDOW
    caches = _new_cache(8)
    worker = RKVWorker()
    algorithm = _register_rkv(worker)
    worker.register_kv_caches(caches)
    worker.begin_step(
        ["req"],
        _metadata([1, 3, 5], length, 1),
        physical_seq_lens=[length],
        is_genuine_decode=[True],
        num_decoded_tokens=[WINDOW],
        num_new_tokens=[1],
    )
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

    def reject_selection(*_args):
        raise RuntimeError("algorithm selection failed")

    algorithm.select_kept_positions = reject_selection

    with pytest.raises(RuntimeError, match="algorithm selection failed"):
        worker.compact()


def test_normal_registration_does_not_require_rkv_compatible_kv_layout():
    worker = RKVWorker()
    worker.register_kv_caches({"layer": torch.empty(2, 8, 4, 1, 8)})

    assert worker._kv_caches_validated is False

    state = SimpleNamespace(
        request_id="td",
        algorithm="rkv",
        config={"budget": 32, "buffer": 16},
        resident_kv_tokens=33,
        has_physical_override=False,
        is_genuine_decode=True,
        num_decoded_tokens=1,
        num_new_tokens=1,
    )
    with pytest.raises(ValueError, match="FlashAttention KV shaped"):
        worker.prepare_forward(SimpleNamespace(attn_metadata=None), [state])


def test_request_local_rkv_configs_stay_independent():
    worker = RKVWorker()
    state_a = SimpleNamespace(
        request_id="a",
        algorithm="rkv",
        config={"budget": 32, "buffer": 16, "window_size": 4},
    )
    state_b = SimpleNamespace(
        request_id="b",
        algorithm="rkv",
        config={"budget": 48, "buffer": 24, "window_size": 8},
    )

    rkv_a = worker._algorithm_from_state(state_a)
    rkv_b = worker._algorithm_from_state(state_b)

    assert rkv_a is not rkv_b
    assert worker._algorithm_for_request("a") is rkv_a
    assert worker._algorithm_for_request("b") is rkv_b


def test_live_request_reuses_algorithm_until_request_reset():
    worker = RKVWorker()
    state = SimpleNamespace(
        request_id="req",
        algorithm="rkv",
        config={"budget": 32, "buffer": 16},
    )

    first = worker._algorithm_from_state(state)
    assert worker._algorithm_from_state(state) is first

    worker.drop_requests({"req"})
    assert worker._algorithm_from_state(state) is not first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_different_rkv_configs_compact_in_same_batch():
    worker = RKVWorker()
    worker.register_kv_caches(_new_cache(10))

    states = [
        SimpleNamespace(
            request_id="a",
            algorithm="rkv",
            config={"budget": 32, "buffer": 8, "window_size": 4},
        ),
        SimpleNamespace(
            request_id="b",
            algorithm="rkv",
            config={"budget": 48, "buffer": 8, "window_size": 8},
        ),
    ]
    for state in states:
        worker._algorithm_from_state(state)

    metadata = SimpleNamespace(
        use_cascade=False,
        query_start_loc=torch.tensor([0, 1, 2], device="cuda"),
        seq_lens=torch.tensor([40, 56], device="cuda"),
        block_table=torch.tensor(
            [[1, 3, 5, 0], [2, 4, 6, 7]],
            device="cuda",
        ),
    )
    worker.begin_step(
        ["a", "b"],
        metadata,
        physical_seq_lens=[40, 56],
        request_rows=[0, 1],
        is_genuine_decode=[True, True],
        num_decoded_tokens=[8, 8],
        num_new_tokens=[1, 1],
    )
    _seed_query_windows(
        worker,
        {
            request_id: {
                name: torch.randn(
                    WINDOW,
                    Q_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for name in LAYER_NAMES
            }
            for request_id in ("a", "b")
        },
    )

    assert worker.compact() == {"a": 32, "b": 48}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rkv_worker_records_only_genuine_decode_frontier_query():
    length = BUDGET + WINDOW
    block_ids = [1, 3, 5]
    caches = _new_cache(8)
    worker = RKVWorker()
    _register_rkv(worker)
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
