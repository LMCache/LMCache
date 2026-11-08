# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

# First Party
from lmcache.integration.vllm.experimental import token_drop_worker as worker_mod
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    install_token_drop_query_hooks,
    remove_token_drop_query_hooks,
)
from lmcache.integration.vllm.experimental.token_drop_worker import TokenDropWorker
from lmcache.v1.platform import torch_device_type

BLOCK_SIZE = 4
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 8
LAYER_NAMES = ["layer.0", "layer.1"]


class _FakeAlgorithm:
    def __init__(self, *, kept=None, observe=False, compact=False):
        self.kept = kept
        self.observe = observe
        self.compact_now = compact
        self.observations = {}
        self.observation_batches = []
        self.observe_calls = []
        self.compact_calls = []
        self.selection_inputs = []

    def should_observe_token_queries(self, phase, decoded_tokens_before_step):
        facts = {
            "phase": phase,
            "decoded_tokens_before_step": decoded_tokens_before_step,
        }
        self.observe_calls.append(facts)
        result = self.observe(facts) if callable(self.observe) else self.observe
        return int(result)

    def observe_token_queries(self, layer_queries):
        self.observation_batches.append(tuple(layer_queries))
        for layer_name, query in layer_queries.items():
            self.observations.setdefault(layer_name, []).append(query.clone())

    def should_compact_kv(self, phase, resident_kv_tokens, decoded_tokens_before_step):
        facts = {
            "phase": phase,
            "resident_kv_tokens": resident_kv_tokens,
            "decoded_tokens_before_step": decoded_tokens_before_step,
        }
        self.compact_calls.append(facts)
        return (
            self.compact_now(facts) if callable(self.compact_now) else self.compact_now
        )

    def select_kept_token_positions(self, kv_by_layer):
        self.selection_inputs.append(kv_by_layer)
        kept = self.kept(kv_by_layer) if callable(self.kept) else self.kept
        if isinstance(kept, dict):
            return {
                name: torch.as_tensor(
                    kept[name], dtype=torch.long, device=view.get_keys().device
                ).T.contiguous()
                for name, view in kv_by_layer.items()
            }
        return {
            name: torch.as_tensor(kept, dtype=torch.long, device=view.get_keys().device)
            .unsqueeze(1)
            .expand(-1, view.get_keys().shape[1])
            .clone()
            for name, view in kv_by_layer.items()
        }


def _install_algorithms(monkeypatch, algorithms):
    monkeypatch.setattr(
        worker_mod,
        "build_token_drop_algorithm",
        lambda spec: algorithms[spec.config["instance"]],
    )


def _state(
    request_id,
    instance,
    *,
    resident,
    num_new=1,
    decoded=1,
    is_decode=True,
    worker_row=0,
):
    return SimpleNamespace(
        request_id=request_id,
        algorithm="fake",
        config={"instance": instance},
        resident_kv_tokens=resident,
        phase=("decode" if is_decode else "prefill"),
        generation=1,
        num_decoded_tokens=decoded,
        num_new_tokens=num_new,
        worker_row=worker_row,
    )


def _new_cache(num_blocks, *, dtype=torch.bfloat16):
    torch.manual_seed(7)
    return {
        name: torch.randn(
            num_blocks,
            2,
            BLOCK_SIZE,
            KV_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=dtype,
        )
        for name in LAYER_NAMES
    }


def _metadata(block_rows, query_lens):
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    max_blocks = max(len(row) for row in block_rows)
    padded = [row + [0] * (max_blocks - len(row)) for row in block_rows]
    return FlashAttentionMetadata(
        num_actual_tokens=starts[-1],
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor(starts, device="cuda"),
        max_seq_len=max(query_lens),
        seq_lens=torch.tensor(query_lens, device="cuda"),
        block_table=torch.tensor(padded, device="cuda"),
        slot_mapping=torch.zeros(starts[-1], device="cuda", dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _slots(block_ids, length):
    positions = torch.arange(length, device="cuda")
    blocks = torch.tensor(block_ids, device="cuda")
    return blocks[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE


def _read_slots(cache, slots, kv_index):
    return cache[:, kv_index][
        slots // BLOCK_SIZE,
        slots % BLOCK_SIZE,
    ]


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
def test_attention_forward_hook_captures_query_and_restores_original():
    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    worker._query_observation_indices = torch.tensor([0, 2], device="cuda")
    layers = {}
    originals = {}

    for layer_name in LAYER_NAMES:

        def original_forward(attn_layer, query, key, value, kv_cache, metadata):
            return query + 1

        impl = SimpleNamespace(forward=original_forward)
        layers[layer_name] = SimpleNamespace(layer_name=layer_name, impl=impl)
        originals[layer_name] = original_forward

    install_token_drop_query_hooks(worker, layers)
    try:
        assert bool(worker._original_attention_forwards)
        query = torch.arange(
            4 * Q_HEADS * HEAD_DIM, device="cuda", dtype=torch.float32
        ).reshape(4, Q_HEADS, HEAD_DIM)
        meta = SimpleNamespace(num_actual_tokens=3)
        for layer_name, layer in layers.items():
            result = layer.impl.forward(layer, query, query, query, None, meta)
            assert torch.equal(result, query + 1)
            assert torch.equal(worker._observed_queries[layer_name], query[[0, 2]])
    finally:
        remove_token_drop_query_hooks(worker)

    assert not bool(worker._original_attention_forwards)
    for layer_name, layer in layers.items():
        assert layer.impl.forward is originals[layer_name]


def test_algorithm_is_request_scoped_and_resettable(monkeypatch):
    first = _FakeAlgorithm()
    second = _FakeAlgorithm()
    created = [first, second]
    monkeypatch.setattr(
        worker_mod,
        "build_token_drop_algorithm",
        lambda _spec: created.pop(0),
    )

    worker = TokenDropWorker()
    state = _state("req", "unused", resident=8)

    assert worker._algorithm_from_state(state) is first
    assert worker._algorithm_from_state(state) is first
    worker.drop_requests({"req"})
    assert worker._algorithm_from_state(state) is second


def test_normal_registration_does_not_require_token_drop_layout():
    worker = TokenDropWorker()
    worker.register_kv_caches({"layer": torch.empty(2, 8, 4, 1, 8)})
    assert worker._kv_caches_validated is False


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
def test_token_drop_accepts_noncontiguous_kv_layout():
    backing = torch.empty(
        2,
        2,
        BLOCK_SIZE,
        KV_HEADS,
        HEAD_DIM * 2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    noncontiguous = backing[..., ::2]
    assert not noncontiguous.is_contiguous()

    worker = TokenDropWorker()
    worker.register_kv_caches({"layer": noncontiguous})

    worker._ensure_kv_caches_compatible()
    assert worker._kv_caches_validated


def test_observation_error_quarantines_algorithm_state(monkeypatch):
    worker = TokenDropWorker()

    class _PartiallyMutating:
        def __init__(self):
            self.history = []

        def observe_token_queries(self, queries):
            self.history.append(queries)
            raise ValueError("partial observation failure")

    algorithm = _PartiallyMutating()
    worker._request_algorithms["td"] = algorithm
    worker._request_generations["td"] = 3
    worker._request_ids = ["td"]
    worker._layer_names = ["layer.0"]
    worker._query_observation_slices = [("td", 0, 1)]
    worker._observed_queries = {"layer.0": torch.ones(1, 2, 8)}
    worker._compact_now = [True]

    monkeypatch.setattr(worker_mod, "current_stream", lambda device: None)
    monkeypatch.setattr(worker_mod, "synchronize_stream", lambda *args: None)

    worker._flush_query_observations()
    assert len(algorithm.history) == 1
    assert worker._compact_now == [False]
    assert worker._failed_generations == {"td": 3}

    # No KV registration is needed because the corrupted generation must not
    # reach the algorithm/attention paths again.
    state = _state("td", "a", resident=8)
    state.generation = 3
    worker.prepare_forward(SimpleNamespace(), [state])
    assert worker._request_ids == []
    worker.drop_requests({"td"})
    assert worker._failed_generations == {}


def test_incomplete_query_capture_quarantines_history():
    worker = TokenDropWorker()
    worker._request_ids = ["td"]
    worker._request_generations["td"] = 7
    worker._layer_names = ["layer.0", "layer.1"]
    worker._query_observation_slices = [("td", 0, 1)]
    worker._observed_queries = {"layer.0": torch.ones(1, 2, 8)}
    worker._compact_now = [True]
    worker._flush_query_observations()
    assert worker._failed_generations == {"td": 7}
    assert worker._compact_now == [False]


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compatible_kv_cache_dtype_accepted(dtype):
    worker = TokenDropWorker()
    kv = torch.empty(2, 2, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    worker.register_kv_caches({"layer": kv})
    worker._ensure_kv_caches_compatible()
    assert worker._kv_caches_validated


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_unsupported_kv_cache_dtype_rejected(dtype):
    worker = TokenDropWorker()
    kv = torch.empty(2, 2, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    worker.register_kv_caches({"layer": kv})
    with pytest.raises(ValueError, match="FP16 or BF16 KV"):
        worker._ensure_kv_caches_compatible()
    assert not worker._kv_caches_validated


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
@pytest.mark.parametrize("mixed_metadata", [False, True])
@pytest.mark.parametrize("observe, compact", [(True, True), (False, False)])
def test_worker_rejects_non_flash_attention_metadata_before_hooks(
    monkeypatch, mixed_metadata, observe, compact
):
    algorithm = _FakeAlgorithm(observe=observe, compact=compact)
    _install_algorithms(monkeypatch, {"a": algorithm})
    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    request_state = _state("td", "a", resident=8)
    unsupported = SimpleNamespace(
        num_actual_tokens=1, use_cascade=False, slot_mapping=torch.zeros(1)
    )
    attn = {"first": _metadata([[3, 1]], [1])} if mixed_metadata else {}
    attn["unsupported"] = unsupported
    context = SimpleNamespace(
        attn_metadata=attn,
        no_compile_layers={},
    )
    with pytest.raises(ValueError, match="requires FlashAttention metadata"):
        worker.prepare_forward(context, [request_state])
    assert not worker._original_attention_forwards
    assert worker._seq_lens is None
    assert worker._request_ids == []
    assert not worker._observed_queries


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
def test_failed_observation_planning_leaves_no_backend_hook(monkeypatch):
    algorithm = _FakeAlgorithm(observe=True, compact=False)
    _install_algorithms(monkeypatch, {"a": algorithm})
    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    state = _state("td", "a", resident=8, worker_row=7)
    # Row 7 is absent from query_start_loc; begin_step must fail.
    context = SimpleNamespace(
        attn_metadata={"group": _metadata([[3, 1]], [1])},
        no_compile_layers={},
    )
    with pytest.raises(ValueError, match="segmentation mismatch"):
        worker.prepare_forward(context, [state])
    assert not worker._original_attention_forwards
    assert worker._seq_lens is None
    assert worker._request_ids == []


@pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")
def test_reject_negative_observation_before_installing_hooks(monkeypatch):
    algorithm = _FakeAlgorithm(observe=-1, compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})
    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    with pytest.raises(ValueError, match="invalid Q observation count"):
        worker.prepare_forward(
            SimpleNamespace(attn_metadata=None, no_compile_layers={}),
            [_state("td", "a", resident=8, worker_row=0)],
        )
    assert not worker._original_attention_forwards
    assert worker._seq_lens is None
