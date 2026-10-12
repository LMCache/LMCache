# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental import token_drop_worker as worker_module
from lmcache.integration.vllm.experimental.token_drop_worker import TokenDropWorker


class _Algorithm:
    def __init__(self):
        self.observations = []

    def should_observe_token_queries(self, phase, decoded_tokens_before_step):
        return 2 if phase == "prefill" else 0

    def observe_token_queries(self, queries_by_layer):
        self.observations.append(queries_by_layer)


def _environment():
    worker = TokenDropWorker()
    worker.register_kv_caches({"layer0": torch.zeros((3, 2, 4, 1, 8))})
    calls = []

    def forward(*args, **kwargs):
        calls.append(args)
        return args[1]

    implementation = SimpleNamespace(forward=forward)
    ctx = SimpleNamespace(
        attn_metadata={
            "layer0": SimpleNamespace(query_start_loc=torch.tensor([0, 3, 7]))
        },
        no_compile_layers={"layer0": SimpleNamespace(impl=implementation)},
    )
    state = SimpleNamespace(
        request_id="td",
        algorithm="rkv",
        config={"budget": 16},
        worker_row=1,
        num_new_tokens=4,
        num_decoded_tokens=0,
        is_genuine_decode=False,
    )
    return worker, ctx, state, implementation, forward, calls


def test_selective_post_rope_queries_and_request_lifecycle(monkeypatch):
    worker, ctx, state, impl, original, calls = _environment()
    algorithms = []

    def factory(spec):
        algorithm = _Algorithm()
        algorithms.append(algorithm)
        return algorithm

    monkeypatch.setattr(worker_module, "build_token_drop_algorithm", factory)
    query = torch.arange(7 * 2 * 8).reshape(7, 2, 8).float()

    worker.prepare_forward(ctx, [state])
    impl.forward(
        SimpleNamespace(layer_name="layer0"),
        query,
        None,
        None,
        None,
        SimpleNamespace(num_actual_tokens=7),
    )
    assert worker.compact() == {}
    assert len(calls) == 1
    assert len(algorithms) == 1
    assert len(algorithms[0].observations) == 1
    assert torch.equal(algorithms[0].observations[0]["layer0"], query[5:7])
    assert impl.forward is original

    worker.prepare_forward(ctx, [state])
    assert len(algorithms) == 1
    worker.compact()
    worker.drop_requests({"td"})
    assert not worker.is_token_drop_request("td")
    worker.prepare_forward(ctx, [state])
    assert len(algorithms) == 2
    worker.compact()


def test_mixed_batch_delivers_only_each_requests_selected_queries(monkeypatch):
    worker, ctx, first, impl, original, _ = _environment()
    second = SimpleNamespace(
        request_id="td2",
        algorithm="rkv",
        config={},
        worker_row=0,
        num_new_tokens=3,
        num_decoded_tokens=0,
        is_genuine_decode=False,
    )
    algorithms = {}

    def factory(spec):
        algorithm = _Algorithm()
        algorithms[len(algorithms)] = algorithm
        return algorithm

    monkeypatch.setattr(worker_module, "build_token_drop_algorithm", factory)
    query = torch.arange(7 * 2 * 8).reshape(7, 2, 8).float()

    worker.prepare_forward(ctx, [first, second])
    impl.forward(
        SimpleNamespace(layer_name="layer0"),
        query,
        None,
        None,
        None,
        SimpleNamespace(num_actual_tokens=7),
    )
    worker.compact()
    assert torch.equal(algorithms[0].observations[0]["layer0"], query[5:7])
    assert torch.equal(algorithms[1].observations[0]["layer0"], query[1:3])
    assert impl.forward is original


def test_normal_step_does_not_install_query_hooks():
    worker, ctx, _, impl, original, _ = _environment()
    worker.prepare_forward(ctx, [])
    assert impl.forward is original
    assert worker.compact() == {}
    assert impl.forward is original


def test_selective_capture_rejects_invalid_count(monkeypatch):
    worker, ctx, state, impl, original, _ = _environment()
    algorithm = _Algorithm()
    algorithm.should_observe_token_queries = lambda *args: -1
    monkeypatch.setattr(
        worker_module, "build_token_drop_algorithm", lambda _spec: algorithm
    )
    with pytest.raises(ValueError, match="nonnegative"):
        worker.prepare_forward(ctx, [state])
    assert impl.forward is original


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pinned_flash_attention_forward_hook_signature(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    worker, ctx, state, _, _, _ = _environment()
    worker.register_kv_caches({"layer0": torch.zeros((3, 2, 4, 1, 8), device=device)})
    ctx.attn_metadata["layer0"].query_start_loc = ctx.attn_metadata[
        "layer0"
    ].query_start_loc.to(device)
    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    ctx.no_compile_layers["layer0"].impl = impl

    algorithm = _Algorithm()
    monkeypatch.setattr(
        worker_module, "build_token_drop_algorithm", lambda _spec: algorithm
    )
    query = torch.arange(7 * 2 * 8, device=device).reshape(7, 2, 8).float()
    output = torch.empty_like(query)
    worker.prepare_forward(ctx, [state])
    impl.forward(
        SimpleNamespace(layer_name="layer0"),
        query,
        None,
        None,
        None,
        None,  # Real FlashAttentionImpl profiling path, no GPU kernel.
        output,
    )
    assert torch.count_nonzero(output) == 0
    worker.compact()
    assert torch.equal(algorithm.observations[0]["layer0"], query[5:7])
    assert impl.forward.__func__ is FlashAttentionImpl.forward


def test_normal_and_td_mixed_batch_only_captures_td_queries(monkeypatch):
    worker, ctx, state, impl, original, calls = _environment()
    # Rows 0 and 2 are normal requests; only row 1 opted into token dropping.
    ctx.attn_metadata["layer0"].query_start_loc = torch.tensor([0, 2, 6, 9])
    algorithm = _Algorithm()
    monkeypatch.setattr(
        worker_module, "build_token_drop_algorithm", lambda _spec: algorithm
    )
    query = torch.arange(9 * 2 * 8).reshape(9, 2, 8).float()

    worker.prepare_forward(ctx, [state])
    result = impl.forward(
        SimpleNamespace(layer_name="layer0"),
        query,
        None,
        None,
        None,
        SimpleNamespace(num_actual_tokens=9),
    )
    assert worker.compact() == {}
    assert result is query
    assert len(calls) == 1
    assert torch.equal(calls[0][1], query)
    assert len(algorithm.observations) == 1
    assert torch.equal(algorithm.observations[0]["layer0"], query[4:6])
    assert impl.forward is original
