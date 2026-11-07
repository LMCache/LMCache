# SPDX-License-Identifier: Apache-2.0
"""Request-local token-drop algorithm and query observation tests."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental import token_drop_worker as worker_module
from lmcache.integration.vllm.experimental.token_drop_worker import TokenDropWorker


class _FakeAlgorithm:
    def __init__(self, *, observe=0, compact=False):
        self.observe_count = observe
        self.should_compact = compact
        self.observe_calls = []
        self.compact_calls = []
        self.observations = []

    def should_observe_token_queries(self, phase, decoded_tokens_before_step):
        self.observe_calls.append((phase, decoded_tokens_before_step))
        return self.observe_count

    def should_compact_kv(self, phase, resident_kv_tokens, decoded_tokens_before_step):
        self.compact_calls.append(
            (phase, resident_kv_tokens, decoded_tokens_before_step)
        )
        return self.should_compact

    def observe_token_queries(self, layer_queries):
        self.observations.append(dict(layer_queries))


def _state(request_id, *, phase="decode", decoded=0, resident=12, num_new=1, row=0):
    return SimpleNamespace(
        request_id=request_id,
        algorithm="fake",
        config={"request": request_id},
        phase=phase,
        num_decoded_tokens=decoded,
        resident_kv_tokens=resident,
        num_new_tokens=num_new,
        worker_row=row,
    )


def _worker(monkeypatch, algorithms, layers=("layer0", "layer1")):
    monkeypatch.setattr(
        worker_module,
        "build_token_drop_algorithm",
        lambda spec: algorithms[spec.config["request"]],
    )
    worker = TokenDropWorker()
    worker.register_kv_caches({name: torch.zeros(1, 2, 4, 2, 3) for name in layers})
    return worker


def test_request_local_algorithms_preserve_phase_and_first_decode_step(monkeypatch):
    algorithms = {
        "prefill": _FakeAlgorithm(observe=0, compact=False),
        "decode": _FakeAlgorithm(observe=0, compact=True),
    }
    worker = _worker(monkeypatch, algorithms)
    states = [
        _state("prefill", phase="prefill", decoded=0, row=0),
        _state("decode", phase="decode", decoded=0, row=1),
    ]
    context = SimpleNamespace(
        attn_metadata={
            "group": SimpleNamespace(
                query_start_loc=torch.tensor([0, 1, 2]),
                use_cascade=False,
            )
        },
        no_compile_layers={},
    )
    worker.prepare_forward(context, states)
    assert algorithms["prefill"].observe_calls == [("prefill", 0)]
    assert algorithms["decode"].observe_calls == [("decode", 0)]
    assert algorithms["decode"].compact_calls == [("decode", 12, 0)]
    assert worker._compact_now is None or worker._compact_now == [False, True]

    # A later decode step reuses the same request-local plugin.
    states[1].num_decoded_tokens = 1
    worker.prepare_forward(
        SimpleNamespace(
            attn_metadata={
                "group": SimpleNamespace(
                    query_start_loc=torch.tensor([0, 1, 2]),
                    block_table=torch.tensor([[0], [1]]),
                )
            },
            no_compile_layers={},
        ),
        states,
    )
    assert worker._request_algorithms["decode"] is algorithms["decode"]
    assert algorithms["decode"].observe_calls[-1] == ("decode", 1)
    assert worker.finish_requests({"decode"}) == {"decode"}
    assert "decode" not in worker._request_algorithms
    assert "prefill" in worker._request_algorithms


def test_query_observation_respects_mixed_batch_rows(monkeypatch):
    algorithms = {
        "a": _FakeAlgorithm(),
        "b": _FakeAlgorithm(),
    }
    worker = _worker(monkeypatch, algorithms)
    for request_id in algorithms:
        worker._algorithm_from_state(_state(request_id))
    worker.begin_step(
        request_ids=["a", "b"],
        attn_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 3, 4]),
            use_cascade=False,
        ),
        request_rows=[1, 2],
        observe_counts=[2, 1],
        compact_now=[True, True],
        num_new_tokens=[2, 1],
    )
    queries = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    for layer_name in ("layer0", "layer1"):
        worker.capture_query(layer_name, queries)
    worker._flush_query_observations()
    assert worker._compact_now == [True, True]
    for name in ("layer0", "layer1"):
        assert torch.equal(algorithms["a"].observations[0][name], queries[1:3])
        assert torch.equal(algorithms["b"].observations[0][name], queries[3:4])


def test_incomplete_layer_capture_disables_compaction_without_observation(monkeypatch):
    algorithm = _FakeAlgorithm()
    worker = _worker(monkeypatch, {"a": algorithm})
    worker._algorithm_from_state(_state("a"))
    worker.begin_step(
        request_ids=["a"],
        attn_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1]),
            use_cascade=False,
        ),
        request_rows=[0],
        observe_counts=[1],
        compact_now=[True],
        num_new_tokens=[1],
    )
    worker.capture_query("layer0", torch.ones(1, 2, 3))
    worker._flush_query_observations()
    assert algorithm.observations == []
    assert worker._compact_now == [False]


def test_flattened_queries_are_captured_in_token_major_order(monkeypatch):
    algorithm = _FakeAlgorithm()
    worker = _worker(monkeypatch, {"a": algorithm}, layers=("layer0",))
    worker._algorithm_from_state(_state("a"))
    worker.begin_step(
        request_ids=["a"],
        attn_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            use_cascade=False,
        ),
        request_rows=[0],
        observe_counts=[2],
        compact_now=[False],
        num_new_tokens=[2],
    )
    queries = torch.arange(12).reshape(2, 6)
    worker.capture_query("layer0", queries)
    worker._flush_query_observations()
    assert torch.equal(algorithm.observations[0]["layer0"], queries.reshape(2, 2, 3))


def test_invalid_observation_count_fails_before_query_capture(monkeypatch):
    worker = _worker(monkeypatch, {"a": _FakeAlgorithm()})
    with pytest.raises(ValueError, match="invalid Q capture count"):
        worker.begin_step(
            request_ids=["a"],
            attn_metadata=SimpleNamespace(
                query_start_loc=torch.tensor([0, 1]),
                use_cascade=False,
            ),
            request_rows=[0],
            observe_counts=[-1],
            compact_now=[True],
            num_new_tokens=[1],
        )


def test_failed_step_preparation_does_not_install_query_hooks(monkeypatch):
    worker = _worker(monkeypatch, {"a": _FakeAlgorithm(observe=1, compact=True)})
    attempts = []
    monkeypatch.setattr(
        worker_module,
        "install_token_drop_query_hooks",
        lambda *_args: attempts.append("install"),
    )
    with pytest.raises(ValueError, match="request/query segmentation mismatch"):
        worker.prepare_forward(
            SimpleNamespace(
                attn_metadata={
                    "group": SimpleNamespace(
                        query_start_loc=torch.tensor([0, 1]),
                        use_cascade=False,
                    )
                },
                no_compile_layers={},
            ),
            [_state("a", row=2)],
        )
    assert attempts == []
    assert worker._request_ids == []
    assert worker._query_observation_indices is None
