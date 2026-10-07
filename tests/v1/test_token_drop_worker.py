# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from lmcache.integration.vllm.experimental import token_drop_worker as worker_mod
from lmcache.integration.vllm.experimental.token_drop_worker import TokenDropWorker

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

    def should_observe_query(self, **facts):
        self.observe_calls.append(facts)
        return self.observe(facts) if callable(self.observe) else self.observe

    def observe_query(self, layer_queries):
        self.observation_batches.append(tuple(layer_queries))
        for layer_name, query in layer_queries.items():
            self.observations.setdefault(layer_name, []).append(query.clone())

    def should_compact(self, **facts):
        self.compact_calls.append(facts)
        return (
            self.compact_now(facts)
            if callable(self.compact_now)
            else self.compact_now
        )

    def select_kept_positions(self, layer_keys):
        self.selection_inputs.append(layer_keys)
        first = next(iter(layer_keys.values()))
        kept = self.kept(layer_keys) if callable(self.kept) else self.kept
        return torch.as_tensor(kept, dtype=torch.long, device=first.device)


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
        has_physical_override=False,
        is_genuine_decode=is_decode,
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
    return SimpleNamespace(
        use_cascade=False,
        query_start_loc=torch.tensor(starts, device="cuda"),
        block_table=torch.tensor(padded, device="cuda"),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_query_observation_forwards_full_request_step_even_for_prefill(monkeypatch):
    algorithm = _FakeAlgorithm(observe=True, compact=False)
    _install_algorithms(monkeypatch, {"a": algorithm})

    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    worker._algorithm_from_state(_state("td", "a", resident=7, num_new=3))

    # Row 0 is a normal request with two tokens; token-drop request is sparse row 1.
    metadata = _metadata([[0], [1, 2]], [2, 3])
    worker.begin_step(
        ["td"],
        metadata,
        physical_seq_lens=[7],
        request_rows=[1],
        observe_query=[True],
        compact_now=[False],
        num_new_tokens=[3],
    )

    query = torch.arange(
        5 * Q_HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    ).view(5, Q_HEADS, HEAD_DIM)
    for name in LAYER_NAMES:
        worker.capture_query(name, query)

    assert worker.compact() == {}
    assert algorithm.observation_batches == [tuple(LAYER_NAMES)]

    for name in LAYER_NAMES:
        assert len(algorithm.observations[name]) == 1
        assert torch.equal(algorithm.observations[name][0], query[2:5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_idle_step_skips_attention_metadata_and_worker_state(monkeypatch):
    algorithm = _FakeAlgorithm(observe=False, compact=False)
    _install_algorithms(monkeypatch, {"a": algorithm})

    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    state = _state("td", "a", resident=8, worker_row=0)
    forward_context = SimpleNamespace(
        attn_metadata=None,
        no_compile_layers={},
    )

    worker.prepare_forward(forward_context, [state])

    assert worker._query_hooks_installed is False
    assert worker._seq_lens is None
    assert worker._compact_now is None
    assert worker.compact() == {}
    assert len(algorithm.observe_calls) == 1
    assert len(algorithm.compact_calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_does_not_depend_on_query_observation(monkeypatch):
    algorithm = _FakeAlgorithm(kept=[0, 2, 4, 6], observe=False, compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})

    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    state = _state("td", "a", resident=8, worker_row=0)
    forward_context = SimpleNamespace(
        attn_metadata={"group": _metadata([[3, 1]], [1])},
        no_compile_layers={},
    )

    worker.prepare_forward(forward_context, [state])
    assert worker._query_hooks_installed is False
    assert worker.compact() == {"td": 4}
    assert algorithm.observations == {}
    assert len(algorithm.compact_calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compaction_applies_algorithm_positions_on_cuda(monkeypatch, dtype):
    kept = [1, 3, 4, 7]
    algorithm = _FakeAlgorithm(kept=kept, compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})

    caches = _new_cache(4, dtype=dtype)
    original = {name: cache.clone() for name, cache in caches.items()}
    block_ids = [3, 1]
    source_slots = _slots(block_ids, 8)[kept]
    destination_slots = _slots(block_ids, 8)[: len(kept)]

    worker = TokenDropWorker()
    worker.register_kv_caches(caches)
    worker._algorithm_from_state(_state("td", "a", resident=8))
    worker.begin_step(
        ["td"],
        _metadata([block_ids], [1]),
        physical_seq_lens=[8],
        observe_query=[False],
        compact_now=[True],
        num_new_tokens=[1],
    )

    assert worker.compact() == {"td": 4}
    for name in LAYER_NAMES:
        for kv_index in (0, 1):
            assert torch.equal(
                _read_slots(caches[name], destination_slots, kv_index),
                _read_slots(original[name], source_slots, kv_index),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_repeated_compaction_uses_current_physical_sequence(monkeypatch):
    algorithm = _FakeAlgorithm(kept=[1, 3, 4, 7], compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})

    caches = _new_cache(4)
    original = {name: cache.clone() for name, cache in caches.items()}
    block_ids = [3, 1]
    original_slots = _slots(block_ids, 8)

    worker = TokenDropWorker()
    worker.register_kv_caches(caches)
    worker._algorithm_from_state(_state("td", "a", resident=8))

    worker.begin_step(
        ["td"],
        _metadata([block_ids], [1]),
        physical_seq_lens=[8],
        observe_query=[False],
        compact_now=[True],
        num_new_tokens=[1],
    )
    assert worker.compact() == {"td": 4}

    algorithm.kept = [1, 3]
    worker.begin_step(
        ["td"],
        _metadata([block_ids], [1]),
        physical_seq_lens=[4],
        observe_query=[False],
        compact_now=[True],
        num_new_tokens=[1],
    )
    assert worker.compact() == {"td": 2}

    destination_slots = original_slots[:2]
    expected_sources = original_slots[[3, 7]]
    for name in LAYER_NAMES:
        for kv_index in (0, 1):
            assert torch.equal(
                _read_slots(caches[name], destination_slots, kv_index),
                _read_slots(original[name], expected_sources, kv_index),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_drop_first_shifts_across_shuffled_block_boundaries(monkeypatch):
    kept = list(range(1, 12))
    algorithm = _FakeAlgorithm(kept=kept, compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})

    caches = _new_cache(5)
    original = {name: cache.clone() for name, cache in caches.items()}
    block_ids = [2, 0, 4]
    slots = _slots(block_ids, 12)

    worker = TokenDropWorker()
    worker.register_kv_caches(caches)
    worker._algorithm_from_state(_state("td", "a", resident=12))
    worker.begin_step(
        ["td"],
        _metadata([block_ids], [1]),
        physical_seq_lens=[12],
        observe_query=[False],
        compact_now=[True],
        num_new_tokens=[1],
    )

    assert worker.compact() == {"td": 11}
    for name in LAYER_NAMES:
        for kv_index in (0, 1):
            assert torch.equal(
                _read_slots(caches[name], slots[:11], kv_index),
                _read_slots(original[name], slots[1:], kv_index),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_algorithms_compact_independently_in_same_batch(monkeypatch):
    algorithms = {
        "a": _FakeAlgorithm(kept=[0, 2, 4, 6], compact=True),
        "b": _FakeAlgorithm(kept=[1, 3, 5], compact=True),
    }
    _install_algorithms(monkeypatch, algorithms)

    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(6))
    worker._algorithm_from_state(_state("a", "a", resident=8, worker_row=0))
    worker._algorithm_from_state(_state("b", "b", resident=8, worker_row=1))
    worker.begin_step(
        ["a", "b"],
        _metadata([[4, 1], [3, 2]], [1, 1]),
        physical_seq_lens=[8, 8],
        request_rows=[0, 1],
        observe_query=[False, False],
        compact_now=[True, True],
        num_new_tokens=[1, 1],
    )

    assert worker.compact() == {"a": 4, "b": 3}


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_token_drop_rejects_noncontiguous_kv_layout():
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

    with pytest.raises(ValueError, match="contiguous NHD"):
        worker._ensure_kv_caches_compatible()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_algorithm_selection_failure_propagates(monkeypatch):
    class _Rejecting(_FakeAlgorithm):
        def select_kept_positions(self, layer_keys):
            raise RuntimeError("algorithm selection failed")

    algorithm = _Rejecting(compact=True)
    _install_algorithms(monkeypatch, {"a": algorithm})

    worker = TokenDropWorker()
    worker.register_kv_caches(_new_cache(4))
    worker._algorithm_from_state(_state("td", "a", resident=8))
    worker.begin_step(
        ["td"],
        _metadata([[3, 1]], [1]),
        physical_seq_lens=[8],
        observe_query=[False],
        compact_now=[True],
        num_new_tokens=[1],
    )

    with pytest.raises(RuntimeError, match="algorithm selection failed"):
        worker.compact()
