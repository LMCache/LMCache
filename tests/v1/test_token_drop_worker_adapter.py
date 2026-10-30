# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmcache.integration.vllm.token_drop_worker_adapter import (
    _compute_slot_mapping_with_physical_positions,
    _find_token_drop_requests,
    _prepare_inputs_with_physical_frontier,
    install_token_drop_worker_adapter,
)


class _FakeBuffer:
    def __init__(self, size: int, dtype: torch.dtype):
        np_dtype = np.int64 if dtype == torch.int64 else np.int32
        self.np = np.zeros(size, dtype=np_dtype)
        self.gpu = torch.zeros(size, dtype=dtype)

    def copy_to_gpu(self, size: int) -> None:
        self.gpu[:size].copy_(torch.from_numpy(self.np[:size]))


class _FakeRunner:
    def __init__(self, req_ids: list[str], logical_frontiers: list[int]):
        self.max_num_tokens = 32
        self.max_num_reqs = 8
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.input_batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            block_table=object(),
            num_computed_tokens_cpu=np.asarray(logical_frontiers, dtype=np.int32),
        )
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32)

    def _make_buffer(self, size: int, *, dtype: torch.dtype):
        return _FakeBuffer(size, dtype)


def _state(
    request_id: str,
    resident: int,
    *,
    override: bool,
    num_new_tokens: int,
):
    return SimpleNamespace(
        request_id=request_id,
        resident_kv_tokens=resident,
        has_physical_override=override,
        num_new_tokens=num_new_tokens,
        worker_row=-1,
    )


def test_find_token_drop_requests_through_nested_connector_metadata() -> None:
    requests = [_state("req", 9, override=True, num_new_tokens=1)]
    metadata = SimpleNamespace(
        metadata=[
            SimpleNamespace(other=True),
            SimpleNamespace(token_drop_requests=requests),
        ]
    )

    assert _find_token_drop_requests(metadata) is requests


def test_prepare_inputs_mixed_batch_overrides_only_token_drop_row() -> None:
    runner = _FakeRunner(
        ["normal-a", "td", "normal-c"],
        [100, 104, 20],
    )
    states = [_state("td", 33, override=True, num_new_tokens=1)]
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    num_scheduled_tokens = np.array([1, 1, 2], dtype=np.int32)
    logical_positions = torch.tensor([100, 104, 20, 21], dtype=torch.int64)
    seen: dict[str, torch.Tensor] = {}

    def original_compute(block_table, num_reqs, query_start_loc, positions):
        seen["positions"] = positions.clone()
        return None

    def original_prepare(runner, scheduler_output, num_scheduled_tokens):
        _compute_slot_mapping_with_physical_positions(
            original_compute,
            runner.input_batch.block_table,
            3,
            torch.tensor([0, 1, 2, 4], dtype=torch.int32),
            logical_positions,
        )
        runner.seq_lens[:3] = torch.tensor([101, 105, 22], dtype=torch.int32)
        return "prepared"

    result = _prepare_inputs_with_physical_frontier(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "prepared"
    assert states[0].worker_row == 1
    assert torch.equal(
        seen["positions"],
        torch.tensor([100, 32, 20, 21], dtype=torch.int64),
    )
    assert torch.equal(logical_positions, torch.tensor([100, 104, 20, 21]))
    assert torch.equal(runner.seq_lens[:3], torch.tensor([101, 33, 22]))


def test_prepare_inputs_without_physical_override_is_exact_vanilla() -> None:
    runner = _FakeRunner(["normal", "td"], [104, 40])
    states = [_state("td", 42, override=False, num_new_tokens=2)]
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    calls = []

    def original_prepare(runner, scheduler_output, num_scheduled_tokens):
        calls.append(num_scheduled_tokens.copy())
        runner.seq_lens[:2] = torch.tensor([105, 42], dtype=torch.int32)
        return "vanilla"

    result = _prepare_inputs_with_physical_frontier(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "vanilla"
    assert len(calls) == 1
    assert states[0].worker_row == 1
    assert torch.equal(runner.seq_lens[:2], torch.tensor([105, 42]))


def test_prepare_inputs_rejects_token_drop_request_absent_from_worker_batch() -> None:
    runner = _FakeRunner(["req-a", "req-b"], [32, 32])
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(
            token_drop_requests=[
                _state("missing", 33, override=True, num_new_tokens=1)
            ]
        )
    )

    with pytest.raises(RuntimeError, match="absent from worker batch"):
        _prepare_inputs_with_physical_frontier(
            lambda *_: None,
            runner,
            scheduler_output,
            np.array([1, 1], dtype=np.int32),
        )


def test_install_worker_adaptor_is_idempotent_for_pinned_vllm() -> None:
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_prepare = GPUModelRunner._prepare_inputs
    original_slot_mapping = MultiGroupBlockTable.compute_slot_mapping
    try:
        install_token_drop_worker_adapter()
        wrapped_prepare = GPUModelRunner._prepare_inputs
        wrapped_slot_mapping = MultiGroupBlockTable.compute_slot_mapping

        assert getattr(
            wrapped_prepare,
            "_lmcache_token_drop_worker_adapter",
            False,
        )
        assert getattr(
            wrapped_slot_mapping,
            "_lmcache_token_drop_worker_adapter",
            False,
        )

        install_token_drop_worker_adapter()
        assert GPUModelRunner._prepare_inputs is wrapped_prepare
        assert MultiGroupBlockTable.compute_slot_mapping is wrapped_slot_mapping
    finally:
        GPUModelRunner._prepare_inputs = original_prepare
        MultiGroupBlockTable.compute_slot_mapping = original_slot_mapping
