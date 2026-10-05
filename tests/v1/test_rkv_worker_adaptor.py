# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmcache.integration.vllm.rkv_worker_adaptor import (
    _compute_slot_mapping_with_physical_positions,
    _find_rkv_requests,
    _prepare_inputs_with_physical_frontier,
    install_rkv_worker_adaptor,
)


class _FakeBuffer:
    def __init__(self, size: int, dtype: torch.dtype):
        np_dtype = np.int64 if dtype == torch.int64 else np.int32
        self.np = np.zeros(size, dtype=np_dtype)
        self.gpu = torch.zeros(size, dtype=dtype)

    def copy_to_gpu(self, size: int) -> None:
        self.gpu[:size].copy_(torch.from_numpy(self.np[:size]))


class _FakeRunner:
    def __init__(self, req_ids: list[str]):
        self.max_num_tokens = 32
        self.max_num_reqs = 8
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.input_batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            block_table=object(),
        )
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        self.positions = torch.tensor([104, 40, 41], dtype=torch.int64)

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
    )


def test_find_rkv_requests_through_nested_connector_metadata() -> None:
    requests = [_state("req", 9, override=True, num_new_tokens=1)]
    metadata = SimpleNamespace(
        metadata=[
            SimpleNamespace(other=True),
            SimpleNamespace(rkv_requests=requests),
        ]
    )

    assert _find_rkv_requests(metadata) is requests


def test_prepare_inputs_presents_physical_positions_to_vanilla_slot_mapping() -> None:
    runner = _FakeRunner(["req-a", "req-b"])
    states = [
        _state("req-b", 42, override=False, num_new_tokens=2),
        _state("req-a", 33, override=True, num_new_tokens=1),
    ]
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(rkv_requests=states)
    )
    num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    seen: dict[str, torch.Tensor] = {}

    def original_compute(block_table, num_reqs, query_start_loc, positions):
        seen["positions"] = positions.clone()
        return None

    def original_prepare(runner, scheduler_output, num_scheduled_tokens):
        # Vanilla vLLM has already computed logical/RoPE positions at this point.
        logical_positions = runner.positions.clone()
        _compute_slot_mapping_with_physical_positions(
            original_compute,
            runner.input_batch.block_table,
            2,
            torch.tensor([0, 1, 3], dtype=torch.int32),
            logical_positions,
        )
        runner.seq_lens[:2] = torch.tensor([105, 42], dtype=torch.int32)
        return "prepared"

    result = _prepare_inputs_with_physical_frontier(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "prepared"
    assert [state.request_id for state in states] == ["req-a", "req-b"]
    assert torch.equal(seen["positions"], torch.tensor([32, 40, 41]))
    assert torch.equal(runner.positions, torch.tensor([104, 40, 41]))
    assert torch.equal(runner.seq_lens[:2], torch.tensor([33, 42]))


def test_prepare_inputs_without_physical_override_is_vanilla() -> None:
    runner = _FakeRunner(["req-a", "req-b"])
    states = [
        _state("req-b", 42, override=False, num_new_tokens=2),
        _state("req-a", 105, override=False, num_new_tokens=1),
    ]
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(rkv_requests=states)
    )
    num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    seen: dict[str, torch.Tensor] = {}

    def original_compute(block_table, num_reqs, query_start_loc, positions):
        seen["positions"] = positions.clone()
        return None

    def original_prepare(runner, scheduler_output, num_scheduled_tokens):
        _compute_slot_mapping_with_physical_positions(
            original_compute,
            runner.input_batch.block_table,
            2,
            torch.tensor([0, 1, 3], dtype=torch.int32),
            runner.positions,
        )
        runner.seq_lens[:2] = torch.tensor([105, 42], dtype=torch.int32)
        return "vanilla"

    result = _prepare_inputs_with_physical_frontier(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "vanilla"
    assert [state.request_id for state in states] == ["req-a", "req-b"]
    assert torch.equal(seen["positions"], runner.positions)
    assert torch.equal(runner.seq_lens[:2], torch.tensor([105, 42]))


def test_prepare_inputs_rejects_missing_worker_request() -> None:
    runner = _FakeRunner(["req-a", "req-b"])
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(
            rkv_requests=[_state("req-a", 33, override=True, num_new_tokens=1)]
        )
    )

    with pytest.raises(RuntimeError, match="missing worker request"):
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
        install_rkv_worker_adaptor()
        wrapped_prepare = GPUModelRunner._prepare_inputs
        wrapped_slot_mapping = MultiGroupBlockTable.compute_slot_mapping

        assert getattr(wrapped_prepare, "_lmcache_rkv_worker_adaptor", False)
        assert getattr(wrapped_slot_mapping, "_lmcache_rkv_worker_adaptor", False)

        install_rkv_worker_adaptor()
        assert GPUModelRunner._prepare_inputs is wrapped_prepare
        assert MultiGroupBlockTable.compute_slot_mapping is wrapped_slot_mapping
    finally:
        GPUModelRunner._prepare_inputs = original_prepare
        MultiGroupBlockTable.compute_slot_mapping = original_slot_mapping
