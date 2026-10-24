# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _compute_slot_mapping_with_kv_write_positions,
    _find_token_drop_states,
    _prepare_inputs_with_resident_kv,
    install_token_drop_worker_kv_address_adapter,
)
from lmcache.v1.platform import torch_device_type

pytestmark = pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")


class _FakeBuffer:
    def __init__(self, size: int, dtype: torch.dtype):
        np_dtype = np.int64 if dtype == torch.int64 else np.int32
        self.np = np.zeros(size, dtype=np_dtype)
        self.gpu = torch.zeros(size, dtype=dtype, device="cuda")

    def copy_to_gpu(self, size: int) -> None:
        self.gpu[:size].copy_(torch.from_numpy(self.np[:size]))


class _FakeBlockTable:
    def __init__(self) -> None:
        self.rows: dict[int, tuple[list[int], ...]] = {}

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        self.rows[row_idx] = block_ids


class _FakeRunner:
    def __init__(self, req_ids: list[str], logical_frontiers: list[int]):
        self.max_num_tokens = 32
        self.max_num_reqs = 8
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.input_batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            block_table=_FakeBlockTable(),
            num_computed_tokens_cpu=np.asarray(logical_frontiers, dtype=np.int32),
        )
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device="cuda")

    def _make_buffer(self, size: int, *, dtype: torch.dtype):
        return _FakeBuffer(size, dtype)


def _state(
    request_id: str,
    resident: int,
    *,
    include_block_ids: bool,
):
    return SimpleNamespace(
        request_id=request_id,
        resident_kv_tokens=resident,
        physical_block_ids=[7, 8, 9] if include_block_ids else None,
        worker_row=-1,
    )


def test_find_token_drop_states_through_nested_connector_metadata() -> None:
    requests = [_state("req", 9, include_block_ids=True)]
    metadata = SimpleNamespace(
        metadata=[
            SimpleNamespace(token_drop_requests=[]),
            SimpleNamespace(metadata=[SimpleNamespace(token_drop_requests=requests)]),
        ]
    )

    assert _find_token_drop_states(metadata) is requests


def test_prepare_inputs_mixed_batch_overrides_only_token_drop_row() -> None:
    runner = _FakeRunner(
        ["normal-a", "td", "normal-c"],
        [100, 104, 20],
    )
    states = [_state("td", 33, include_block_ids=True)]
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
        _compute_slot_mapping_with_kv_write_positions(
            original_compute,
            runner.input_batch.block_table,
            3,
            torch.tensor([0, 1, 2, 4], dtype=torch.int32),
            logical_positions,
        )
        runner.seq_lens[:3] = torch.tensor([999, 105, 888], dtype=torch.int32)
        return "prepared"

    result = _prepare_inputs_with_resident_kv_addressing(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "prepared"
    assert states[0].worker_row == 1
    assert torch.equal(
        seen["positions"],
        torch.tensor([100, 32, 20, 21], dtype=torch.int64, device="cuda"),
    )
    assert torch.equal(logical_positions, torch.tensor([100, 104, 20, 21]))
    assert torch.equal(runner.seq_lens[:3], torch.tensor([999, 33, 888], device="cuda"))
    assert runner.input_batch.block_table.rows == {1: ([7, 8, 9],)}


def test_matching_resident_length_without_block_refresh_uses_normal_path() -> None:
    runner = _FakeRunner(["normal", "td"], [104, 40])
    states = [_state("td", 42, include_block_ids=False)]
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    calls = []

    def original_prepare(runner, scheduler_output, num_scheduled_tokens):
        calls.append(num_scheduled_tokens.copy())
        runner.seq_lens[:2] = torch.tensor([105, 42], dtype=torch.int32)
        return "vanilla"

    result = _prepare_inputs_with_resident_kv_addressing(
        original_prepare,
        runner,
        scheduler_output,
        num_scheduled_tokens,
    )

    assert result == "vanilla"
    assert len(calls) == 1
    assert states[0].worker_row == 1
    assert torch.equal(runner.seq_lens[:2], torch.tensor([105, 42], device="cuda"))


def test_compacted_resident_length_applies_without_new_block_ids() -> None:
    runner = _FakeRunner(["normal", "td"], [100, 104])
    states = [_state("td", 33, include_block_ids=False)]
    output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    logical_positions = torch.tensor([100, 104], dtype=torch.int64)
    captured_positions = []

    def original_compute(_table, _num_reqs, _starts, positions):
        captured_positions.append(positions.clone())

    def original_prepare(_runner, _output, _scheduled):
        _compute_slot_mapping_with_kv_write_positions(
            original_compute,
            runner.input_batch.block_table,
            2,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            logical_positions,
        )
        runner.seq_lens[:2] = torch.tensor([101, 105], device="cuda")
        return "prepared"

    result = _prepare_inputs_with_resident_kv_addressing(
        original_prepare, runner, output, np.array([1, 1], dtype=np.int32)
    )

    assert result == "prepared"
    assert torch.equal(
        captured_positions[0], torch.tensor([100, 32], device="cuda")
    )
    assert torch.equal(runner.seq_lens[:2], torch.tensor([101, 33], device="cuda"))
    assert logical_positions.tolist() == [100, 104]
    assert runner.input_batch.block_table.rows == {}


def test_prepare_inputs_rejects_token_drop_request_absent_from_worker_batch() -> None:
    runner = _FakeRunner(["req-a", "req-b"], [32, 32])
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(
            token_drop_requests=[_state("missing", 33, include_block_ids=True)]
        )
    )

    with pytest.raises(RuntimeError, match="absent from worker batch"):
        _prepare_inputs_with_resident_kv_addressing(
            lambda *_: (torch.empty(0), None),
            runner,
            scheduler_output,
            np.array([1, 1], dtype=np.int32),
        )


def test_worker_kv_address_adapter_installation_is_idempotent() -> None:
    vllm = pytest.importorskip("vllm")
    if vllm.__version__ != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_prepare = GPUModelRunner._prepare_inputs
    original_slot_mapping = MultiGroupBlockTable.compute_slot_mapping
    try:
        install_token_drop_worker_kv_address_adapter()
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

        install_token_drop_worker_kv_address_adapter()
        assert GPUModelRunner._prepare_inputs is wrapped_prepare
        assert MultiGroupBlockTable.compute_slot_mapping is wrapped_slot_mapping
    finally:
        GPUModelRunner._prepare_inputs = original_prepare
        MultiGroupBlockTable.compute_slot_mapping = original_slot_mapping


def test_kv_write_positions_override_is_cleared_after_prepare_failure() -> None:
    runner = _FakeRunner(["normal", "td"], [10, 104])
    states = [_state("td", 33, include_block_ids=True)]
    output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    scheduled = np.array([1, 1], dtype=np.int32)
    positions = torch.tensor([10, 104], dtype=torch.int64)
    captured = []

    def record_slot_mapping(_table, _num_reqs, _starts, mapped_positions):
        captured.append(mapped_positions.clone())

    def failing_prepare(_runner, _output, _scheduled):
        _compute_slot_mapping_with_kv_write_positions(
            record_slot_mapping,
            runner.input_batch.block_table,
            2,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            positions,
        )
        raise RuntimeError("injected prepare failure")

    with pytest.raises(RuntimeError, match="injected prepare failure"):
        _prepare_inputs_with_resident_kv_addressing(
            failing_prepare, runner, output, scheduled
        )

    _compute_slot_mapping_with_kv_write_positions(
        record_slot_mapping,
        runner.input_batch.block_table,
        2,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        positions,
    )
    assert torch.equal(captured[0], torch.tensor([10, 32], device="cuda"))
    assert torch.equal(captured[1], positions)


def test_worker_row_assignment_does_not_reorder_connector_metadata() -> None:
    runner = _FakeRunner(["td-b", "normal", "td-a"], [80, 10, 100])
    states = [
        _state("td-a", 101, include_block_ids=False),
        _state("td-b", 21, include_block_ids=True),
    ]
    output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )

    def original_prepare(runner, _output, _scheduled):
        runner.seq_lens[:3] = torch.tensor([81, 11, 101], device="cuda")
        return "prepared"

    assert (
        _prepare_inputs_with_resident_kv_addressing(
            original_prepare, runner, output, np.array([1, 1, 1])
        )
        == "prepared"
    )
    assert [state.request_id for state in states] == ["td-a", "td-b"]
    assert [state.worker_row for state in states] == [2, 0]
    assert torch.equal(runner.seq_lens[:3], torch.tensor([21, 11, 101], device="cuda"))


def test_pinned_vllm_cuda_slot_mapping_separates_normal_and_token_drop_rows() -> None:
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.worker.block_table import MultiGroupBlockTable

    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cuda"),
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks=[8],
    )
    runner = _FakeRunner(["normal", "td"], [100, 104])
    runner.input_batch.block_table = table
    table.add_row((list(range(30, 37)),), 0)

    states = [_state("td", 33, include_block_ids=True)]
    output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )
    logical_positions = torch.tensor([100, 104], dtype=torch.int64, device="cuda")
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")

    def original_prepare(_runner, _output, _scheduled):
        table.commit_block_table(2)
        _compute_slot_mapping_with_kv_write_positions(
            MultiGroupBlockTable.compute_slot_mapping,
            table,
            2,
            query_start_loc,
            logical_positions,
        )
        runner.seq_lens[:2] = logical_positions.to(torch.int32) + 1
        return "prepared"

    assert (
        _prepare_inputs_with_resident_kv_addressing(
            original_prepare, runner, output, np.array([1, 1])
        )
        == "prepared"
    )
    # Normal: logical 100 -> block 36, offset 4.
    # TD: logical 104 -> resident position 32 -> block 9, offset 0.
    assert table[0].slot_mapping.gpu[:2].tolist() == [36 * 16 + 4, 9 * 16]
    assert logical_positions.tolist() == [100, 104]
    assert runner.seq_lens[:2].tolist() == [101, 33]
    assert states[0].worker_row == 1

    # Next step: P still differs from L, but scheduler block IDs are unchanged.
    states[0].physical_block_ids = None
    states[0].resident_kv_tokens = 34
    runner.input_batch.num_computed_tokens_cpu[:] = [101, 105]
    logical_positions = torch.tensor([101, 105], dtype=torch.int64, device="cuda")

    assert (
        _prepare_inputs_with_resident_kv_addressing(
            original_prepare, runner, output, np.array([1, 1])
        )
        == "prepared"
    )
    assert table[0].slot_mapping.gpu[:2].tolist() == [36 * 16 + 5, 9 * 16 + 1]
    assert runner.seq_lens[:2].tolist() == [102, 34]
    assert logical_positions.tolist() == [101, 105]
