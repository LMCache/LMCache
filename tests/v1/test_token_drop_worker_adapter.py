# SPDX-License-Identifier: Apache-2.0
"""Single-hook worker KV remapping: CPU contract tests and pinned CUDA regression."""

# Standard
from types import SimpleNamespace

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _find_token_drop_states,
    _prepare_inputs_with_resident_kv_addressing,
    install_token_drop_worker_kv_address_adapter,
)


class _FakeBlockTable:
    def __init__(self):
        self.rows = {}
        self.mapping_positions = []

    def add_row(self, block_ids, row_idx):
        self.rows[row_idx] = block_ids

    def compute_slot_mapping(self, _num_reqs, _starts, positions):
        self.mapping_positions.append(positions.clone())


class _FakeRunner:
    def __init__(self, req_ids, logical_frontiers, *, device="cpu"):
        self.max_num_tokens = 16
        self.input_batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            block_table=_FakeBlockTable(),
            num_computed_tokens_cpu=np.asarray(logical_frontiers, dtype=np.int32),
        )
        self.positions = torch.zeros(16, dtype=torch.int64, device=device)
        self.seq_lens = torch.zeros(8, dtype=torch.int32, device=device)
        self.query_pos = SimpleNamespace(
            gpu=torch.zeros(16, dtype=torch.int64, device=device)
        )
        self.query_start_loc = SimpleNamespace(
            gpu=torch.zeros(9, dtype=torch.int32, device=device)
        )


def _state(req_id, resident, ids=None):
    return SimpleNamespace(
        request_id=req_id,
        resident_kv_tokens=resident,
        physical_block_ids=ids,
        worker_row=-1,
    )


def _output(states):
    return SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(token_drop_requests=states)
    )


def _native_prepare(runner, positions):
    def run(_runner, _output, scheduled):
        num_tokens = int(scheduled.sum())
        assert len(positions) == num_tokens
        starts = np.r_[0, np.cumsum(scheduled)]
        runner.query_start_loc.gpu[: len(starts)] = torch.tensor(
            starts, dtype=torch.int32, device=runner.positions.device
        )
        offsets = [i for n in scheduled for i in range(int(n))]
        runner.query_pos.gpu[:num_tokens] = torch.tensor(
            offsets, dtype=torch.int64, device=runner.positions.device
        )
        runner.positions[:num_tokens] = torch.tensor(
            positions, dtype=torch.int64, device=runner.positions.device
        )
        for i, scheduled_count in enumerate(scheduled):
            runner.seq_lens[i] = runner.input_batch.num_computed_tokens_cpu[i] + int(
                scheduled_count
            )
        runner.input_batch.block_table.compute_slot_mapping(
            runner.input_batch.num_reqs,
            runner.query_start_loc.gpu[: len(starts)],
            runner.positions[:num_tokens],
        )
        return ("prepared", None)

    return run


def test_nested_multiconnector_metadata_skips_empty_children():
    state = _state("td", 33)
    metadata = SimpleNamespace(
        metadata=[
            SimpleNamespace(token_drop_requests=[]),
            SimpleNamespace(metadata=[SimpleNamespace(token_drop_requests=[state])]),
        ]
    )
    assert _find_token_drop_states(metadata) == [state]


def test_mixed_batch_preserves_native_normal_positions_and_logical_rope():
    runner = _FakeRunner(["normal-a", "td", "normal-b"], [100, 104, 20])
    state = _state("td", 33, ids=[7, 8, 9])
    scheduled = np.array([1, 1, 2], dtype=np.int32)
    normal_rope_positions = [117, 104, 20, 21]
    result = _prepare_inputs_with_resident_kv_addressing(
        _native_prepare(runner, normal_rope_positions),
        runner,
        _output([state]),
        scheduled,
    )
    assert result == ("prepared", None)
    calls = runner.input_batch.block_table.mapping_positions
    assert len(calls) == 2  # native mapping followed by physical remapping
    assert calls[0].tolist() == normal_rope_positions
    assert calls[1].tolist() == [117, 32, 20, 21]
    assert runner.positions[:4].tolist() == normal_rope_positions
    assert runner.seq_lens[:3].tolist() == [101, 33, 22]
    assert runner.input_batch.block_table.rows == {1: ([7, 8, 9],)}
    assert state.worker_row == 1


def test_compacted_second_step_without_new_block_ids_remaps_again():
    runner = _FakeRunner(["normal", "td"], [100, 104])
    state = _state("td", 33, ids=[7, 8, 9])
    output = _output([state])
    _prepare_inputs_with_resident_kv_addressing(
        _native_prepare(runner, [100, 104]),
        runner,
        output,
        np.array([1, 1], dtype=np.int32),
    )
    state.physical_block_ids = None
    state.resident_kv_tokens = 34
    runner.input_batch.num_computed_tokens_cpu[:] = [101, 105]
    _prepare_inputs_with_resident_kv_addressing(
        _native_prepare(runner, [101, 105]),
        runner,
        output,
        np.array([1, 1], dtype=np.int32),
    )
    assert runner.input_batch.block_table.mapping_positions[-1].tolist() == [101, 33]
    assert runner.seq_lens[:2].tolist() == [102, 34]
    assert runner.input_batch.block_table.rows == {1: ([7, 8, 9],)}


def test_no_compaction_uses_only_native_slot_mapping():
    runner = _FakeRunner(["normal", "td"], [10, 104])
    state = _state("td", 105, ids=[7, 8, 9])
    _prepare_inputs_with_resident_kv_addressing(
        _native_prepare(runner, [10, 104]),
        runner,
        _output([state]),
        np.array([1, 1], dtype=np.int32),
    )
    assert len(runner.input_batch.block_table.mapping_positions) == 1
    assert runner.input_batch.block_table.rows == {1: ([7, 8, 9],)}
    assert not hasattr(runner, "_lmcache_token_drop_kv_write_positions")


@pytest.mark.parametrize(
    "state",
    [
        _state("missing", 33, ids=[7]),
        _state("td", 0, ids=[7]),
        _state("td", 33, ids=[]),
    ],
)
def test_invalid_metadata_fails_before_native_prepare_or_block_table_change(state):
    runner = _FakeRunner(["normal", "td"], [10, 104])
    invoked = []
    with pytest.raises(RuntimeError):
        _prepare_inputs_with_resident_kv_addressing(
            lambda *_: invoked.append(True),
            runner,
            _output([state]),
            np.array([1, 1], dtype=np.int32),
        )
    assert not invoked
    assert runner.input_batch.block_table.rows == {}


def test_worker_adapter_installs_only_prepare_inputs_and_is_idempotent():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_prepare = GPUModelRunner._prepare_inputs
    original_slot_mapper = MultiGroupBlockTable.compute_slot_mapping
    try:
        install_token_drop_worker_kv_address_adapter()
        wrapped = GPUModelRunner._prepare_inputs
        assert wrapped is not original_prepare
        assert MultiGroupBlockTable.compute_slot_mapping is original_slot_mapper
        install_token_drop_worker_kv_address_adapter()
        assert GPUModelRunner._prepare_inputs is wrapped
    finally:
        GPUModelRunner._prepare_inputs = original_prepare


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_cuda_real_slot_mapping_two_steps():
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
    runner = _FakeRunner(["normal", "td"], [100, 104], device="cuda")
    runner.input_batch.block_table = table
    table.add_row((list(range(30, 37)),), 0)
    state = _state("td", 33, ids=[7, 8, 9])
    output = _output([state])

    def native(positions):
        def run(_runner, _output, scheduled):
            table.commit_block_table(2)
            return _native_prepare(runner, positions)(_runner, _output, scheduled)

        return run

    _prepare_inputs_with_resident_kv_addressing(
        native([100, 104]), runner, output, np.array([1, 1], dtype=np.int32)
    )
    assert table[0].slot_mapping.gpu[:2].tolist() == [36 * 16 + 4, 9 * 16]
    assert runner.positions[:2].tolist() == [100, 104]
    assert runner.seq_lens[:2].tolist() == [101, 33]

    state.physical_block_ids = None
    state.resident_kv_tokens = 34
    runner.input_batch.num_computed_tokens_cpu[:] = [101, 105]
    _prepare_inputs_with_resident_kv_addressing(
        native([101, 105]), runner, output, np.array([1, 1], dtype=np.int32)
    )
    assert table[0].slot_mapping.gpu[:2].tolist() == [36 * 16 + 5, 9 * 16 + 1]
    assert runner.positions[:2].tolist() == [101, 105]
    assert runner.seq_lens[:2].tolist() == [102, 34]
