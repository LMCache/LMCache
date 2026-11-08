# SPDX-License-Identifier: Apache-2.0
"""Focused S4 tests: physical slot addressing without altering logical RoPE."""

# Standard
from types import SimpleNamespace

# Third Party
import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _find_token_drop_requests,
    _prepare_inputs_with_physical_frontier,
    install_token_drop_worker_adapter,
)


class FakeBlockTable:
    def __init__(self):
        self.block_tables = [SimpleNamespace(block_size=4)]
        self.rows = {}
        self.mapped = []

    def add_row(self, block_ids, row_idx):
        self.rows[row_idx] = block_ids

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.mapped.append(positions.clone())


class FakeRunner:
    def __init__(self, ids, logical):
        self.input_batch = SimpleNamespace(
            num_reqs=len(ids),
            req_ids=ids,
            num_computed_tokens_cpu=np.array(logical, dtype=np.int32),
            block_table=FakeBlockTable(),
        )
        self.positions = torch.zeros(32, dtype=torch.long)
        self.seq_lens = torch.zeros(8, dtype=torch.int32)
        self.optimistic_seq_lens_cpu = torch.zeros(8, dtype=torch.int32)


def state(request_id="td", resident=9, physical=True):
    return SimpleNamespace(
        request_id=request_id,
        resident_kv_tokens=resident,
        physical_block_ids=list(range(7, 16)) if physical else None,
        worker_row=-1,
    )


def metadata(states):
    return SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(
            token_drop_requests=states,
        )
    )


def test_nested_metadata_found():
    states = [state()]
    meta = SimpleNamespace(
        metadata=[
            SimpleNamespace(other=True),
            SimpleNamespace(token_drop_requests=states),
        ]
    )
    assert _find_token_drop_requests(meta) is states


def test_mixed_batch_slot_mapping_uses_p_but_rope_stays_l():
    runner = FakeRunner(["normal-a", "td", "normal-b"], [100, 104, 20])
    states = [state(resident=33)]
    scheduled = np.array([1, 1, 2], dtype=np.int32)
    logical_positions = torch.tensor([100, 104, 20, 21], dtype=torch.long)

    def original(model, output, counts):
        model.input_batch.block_table.compute_slot_mapping(3, None, logical_positions)
        model.seq_lens[:3] = torch.tensor([101, 105, 22])
        model.optimistic_seq_lens_cpu[:3] = torch.tensor([101, 105, 22])
        return "native"

    assert (
        _prepare_inputs_with_physical_frontier(
            original, runner, metadata(states), scheduled
        )
        == "native"
    )
    table = runner.input_batch.block_table
    assert states[0].worker_row == 1
    assert table.rows == {1: (list(range(7, 16)),)}
    assert torch.equal(table.mapped[0], torch.tensor([100, 32, 20, 21]))
    assert torch.equal(logical_positions, torch.tensor([100, 104, 20, 21]))
    assert torch.equal(runner.seq_lens[:3], torch.tensor([101, 33, 22]))
    assert torch.equal(runner.optimistic_seq_lens_cpu[:3], torch.tensor([101, 33, 22]))
    assert "compute_slot_mapping" not in vars(table)


def test_first_prefill_has_no_interception():
    runner = FakeRunner(["normal", "td"], [100, 40])
    states = [state(resident=42, physical=False)]
    scheduled = np.array([1, 2], dtype=np.int32)

    def native(model, *_):
        assert "compute_slot_mapping" not in vars(model.input_batch.block_table)
        return "unchanged"

    assert (
        _prepare_inputs_with_physical_frontier(
            native, runner, metadata(states), scheduled
        )
        == "unchanged"
    )
    assert states[0].worker_row == 1
    assert runner.input_batch.block_table.rows == {}


def test_mapping_restored_when_native_raises():
    runner = FakeRunner(["td"], [80])
    table = runner.input_batch.block_table
    original = vars(table).get("compute_slot_mapping")
    assert original is None

    def failing(*args):
        assert "compute_slot_mapping" in vars(table)
        raise RuntimeError("native failure")

    with pytest.raises(RuntimeError, match="native failure"):
        _prepare_inputs_with_physical_frontier(
            failing,
            runner,
            metadata([state(resident=33)]),
            np.array([1], dtype=np.int32),
        )
    assert "compute_slot_mapping" not in vars(table)


@pytest.mark.parametrize(
    "bad",
    [
        state(request_id="unknown"),
        state(resident=0),
        state(resident=80),
    ],
)
def test_invalid_metadata_never_calls_native_or_installs_interception(bad):
    runner = FakeRunner(["td"], [40])
    table = runner.input_batch.block_table
    with pytest.raises(RuntimeError):
        _prepare_inputs_with_physical_frontier(
            lambda *_: pytest.fail("Native method called despite invalid metadata"),
            runner,
            metadata([bad]),
            np.array([1], dtype=np.int32),
        )
    assert "compute_slot_mapping" not in vars(table)
    assert table.rows == {}


def test_install_worker_hook_does_not_patch_global_slot_mapping():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__ != "0.25.1":
        pytest.skip("Requires pinned vLLM 0.25.1")
    # Third Party
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    prepare = GPUModelRunner._prepare_inputs
    mapping = MultiGroupBlockTable.compute_slot_mapping
    try:
        install_token_drop_worker_adapter()
        wrapped = GPUModelRunner._prepare_inputs
        assert wrapped is not prepare
        assert MultiGroupBlockTable.compute_slot_mapping is mapping
        install_token_drop_worker_adapter()
        assert GPUModelRunner._prepare_inputs is wrapped
    finally:
        GPUModelRunner._prepare_inputs = prepare


def test_missing_physical_ids_after_compaction_fails_before_native():
    runner = FakeRunner(["td"], [100])
    td = state(resident=33, physical=False)
    with pytest.raises(RuntimeError, match="without authoritative private block ids"):
        _prepare_inputs_with_physical_frontier(
            lambda *_: pytest.fail("Native called with wrong KV mapping"),
            runner,
            metadata([td]),
            np.array([1], dtype=np.int32),
        )
    assert runner.input_batch.block_table.rows == {}


def test_sparse_td_metadata_does_not_reorder_scheduler_states():
    runner = FakeRunner(["td-b", "normal", "td-a"], [10, 20, 100])
    states = [state("td-a", 101, physical=False), state("td-b", 11, physical=False)]
    assert (
        _prepare_inputs_with_physical_frontier(
            lambda *_: "native",
            runner,
            metadata(states),
            np.array([1, 1, 1], dtype=np.int32),
        )
        == "native"
    )
    assert [s.request_id for s in states] == ["td-a", "td-b"]
    assert [s.worker_row for s in states] == [2, 0]


def test_multi_connector_skips_empty_child_before_active_td():
    states = [state()]
    meta = SimpleNamespace(
        metadata=[
            SimpleNamespace(token_drop_requests=[]),
            SimpleNamespace(token_drop_requests=states),
        ]
    )
    assert _find_token_drop_requests(meta) is states


def test_multi_connector_rejects_multiple_active_td_children():
    meta = SimpleNamespace(
        metadata=[
            SimpleNamespace(token_drop_requests=[state("one")]),
            SimpleNamespace(token_drop_requests=[state("two")]),
        ]
    )
    with pytest.raises(RuntimeError, match="Multiple active token-drop connectors"):
        _find_token_drop_requests(meta)
