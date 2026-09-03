# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.gpu_connector.neuron_kv_staging as staging_mod


def _capture_multi_layer(monkeypatch, recorded: dict[str, Any]):
    """Patch ``multi_layer_kv_transfer`` to record staged tensors.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param recorded: Dict populated with the captured call arguments; the
        staged layer tensors are stored under ``"staged"`` (as clones) so the
        gather output can be inspected, and the same list object under
        ``"staged_ref"`` so a test can mutate it to simulate an H2D unpack.
    """

    def fake_multi_layer_kv_transfer(
        key_value,
        key_value_ptrs,
        slot_mapping,
        paged_memory_device,
        page_buffer_size,
        direction,
        engine_kv_format,
        block_size=0,
        head_size=0,
        skip_prefix_n_tokens=0,
        block_stride_elems=0,
    ):
        recorded["staged"] = [t.clone() for t in key_value_ptrs]
        recorded["staged_ref"] = key_value_ptrs
        recorded["slot_mapping"] = slot_mapping.tolist()
        recorded["page_buffer_size"] = page_buffer_size
        recorded["direction"] = int(direction)
        recorded["fmt"] = int(engine_kv_format)
        recorded["block_size"] = block_size
        recorded["head_size"] = head_size

    monkeypatch.setattr(
        staging_mod.device_ops, "multi_layer_kv_transfer", fake_multi_layer_kv_transfer
    )


def test_compact_slot_mapping_remaps_blocks_and_preserves_invalid_slots():
    stager = staging_mod.NeuronKVBlockStager()
    slots = torch.tensor([-1, 4, 5, 12, 13], dtype=torch.long)

    selected_blocks, compact = stager._compact_slot_mapping(slots, block_size=4)

    assert selected_blocks == [1, 3]
    assert compact.tolist() == [-1, 0, 1, 4, 5]


def test_selection_block_indexed_hnd_two_major_uses_block_axis():
    stager = staging_mod.NeuronKVBlockStager()

    dim, indices = stager._selection(
        torch.device("cpu"),
        [1, 3],
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
    )

    assert dim == 1
    assert indices.tolist() == [1, 3]


def test_selection_token_indexed_expands_blocks_to_token_slots():
    stager = staging_mod.NeuronKVBlockStager()

    dim, indices = stager._selection(
        torch.device("cpu"),
        [1, 3],
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        block_size=2,
    )

    assert dim == 0
    assert indices.tolist() == [2, 3, 6, 7]


def test_transfer_into_key_value_gathers_only_selected_blocks(monkeypatch):
    recorded: dict[str, Any] = {}
    _capture_multi_layer(monkeypatch, recorded)

    stager = staging_mod.NeuronKVBlockStager()
    key_value = torch.empty((2, 1, 4, 6), dtype=torch.float32)
    # [2, num_blocks, num_heads, block_size, head_size]; unique per block.
    layer = torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 2)
    slots = torch.tensor([2, 3, 6, 7], dtype=torch.long)  # blocks 1 and 3

    stager.transfer_into_key_value(
        key_value=key_value,
        layer_tensors=[layer],
        slot_mapping=slots,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
    )

    assert recorded["slot_mapping"] == [0, 1, 2, 3]
    assert recorded["page_buffer_size"] == 4
    assert recorded["direction"] == int(lmcache_native.TransferDirection.D2H)
    staged = recorded["staged"]
    assert len(staged) == 1
    assert tuple(staged[0].shape) == (2, 2, 3, 2, 2)
    # Staged data must equal the source layer's blocks 1 and 3.
    expected = layer.index_select(1, torch.tensor([1, 3]))
    assert torch.equal(staged[0], expected)


def test_transfer_from_key_value_scatters_into_selected_blocks(monkeypatch):
    stager = staging_mod.NeuronKVBlockStager()
    key_value = torch.empty((2, 1, 4, 6), dtype=torch.float32)
    layer = torch.zeros((2, 4, 3, 2, 2), dtype=torch.float32)
    slots = torch.tensor([2, 3, 6, 7], dtype=torch.long)  # blocks 1 and 3

    # Simulate the CPU unpack filling each staged buffer with known values.
    def fill_staged(key_value, key_value_ptrs, *_args, **_kwargs):
        for staged in key_value_ptrs:
            staged.copy_(torch.ones_like(staged))

    monkeypatch.setattr(staging_mod.device_ops, "multi_layer_kv_transfer", fill_staged)

    stager.transfer_from_key_value(
        key_value=key_value,
        layer_tensors=[layer],
        slot_mapping=slots,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
    )

    # Blocks 1 and 3 must now be ones; blocks 0 and 2 untouched (zeros).
    assert torch.equal(layer[:, 1], torch.ones_like(layer[:, 1]))
    assert torch.equal(layer[:, 3], torch.ones_like(layer[:, 3]))
    assert torch.equal(layer[:, 0], torch.zeros_like(layer[:, 0]))
    assert torch.equal(layer[:, 2], torch.zeros_like(layer[:, 2]))


def test_transfer_into_key_value_requires_cpu_destination():
    stager = staging_mod.NeuronKVBlockStager()
    with pytest.raises(ValueError):
        stager.transfer_into_key_value(
            key_value=torch.empty((2, 1, 4, 6), device="meta"),
            layer_tensors=[torch.empty((2, 4, 3, 2, 2))],
            slot_mapping=torch.tensor([0], dtype=torch.long),
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            block_size=2,
            head_size=2,
        )


def test_transfer_into_key_value_empty_inputs_are_noops(monkeypatch):
    recorded: dict[str, Any] = {}
    _capture_multi_layer(monkeypatch, recorded)

    stager = staging_mod.NeuronKVBlockStager()
    key_value = torch.empty((2, 1, 4, 6), dtype=torch.float32)

    stager.transfer_into_key_value(
        key_value=key_value,
        layer_tensors=[],
        slot_mapping=torch.tensor([], dtype=torch.long),
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
    )

    assert "staged" not in recorded
