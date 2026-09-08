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


def test_compact_slot_mapping_handles_scattered_unsorted_blocks():
    """Real slot mappings come from the block allocator, not a dense range."""
    stager = staging_mod.NeuronKVBlockStager()
    # Blocks 9, 2 and 5, visited out of order, with a prefix-cache gap.
    slots = torch.tensor([-1, -1, 37, 36, 11, 22, 8], dtype=torch.long)

    selected_blocks, compact = stager._compact_slot_mapping(slots, block_size=4)

    assert selected_blocks == [2, 5, 9]
    # 37 -> block 9 off 1 -> compact block 2; 36 -> block 9 off 0;
    # 11 -> block 2 off 3 -> compact block 0; 22 -> block 5 off 2 -> compact 1;
    # 8 -> block 2 off 0 -> compact block 0.
    assert compact.tolist() == [-1, -1, 9, 8, 3, 6, 0]


def test_compact_slot_mapping_all_invalid_returns_no_blocks():
    stager = staging_mod.NeuronKVBlockStager()
    slots = torch.tensor([-1, -1], dtype=torch.long)

    selected_blocks, compact = stager._compact_slot_mapping(slots, block_size=4)

    assert selected_blocks == []
    assert compact.tolist() == [-1, -1]


def test_contiguous_runs_collapses_adjacent_indices():
    """Runs are what keep a transfer from touching the whole paged cache."""
    runs = staging_mod.NeuronKVBlockStager._contiguous_runs([0, 1, 2, 7, 9, 10])

    assert runs == [(0, 3), (7, 1), (9, 2)]


def test_contiguous_runs_empty_selection():
    assert staging_mod.NeuronKVBlockStager._contiguous_runs([]) == []


def test_contiguous_runs_fully_scattered_indices():
    runs = staging_mod.NeuronKVBlockStager._contiguous_runs([1, 3, 5])

    assert runs == [(1, 1), (3, 1), (5, 1)]


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
        skip_prefix_n_tokens=0,
    )

    # Blocks 1 and 3 must now be ones; blocks 0 and 2 untouched (zeros).
    assert torch.equal(layer[:, 1], torch.ones_like(layer[:, 1]))
    assert torch.equal(layer[:, 3], torch.ones_like(layer[:, 3]))
    assert torch.equal(layer[:, 0], torch.zeros_like(layer[:, 0]))
    assert torch.equal(layer[:, 2], torch.zeros_like(layer[:, 2]))


def _multi_layer_args(num_layers: int) -> dict[str, Any]:
    """Build a call for ``num_layers`` identical layers in an HND-two-major cache.

    :param num_layers: Number of per-layer KV tensors to generate.
    :returns: Keyword arguments for either transfer method.
    """
    return {
        "key_value": torch.empty((2, num_layers, 4, 6), dtype=torch.float32),
        "layer_tensors": [
            torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 2)
            for _ in range(num_layers)
        ],
        "slot_mapping": torch.tensor([2, 3, 6, 7], dtype=torch.long),
        "engine_kv_format": lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        "block_size": 2,
        "head_size": 2,
    }


def test_all_layers_stage_through_a_single_buffer(monkeypatch):
    """Every layer must land in one allocation, so one host copy covers them all.

    This is the property the staging exists for: host transfer count on Neuron
    is a fixed cost per copy, so staging per layer reintroduces the overhead the
    gather was meant to remove.
    """
    recorded: dict[str, Any] = {}
    _capture_multi_layer(monkeypatch, recorded)

    stager = staging_mod.NeuronKVBlockStager()
    stager.transfer_into_key_value(**_multi_layer_args(num_layers=8))

    staged = recorded["staged_ref"]
    assert len(staged) == 8
    storages = {tensor.untyped_storage().data_ptr() for tensor in staged}
    assert len(storages) == 1, "staged layers must be views of one buffer"


def test_staging_buffers_are_reused_across_chunks_of_equal_geometry(monkeypatch):
    """Steady-state serving must not reallocate; only new geometry may."""
    recorded: dict[str, Any] = {}
    _capture_multi_layer(monkeypatch, recorded)
    stager = staging_mod.NeuronKVBlockStager()

    stager.transfer_into_key_value(**_multi_layer_args(num_layers=4))
    first = recorded["staged_ref"][0].untyped_storage().data_ptr()

    stager.transfer_into_key_value(**_multi_layer_args(num_layers=4))
    second = recorded["staged_ref"][0].untyped_storage().data_ptr()
    assert first == second

    # A different layer count is different geometry and gets its own buffer.
    stager.transfer_into_key_value(**_multi_layer_args(num_layers=5))
    assert recorded["staged_ref"][0].untyped_storage().data_ptr() != first


def test_reused_buffer_does_not_leak_the_previous_chunk(monkeypatch):
    """A reused buffer is fully overwritten, so no stale KV can be stored."""
    recorded: dict[str, Any] = {}
    _capture_multi_layer(monkeypatch, recorded)
    stager = staging_mod.NeuronKVBlockStager()

    args = _multi_layer_args(num_layers=2)
    stager.transfer_into_key_value(**args)

    # Second chunk selects different blocks from a different source tensor.
    args["layer_tensors"] = [torch.full((2, 4, 3, 2, 2), 7.0) for _ in range(2)]
    args["slot_mapping"] = torch.tensor([0, 1, 4, 5], dtype=torch.long)  # blocks 0, 2
    stager.transfer_into_key_value(**args)

    for staged in recorded["staged"]:
        assert torch.equal(staged, torch.full_like(staged, 7.0))


def test_partially_covered_block_keeps_its_uncovered_slots(monkeypatch):
    """A partial tail block must not have its unmapped slots overwritten.

    Transfers are block-granular but the unpack only fills the token slots named
    in the slot mapping. The tail block of any request whose length is not a
    multiple of ``block_size`` is partially covered, and its remaining slots hold
    live KV for tokens vLLM has not generated yet. Scattering the staging buffer
    over them wholesale corrupts the cache with whatever that buffer last held --
    a previous chunk's KV, which is plausible enough to produce bad output rather
    than an error.
    """
    stager = staging_mod.NeuronKVBlockStager()
    # [2, num_blocks, num_heads, block_size, head_size] with block_size 2.
    layer = torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 2)
    original = layer.clone()
    # Slot 2 only: block 1, offset 0. Offset 1 of block 1 is not mapped.
    slots = torch.tensor([2], dtype=torch.long)

    def fill_covered_slot_only(key_value, key_value_ptrs, *_args, **_kwargs):
        """Mimic the native unpack: write only the mapped slot, leave the rest."""
        for staged in key_value_ptrs:
            # staged is [2, 1 block, num_heads, block_size, head_size].
            staged[:, :, :, 0, :] = 1.0

    monkeypatch.setattr(
        staging_mod.device_ops, "multi_layer_kv_transfer", fill_covered_slot_only
    )

    stager.transfer_from_key_value(
        key_value=torch.empty((2, 1, 1, 6), dtype=torch.float32),
        layer_tensors=[layer],
        slot_mapping=slots,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
        skip_prefix_n_tokens=0,
    )

    # The mapped slot took the retrieved value.
    assert torch.equal(layer[:, 1, :, 0, :], torch.ones_like(layer[:, 1, :, 0, :]))
    # The unmapped slot of the same block is unchanged.
    assert torch.equal(layer[:, 1, :, 1, :], original[:, 1, :, 1, :])
    # Other blocks are untouched.
    for block in (0, 2, 3):
        assert torch.equal(layer[:, block], original[:, block])


def test_skipped_prefix_tokens_are_not_written(monkeypatch):
    """Prefix-cached leading tokens must survive a retrieve untouched.

    ``skip_prefix_n_tokens`` names tokens vLLM already holds. Their blocks can be
    shared with other running requests via prefix caching, so writing them races
    with live readers and corrupts KV that belongs to a different request. The
    connector computed this value and the Neuron path dropped it on the floor.
    """
    stager = staging_mod.NeuronKVBlockStager()
    layer = torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 2)
    original = layer.clone()
    # Blocks 1 and 3. The first two tokens (block 1) are prefix-cached.
    slots = torch.tensor([2, 3, 6, 7], dtype=torch.long)

    def fill_after_skip(
        key_value, key_value_ptrs, *_args, skip_prefix_n_tokens=0, **_kwargs
    ):
        """Mimic the native unpack honouring the skip: leave the first N slots."""
        for staged in key_value_ptrs:
            # staged is [2, 2 blocks, num_heads, block_size, head_size]; flatten
            # the block and within-block axes to index tokens of the chunk.
            flat = staged.permute(0, 2, 1, 3, 4).reshape(2, 3, 4, 2)
            flat[:, :, skip_prefix_n_tokens:, :] = 1.0
            staged.copy_(flat.reshape(2, 3, 2, 2, 2).permute(0, 2, 1, 3, 4))

    monkeypatch.setattr(
        staging_mod.device_ops, "multi_layer_kv_transfer", fill_after_skip
    )

    stager.transfer_from_key_value(
        key_value=torch.empty((2, 1, 4, 6), dtype=torch.float32),
        layer_tensors=[layer],
        slot_mapping=slots,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
        skip_prefix_n_tokens=2,
    )

    # Block 1 holds the two skipped tokens and must be exactly as it was.
    assert torch.equal(layer[:, 1], original[:, 1])
    # Block 3 holds the retrieved tokens.
    assert torch.equal(layer[:, 3], torch.ones_like(layer[:, 3]))
    # Unselected blocks untouched.
    assert torch.equal(layer[:, 0], original[:, 0])
    assert torch.equal(layer[:, 2], original[:, 2])


def test_skip_prefix_n_tokens_is_forwarded_to_the_unpack(monkeypatch):
    """The value must reach the unpack, not just gate the seeding."""
    seen: dict[str, Any] = {}

    def record(key_value, key_value_ptrs, *_args, skip_prefix_n_tokens=0, **_kwargs):
        seen["skip"] = skip_prefix_n_tokens

    monkeypatch.setattr(staging_mod.device_ops, "multi_layer_kv_transfer", record)

    staging_mod.NeuronKVBlockStager().transfer_from_key_value(
        key_value=torch.empty((2, 1, 4, 6), dtype=torch.float32),
        layer_tensors=[torch.zeros((2, 4, 3, 2, 2), dtype=torch.float32)],
        slot_mapping=torch.tensor([2, 3, 6, 7], dtype=torch.long),
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
        skip_prefix_n_tokens=3,
    )

    assert seen["skip"] == 3


def test_negative_skip_prefix_n_tokens_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        staging_mod.NeuronKVBlockStager().transfer_from_key_value(
            key_value=torch.empty((2, 1, 4, 6), dtype=torch.float32),
            layer_tensors=[torch.zeros((2, 4, 3, 2, 2), dtype=torch.float32)],
            slot_mapping=torch.tensor([2, 3], dtype=torch.long),
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            block_size=2,
            head_size=2,
            skip_prefix_n_tokens=-1,
        )


def test_transfer_from_key_value_scatters_token_indexed_layout(monkeypatch):
    """The scatter must also be correct for a layout indexed on the token axis."""
    stager = staging_mod.NeuronKVBlockStager()
    # NL_X_NB_BS_HS: [num_blocks * block_size, num_heads, head_size], dim 0.
    layer = torch.zeros((8, 3, 2), dtype=torch.float32)

    def fill_staged(key_value, key_value_ptrs, *_args, **_kwargs):
        for staged in key_value_ptrs:
            staged.copy_(torch.ones_like(staged))

    monkeypatch.setattr(staging_mod.device_ops, "multi_layer_kv_transfer", fill_staged)

    stager.transfer_from_key_value(
        key_value=torch.empty((2, 1, 4, 6), dtype=torch.float32),
        layer_tensors=[layer],
        slot_mapping=torch.tensor([2, 3, 6, 7], dtype=torch.long),  # blocks 1 and 3
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        block_size=2,
        head_size=2,
        skip_prefix_n_tokens=0,
    )

    # Token slots 2,3 and 6,7 written; 0,1 and 4,5 untouched.
    assert torch.equal(layer[2:4], torch.ones_like(layer[2:4]))
    assert torch.equal(layer[6:8], torch.ones_like(layer[6:8]))
    assert torch.equal(layer[0:2], torch.zeros_like(layer[0:2]))
    assert torch.equal(layer[4:6], torch.zeros_like(layer[4:6]))


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
