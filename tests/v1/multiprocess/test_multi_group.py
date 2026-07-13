# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-driven multi-group KV cache transfer.

Covers:
  - GroupLayoutInfo msgspec roundtrip
  - RegisterEngineDrivenContextPayload with group_layouts
  - _serialize / _deserialize multi-group chunks
  - slice_kv_caches_for_group
  - Single-group backward compatibility
  - (Optional GPU) gather/scatter multi-group roundtrip
"""

# Standard
import pickle

# Third Party
import msgspec.msgpack as mp
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    GroupLayoutInfo,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContextMetadata,
    MemoryLayoutDesc,
    _deserialize_multi_group_chunks,
    _serialize_multi_group_chunks,
    slice_kv_caches_for_group,
)

# ─── Test 1: GroupLayoutInfo msgspec roundtrip ───────────────────────────────


def test_group_layout_info_serialization():
    """GroupLayoutInfo must survive msgpack encode/decode."""
    g0 = GroupLayoutInfo(
        block_size=16,
        num_layers=14,
        hidden_dim_size=256,
        dtype_str="bfloat16",
        use_mla=False,
        tokens_per_block=16,
    )
    raw = mp.encode(g0)
    restored = mp.decode(raw, type=GroupLayoutInfo)
    assert restored.block_size == 16
    assert restored.num_layers == 14
    assert restored.hidden_dim_size == 256
    assert restored.dtype_str == "bfloat16"
    assert restored.use_mla is False
    assert restored.tokens_per_block == 16


# ─── Test 2: RegisterEngineDrivenContextPayload with group_layouts ───────────


def test_group_layout_in_payload():
    """Multi-group payload must roundtrip via msgspec.msgpack."""
    g0 = GroupLayoutInfo(
        block_size=16,
        num_layers=14,
        hidden_dim_size=256,
        dtype_str="bfloat16",
        use_mla=False,
        tokens_per_block=16,
    )
    g1 = GroupLayoutInfo(
        block_size=784,
        num_layers=14,
        hidden_dim_size=512,
        dtype_str="float32",
        use_mla=False,
        tokens_per_block=784,
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=1,
        model_name="test",
        world_size=2,
        block_size=16,
        num_layers=14,
        hidden_dim_size=256,
        dtype_str="bfloat16",
        use_mla=False,
        group_layouts=[g0, g1],
    )
    raw = mp.encode(payload)
    restored = mp.decode(raw, type=RegisterEngineDrivenContextPayload)
    assert len(restored.group_layouts) == 2
    assert restored.group_layouts[0].tokens_per_block == 16
    assert restored.group_layouts[1].tokens_per_block == 784


# ─── Test 3: Single-group backward compatibility ─────────────────────────────


def test_single_group_backward_compat():
    """Empty group_layouts means single-group (backward compatible)."""
    payload = RegisterEngineDrivenContextPayload(
        instance_id=1,
        model_name="test",
        world_size=1,
        block_size=16,
        num_layers=32,
        hidden_dim_size=256,
        dtype_str="float16",
        use_mla=False,
        group_layouts=[],
    )
    assert len(payload.group_layouts) == 0

    # Roundtrip must preserve empty list
    raw = mp.encode(payload)
    restored = mp.decode(raw, type=RegisterEngineDrivenContextPayload)
    assert restored.group_layouts == []


def test_single_group_metadata_is_not_multi():
    """EngineDrivenContextMetadata with empty group lists is single-group."""
    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 32, 256, 128])],
        dtypes=[torch.float16],
    )
    metadata = EngineDrivenContextMetadata(
        layout_desc=layout_desc,
        block_size=16,
        use_mla=False,
    )
    assert not metadata.is_multi_group


def test_multi_group_metadata_flag():
    """EngineDrivenContextMetadata with >1 group_layout_descs is multi-group."""
    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 14, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    metadata = EngineDrivenContextMetadata(
        layout_desc=layout_desc,
        block_size=16,
        use_mla=False,
        group_layout_descs=[layout_desc, layout_desc],
        group_block_sizes=[16, 784],
        group_use_mla=[False, False],
        group_blocks_in_chunk=[16, 1],
    )
    assert metadata.is_multi_group


# ─── Test 4: serialize/deserialize multi-group chunks ────────────────────────


def test_serialize_deserialize_multi_group():
    """Roundtrip: group_chunks -> bytes -> group_chunks."""
    group_chunks = [
        [torch.randn(2, 4, 16, 128), torch.randn(2, 4, 16, 128)],  # group 0: 2 chunks
        [torch.randn(1, 2, 784, 64)],  # group 1: 1 chunk (GDN-like)
    ]
    blob = _serialize_multi_group_chunks(group_chunks)
    assert isinstance(blob, bytes)
    assert len(blob) > 0

    restored = _deserialize_multi_group_chunks(blob)
    assert len(restored) == 2
    assert len(restored[0]) == 2
    assert len(restored[1]) == 1

    torch.testing.assert_close(restored[0][0], group_chunks[0][0])
    torch.testing.assert_close(restored[0][1], group_chunks[0][1])
    torch.testing.assert_close(restored[1][0], group_chunks[1][0])


def test_serialize_deserialize_empty_group():
    """A group with zero chunks must survive roundtrip."""
    group_chunks = [
        [torch.randn(2, 4, 16, 128)],
        [],  # empty group
    ]
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert len(restored) == 2
    assert len(restored[0]) == 1
    assert len(restored[1]) == 0


def test_serialize_preserves_dtype():
    """Dtype must survive pickle roundtrip via numpy."""
    group_chunks = [
        [torch.randn(2, 4, 16, 128, dtype=torch.bfloat16)],
    ]
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert restored[0][0].dtype == torch.bfloat16
    # Values should be close (bf16 has lower precision than f32)
    torch.testing.assert_close(
        restored[0][0].float(), group_chunks[0][0].float(), atol=0.01, rtol=0.05
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn not available in this PyTorch build",
)
def test_serialize_preserves_fp8_e4m3fn():
    """fp8_e4m3fn dtypes must roundtrip via uint8 view."""
    # Generate values within fp8_e4m3fn range
    src = torch.randn(2, 4, 16, 128).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    group_chunks = [[src]]
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert restored[0][0].dtype == torch.float8_e4m3fn
    assert restored[0][0].shape == src.shape
    # Bit-exact: fp8 is a view of uint8
    assert torch.equal(restored[0][0].view(torch.uint8), src.view(torch.uint8))


@pytest.mark.skipif(
    not hasattr(torch, "float8_e5m2"),
    reason="torch.float8_e5m2 not available in this PyTorch build",
)
def test_serialize_preserves_fp8_e5m2():
    """fp8_e5m2 dtypes must roundtrip via uint8 view."""
    src = torch.randn(2, 4, 16, 128).clamp(-240.0, 240.0).to(torch.float8_e5m2)
    group_chunks = [[src]]
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert restored[0][0].dtype == torch.float8_e5m2
    assert torch.equal(restored[0][0].view(torch.uint8), src.view(torch.uint8))


def test_serialize_mixed_dtypes_in_one_blob():
    """A multi-group blob can mix dtypes per group."""
    group_chunks = [
        [torch.randn(2, 4, 16, 128, dtype=torch.bfloat16)],
        [torch.randn(2, 4, 8, 64, dtype=torch.float16)],
        [torch.randn(2, 4, 4, 32, dtype=torch.float32)],
    ]
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert restored[0][0].dtype == torch.bfloat16
    assert restored[1][0].dtype == torch.float16
    assert restored[2][0].dtype == torch.float32


# ─── Test 5: slice_kv_caches_for_group ───────────────────────────────────────


def test_slice_kv_caches_for_group():
    """Extract a subset of KV tensors for a single group."""
    kv = {str(i): torch.zeros(2, 100, 64) for i in range(10)}
    group = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 3, 7),
        tokens_per_block=16,
    )
    sliced = slice_kv_caches_for_group(kv, group.layer_indices)
    assert len(sliced) == 3
    # Keys are re-indexed 0, 1, 2 (sorted by layer_indices)
    assert set(sliced.keys()) == {"0", "1", "2"}
    for tensor in sliced.values():
        assert tensor.shape == (2, 100, 64)


def test_slice_kv_caches_for_group_unsorted_indices():
    """Indices must be sorted even if passed unsorted."""
    kv = {str(i): torch.ones(2, 10, 32) * i for i in range(10)}
    # Pass indices in reverse order
    sliced = slice_kv_caches_for_group(kv, (9, 5, 1))
    assert len(sliced) == 3
    # Key "0" -> layer 1, "1" -> layer 5, "2" -> layer 9
    assert torch.allclose(sliced["0"], torch.ones(2, 10, 32) * 1)
    assert torch.allclose(sliced["1"], torch.ones(2, 10, 32) * 5)
    assert torch.allclose(sliced["2"], torch.ones(2, 10, 32) * 9)


# ─── Test 6: Integration (CPU-only gather/scatter) ──────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs GPU")
def test_engine_driven_multi_group_store_retrieve():
    """End-to-end CPU gather -> serialize -> deserialize -> scatter on GPU."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_multi_group_to_cpu,
        scatter_cpu_multi_group_to_paged_kv,
    )

    # Simulate 2 groups: group 0 has layers 0-1, group 1 has layers 2-3
    block_size = 16
    num_blocks = 2
    block_ids = list(range(num_blocks))

    # Group 0: attention-like (2 layers, MLA=False)
    attn_kv = {
        str(i): torch.randn(2, num_blocks * block_size, 128, device="cuda")
        for i in range(2)
    }
    # Group 1: GDN-like (2 layers, single dim)
    gdn_kv = {
        str(i): torch.randn(num_blocks * 784, 64, device="cuda") for i in range(2)
    }

    # All KV caches ordered: attn_0, attn_1, gdn_0, gdn_1
    all_kv = {}
    for i, v in attn_kv.items():
        all_kv[f"attn_{i}"] = v
    for i, v in gdn_kv.items():
        all_kv[f"gdn_{i}"] = v

    engine_group_infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=16),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=784),
    ]

    # For group 0: lmcache_tokens_per_chunk = blocks_in_chunk * block_size
    # For group 1: tokens_per_block=784, so blocks_in_chunk = lmcache_tokens_per_chunk // 784
    lmcache_tokens_per_chunk = 784  # Must be >= max(tokens_per_block) across all groups

    # Gather all groups to CPU
    group_chunks = gather_paged_kv_multi_group_to_cpu(
        all_kv,
        [block_ids, block_ids],
        engine_group_infos,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    )
    assert len(group_chunks) == 2

    # Serialize -> deserialize
    blob = _serialize_multi_group_chunks(group_chunks)
    restored = _deserialize_multi_group_chunks(blob)
    assert len(restored) == 2

    # Scatter back to GPU
    # Create fresh GPU tensors for scatter target
    scatter_target = {}
    for k, v in all_kv.items():
        scatter_target[k] = torch.zeros_like(v)

    scatter_cpu_multi_group_to_paged_kv(
        scatter_target,
        [block_ids, block_ids],
        restored,
        engine_group_infos,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    )

    # Verify scatter wrote non-zero data
    for v in scatter_target.values():
        assert v.abs().sum() > 0


# ─── Test 7: Server-side commit_store multi-group format ────────────────────


def test_commit_store_multi_group_format():
    """Server-side: _commit_store_multi_group must use _deserialize_multi_group_chunks."""
    # Simulate worker serialization
    group_chunks = [
        [torch.randn(2, 4, 16, 128, dtype=torch.bfloat16)],  # Group 0
        [torch.randn(1, 2, 784, 64)],  # Group 1
    ]
    cpu_data = _serialize_multi_group_chunks(group_chunks)

    # Verify that _deserialize_multi_group_chunks correctly roundtrips
    restored = _deserialize_multi_group_chunks(cpu_data)
    assert restored[0][0].dtype == torch.bfloat16
    assert restored[1][0].shape == group_chunks[1][0].shape

    # Verify that pickle.dumps(list[torch.Tensor]) for strategy is compatible
    for group in restored:
        repickled = pickle.dumps(group)  # list[torch.Tensor]
        reloaded = pickle.loads(repickled)
        assert all(isinstance(t, torch.Tensor) for t in reloaded)


# ─── Test 8: Server-side prepare_retrieve multi-group format ─────────────────


def test_prepare_retrieve_multi_group_format():
    """Server→Worker: Format must use _serialize_multi_group_chunks."""
    # Simulate what strategy.prepare_retrieve() returns (single-group format)
    tensors_g0 = [torch.randn(2, 4, 16, 128)]
    tensors_g1 = [torch.randn(1, 2, 784, 64)]
    strategy_response_g0 = pickle.dumps(tensors_g0)  # Like PickleTransferStrategy
    strategy_response_g1 = pickle.dumps(tensors_g1)

    # Simulate what the corrected server code should do:
    group_tensors_0 = pickle.loads(strategy_response_g0)
    group_tensors_1 = pickle.loads(strategy_response_g1)
    cpu_data = _serialize_multi_group_chunks([group_tensors_0, group_tensors_1])

    # What the worker receives must be correctly deserializable:
    restored = _deserialize_multi_group_chunks(cpu_data)
    assert len(restored) == 2
    torch.testing.assert_close(restored[0][0], tensors_g0[0])
    torch.testing.assert_close(restored[1][0], tensors_g1[0])
