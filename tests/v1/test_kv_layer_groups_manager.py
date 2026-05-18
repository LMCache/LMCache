# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import (
    KVLayerGroupInfo,
    KVLayerGroupsManager,
    format_kvcache_shape_spec,
    parse_kvcache_shape_spec,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)


def _build_manager(
    tensors: list[torch.Tensor],
    *,
    num_blocks: int,
    layout_hints: LayoutHints | None = None,
    lmcache_logical_chunk_size: int = 256,
) -> KVLayerGroupsManager:
    """Build a manager using the per-layer NHD format.

    Tensors in these tests have shape ``[2, NB, BS, NH, HS]`` — the
    canonical vLLM flash-attention per-layer NHD layout matched by
    ``GPUKVFormat.NL_X_TWO_NB_BS_NH_HS``. ``bs`` is discovered
    per-layer from the tensor shapes, so callers no longer pass it.
    """
    # First Party
    import lmcache.c_ops as lmc_ops

    return KVLayerGroupsManager(
        tensors,
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        num_blocks=num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=lmcache_logical_chunk_size,
    )


class TestKVLayerGroupsManager:
    """Tests for KVLayerGroupsManager construction and lookups."""

    def test_build_empty(self):
        manager = _build_manager([], num_blocks=32)
        assert manager.kv_layer_groups == []

    def test_build_single_layer(self):
        tensors = [torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)]
        manager = _build_manager(tensors, num_blocks=32)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert isinstance(group, KVLayerGroupInfo)
        assert group.layer_indices == [0]
        assert group.shape_desc.kv_size == 2
        assert group.shape_desc.nh == 8
        assert group.shape_desc.hs == 64
        assert group.shape_desc.nl == 1
        assert group.shape_desc.nb == 32
        assert group.shape_desc.bs == 256
        assert group.dtype == torch.float16

    def test_build_multiple_layers_same_shape(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(tensors, num_blocks=32)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape_desc.nl == 3
        assert group.shape_desc.nh == 8

    def test_build_different_shapes(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 2
        group1, group2 = manager.kv_layer_groups
        assert group1.layer_indices == [0, 2]
        assert group1.shape_desc.nh == 8
        assert group2.layer_indices == [1]
        assert group2.shape_desc.nh == 16

    def test_build_different_dtypes(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 2
        group1, group2 = manager.kv_layer_groups
        assert group1.layer_indices == [0, 2]
        assert group1.dtype == torch.float16
        assert group2.layer_indices == [1]
        assert group2.dtype == torch.float32

    def test_build_mixed_differences(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # nh=8, f16
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),  # nh=8, f32
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),  # nh=16, f16
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # nh=8, f16
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float32),  # nh=16, f32
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 4

        groups_by_key = {(g.shape_desc.nh, g.dtype): g for g in manager.kv_layer_groups}
        assert groups_by_key[(8, torch.float16)].layer_indices == [0, 3]
        assert groups_by_key[(8, torch.float32)].layer_indices == [1]
        assert groups_by_key[(16, torch.float16)].layer_indices == [2]
        assert groups_by_key[(16, torch.float32)].layer_indices == [4]

    def test_get_shape_desc_by_group_idx(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, num_blocks=32)

        sd0 = manager.get_shape_desc(0)
        assert sd0.nh == 8
        assert sd0.hs == 64
        assert sd0.nl == 1

        sd1 = manager.get_shape_desc(1)
        assert sd1.nh == 16
        assert sd1.hs == 64


class TestPerLayerLogicalBlockSize:
    """Tests for the ``per_layer_logical_block_size`` LayoutHints field.

    The hint extends LayerGroupIdentity with the engine's scheduler-side
    block size: layers with the same physical shape but different
    scheduler block sizes (e.g. vLLM's hybrid KV cache manager handing
    different ``KVCacheGroupSpec.block_size`` values to different
    layers) end up in distinct LMCache groups.
    """

    def test_hint_absent_collapses_to_5tuple(self):
        """No hint = legacy behavior: layers with matching physical
        identity collapse to a single group."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert group.layer_indices == [0, 1, 2, 3]
        # logical_block_size defaults to physical bs when no hint
        assert group.logical_block_size == 64
        assert group.shape_desc.bs == 64
        assert group.compress_ratio == 1

    def test_hint_splits_layers_by_logical_bs(self):
        """Two layers with same physical shape but different logical
        block sizes split into distinct groups."""
        # 4 layers all with physical bs=64. Hint says first 2 layers
        # use scheduler block size 256 (i.e. compressed: 4 logical
        # tokens per physical slot) and the other 2 use 64 (uncompressed).
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        layout_hints = {
            "per_layer_logical_block_size": [256, 256, 64, 64],
        }
        manager = _build_manager(
            tensors,
            num_blocks=32,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=1024,
        )
        assert len(manager.kv_layer_groups) == 2
        groups_by_logical_bs = {
            g.logical_block_size: g for g in manager.kv_layer_groups
        }
        compressed = groups_by_logical_bs[256]
        uncompressed = groups_by_logical_bs[64]
        assert compressed.layer_indices == [0, 1]
        assert compressed.compress_ratio == 4
        assert compressed.physical_chunk_size == 256
        assert uncompressed.layer_indices == [2, 3]
        assert uncompressed.compress_ratio == 1
        assert uncompressed.physical_chunk_size == 1024

    def test_hint_length_mismatch_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="length"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_logical_block_size": [64]},
            )

    def test_hint_non_positive_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="non-positive"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_logical_block_size": [64, 0]},
            )

    def test_logical_bs_not_multiple_of_physical_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(1)
        ]
        with pytest.raises(ValueError, match="multiple"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_logical_block_size": [100]},
                lmcache_logical_chunk_size=400,
            )


class TestPerLayerKvCacheGroupId:
    """Tests for the ``per_layer_kv_cache_group_id`` LayoutHints field.

    The hint adds the engine's per-layer block-ID namespace to
    LayerGroupIdentity, so layers whose physical-and-logical identity
    matches but whose engine-side block IDs come from disjoint pools
    end up in distinct LMCache groups.
    """

    def test_hint_absent_collapses_namespaces(self):
        """No hint = legacy behavior: every layer namespace is 0 and
        physically-identical layers collapse to a single group."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 1
        assert manager.kv_layer_groups[0].kv_cache_group_id == 0

    def test_hint_splits_layers_by_namespace(self):
        """Two sets of identical-shape layers from disjoint engine
        namespaces split into distinct LMCache groups."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        # Layers 0, 1 in namespace 1; layers 2, 3 in namespace 2.
        layout_hints = {
            "per_layer_kv_cache_group_id": [1, 1, 2, 2],
        }
        manager = _build_manager(
            tensors,
            num_blocks=32,
            layout_hints=layout_hints,
        )
        assert len(manager.kv_layer_groups) == 2
        groups_by_ns = {g.kv_cache_group_id: g for g in manager.kv_layer_groups}
        assert groups_by_ns[1].layer_indices == [0, 1]
        assert groups_by_ns[2].layer_indices == [2, 3]

    def test_namespace_combined_with_logical_bs(self):
        """Both hints together: layers must match on every identity
        field including namespace AND logical_block_size."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        # 4 layers all physical-identical. Namespace 1 with logical
        # bs 64; namespace 1 with logical bs 256; namespace 2 with
        # logical bs 64; namespace 2 with logical bs 64. Expected
        # groups: {(ns=1, lbs=64): [0]}, {(ns=1, lbs=256): [1]},
        # {(ns=2, lbs=64): [2, 3]}.
        layout_hints = {
            "per_layer_kv_cache_group_id": [1, 1, 2, 2],
            "per_layer_logical_block_size": [64, 256, 64, 64],
        }
        manager = _build_manager(
            tensors,
            num_blocks=32,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=1024,
        )
        assert len(manager.kv_layer_groups) == 3
        groups_by_id = {
            (g.kv_cache_group_id, g.logical_block_size): g
            for g in manager.kv_layer_groups
        }
        assert groups_by_id[(1, 64)].layer_indices == [0]
        assert groups_by_id[(1, 256)].layer_indices == [1]
        assert groups_by_id[(2, 64)].layer_indices == [2, 3]

    def test_namespace_length_mismatch_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="length"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_kv_cache_group_id": [1]},
            )

    def test_namespace_negative_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="negative"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_kv_cache_group_id": [0, -1]},
            )


class TestPerLayerSlidingWindow:
    """Tests for the ``per_layer_sliding_window`` LayoutHints field.

    The hint adds the SWA window size to LayerGroupIdentity so layers
    with different windows end up in distinct LMCache groups, and
    activates the SWA-suffix-only optimization for groups with
    non-zero windows.
    """

    def test_hint_absent_collapses_to_full(self):
        """No hint = legacy behavior: all layers treated as full-attention
        (sliding_window=0)."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kv_layer_groups) == 1
        assert manager.kv_layer_groups[0].sliding_window == 0

    def test_hint_splits_layers_by_window(self):
        """Two sets of identical-shape layers with different windows
        end up in distinct groups."""
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(4)
        ]
        layout_hints = {
            "per_layer_sliding_window": [128, 128, 0, 0],
        }
        manager = _build_manager(
            tensors,
            num_blocks=32,
            layout_hints=layout_hints,
        )
        assert len(manager.kv_layer_groups) == 2
        groups_by_window = {g.sliding_window: g for g in manager.kv_layer_groups}
        assert groups_by_window[128].layer_indices == [0, 1]
        assert groups_by_window[0].layer_indices == [2, 3]

    def test_window_length_mismatch_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="length"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_sliding_window": [128]},
            )

    def test_window_negative_raises(self):
        tensors = [
            torch.randn(2, 32, 64, 8, 128, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="negative"):
            _build_manager(
                tensors,
                num_blocks=32,
                layout_hints={"per_layer_sliding_window": [128, -1]},
            )


class TestParseKvcacheShapeSpec:
    """Test cases for parse_kvcache_shape_spec function."""

    def test_single_group(self):
        """Test parsing a single group spec."""
        groups = parse_kvcache_shape_spec("(2,1024,16,8,128):float16:32")
        assert len(groups) == 1
        g = groups[0]
        assert g.num_layers == 32
        assert g.shape_desc.kv_size == 2
        assert g.shape_desc.nb == 1024
        assert g.shape_desc.bs == 16
        assert g.shape_desc.nh == 8
        assert g.shape_desc.hs == 128
        assert g.shape_desc.nl == 32
        assert g.dtype == torch.float16
        assert g.layer_indices == list(range(32))

    def test_multiple_groups(self):
        """Test parsing multiple groups separated by semicolons."""
        spec = "(2,1024,16,8,128):float16:30;(2,1024,16,4,64):bfloat16:2"
        groups = parse_kvcache_shape_spec(spec)
        assert len(groups) == 2

        # First group: 30 layers
        assert groups[0].num_layers == 30
        assert groups[0].dtype == torch.float16
        assert groups[0].layer_indices == list(range(30))

        # Second group: 2 layers, offset by 30
        assert groups[1].num_layers == 2
        assert groups[1].dtype == torch.bfloat16
        assert groups[1].shape_desc.nh == 4
        assert groups[1].shape_desc.hs == 64
        assert groups[1].layer_indices == [30, 31]

    def test_empty_spec_raises(self):
        """Test that empty spec raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_kvcache_shape_spec("")

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid group spec"):
            parse_kvcache_shape_spec("bad_format")

    def test_unrecognized_dtype_raises(self):
        """Test that unrecognized dtype raises with helpful message."""
        with pytest.raises(ValueError, match="Unrecognized dtype"):
            parse_kvcache_shape_spec("(2,1024,16,8,128):float64:32")

    def test_invalid_number_raises(self):
        """Test that non-numeric shape values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid number"):
            parse_kvcache_shape_spec("(2,abc,16,8,128):float16:32")

    def test_whitespace_handling(self):
        """Test that whitespace around group separators is handled."""
        groups = parse_kvcache_shape_spec(
            " (2,1024,16,8,128):float16:4 ; (2,1024,16,4,64):bfloat16:2 "
        )
        assert len(groups) == 2
        assert groups[0].num_layers == 4
        assert groups[1].num_layers == 2

    def test_no_valid_groups_raises(self):
        """Test that spec with only separators raises."""
        with pytest.raises(ValueError, match="No valid layer groups"):
            parse_kvcache_shape_spec(";;;")


class TestFormatKvcacheShapeSpec:
    """Test cases for format_kvcache_shape_spec function."""

    def test_single_group(self):
        spec = "(2,1024,16,8,128):float16:32"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_multiple_groups(self):
        spec = "(2,1024,16,8,128):float16:30;(1,512,8,4,64):bfloat16:2"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_uint8_dtype(self):
        spec = "(2,1024,16,8,128):uint8:32"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_round_trip_normalizes_whitespace(self):
        """format() always produces the canonical (whitespace-free) form."""
        messy = " (2,1024,16,8,128):float16:4 ; (2,1024,16,4,64):bfloat16:2 "
        canonical = "(2,1024,16,8,128):float16:4;(2,1024,16,4,64):bfloat16:2"
        assert format_kvcache_shape_spec(parse_kvcache_shape_spec(messy)) == canonical

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="empty"):
            format_kvcache_shape_spec([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
