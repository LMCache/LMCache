# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
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
