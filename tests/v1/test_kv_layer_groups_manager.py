# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)


def _build_manager(
    kv_caches: Dict[str, torch.Tensor],
    *,
    num_blocks: int,
    block_size: int,
) -> KVLayerGroupsManager:
    """Build a manager from a name→tensor dict using the per-layer NHD format.

    Tensors in these tests have shape ``[2, NB, BS, NH, HS]`` — the
    canonical vLLM flash-attention per-layer NHD layout matched by
    ``GPUKVFormat.NL_X_TWO_NB_BS_NH_HS``.
    """
    # First Party
    import lmcache.c_ops as lmc_ops

    return KVLayerGroupsManager(
        list(kv_caches.values()),
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        num_blocks=num_blocks,
        block_size=block_size,
        layer_names=list(kv_caches.keys()),
    )


class TestKVLayerGroupsManager:
    """Tests for KVLayerGroupsManager construction and lookups."""

    def test_build_kv_layer_groups_empty(self):
        """Test building layer groups with empty kv_caches."""
        manager = _build_manager({}, num_blocks=32, block_size=256)
        assert manager.kv_layer_groups == []

    def test_build_kv_layer_groups_single_layer(self):
        """Test building layer groups with a single layer."""
        kv_caches = {"layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)}
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert isinstance(group, KVLayerGroupInfo)
        assert group.layer_names == ["layer_0"]
        assert group.layer_indices == [0]
        assert group.shape_desc.kv_size == 2
        assert group.shape_desc.nh == 8
        assert group.shape_desc.hs == 64
        assert group.shape_desc.nl == 1
        assert group.shape_desc.nb == 32
        assert group.shape_desc.bs == 256
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_multiple_layers_same_shape(self):
        """Test building layer groups with multiple layers having the same shape."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert group.layer_names == ["layer_0", "layer_1", "layer_2"]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape_desc.nl == 3
        assert group.shape_desc.nh == 8
        assert group.shape_desc.hs == 64
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_different_shapes(self):
        """Test building layer groups with layers having different head counts."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)
        assert len(manager.kv_layer_groups) == 2

        group1 = manager.kv_layer_groups[0]
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.layer_indices == [0, 2]
        assert group1.shape_desc.nh == 8

        group2 = manager.kv_layer_groups[1]
        assert group2.layer_names == ["layer_1"]
        assert group2.layer_indices == [1]
        assert group2.shape_desc.nh == 16

    def test_build_kv_layer_groups_different_dtypes(self):
        """Test building layer groups with layers having different dtypes."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)
        assert len(manager.kv_layer_groups) == 2

        group1 = manager.kv_layer_groups[0]
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.dtype == torch.float16

        group2 = manager.kv_layer_groups[1]
        assert group2.layer_names == ["layer_1"]
        assert group2.dtype == torch.float32

    def test_build_kv_layer_groups_mixed_differences(self):
        """Test building layer groups with mixed shape and dtype differences."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            "layer_2": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            "layer_3": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_4": torch.randn(2, 32, 256, 16, 64, dtype=torch.float32),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)
        assert len(manager.kv_layer_groups) == 4

        groups_by_key = {(g.shape_desc.nh, g.dtype): g for g in manager.kv_layer_groups}
        assert set(groups_by_key[(8, torch.float16)].layer_names) == {
            "layer_0",
            "layer_3",
        }
        assert groups_by_key[(8, torch.float32)].layer_names == ["layer_1"]
        assert groups_by_key[(16, torch.float16)].layer_names == ["layer_2"]
        assert groups_by_key[(16, torch.float32)].layer_names == ["layer_4"]

    def test_build_kv_layer_groups_preserves_order(self):
        """Groups are sorted by the first layer index to maintain order."""
        kv_caches = {
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)
        assert len(manager.kv_layer_groups) == 2

        group1 = manager.kv_layer_groups[0]
        assert group1.layer_indices[0] == 0
        assert set(group1.layer_names) == {"layer_2", "layer_0"}
        assert set(group1.layer_indices) == {0, 1}

        group2 = manager.kv_layer_groups[1]
        assert group2.layer_indices == [2]
        assert group2.layer_names == ["layer_1"]

    def test_build_stores_format_and_topology(self):
        """The manager records gpu_kv_format, num_blocks, block_size at build."""
        # First Party
        import lmcache.c_ops as lmc_ops

        kv_caches = {"layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)}
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)
        assert manager.gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert manager.num_blocks == 32
        assert manager.block_size == 256

    def test_get_group_methods_after_build(self):
        """Test get_group_by_layer_idx and get_group_by_layer_name."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)

        group0 = manager.get_group_by_layer_idx(0)
        assert group0 is not None
        assert group0.contains_layer(0)
        assert group0.contains_layer(1)
        assert not group0.contains_layer(2)

        group2 = manager.get_group_by_layer_idx(2)
        assert group2 is not None
        assert group2.contains_layer(2)
        assert not group2.contains_layer(0)

        group_layer0 = manager.get_group_by_layer_name("layer_0")
        assert group_layer0 is not None
        assert group_layer0.contains_layer_name("layer_0")

        assert manager.get_layer_dtype(0) == torch.float16
        assert manager.get_layer_dtype(2) == torch.float16

    def test_get_shape_desc_by_group_idx(self):
        """Test that get_shape_desc returns the expected per-group descriptor."""
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        }
        manager = _build_manager(kv_caches, num_blocks=32, block_size=256)

        sd0 = manager.get_shape_desc(0)
        assert sd0.nh == 8
        assert sd0.hs == 64
        assert sd0.nl == 1

        sd1 = manager.get_shape_desc(1)
        assert sd1.nh == 16
        assert sd1.hs == 64
        assert sd1.nl == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
