# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager


class TestKVLayerGroupsManager:
    """Test cases for KVLayerGroupsManager.build_kv_layer_groups method."""

    def test_build_kv_layer_groups_empty(self):
        """Test building layer groups with empty kv_caches."""
        manager = KVLayerGroupsManager()
        kv_caches: Dict[str, torch.Tensor] = {}

        manager.build_kv_layer_groups(kv_caches)

        # Should not build any groups for empty kv_caches
        assert manager.kv_layer_groups == []

    def test_build_kv_layer_groups_single_layer(self):
        """Test building layer groups with a single layer."""
        manager = KVLayerGroupsManager()

        # Create a single layer KV cache
        kv_caches = {"layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)}

        manager.build_kv_layer_groups(kv_caches)

        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert isinstance(group, KVLayerGroupInfo)
        assert group.layer_names == ["layer_0"]
        assert group.layer_indices == [0]
        assert group.shape == (2, 32, 256, 8, 64)
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_multiple_layers_same_shape(self):
        """Test building layer groups with multiple layers having the same shape."""
        manager = KVLayerGroupsManager()

        # Create multiple layers with identical shape and dtype
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)

        assert manager.kv_layer_groups is not None
        assert (
            len(manager.kv_layer_groups) == 1
        )  # All layers should be in the same group
        group = manager.kv_layer_groups[0]
        assert group.layer_names == ["layer_0", "layer_1", "layer_2"]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape == (2, 32, 256, 8, 64)
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_different_shapes(self):
        """Test building layer groups with layers having different shapes."""
        manager = KVLayerGroupsManager()

        # Create layers with different shapes
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # 8 heads
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),  # 16 heads
            "layer_2": torch.randn(
                2, 32, 256, 8, 64, dtype=torch.float16
            ),  # Same as layer_0
        }

        manager.build_kv_layer_groups(kv_caches)

        # Should create two groups: one for layers with 8 heads, one for 16 heads
        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 2

        # Sort groups by first layer index for consistent testing
        manager.kv_layer_groups.sort(key=lambda g: g.layer_indices[0])

        # First group: layers 0 and 2 (8 heads)
        group1 = manager.kv_layer_groups[0]
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.layer_indices == [0, 2]
        assert group1.shape == (2, 32, 256, 8, 64)

        # Second group: layer 1 (16 heads)
        group2 = manager.kv_layer_groups[1]
        assert group2.layer_names == ["layer_1"]
        assert group2.layer_indices == [1]
        assert group2.shape == (2, 32, 256, 16, 64)

    def test_build_kv_layer_groups_different_dtypes(self):
        """Test building layer groups with layers having different dtypes."""
        manager = KVLayerGroupsManager()

        # Create layers with different dtypes but same shape
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            "layer_2": torch.randn(
                2, 32, 256, 8, 64, dtype=torch.float16
            ),  # Same as layer_0
        }

        manager.build_kv_layer_groups(kv_caches)

        # Should create two groups: one for float16, one for float32
        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 2

        # Sort groups by first layer index for consistent testing
        manager.kv_layer_groups.sort(key=lambda g: g.layer_indices[0])

        # First group: layers 0 and 2 (float16)
        group1 = manager.kv_layer_groups[0]
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.layer_indices == [0, 2]
        assert group1.dtype == torch.float16

        # Second group: layer 1 (float32)
        group2 = manager.kv_layer_groups[1]
        assert group2.layer_names == ["layer_1"]
        assert group2.layer_indices == [1]
        assert group2.dtype == torch.float32

    def test_build_kv_layer_groups_mixed_differences(self):
        """Test building layer groups with mixed shape and dtype differences."""
        manager = KVLayerGroupsManager()

        # Create layers with various combinations of shape and dtype
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # Group A
            "layer_1": torch.randn(
                2, 32, 256, 8, 64, dtype=torch.float32
            ),  # Group B (different dtype)
            "layer_2": torch.randn(
                2, 32, 256, 16, 64, dtype=torch.float16
            ),  # Group C (different shape)
            "layer_3": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # Group A
            "layer_4": torch.randn(
                2, 32, 256, 16, 64, dtype=torch.float32
            ),  # Group D (different shape+dtype)
        }

        manager.build_kv_layer_groups(kv_caches)

        # Should create four distinct groups
        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 4

        # Sort groups by first layer index for consistent testing
        manager.kv_layer_groups.sort(key=lambda g: g.layer_indices[0])

        # Verify each group
        groups_by_key = {}
        for group in manager.kv_layer_groups:
            key = (group.shape, group.dtype)
            groups_by_key[key] = group

        # Group A: shape=(2,32,256,8,64), dtype=float16
        group_a = groups_by_key[((2, 32, 256, 8, 64), torch.float16)]
        assert set(group_a.layer_names) == {"layer_0", "layer_3"}
        assert set(group_a.layer_indices) == {0, 3}

        # Group B: shape=(2,32,256,8,64), dtype=float32
        group_b = groups_by_key[((2, 32, 256, 8, 64), torch.float32)]
        assert group_b.layer_names == ["layer_1"]
        assert group_b.layer_indices == [1]

        # Group C: shape=(2,32,256,16,64), dtype=float16
        group_c = groups_by_key[((2, 32, 256, 16, 64), torch.float16)]
        assert group_c.layer_names == ["layer_2"]
        assert group_c.layer_indices == [2]

        # Group D: shape=(2,32,256,16,64), dtype=float32
        group_d = groups_by_key[((2, 32, 256, 16, 64), torch.float32)]
        assert group_d.layer_names == ["layer_4"]
        assert group_d.layer_indices == [4]

    def test_build_kv_layer_groups_preserves_order(self):
        """Test layer groups are sorted by the first layer index to maintain order."""
        manager = KVLayerGroupsManager()

        # Create layers in non-sequential order to test sorting
        kv_caches = {
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # Index 0
            "layer_0": torch.randn(
                2, 32, 256, 8, 64, dtype=torch.float16
            ),  # Index 1 (same shape as layer_2)
            "layer_1": torch.randn(
                2, 32, 256, 16, 64, dtype=torch.float16
            ),  # Index 2 (different shape)
        }

        manager.build_kv_layer_groups(kv_caches)

        # Should create two groups
        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 2

        # Groups should be sorted by first layer index
        # layer_2 (index 0) and layer_0 (index 1) have same shape -> same group
        # layer_1 (index 2) has different shape -> separate group

        # Sort groups by first layer index for consistent testing
        manager.kv_layer_groups.sort(key=lambda g: g.layer_indices[0])

        # First group: contains layers with indices 0 and 1 (same shape)
        group1 = manager.kv_layer_groups[0]
        assert group1.layer_indices[0] == 0  # First layer index in this group
        assert set(group1.layer_names) == {"layer_2", "layer_0"}
        assert set(group1.layer_indices) == {0, 1}

        # Second group: contains layer with index 2 (different shape)
        group2 = manager.kv_layer_groups[1]
        assert group2.layer_indices[0] == 2  # First layer index in this group
        assert group2.layer_names == ["layer_1"]
        assert group2.layer_indices == [2]
        assert group2.layer_indices == [2]

    def test_build_kv_layer_groups_with_build_kv_layer_groups(self):
        """Test the complete workflow with build_kv_layer_groups."""
        manager = KVLayerGroupsManager()

        # Create test data
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        # Build the groups
        manager.build_kv_layer_groups(kv_caches)

        # Verify the manager now has the groups
        assert manager.kv_layer_groups is not None
        assert len(manager.kv_layer_groups) == 1
        assert manager.kv_layer_groups[0].layer_names == ["layer_0", "layer_1"]

    def test_get_group_methods_after_build(self):
        """Test get_group_by_layer_idx and get_group_by_layer_name."""
        manager = KVLayerGroupsManager()

        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_2": torch.randn(
                2, 32, 256, 16, 64, dtype=torch.float16
            ),  # Different shape
        }

        manager.build_kv_layer_groups(kv_caches)

        # Test get_group_by_layer_idx
        group0 = manager.get_group_by_layer_idx(0)
        assert group0 is not None
        assert group0.contains_layer(0)
        assert group0.contains_layer(1)
        assert not group0.contains_layer(2)

        group2 = manager.get_group_by_layer_idx(2)
        assert group2 is not None
        assert group2.contains_layer(2)
        assert not group2.contains_layer(0)

        # Test get_group_by_layer_name
        group_layer0 = manager.get_group_by_layer_name("layer_0")
        assert group_layer0 is not None
        assert group_layer0.contains_layer_name("layer_0")
        assert group_layer0.contains_layer_name("layer_1")
        assert not group_layer0.contains_layer_name("layer_2")

        group_layer2 = manager.get_group_by_layer_name("layer_2")
        assert group_layer2 is not None
        assert group_layer2.contains_layer_name("layer_2")
        assert not group_layer2.contains_layer_name("layer_0")

        # Test get_layer_shape and get_layer_dtype
        assert manager.get_layer_shape(0) == (2, 32, 256, 8, 64)
        assert manager.get_layer_shape(2) == (2, 32, 256, 16, 64)
        assert manager.get_layer_dtype(0) == torch.float16
        assert manager.get_layer_dtype(2) == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
