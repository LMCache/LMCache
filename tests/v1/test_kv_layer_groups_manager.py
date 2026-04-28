# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)


def _build_manager(
    tensors: list[torch.Tensor],
    *,
    num_blocks: int,
    block_size: int,
) -> KVLayerGroupsManager:
    """Build a manager using the per-layer NHD format.

    Tensors in these tests have shape ``[2, NB, BS, NH, HS]`` — the
    canonical vLLM flash-attention per-layer NHD layout matched by
    ``GPUKVFormat.NL_X_TWO_NB_BS_NH_HS``.
    """
    # First Party
    import lmcache.c_ops as lmc_ops

    return KVLayerGroupsManager(
        tensors,
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        num_blocks=num_blocks,
        block_size=block_size,
    )


class TestKVLayerGroupsManager:
    """Tests for KVLayerGroupsManager construction and lookups."""

    def test_build_empty(self):
        manager = _build_manager([], num_blocks=32, block_size=256)
        assert manager.kv_layer_groups == []

    def test_build_single_layer(self):
        tensors = [torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)]
        manager = _build_manager(tensors, num_blocks=32, block_size=256)

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
        manager = _build_manager(tensors, num_blocks=32, block_size=256)

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
        manager = _build_manager(tensors, num_blocks=32, block_size=256)
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
        manager = _build_manager(tensors, num_blocks=32, block_size=256)
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
        manager = _build_manager(tensors, num_blocks=32, block_size=256)
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
        manager = _build_manager(tensors, num_blocks=32, block_size=256)

        sd0 = manager.get_shape_desc(0)
        assert sd0.nh == 8
        assert sd0.hs == 64
        assert sd0.nl == 1

        sd1 = manager.get_shape_desc(1)
        assert sd1.nh == 16
        assert sd1.hs == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
