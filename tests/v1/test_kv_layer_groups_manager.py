# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import (
    EXCLUDED_ENGINE_GROUP,
    KernelGroupIdentity,
    KernelGroupInfo,
    KVLayerGroupInfo,
    KVLayerGroupsManager,
    LayerGroupIdentity,
    ObjectGroupInfo,
    format_kvcache_shape_spec,
    parse_kvcache_shape_spec,
)
from lmcache.v1.multiprocess.group_view import LMCacheGroupView

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)


def _build_manager(
    tensors: list[torch.Tensor],
    *,
    num_blocks: int,
    layout_hints: LayoutHints | None = None,
    group_views: Sequence[LMCacheGroupView] = (),
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
        group_views=group_views,
    )


class TestKVLayerGroupsManager:
    """Tests for KVLayerGroupsManager construction and lookups."""

    def test_build_empty(self):
        manager = _build_manager([], num_blocks=32)
        assert manager.kernel_groups == []

    def test_build_single_layer(self):
        tensors = [torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)]
        manager = _build_manager(tensors, num_blocks=32)

        assert len(manager.kernel_groups) == 1
        group = manager.kernel_groups[0]
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

        assert len(manager.kernel_groups) == 1
        group = manager.kernel_groups[0]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape_desc.nl == 3
        assert group.shape_desc.nh == 8
        assert group.engine_group_idx == 0

    def test_build_splits_same_shape_by_engine_group_idx(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(4)
        ]
        manager = _build_manager(
            tensors,
            num_blocks=32,
            group_views=[
                LMCacheGroupView(0, (0, 2)),
                LMCacheGroupView(1, (1, 3)),
            ],
        )

        assert len(manager.kernel_groups) == 2
        groups_by_engine_group_idx = {
            group.engine_group_idx: group for group in manager.kernel_groups
        }
        assert groups_by_engine_group_idx[0].layer_indices == [0, 2]
        assert groups_by_engine_group_idx[1].layer_indices == [1, 3]

    def test_build_rejects_bad_group_views(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="outside registered layer"):
            _build_manager(
                tensors,
                num_blocks=32,
                group_views=[LMCacheGroupView(0, (2,))],
            )

    def test_build_different_shapes(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert len(manager.kernel_groups) == 2
        group1, group2 = manager.kernel_groups
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
        assert len(manager.kernel_groups) == 2
        group1, group2 = manager.kernel_groups
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
        assert len(manager.kernel_groups) == 4

        groups_by_key = {(g.shape_desc.nh, g.dtype): g for g in manager.kernel_groups}
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


class TestDeriveCompressionMetadata:
    """``(compress_ratio, physical_chunk_size)`` derivation: ``1`` when there is
    no engine block size, else ``ie_logical_block_size // bs`` (e.g. DeepSeek V4
    compression where ``bs < logical``), with divisibility enforced.
    """

    def _derive(self, bs: int, logical: "int | None", chunk: int = 256):
        return KVLayerGroupsManager._derive_compression_metadata(
            group_idx=0,
            bs=bs,
            ie_logical_block_size=logical,
            lmcache_logical_chunk_size=chunk,
        )

    def test_one_to_one(self):
        assert self._derive(bs=16, logical=16) == (1, 256)

    def test_no_block_size_info(self):
        assert self._derive(bs=16, logical=None) == (1, 256)

    def test_compression_bs_lt_logical(self):
        # bs=8 packs 2 logical tokens per physical slot (DeepSeek V4 style).
        assert self._derive(bs=8, logical=16) == (2, 128)

    def test_not_divisible_raises(self):
        # Divisibility is enforced loudly (e.g. bs=6 does not divide 16).
        with pytest.raises(ValueError, match="must be a multiple of"):
            self._derive(bs=6, logical=16)


class TestKernelGroupIdentity:
    """The grouping key is a named tuple; ``LayerGroupIdentity`` is its alias."""

    def test_fields_and_alias(self):
        ident = KernelGroupIdentity(
            kv_size=2,
            num_heads=8,
            head_size=64,
            block_size=16,
            engine_group_idx=0,
            dtype=torch.float16,
        )
        assert ident.kv_size == 2
        assert ident.num_heads == 8
        assert ident.head_size == 64
        assert ident.block_size == 16
        assert ident.engine_group_idx == 0
        assert ident.dtype == torch.float16
        assert LayerGroupIdentity is KernelGroupIdentity

    def test_hashable_as_dict_key(self):
        ident = KernelGroupIdentity(2, 8, 64, 16, 0, torch.float16)
        assert {ident: "x"}[ident] == "x"

    def test_excluded_engine_group_sentinel(self):
        assert EXCLUDED_ENGINE_GROUP == -1


class TestKernelAndObjectGroups:
    """Kernel-group accessors, deprecated aliases, and the (currently single)
    object-group layout."""

    def test_kernel_groups_match_deprecated_alias(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(tensors, num_blocks=32)
        # The deprecated alias must still return the live list, not a bound
        # method (regression guard for the @property/@deprecate ordering).
        assert isinstance(manager.kv_layer_groups, list)
        assert manager.kernel_groups is manager.kv_layer_groups
        assert manager.num_kernel_groups == manager.num_groups
        assert manager.num_kernel_groups == len(manager.kernel_groups)
        assert all(isinstance(g, KernelGroupInfo) for g in manager.kernel_groups)

    def test_single_object_group_covers_all_kernel_groups(self):
        # Two distinct kernel groups (different num_heads) still share one
        # object group under the current single-object-group assumption.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, num_blocks=32)
        assert manager.num_kernel_groups == 2
        assert manager.num_object_groups == 1
        obj = manager.object_groups[0]
        assert isinstance(obj, ObjectGroupInfo)
        assert obj.kernel_group_indices == list(range(manager.num_kernel_groups))

    def test_empty_manager_has_no_groups(self):
        # Empty registration returns early in __init__; both group lists must
        # still be initialized (regression guard for missing _object_groups).
        manager = _build_manager([], num_blocks=32)
        assert manager.kernel_groups == []
        assert manager.num_kernel_groups == 0
        assert manager.object_groups == []
        assert manager.num_object_groups == 0

    def test_excluded_layer_left_out_of_all_groups(self):
        # Layer 2 is referenced by no group view, so it is excluded entirely.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(
            tensors,
            num_blocks=32,
            group_views=[LMCacheGroupView(0, (0, 1))],
        )
        grouped = sorted(
            idx for group in manager.kernel_groups for idx in group.layer_indices
        )
        assert grouped == [0, 1]

    def test_calculate_num_blocks_uncompressed(self):
        # bs=16, compress_ratio=1 -> 256 tokens span 16 blocks.
        tensors = [torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(tensors, num_blocks=32)
        assert manager.calculate_num_blocks(0, 256) == 16

    def test_calculate_num_blocks_compressed(self):
        # bs=8, ie_logical_block_size=16 -> compress_ratio=2;
        # 256 logical tokens -> 128 physical slots -> 128 // 8 = 16 blocks.
        tensors = [torch.randn(2, 32, 8, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(
            tensors,
            num_blocks=32,
            layout_hints={"inference_engine_logical_block_size": 16},
        )
        assert manager.calculate_num_blocks(0, 256) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
