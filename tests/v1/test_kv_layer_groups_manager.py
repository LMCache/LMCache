# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence

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

        assert len(manager.kv_layer_groups) == 2
        groups_by_engine_group_idx = {
            group.engine_group_idx: group for group in manager.kv_layer_groups
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


class TestPerGroupCompressionMetadata:
    """Per-group compress_ratio / physical_chunk_size derivation.

    Exercises the hybrid-KV-cache-manager case where the engine's groups have
    *different* logical block sizes, carried per layer via
    ``LayoutHints["per_layer_inference_engine_logical_block_size"]``.
    """

    def _v4_like_tensors(self) -> list[torch.Tensor]:
        """Two engine groups with different physical block sizes.

        - layers 0,1: physical bs=64 (V4 SWA-like: logical 64 -> ratio 1)
        - layer  2:   physical bs=4  (V4 state-like: logical 4 -> ratio 1)

        The shapes also differ in ``hs`` so each engine group becomes its own
        identity even before the engine-group split.
        """
        return [
            torch.randn(1, 32, 64, 1, 128, dtype=torch.float16),  # bs=64
            torch.randn(1, 32, 64, 1, 128, dtype=torch.float16),  # bs=64
            torch.randn(1, 32, 4, 1, 512, dtype=torch.float16),  # bs=4
        ]

    def test_per_group_logical_block_size_drives_ratio(self):
        """Each group's compress_ratio uses its own logical block size."""
        tensors = self._v4_like_tensors()
        # logical block sizes: groups [0,1] = 64, group [2] = 4
        layout_hints: LayoutHints = {
            "inference_engine_logical_block_size": 4,  # GCD scalar (wrong per group)
            "per_layer_inference_engine_logical_block_size": [64, 64, 4],
        }
        manager = KVLayerGroupsManager(
            tensors,
            gpu_kv_format=_nhd_format(),
            num_blocks=32,
            layout_hints=layout_hints,
            group_views=[
                LMCacheGroupView(0, (0, 1)),
                LMCacheGroupView(1, (2,)),
            ],
            lmcache_logical_chunk_size=1024,
        )
        by_engine = {g.engine_group_idx: g for g in manager.kv_layer_groups}
        swa = by_engine[0]
        state = by_engine[1]
        # SWA group: physical bs 64, logical 64 -> ratio 1, full chunk 1024 slots
        assert swa.compress_ratio == 1
        assert swa.logical_block_size == 64
        assert swa.physical_chunk_size == 1024
        assert swa.blocks_per_chunk == 16  # 1024 / 64
        # State group: physical bs 4, logical 4 -> ratio 1, 256 blocks/chunk
        assert state.compress_ratio == 1
        assert state.logical_block_size == 4
        assert state.physical_chunk_size == 1024
        assert state.blocks_per_chunk == 256  # 1024 / 4

    def test_compressed_group_ratio_gt_one(self):
        """logical block size larger than physical bs -> compress_ratio > 1."""
        tensors = [torch.randn(1, 32, 64, 1, 128, dtype=torch.float16)]
        layout_hints: LayoutHints = {
            "per_layer_inference_engine_logical_block_size": [256],
        }
        manager = KVLayerGroupsManager(
            tensors,
            gpu_kv_format=_nhd_format(),
            num_blocks=32,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=1024,
        )
        g = manager.kv_layer_groups[0]
        assert g.compress_ratio == 4  # 256 / 64
        assert g.logical_block_size == 256
        assert g.physical_chunk_size == 256  # 1024 / 4
        assert g.blocks_per_chunk == 4  # 256 / 64

    def test_scalar_fallback_when_no_per_layer_hint(self):
        """Absent per-layer hint -> scalar inference_engine_logical_block_size."""
        tensors = [torch.randn(1, 32, 64, 1, 128, dtype=torch.float16)]
        layout_hints: LayoutHints = {
            "inference_engine_logical_block_size": 256,
        }
        manager = KVLayerGroupsManager(
            tensors,
            gpu_kv_format=_nhd_format(),
            num_blocks=32,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=1024,
        )
        g = manager.kv_layer_groups[0]
        assert g.compress_ratio == 4  # 256 / 64 (from the scalar)

    def test_zero_per_layer_entry_falls_back_to_scalar(self):
        """A 0 entry in the per-layer list defers to the scalar for that group."""
        tensors = [torch.randn(1, 32, 64, 1, 128, dtype=torch.float16)]
        layout_hints: LayoutHints = {
            "inference_engine_logical_block_size": 256,
            "per_layer_inference_engine_logical_block_size": [0],
        }
        manager = KVLayerGroupsManager(
            tensors,
            gpu_kv_format=_nhd_format(),
            num_blocks=32,
            layout_hints=layout_hints,
            lmcache_logical_chunk_size=1024,
        )
        g = manager.kv_layer_groups[0]
        assert g.compress_ratio == 4  # falls back to scalar 256

    def test_no_hint_at_all_treats_as_uncompressed(self):
        """No layout hints -> compress_ratio 1 (physical == logical)."""
        tensors = [torch.randn(1, 32, 64, 1, 128, dtype=torch.float16)]
        manager = KVLayerGroupsManager(
            tensors,
            gpu_kv_format=_nhd_format(),
            num_blocks=32,
            layout_hints=None,
            lmcache_logical_chunk_size=1024,
        )
        g = manager.kv_layer_groups[0]
        assert g.compress_ratio == 1
        assert g.logical_block_size == 64
        assert g.blocks_per_chunk == 16


class TestNumSuffixBlocksPerChunk:
    """``num_suffix_blocks_per_chunk`` / SWA-suffix helper math."""

    def _group(
        self,
        *,
        bs: int,
        compress_ratio: int,
        physical_chunk_size: int,
        sliding_window: int,
    ) -> KVLayerGroupInfo:
        # First Party
        import lmcache.c_ops as lmc_ops

        sd = lmc_ops.PageBufferShapeDesc()
        sd.kv_size = 1
        sd.nl = 1
        sd.nb = 32
        sd.bs = bs
        sd.nh = 1
        sd.hs = 128
        sd.element_size = 2
        return KVLayerGroupInfo(
            layer_indices=[0],
            shape_desc=sd,
            dtype=torch.float16,
            compress_ratio=compress_ratio,
            physical_chunk_size=physical_chunk_size,
            sliding_window=sliding_window,
        )

    def test_full_attention_returns_none(self):
        g = self._group(
            bs=64, compress_ratio=4, physical_chunk_size=256, sliding_window=0
        )
        assert g.num_suffix_blocks_per_chunk is None

    def test_swa_ceil(self):
        # logical bs = 64*1 = 64, window 128 -> ceil(128/64) = 2
        g = self._group(
            bs=64, compress_ratio=1, physical_chunk_size=1024, sliding_window=128
        )
        assert g.logical_block_size == 64
        assert g.blocks_per_chunk == 16
        assert g.num_suffix_blocks_per_chunk == 2

    def test_swa_ceil_rounds_up(self):
        # logical bs 4, window 8 -> ceil(8/4) = 2
        g = self._group(
            bs=4, compress_ratio=1, physical_chunk_size=1024, sliding_window=8
        )
        assert g.blocks_per_chunk == 256
        assert g.num_suffix_blocks_per_chunk == 2

    def test_window_larger_than_chunk_capped(self):
        # logical bs 64, window 4096 -> ceil = 64, but capped at blocks_per_chunk 16
        g = self._group(
            bs=64, compress_ratio=1, physical_chunk_size=1024, sliding_window=4096
        )
        assert g.num_suffix_blocks_per_chunk == 16


class TestStoragePerChunkHelpers:
    """SWA-suffix storage helpers that the transfer loops + buffers share."""

    def _group(
        self,
        *,
        bs: int,
        compress_ratio: int,
        physical_chunk_size: int,
        sliding_window: int,
    ) -> KVLayerGroupInfo:
        # First Party
        import lmcache.c_ops as lmc_ops

        sd = lmc_ops.PageBufferShapeDesc()
        sd.kv_size = 1
        sd.nl = 1
        sd.nb = 32
        sd.bs = bs
        sd.nh = 1
        sd.hs = 128
        sd.element_size = 2
        return KVLayerGroupInfo(
            layer_indices=[0],
            shape_desc=sd,
            dtype=torch.float16,
            compress_ratio=compress_ratio,
            physical_chunk_size=physical_chunk_size,
            sliding_window=sliding_window,
        )

    def test_full_attention_stores_whole_chunk(self):
        # MLA full-attn: physical bs 64, logical 256, chunk 256 slots, 4 blk/chunk
        g = self._group(
            bs=64, compress_ratio=4, physical_chunk_size=256, sliding_window=0
        )
        assert g.storage_blocks_per_chunk == g.blocks_per_chunk == 4
        assert g.storage_slots_per_chunk == g.physical_chunk_size == 256
        assert g.storage_tokens_per_chunk == 1024  # 4 blocks * logical 256
        assert g.chunk_suffix_offset_blocks == 0

    def test_swa_stores_only_trailing_window(self):
        # SWA: physical bs 64, logical 64, window 128, chunk 1024 slots, 16 blk/chunk
        g = self._group(
            bs=64, compress_ratio=1, physical_chunk_size=1024, sliding_window=128
        )
        assert g.blocks_per_chunk == 16
        assert g.storage_blocks_per_chunk == 2  # ceil(128/64)
        assert g.storage_slots_per_chunk == 2 * 64  # 128 physical slots
        assert g.storage_tokens_per_chunk == 2 * 64  # logical_bs == bs here
        assert g.chunk_suffix_offset_blocks == 14  # 16 - 2

    def test_state_group_large_shrink(self):
        # C4A state: physical bs 4, logical 4, window 8, 256 blk/chunk
        g = self._group(
            bs=4, compress_ratio=1, physical_chunk_size=1024, sliding_window=8
        )
        assert g.blocks_per_chunk == 256
        assert g.storage_blocks_per_chunk == 2  # ceil(8/4)
        assert g.storage_slots_per_chunk == 8
        assert g.chunk_suffix_offset_blocks == 254  # 256 - 2

    def test_kernel_chunk_size_invariant(self):
        """storage_slots_per_chunk must equal storage_blocks_per_chunk * bs.

        The kernel asserts num_blocks_per_object * bs == lmcache_chunk_size; we
        pass storage_blocks_per_chunk block IDs and storage_slots_per_chunk as
        lmcache_chunk_size, so this identity is what keeps it valid.
        """
        for g in [
            self._group(bs=64, compress_ratio=4, physical_chunk_size=256,
                        sliding_window=0),
            self._group(bs=64, compress_ratio=1, physical_chunk_size=1024,
                        sliding_window=128),
            self._group(bs=4, compress_ratio=1, physical_chunk_size=1024,
                        sliding_window=8),
            self._group(bs=8, compress_ratio=1, physical_chunk_size=1024,
                        sliding_window=128),
        ]:
            assert (
                g.storage_slots_per_chunk
                == g.storage_blocks_per_chunk * g.shape_desc.bs
            )


def _nhd_format():
    # First Party
    import lmcache.c_ops as lmc_ops

    return lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS


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
