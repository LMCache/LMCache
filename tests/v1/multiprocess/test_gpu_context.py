# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPUCacheContext.get_tmp_chunk_gpu_buffer,
get_tmp_chunk_gpu_buffer_batched and get_tmp_gpu_buffer_flat — verifying
contiguity, shape, non-overlapping guarantees, and multi-group layout.

These tests construct a minimal GPUCacheContext-like object that has
just the fields the buffer methods need, avoiding the full KVCache /
CudaIPCWrapper construction.
"""

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# First Party
from lmcache.v1.kv_layer_groups import (  # noqa: E402
    KVLayerGroupInfo,
    KVLayerGroupsManager,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    num_layers: int = 4,
    num_heads: int = 8,
    head_size: int = 128,
    is_mla: bool = False,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> GPUCacheContext:
    """Build a GPUCacheContext with a single KV layer group by directly
    setting internal fields, bypassing the KVCache/IPC wrapper construction."""
    ctx = object.__new__(GPUCacheContext)
    ctx.is_mla_ = is_mla
    ctx.num_layers_ = num_layers
    ctx.max_batch_size = 4

    hidden_dim = num_heads * head_size
    ctx.hidden_dim_sizes_ = [hidden_dim]

    # Build a minimal KVLayerGroupsManager with a single group
    kv_size = 1 if is_mla else 2
    dummy_shape = torch.Size([kv_size, 1, 1, num_heads, head_size])
    group = KVLayerGroupInfo(
        layer_names=[str(i) for i in range(num_layers)],
        layer_indices=list(range(num_layers)),
        shape=dummy_shape,
        dtype=dtype,
    )
    manager = KVLayerGroupsManager(kv_layer_groups=[group])
    ctx.kv_layer_groups_manager_ = manager

    # Build flat tmp_gpu_buffer_ with prefix-sum offsets (new layout)
    ctx.tmp_chunk_group_offsets_ = [0]
    for gidx, grp in enumerate(manager.kv_layer_groups):
        shape = ctx.get_kv_buffer_shape(chunk_size, gidx)
        byte_size = shape.numel() * grp.dtype.itemsize
        ctx.tmp_chunk_group_offsets_.append(
            ctx.tmp_chunk_group_offsets_[-1] + byte_size
        )
    ctx.tmp_chunk_bytes_ = ctx.tmp_chunk_group_offsets_[-1]
    ctx.lmcache_chunk_size = chunk_size
    ctx.tmp_gpu_buffer_ = torch.empty(
        ctx.tmp_chunk_bytes_ * ctx.max_batch_size,
        dtype=torch.uint8,
        device="cuda",
    )
    return ctx


def _make_context_multi_group(
    groups: list[dict],
    chunk_size: int = 256,
    is_mla: bool = False,
) -> GPUCacheContext:
    """Build a GPUCacheContext with multiple KV layer groups.

    Args:
        groups: List of dicts, each with keys:
            - num_layers (int)
            - num_heads  (int)
            - head_size  (int)
            - dtype      (torch.dtype, optional, default bfloat16)
        chunk_size: Tokens per chunk.
        is_mla: Whether to use MLA (kv_dim=1) layout.
    """
    ctx = object.__new__(GPUCacheContext)
    ctx.is_mla_ = is_mla
    ctx.max_batch_size = 4

    kv_size = 1 if is_mla else 2
    kv_layer_groups = []
    hidden_dim_sizes = []
    layer_offset = 0
    for g in groups:
        nl = g["num_layers"]
        nh = g["num_heads"]
        hs = g["head_size"]
        dt = g.get("dtype", torch.bfloat16)
        hidden_dim = nh * hs
        hidden_dim_sizes.append(hidden_dim)
        dummy_shape = torch.Size([kv_size, 1, 1, nh, hs])
        kv_layer_groups.append(
            KVLayerGroupInfo(
                layer_names=[str(i) for i in range(layer_offset, layer_offset + nl)],
                layer_indices=list(range(layer_offset, layer_offset + nl)),
                shape=dummy_shape,
                dtype=dt,
            )
        )
        layer_offset += nl

    ctx.num_layers_ = layer_offset
    ctx.hidden_dim_sizes_ = hidden_dim_sizes
    manager = KVLayerGroupsManager(kv_layer_groups=kv_layer_groups)
    ctx.kv_layer_groups_manager_ = manager

    # Build flat tmp_gpu_buffer_ with prefix-sum offsets
    ctx.tmp_chunk_group_offsets_ = [0]
    for gidx, grp in enumerate(manager.kv_layer_groups):
        shape = ctx.get_kv_buffer_shape(chunk_size, gidx)
        byte_size = shape.numel() * grp.dtype.itemsize
        ctx.tmp_chunk_group_offsets_.append(
            ctx.tmp_chunk_group_offsets_[-1] + byte_size
        )
    ctx.tmp_chunk_bytes_ = ctx.tmp_chunk_group_offsets_[-1]
    ctx.lmcache_chunk_size = chunk_size
    ctx.tmp_gpu_buffer_ = torch.empty(
        ctx.tmp_chunk_bytes_ * ctx.max_batch_size,
        dtype=torch.uint8,
        device="cuda",
    )
    return ctx


# ---------------------------------------------------------------------------
# get_tmp_chunk_gpu_buffer tests
# ---------------------------------------------------------------------------


class TestGetTmpChunkGpuBuffer:
    def test_contiguity(self) -> None:
        ctx = _make_context(chunk_size=256)
        buf = ctx.get_tmp_chunk_gpu_buffer()
        assert buf.is_contiguous(), "Buffer not contiguous"

    def test_shape(self) -> None:
        ctx = _make_context(chunk_size=256)
        buf = ctx.get_tmp_chunk_gpu_buffer()
        expected = ctx.get_kv_buffer_shape(256)
        assert buf.shape == expected

    def test_shape_mla(self) -> None:
        ctx = _make_context(is_mla=True, num_heads=1, head_size=576, chunk_size=256)
        buf = ctx.get_tmp_chunk_gpu_buffer()
        expected = ctx.get_kv_buffer_shape(256)
        assert buf.shape == expected
        assert buf.shape[0] == 1  # kv_dim=1 for MLA

    def test_repeated_calls_same_ptr(self) -> None:
        """Two calls should return the same base pointer (same pre-allocated slot)."""
        ctx = _make_context(chunk_size=256)
        buf1 = ctx.get_tmp_chunk_gpu_buffer()
        buf2 = ctx.get_tmp_chunk_gpu_buffer()
        assert buf1.data_ptr() == buf2.data_ptr()

    def test_write_read_roundtrip(self) -> None:
        """Write a pattern, read it back to verify the view is correct."""
        ctx = _make_context(num_layers=2, num_heads=2, head_size=16, chunk_size=32)
        buf = ctx.get_tmp_chunk_gpu_buffer()
        buf.fill_(42.0)
        assert buf.to(torch.float32).sum().item() == pytest.approx(
            42.0 * buf.numel(), rel=1e-3
        )


# ---------------------------------------------------------------------------
# get_tmp_chunk_gpu_buffer_batched tests
# ---------------------------------------------------------------------------


class TestGetTmpChunkGpuBufferBatched:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_contiguity(self, batch_size: int) -> None:
        ctx = _make_context(chunk_size=256)
        buffers = ctx.get_tmp_chunk_gpu_buffer_batched(batch_size)
        assert len(buffers) == batch_size
        for i, buf in enumerate(buffers):
            assert buf.is_contiguous(), f"Buffer {i} not contiguous"

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_shapes(self, batch_size: int) -> None:
        ctx = _make_context(chunk_size=256)
        buffers = ctx.get_tmp_chunk_gpu_buffer_batched(batch_size)
        expected_shape = ctx.get_kv_buffer_shape(256)
        for buf in buffers:
            assert buf.shape == expected_shape

    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_non_overlapping(self, batch_size: int) -> None:
        """Buffers in a batch must not overlap in memory."""
        ctx = _make_context(chunk_size=256)
        buffers = ctx.get_tmp_chunk_gpu_buffer_batched(batch_size)
        for i in range(len(buffers)):
            for j in range(i + 1, len(buffers)):
                start_i = buffers[i].data_ptr()
                end_i = start_i + buffers[i].nelement() * buffers[i].element_size()
                start_j = buffers[j].data_ptr()
                end_j = start_j + buffers[j].nelement() * buffers[j].element_size()
                assert end_i <= start_j or end_j <= start_i, (
                    f"Buffers {i} and {j} overlap"
                )

    def test_write_isolation(self) -> None:
        """Writing to one buffer must not affect another."""
        ctx = _make_context(num_layers=2, num_heads=2, head_size=16, chunk_size=32)
        buffers = ctx.get_tmp_chunk_gpu_buffer_batched(4)

        # Write distinct values to each buffer
        for i, buf in enumerate(buffers):
            buf.fill_(float(i + 1))

        # Verify each buffer has its own value
        for i, buf in enumerate(buffers):
            expected = float(i + 1)
            assert buf.to(torch.float32).min().item() == pytest.approx(
                expected, rel=1e-3
            )
            assert buf.to(torch.float32).max().item() == pytest.approx(
                expected, rel=1e-3
            )

    def test_batch_exceeds_max_raises(self) -> None:
        ctx = _make_context(chunk_size=256)
        with pytest.raises(ValueError, match="exceeds max"):
            ctx.get_tmp_chunk_gpu_buffer_batched(5)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_mla(self, batch_size: int) -> None:
        ctx = _make_context(is_mla=True, num_heads=1, head_size=576, chunk_size=256)
        buffers = ctx.get_tmp_chunk_gpu_buffer_batched(batch_size)
        for buf in buffers:
            assert buf.is_contiguous()
            assert buf.shape[0] == 1  # kv_dim=1 for MLA

    def test_consistent_with_single(self) -> None:
        """get_tmp_chunk_gpu_buffer_batched(1)[0] should have the same data_ptr
        and shape as get_tmp_chunk_gpu_buffer()."""
        ctx = _make_context(chunk_size=256)
        single = ctx.get_tmp_chunk_gpu_buffer()
        batched = ctx.get_tmp_chunk_gpu_buffer_batched(1)
        assert len(batched) == 1
        assert batched[0].data_ptr() == single.data_ptr()
        assert batched[0].shape == single.shape


# ---------------------------------------------------------------------------
# Multi-group tests
# ---------------------------------------------------------------------------


class TestMultiGroup:
    """Tests for multi-group flat buffer layout."""

    GROUPS_SAME_DTYPE = [
        {"num_layers": 4, "num_heads": 8, "head_size": 128, "dtype": torch.bfloat16},
        {"num_layers": 4, "num_heads": 8, "head_size": 128, "dtype": torch.bfloat16},
    ]
    GROUPS_DIFF_DTYPE = [
        {"num_layers": 4, "num_heads": 8, "head_size": 128, "dtype": torch.bfloat16},
        {"num_layers": 2, "num_heads": 4, "head_size": 64, "dtype": torch.float16},
    ]

    def test_prefix_sum_length(self) -> None:
        """tmp_chunk_group_offsets_ should have num_groups+1 entries."""
        ctx = _make_context_multi_group(self.GROUPS_SAME_DTYPE)
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)
        assert len(ctx.tmp_chunk_group_offsets_) == num_groups + 1

    def test_prefix_sum_monotone(self) -> None:
        """Offsets must be strictly increasing."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE)
        offsets = ctx.tmp_chunk_group_offsets_
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1], (
                f"Offset not increasing at index {i}: {offsets}"
            )

    def test_flat_buffer_total_size(self) -> None:
        """tmp_gpu_buffer_ byte count == tmp_chunk_bytes_ * max_batch_size."""
        ctx = _make_context_multi_group(self.GROUPS_SAME_DTYPE)
        assert ctx.tmp_gpu_buffer_.numel() == ctx.tmp_chunk_bytes_ * ctx.max_batch_size

    def test_groups_non_overlapping_in_chunk(self) -> None:
        """Within a single chunk, different groups must occupy disjoint byte ranges."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE)
        offsets = ctx.tmp_chunk_group_offsets_
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                # [offsets[i], offsets[i+1]) vs [offsets[j], offsets[j+1])
                assert offsets[i + 1] <= offsets[j] or offsets[j + 1] <= offsets[i], (
                    f"Groups {i} and {j} overlap in chunk layout"
                )

    def test_get_tmp_chunk_gpu_buffer_shape_per_group(self) -> None:
        """get_tmp_chunk_gpu_buffer returns the correct shape for each group."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE, chunk_size=256)
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)
        for gidx in range(num_groups):
            buf = ctx.get_tmp_chunk_gpu_buffer(group_idx=gidx)
            expected = ctx.get_kv_buffer_shape(256, gidx)
            assert buf.shape == expected, (
                f"Group {gidx}: expected {expected}, got {buf.shape}"
            )

    def test_get_tmp_chunk_gpu_buffer_dtype_per_group(self) -> None:
        """get_tmp_chunk_gpu_buffer returns the correct dtype for each group."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE, chunk_size=256)
        groups = ctx.kv_layer_groups_manager_.kv_layer_groups
        for gidx, grp in enumerate(groups):
            buf = ctx.get_tmp_chunk_gpu_buffer(group_idx=gidx)
            assert buf.dtype == grp.dtype, (
                f"Group {gidx}: expected dtype {grp.dtype}, got {buf.dtype}"
            )

    def test_groups_data_ptr_matches_offsets(self) -> None:
        """data_ptr of each group's buffer should equal base + group offset."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE, chunk_size=256)
        base_ptr = ctx.tmp_gpu_buffer_.data_ptr()
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)
        for gidx in range(num_groups):
            buf = ctx.get_tmp_chunk_gpu_buffer(group_idx=gidx)
            expected_ptr = base_ptr + ctx.tmp_chunk_group_offsets_[gidx]
            assert buf.data_ptr() == expected_ptr, (
                f"Group {gidx}: expected ptr offset "
                f"{ctx.tmp_chunk_group_offsets_[gidx]}, "
                f"got {buf.data_ptr() - base_ptr}"
            )

    def test_write_isolation_across_groups(self) -> None:
        """Writing to one group's buffer must not corrupt another group."""
        ctx = _make_context_multi_group(self.GROUPS_SAME_DTYPE, chunk_size=64)
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)
        buffers = [ctx.get_tmp_chunk_gpu_buffer(group_idx=g) for g in range(num_groups)]

        for i, buf in enumerate(buffers):
            buf.fill_(float(i + 1))

        for i, buf in enumerate(buffers):
            expected = float(i + 1)
            assert buf.to(torch.float32).min().item() == pytest.approx(
                expected, rel=1e-3
            ), f"Group {i} was corrupted"
            assert buf.to(torch.float32).max().item() == pytest.approx(
                expected, rel=1e-3
            ), f"Group {i} was corrupted"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batched_non_overlapping_across_groups_and_chunks(
        self, batch_size: int
    ) -> None:
        """All (group, chunk_idx) combinations must occupy disjoint memory."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE, chunk_size=256)
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)

        # Collect (data_ptr, end_ptr) for every (group, chunk) combination
        regions: list[tuple[int, int, str]] = []
        for gidx in range(num_groups):
            bufs = ctx.get_tmp_chunk_gpu_buffer_batched(batch_size, group_idx=gidx)
            for cidx, buf in enumerate(bufs):
                start = buf.data_ptr()
                end = start + buf.nelement() * buf.element_size()
                regions.append((start, end, f"group={gidx},chunk={cidx}"))

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                s_i, e_i, label_i = regions[i]
                s_j, e_j, label_j = regions[j]
                assert e_i <= s_j or e_j <= s_i, (
                    f"Overlap between {label_i} and {label_j}"
                )

    def test_flat_buffer_covers_all_groups(self) -> None:
        """get_tmp_gpu_buffer_flat covers the full chunk (all groups)."""
        ctx = _make_context_multi_group(self.GROUPS_DIFF_DTYPE, chunk_size=256)
        flat = ctx.get_tmp_gpu_buffer_flat(chunk_idx=0)
        assert flat.numel() == ctx.tmp_chunk_bytes_
        assert flat.dtype == torch.uint8

    def test_flat_buffer_chunk_idx_raises(self) -> None:
        """chunk_idx >= max_batch_size should raise ValueError."""
        ctx = _make_context_multi_group(self.GROUPS_SAME_DTYPE)
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            ctx.get_tmp_gpu_buffer_flat(chunk_idx=ctx.max_batch_size)

    def test_flat_buffer_contains_group_data(self) -> None:
        """Data written via get_tmp_chunk_gpu_buffer should be visible in flat view."""
        ctx = _make_context_multi_group(self.GROUPS_SAME_DTYPE, chunk_size=64)
        num_groups = len(ctx.kv_layer_groups_manager_.kv_layer_groups)

        # Fill each group with a distinct byte value
        for gidx in range(num_groups):
            buf = ctx.get_tmp_chunk_gpu_buffer(group_idx=gidx)
            # Use view(torch.uint8) to fill raw bytes
            buf.view(torch.uint8).fill_(gidx + 1)

        flat = ctx.get_tmp_gpu_buffer_flat(chunk_idx=0)
        for gidx in range(num_groups):
            g_start = ctx.tmp_chunk_group_offsets_[gidx]
            g_end = ctx.tmp_chunk_group_offsets_[gidx + 1]
            region = flat[g_start:g_end]
            assert region.min().item() == gidx + 1, (
                f"Group {gidx} flat region has wrong min value"
            )
            assert region.max().item() == gidx + 1, (
                f"Group {gidx} flat region has wrong max value"
            )
