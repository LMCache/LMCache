# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPUCacheContext.get_tmp_gpu_buffer and
get_tmp_gpu_buffer_batched — verifying contiguity, shape, and
non-overlapping guarantees.

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
    """Build a GPUCacheContext by directly setting internal fields,
    bypassing the KVCache/IPC wrapper construction."""
    ctx = object.__new__(GPUCacheContext)
    ctx.is_mla_ = is_mla
    ctx.num_layers_ = num_layers
    ctx.hidden_dim_size_ = num_heads * head_size
    ctx.max_batch_size = 4

    kv_dim = 1 if is_mla else 2
    total_tokens = chunk_size * ctx.max_batch_size
    shape = torch.Size((kv_dim, num_layers, total_tokens, num_heads * head_size))
    ctx.tmp_gpu_buffer_ = torch.empty(shape, dtype=dtype, device="cuda")
    return ctx


# ---------------------------------------------------------------------------
# get_tmp_gpu_buffer tests
# ---------------------------------------------------------------------------


class TestGetTmpGpuBuffer:
    @pytest.mark.parametrize("num_tokens", [64, 128, 256, 512, 1024])
    def test_contiguity(self, num_tokens: int) -> None:
        ctx = _make_context()
        buf = ctx.get_tmp_gpu_buffer(num_tokens)
        assert buf.is_contiguous(), f"Buffer not contiguous for num_tokens={num_tokens}"

    @pytest.mark.parametrize("num_tokens", [64, 128, 256])
    def test_shape(self, num_tokens: int) -> None:
        ctx = _make_context()
        buf = ctx.get_tmp_gpu_buffer(num_tokens)
        expected = ctx.get_kv_buffer_shape(num_tokens)
        assert buf.shape == expected

    @pytest.mark.parametrize("num_tokens", [64, 128, 256])
    def test_shape_mla(self, num_tokens: int) -> None:
        ctx = _make_context(is_mla=True, num_heads=1, head_size=576)
        buf = ctx.get_tmp_gpu_buffer(num_tokens)
        expected = ctx.get_kv_buffer_shape(num_tokens)
        assert buf.shape == expected
        assert buf.shape[0] == 1  # kv_dim=1 for MLA

    def test_different_sizes_dont_alias(self) -> None:
        """Two calls with different sizes should start at the same base
        (they reuse the same pre-allocated buffer)."""
        ctx = _make_context()
        buf_small = ctx.get_tmp_gpu_buffer(64)
        buf_large = ctx.get_tmp_gpu_buffer(128)
        assert buf_small.data_ptr() == buf_large.data_ptr()

    def test_write_read_roundtrip(self) -> None:
        """Write a pattern, read it back to verify the view is correct."""
        ctx = _make_context(num_layers=2, num_heads=2, head_size=16)
        buf = ctx.get_tmp_gpu_buffer(32)
        buf.fill_(42.0)
        assert buf.to(torch.float32).sum().item() == pytest.approx(
            42.0 * buf.numel(), rel=1e-3
        )


# ---------------------------------------------------------------------------
# get_tmp_gpu_buffer_batched tests
# ---------------------------------------------------------------------------


class TestGetTmpGpuBufferBatched:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_contiguity(self, batch_size: int) -> None:
        ctx = _make_context()
        buffers = ctx.get_tmp_gpu_buffer_batched(256, batch_size)
        assert len(buffers) == batch_size
        for i, buf in enumerate(buffers):
            assert buf.is_contiguous(), f"Buffer {i} not contiguous"

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_shapes(self, batch_size: int) -> None:
        ctx = _make_context()
        buffers = ctx.get_tmp_gpu_buffer_batched(256, batch_size)
        expected_shape = ctx.get_kv_buffer_shape(256)
        for buf in buffers:
            assert buf.shape == expected_shape

    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_non_overlapping(self, batch_size: int) -> None:
        """Buffers in a batch must not overlap in memory."""
        ctx = _make_context()
        buffers = ctx.get_tmp_gpu_buffer_batched(256, batch_size)
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
        ctx = _make_context(num_layers=2, num_heads=2, head_size=16)
        buffers = ctx.get_tmp_gpu_buffer_batched(32, 4)

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
        ctx = _make_context()
        with pytest.raises(AssertionError, match="exceeds max"):
            ctx.get_tmp_gpu_buffer_batched(256, 5)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
    def test_mla(self, batch_size: int) -> None:
        ctx = _make_context(is_mla=True, num_heads=1, head_size=576)
        buffers = ctx.get_tmp_gpu_buffer_batched(256, batch_size)
        for buf in buffers:
            assert buf.is_contiguous()
            assert buf.shape[0] == 1  # kv_dim=1 for MLA

    def test_consistent_with_single(self) -> None:
        """get_tmp_gpu_buffer_batched(n, 1)[0] should have the same data_ptr
        and shape as get_tmp_gpu_buffer(n)."""
        ctx = _make_context()
        single = ctx.get_tmp_gpu_buffer(256)
        batched = ctx.get_tmp_gpu_buffer_batched(256, 1)
        assert len(batched) == 1
        assert batched[0].data_ptr() == single.data_ptr()
        assert batched[0].shape == single.shape
