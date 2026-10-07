# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused scaled FP8 KV serializer.

The scaled path must never be *less* accurate than the current scale-free
`.to(torch.float8_e4m3fn)`, and the kernels must round-trip within FP8
tolerance.
"""

import pytest
import torch

pytest.importorskip("lmcache.cuda_ops")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

from lmcache import cuda_ops  # noqa: E402


def _kv_like(num_tokens=64, num_heads=8, head_size=128, seed=0):
    """KV-shaped tensor with occasional large outliers."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        num_tokens,
        num_heads,
        head_size,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    flat = x.view(-1)
    flat[::977] *= 40.0  # outliers, like real post-RoPE K
    return x


def _rel_rms(deq: torch.Tensor, ref: torch.Tensor) -> float:
    err = (deq.float() - ref.float()).pow(2).sum().sqrt()
    return (err / ref.float().pow(2).sum().sqrt()).item() * 100.0


@pytest.mark.parametrize("mode,block", [(0, 128), (1, 128), (2, 128)])
def test_scaled_beats_scale_free(mode, block):
    x = _kv_like()
    n = x.numel()

    # current path
    ref_err = _rel_rms(x.to(torch.float8_e4m3fn).to(torch.float32), x)

    # fused path
    xf = x.to(torch.float32).contiguous()
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    n_scales = 1 if mode == 0 else (n // block if mode == 2 else x.size(-1))
    scales = torch.empty(max(1, n_scales), dtype=torch.float32, device="cuda")
    cuda_ops.fp8_quantize_scaled(xf, fp8, scales, mode, block, 0.0)

    deq = fp8.to(torch.float32)
    if mode == 0:
        deq = deq * scales[0]
    elif mode == 1:
        deq = deq * scales.view(*([1] * (x.dim() - 1)), -1)
    else:
        deq = deq * scales.repeat_interleave(block).view_as(deq)

    assert _rel_rms(deq, x) <= ref_err + 1e-6


def test_scale_uses_actual_amax():
    """A small tensor must use its amax, not the e4m3 hard limit."""
    x = torch.full((256, 128), 2.0, device="cuda", dtype=torch.float32)
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(-1), dtype=torch.float32, device="cuda")
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 1, 128, 0.0)

    # scales[] holds the dequant multiplier amax/448, not the quant inverse
    assert torch.allclose(scales, torch.full_like(scales, 2.0 / 448.0), atol=1e-6)


def test_zero_tensor_no_div_by_zero():
    x = torch.zeros((64, 128), device="cuda", dtype=torch.float32)
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(-1), dtype=torch.float32, device="cuda")
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 1, 128, 0.0)
    assert torch.isfinite(scales).all()
    assert torch.all(fp8.to(torch.float32) == 0.0)


def test_amax_ceiling():
    """ceiling forces a minimum scale so tensors stay comparable."""
    x = torch.full((64, 128), 0.5, device="cuda", dtype=torch.float32)
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(-1), dtype=torch.float32, device="cuda")
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 1, 128, 100.0)
    assert torch.allclose(scales, torch.full_like(scales, 100.0 / 448.0), atol=1e-6)


def test_indivisible_tensor_falls_back_to_rowwise():
    """n not divisible by block_size must not silently corrupt."""
    x = torch.randn(37, 100, device="cuda", dtype=torch.float32)
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(100, dtype=torch.float32, device="cuda")
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 1, 128, 0.0)
    deq = fp8.to(torch.float32) * scales.view(1, -1)
    assert _rel_rms(deq, x) < 20.0
