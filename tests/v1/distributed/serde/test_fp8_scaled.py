# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the scaled FP8 serde path (``FP8Serde(scaled=True)``).

The kernel-backed accuracy tests require a CUDA device and the built
``lmcache.cuda_ops`` extension; they are skipped on CPU-only CI. The
size-estimate, factory-kwarg, and CPU-fallback tests run on CPU.

Covers the reviewer comments on LMCache PR #5527:
  * scale metadata round-trip (serialize writes scales + metadata;
    deserialize applies scales) — serializer -> deserializer regression
  * factory forwards ``scaled`` / ``block_size`` / ``amax_ceiling`` kwargs
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.fp8 import (
    _create_fp8_serde,
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
    _n_scales,
)

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)
_FP8 = torch.float8_e4m3fn


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing the ``.tensor`` attribute used by fp8 serde."""

    tensor: Optional[torch.Tensor]


def _has_cuda_ops() -> bool:
    """True only when the CUDA extension and a GPU are both present."""
    try:
        from lmcache import cuda_ops  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


_REQUIRES_CUDA = pytest.mark.skipif(
    not _has_cuda_ops(), reason="scaled FP8 path needs CUDA + lmcache.cuda_ops"
)


def _roundtrip(
    tensor: torch.Tensor,
    scaled: bool,
    block_size: int = 128,
    amax_ceiling: float = 0.0,
) -> torch.Tensor:
    """Serialize ``tensor`` to fp8 and deserialize back; return recovered."""
    serializer = Fp8QuantizationSerializer(
        fp8_dtype=_FP8, scaled=scaled, block_size=block_size,
        amax_ceiling=amax_ceiling,
    )
    deserializer = Fp8QuantizationDeserializer(
        fp8_dtype=_FP8, scaled=scaled, block_size=block_size,
    )
    layout = MemoryLayoutDesc(shapes=[tensor.shape], dtypes=[tensor.dtype])
    size = serializer.estimate_serialized_size(layout)

    src = _FakeMemoryObj(tensor=tensor)
    temp = _FakeMemoryObj(tensor=torch.zeros(size, dtype=torch.uint8))
    n = serializer.serialize(src, temp, _TEST_KEY)
    assert 0 < n <= size

    recovered = _FakeMemoryObj(
        tensor=torch.zeros(tensor.shape, dtype=tensor.dtype)
    )
    deserializer.deserialize(temp, recovered, _TEST_KEY)
    assert recovered.tensor is not None
    return recovered.tensor


def _rel_rms_err(recovered: torch.Tensor, original: torch.Tensor) -> float:
    diff = recovered.float() - original.float()
    return float(
        torch.sqrt(torch.mean(diff * diff)) / (torch.sqrt(torch.mean(original.float() * original.float())) + 1e-12)
    )


# =============================================================================
# estimate_serialized_size (CPU-runnable)
# =============================================================================


def test_estimate_serialized_size_scale_free_unchanged() -> None:
    """Scale-free estimate stays exactly 1 byte per element (no regression)."""
    serializer = Fp8QuantizationSerializer(scaled=False)
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])], dtypes=[torch.bfloat16]
    )
    assert serializer.estimate_serialized_size(layout) == 2 * 4 * 256 * 128


def test_estimate_serialized_size_scaled_includes_metadata() -> None:
    """Scaled estimate adds [u32 count] + [f32 scales] on top of fp8 bytes."""
    serializer = Fp8QuantizationSerializer(scaled=True, block_size=128)
    shape = torch.Size([2, 4, 256, 128])  # 262144 elems; divisible by 128
    n = 2 * 4 * 256 * 128
    n_scales = n // 128
    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[torch.bfloat16])
    expected = 4 + 4 * n_scales + n
    assert serializer.estimate_serialized_size(layout) == expected


def test_estimate_serialized_size_scaled_rowwise_fallback() -> None:
    """Non-divisible size falls back to rowwise: one scale per last-dim slice."""
    serializer = Fp8QuantizationSerializer(scaled=True, block_size=128)
    shape = torch.Size([3, 100])  # 300 elems, not divisible by 128; cols=100
    n = 300
    expected_n_scales = _n_scales(n, 128, 100)  # rowwise: n // cols = 3
    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[torch.bfloat16])
    assert serializer.estimate_serialized_size(layout) == 4 + 4 * expected_n_scales + n


# =============================================================================
# factory forwards kwargs (CPU-runnable)
# =============================================================================


def test_factory_forwards_scaled_kwargs() -> None:
    """``scaled``/``block_size``/``amax_ceiling`` reach serializer+deserializer."""
    processor = _create_fp8_serde(
        {
            "fp8_dtype": "float8_e4m3fn",
            "scaled": True,
            "block_size": 64,
            "amax_ceiling": 100.0,
            "max_workers": 2,
        }
    )
    ser = processor._serializer  # type: ignore[attr-defined]
    de = processor._deserializer  # type: ignore[attr-defined]
    assert ser._scaled is True
    assert de._scaled is True
    assert ser._block_size == 64
    assert de._block_size == 64
    assert ser._amax_ceiling == 100.0


def test_factory_defaults_unchanged() -> None:
    """Default factory config stays scale-free (existing users unaffected)."""
    processor = _create_fp8_serde({"fp8_dtype": "float8_e4m3fn"})
    ser = processor._serializer  # type: ignore[attr-defined]
    assert ser._scaled is False
    assert ser._block_size == 128
    assert ser._amax_ceiling == 0.0


# =============================================================================
# CPU fallback: scaled on CPU must degrade gracefully to scale-free
# =============================================================================


def test_scaled_cpu_falls_back_to_scale_free_roundtrip() -> None:
    """On CPU (no kernel), scaled serialize/deserialize still round-trips."""
    shape = torch.Size([2, 4, 64, 128])
    original = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    recovered = _roundtrip(original, scaled=True)
    corr = torch.corrcoef(
        torch.stack([recovered.float().flatten(), original.float().flatten()])
    )[0, 1].item()
    assert corr > 0.99, f"CPU fallback round-trip correlation too low: {corr:.4f}"


# =============================================================================
# Kernel-backed accuracy tests (require CUDA + cuda_ops)
# =============================================================================


@_REQUIRES_CUDA
def test_scaled_roundtrip_restores_magnitude() -> None:
    """scaled=True restores original magnitude, not the 0..448 normalized range."""
    from lmcache import cuda_ops

    shape = torch.Size([64, 128])  # 8192 elems, divisible by 128 -> blockwise
    original = torch.randn(shape, device="cuda", generator=torch.Generator().manual_seed(1)) * 40.0
    recovered = _roundtrip(original, scaled=True, block_size=128)
    rel_err = _rel_rms_err(recovered, original)
    # With amax ~40 the scaled path must be far below saturation error.
    assert rel_err < 0.05, f"scaled round-trip rel RMS too high: {rel_err:.4%}"

    # Sanity: an all-ones block must round-trip to ~1, not ~448.
    ones = torch.ones(128, device="cuda")
    recovered_ones = _roundtrip(ones, scaled=True, block_size=128)
    assert torch.allclose(recovered_ones, torch.ones_like(recovered_ones), atol=1e-2)


@_REQUIRES_CUDA
def test_scaled_never_worse_than_scale_free() -> None:
    """Scaled blockwise rel RMS must not exceed the scale-free path on outlier data."""
    from lmcache import cuda_ops

    # KV-like distribution: mostly small values with a few large outliers.
    gen = torch.Generator(device="cuda").manual_seed(2)
    original = torch.randn(2048, 128, device="cuda", generator=gen) * 5.0
    original[0, 0] = 40.0  # one outlier compresses scale-free range

    recovered_scaled = _roundtrip(original, scaled=True, block_size=128)
    recovered_free = _roundtrip(original, scaled=False)
    err_scaled = _rel_rms_err(recovered_scaled, original)
    err_free = _rel_rms_err(recovered_free, original)
    assert err_scaled <= err_free + 1e-6, (
        f"scaled ({err_scaled:.4%}) worse than scale-free ({err_free:.4%})"
    )


@_REQUIRES_CUDA
def test_scale_tracks_actual_amax() -> None:
    """Each block's scale equals block_amax / 448 (not the e4m3 limit)."""
    from lmcache import cuda_ops

    shape = torch.Size([4, 128])  # 4 blocks of 128
    original = torch.randn(shape, device="cuda", generator=torch.Generator().manual_seed(3)) * 10.0
    x = original.contiguous().to(torch.float32)
    n = x.numel()
    fp8 = torch.empty_like(x, dtype=_FP8)
    scales = torch.empty(n // 128, dtype=torch.float32, device=x.device)
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 2, 128, 0.0)

    for g in range(scales.numel()):
        block = x[g * 128 : (g + 1) * 128]
        amax = float(torch.max(torch.abs(block)))
        expected_scale = (amax / 448.0) if amax > 0.015625 else 1.0
        assert torch.isclose(
            scales[g], torch.tensor(expected_scale, device="cuda"), rtol=1e-5, atol=1e-8
        ), f"block {g}: scale={scales[g].item():.6f} expected {expected_scale:.6f} (amax={amax})"


@_REQUIRES_CUDA
def test_zero_tensor_does_not_divide_by_zero() -> None:
    """An all-zero tensor must not crash and must round-trip to zeros."""
    from lmcache import cuda_ops

    x = torch.zeros(256, device="cuda")
    fp8 = torch.empty_like(x, dtype=_FP8)
    scales = torch.empty(2, dtype=torch.float32, device=x.device)
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 2, 128, 0.0)
    assert torch.all(fp8.to(torch.float32) == 0.0)
    recovered = _roundtrip(torch.zeros(256, device="cuda"), scaled=True, block_size=128)
    assert torch.all(recovered == 0.0)


@_REQUIRES_CUDA
def test_amax_ceiling_pins_scale() -> None:
    """When amax_ceiling > tensor amax, scale is pinned at ceiling / 448."""
    from lmcache import cuda_ops

    x = torch.full((128,), 5.0, device="cuda")  # amax = 5 << ceiling
    fp8 = torch.empty_like(x, dtype=_FP8)
    scales = torch.empty(1, dtype=torch.float32, device=x.device)
    cuda_ops.fp8_quantize_scaled(x, fp8, scales, 2, 128, 100.0)
    expected = 100.0 / 448.0
    assert torch.isclose(scales[0], torch.tensor(expected, device="cuda"), rtol=1e-5)


@_REQUIRES_CUDA
def test_non_divisible_block_size_does_not_corrupt() -> None:
    """Size not divisible by block_size (rowwise fallback) round-trips cleanly."""
    shape = torch.Size([7, 100])  # 700 elems, not divisible by 128
    original = torch.randn(shape, device="cuda", generator=torch.Generator().manual_seed(4)) * 20.0
    recovered = _roundtrip(original, scaled=True, block_size=128)
    assert torch.isfinite(recovered).all()
    rel_err = _rel_rms_err(recovered, original)
    assert rel_err < 0.1, f"non-divisible round-trip rel RMS too high: {rel_err:.4%}"
