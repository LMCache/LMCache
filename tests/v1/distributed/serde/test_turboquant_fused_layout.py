# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for TurboQuant serde with fused blocks-first K/V layout.

vLLM >= 0.26 attention backends produce fused KV cache tensors with
``kv_size == 1`` and the per-head ``[K | V]`` pair packed in the trailing
dimension (``hidden_dim = num_heads * 2 * head_dim``). TurboQuant serde
originally hardcoded the split K/V layout (``kv_size == 2``) and rejected
fused tensors with ``ValueError``. These tests lock in fused-layout support
and guard the split-layout path from regressions.
"""

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantDeserializer,
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)
from lmcache.v1.distributed.serde.turboquant.turboquant import (
    _raw_group_nbytes,
    _validate_layout_shape,
)
from lmcache.v1.memory_management import MemoryObj

# Same correlation thresholds as the existing split-layout round-trip tests.
_FUSED_CORR_LOWER_BOUND = {
    "turboquant_k8v4": 0.95,
    "turboquant_4bit_nc": 0.90,
    "turboquant_k3v4_nc": 0.85,
    "turboquant_3bit_nc": 0.80,
}

_HEAD_DIM = 128
_NUM_HEADS = 4
# Fused hidden dim: num_heads * 2 * head_dim.
_FUSED_HIDDEN = _NUM_HEADS * 2 * _HEAD_DIM
# Split hidden dim: num_heads * head_dim.
_SPLIT_HIDDEN = _NUM_HEADS * _HEAD_DIM


def _make_turboquant_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="turboquant_fused_test_model",
        kv_rank=0,
    )


def _fused_shape(num_layers: int = 3, num_tokens: int = 20) -> torch.Size:
    return torch.Size([1, num_layers, num_tokens, _FUSED_HIDDEN])


def _split_shape(num_layers: int = 3, num_tokens: int = 20) -> torch.Size:
    return torch.Size([2, num_layers, num_tokens, _SPLIT_HIDDEN])


def _make_fused_cfg(
    preset: str = "turboquant_k8v4",
    **kwargs,
) -> TurboQuantSerdeConfig:
    return TurboQuantSerdeConfig(
        preset=preset,
        head_dim=_HEAD_DIM,
        block_size=16,
        **kwargs,
    )


# =============================================================================
# Fused layout validation (no device required)
# =============================================================================


def test_validate_layout_shape_accepts_fused_kv() -> None:
    cfg = _make_fused_cfg()
    num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
        _fused_shape(), cfg
    )
    assert (num_layers, num_tokens, num_heads, head_dim) == (
        3,
        20,
        _NUM_HEADS,
        _HEAD_DIM,
    )


def test_validate_layout_shape_split_kv_unchanged() -> None:
    cfg = _make_fused_cfg()
    num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
        _split_shape(), cfg
    )
    assert (num_layers, num_tokens, num_heads, head_dim) == (
        3,
        20,
        _NUM_HEADS,
        _HEAD_DIM,
    )


@pytest.mark.parametrize("kv_size", [0, 3, -1])
def test_validate_layout_shape_rejects_unsupported_kv_size(kv_size: int) -> None:
    cfg = _make_fused_cfg()
    shape = torch.Size([kv_size, 3, 20, _FUSED_HIDDEN])
    with pytest.raises(ValueError, match="kv_size 1 \\(fused K/V\\) or 2"):
        _validate_layout_shape(shape, cfg)


def test_validate_layout_shape_rejects_fused_bad_hidden_dim() -> None:
    cfg = _make_fused_cfg()
    # A kv_size==1 tensor whose hidden dim is not a multiple of 2 * head_dim
    # is not a fused K/V tensor (e.g. an MLA index cache) and must be
    # rejected rather than silently mis-split.
    shape = torch.Size([1, 3, 20, _FUSED_HIDDEN + _HEAD_DIM])
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_layout_shape(shape, cfg)


def test_validate_layout_shape_rejects_split_bad_hidden_dim() -> None:
    cfg = _make_fused_cfg()
    shape = torch.Size([2, 3, 20, _SPLIT_HIDDEN + 1])
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_layout_shape(shape, cfg)


# =============================================================================
# Serialized size accounting (no device required)
# =============================================================================


def test_estimate_serialized_size_accepts_fused_kv() -> None:
    """The fused layout must no longer be rejected at size estimation."""
    cfg = _make_fused_cfg()
    serializer = TurboQuantSerializer(cfg)
    layout = MemoryLayoutDesc(
        shapes=[_fused_shape()],
        dtypes=[torch.bfloat16],
    )
    # No exception raised; expected size matches the raw split-equivalent.
    assert serializer.estimate_serialized_size(layout) > 0


def test_serialized_size_fused_matches_split() -> None:
    """Equivalent fused and split layouts must serialize to the same size."""
    num_layers, num_tokens = 4, 32
    cfg = _make_fused_cfg(
        skip_first_layers=2,
        skip_last_layers=2,
    )
    serializer = TurboQuantSerializer(cfg)
    fused = MemoryLayoutDesc(
        shapes=[_fused_shape(num_layers, num_tokens)],
        dtypes=[torch.bfloat16],
    )
    split = MemoryLayoutDesc(
        shapes=[_split_shape(num_layers, num_tokens)],
        dtypes=[torch.bfloat16],
    )
    assert serializer.estimate_serialized_size(
        fused
    ) == serializer.estimate_serialized_size(split)


def test_raw_group_nbytes_fused_matches_split() -> None:
    num_layers, num_tokens = 4, 32
    dtype = torch.float16
    fused_bytes = _raw_group_nbytes(
        _fused_shape(num_layers, num_tokens), dtype, num_layers
    )
    split_bytes = _raw_group_nbytes(
        _split_shape(num_layers, num_tokens), dtype, num_layers
    )
    assert fused_bytes == split_bytes
    assert fused_bytes == num_layers * num_tokens * _FUSED_HIDDEN * dtype.itemsize


def test_fused_asymmetric_presets_use_distinct_kv_bits() -> None:
    """Guard that the asymmetric presets really differ, so the fused
    round-trip below exercises separate K/V quantization paths."""
    for preset in ("turboquant_k8v4", "turboquant_k3v4_nc"):
        cfg = _make_fused_cfg(preset)
        assert cfg.key_quant_bits != cfg.value_quant_bits, (
            f"{preset} must be asymmetric to exercise K/V split"
        )


# =============================================================================
# Fused round-trip (GPU)
# =============================================================================


class _FakeMemoryObj:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
@pytest.mark.parametrize("preset", list(_FUSED_CORR_LOWER_BOUND))
def test_turboquant_fused_direct_roundtrip_cuda(preset: str) -> None:
    """Direct GPU round-trip through TurboQuant serializer/deserializer on a
    fused K/V tensor (kv_size == 1).

    Verifies:
    - the reconstructed shape exactly matches the input fused shape;
    - the reconstructed dtype matches the input dtype;
    - both the K half and the V half of the trailing dimension reconstruct
      within the preset's correlation bound (catches implementations that
      wrongly quantize fused K/V as a single tensor).
    """
    device = torch.device(f"{torch_device_type}:0")
    dtype = torch.float16

    num_layers = 4
    num_tokens = 128
    cfg = _make_fused_cfg(preset, skip_first_layers=0, skip_last_layers=0)

    torch.manual_seed(2026)
    shape = torch.Size([1, num_layers, num_tokens, _NUM_HEADS * 2 * _HEAD_DIM])
    original = torch.randn(shape, dtype=dtype, device=device)

    serializer = TurboQuantSerializer(cfg)
    deserializer = TurboQuantDeserializer(cfg)

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    n_bytes = serializer.estimate_serialized_size(layout)

    compressed = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    recovered = torch.empty_like(original)

    written = serializer.serialize(
        cast(MemoryObj, _FakeMemoryObj(original)),
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        _make_turboquant_object_key(0),
    )
    assert written == n_bytes

    deserializer.deserialize(
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        cast(MemoryObj, _FakeMemoryObj(recovered)),
        _make_turboquant_object_key(0),
    )

    assert recovered.shape == original.shape
    assert recovered.dtype == original.dtype

    orig_view = original[0].view(num_layers, num_tokens, _NUM_HEADS, 2 * _HEAD_DIM)
    rec_view = recovered[0].view(num_layers, num_tokens, _NUM_HEADS, 2 * _HEAD_DIM)

    corr_lower_bound = _FUSED_CORR_LOWER_BOUND[preset]
    full_corr = torch.corrcoef(
        torch.stack([original.float().flatten(), recovered.float().flatten()])
    )[0, 1].item()
    k_corr = torch.corrcoef(
        torch.stack(
            [
                orig_view[..., :_HEAD_DIM].float().flatten(),
                rec_view[..., :_HEAD_DIM].float().flatten(),
            ]
        )
    )[0, 1].item()
    v_corr = torch.corrcoef(
        torch.stack(
            [
                orig_view[..., _HEAD_DIM:].float().flatten(),
                rec_view[..., _HEAD_DIM:].float().flatten(),
            ]
        )
    )[0, 1].item()

    assert full_corr > corr_lower_bound, (
        f"low corr for preset={preset}: full={full_corr}, k={k_corr}, v={v_corr}"
    )
    assert k_corr > corr_lower_bound, f"low K-half corr for preset={preset}: k={k_corr}"
    assert v_corr > corr_lower_bound, f"low V-half corr for preset={preset}: v={v_corr}"


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires torch_device_type",
)
def test_turboquant_fused_skipped_layers_raw_exact_cuda() -> None:
    """Layers outside the quantized range are stored raw and must round-trip
    bit-exactly for the fused layout too."""
    device = torch.device(f"{torch_device_type}:0")
    dtype = torch.float16

    num_layers = 4
    num_tokens = 64
    cfg = _make_fused_cfg("turboquant_k8v4")
    # Default skip_first_layers=2 / skip_last_layers=2 leaves no layers to
    # quantize; the whole tensor is stored raw.
    assert cfg.skip_first_layers + cfg.skip_last_layers >= num_layers

    torch.manual_seed(7)
    shape = torch.Size([1, num_layers, num_tokens, _NUM_HEADS * 2 * _HEAD_DIM])
    original = torch.randn(shape, dtype=dtype, device=device)

    serializer = TurboQuantSerializer(cfg)
    deserializer = TurboQuantDeserializer(cfg)

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    n_bytes = serializer.estimate_serialized_size(layout)

    compressed = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    recovered = torch.empty_like(original)

    serializer.serialize(
        cast(MemoryObj, _FakeMemoryObj(original)),
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        _make_turboquant_object_key(0),
    )
    deserializer.deserialize(
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        cast(MemoryObj, _FakeMemoryObj(recovered)),
        _make_turboquant_object_key(0),
    )

    assert recovered.shape == original.shape
    assert torch.equal(recovered, original)
