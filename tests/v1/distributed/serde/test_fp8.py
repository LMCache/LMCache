# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Fp8QuantizationSerializer / Fp8QuantizationDeserializer.

These tests use BytesBufferMemoryObj and TensorMemoryObj directly so they
do not need an BaseL1Manager or GPU; they verify the pure transform logic.
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
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)

# Every byte value, so the NaN and signed-zero codes are covered too.
_ALL_BYTES = torch.arange(256, dtype=torch.uint8)
_ONE_BYTE_KV_DTYPES = ("uint8", "float8_e4m3fn", "float8_e5m2")
_FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")
_WIDE_KV_DTYPES = ("float16", "bfloat16", "float32")

# Exact in every dtype above; bytes hand-checked against the fp8 bit layouts.
_QUANT_VALUES = [0.0, -0.0, 0.5, 1.0, -2.0, 448.0]
_QUANT_BYTES = {
    "float8_e4m3fn": [0x00, 0x80, 0x30, 0x38, 0xC0, 0x7E],
    "float8_e5m2": [0x00, 0x80, 0x38, 0x3C, 0xC0, 0x5F],
}


def _torch_dtype(name: str) -> torch.dtype:
    """Return ``torch.<name>``, skipping the test if this torch lacks it."""
    dtype = getattr(torch, name, None)
    if dtype is None:
        pytest.skip(f"torch {torch.__version__} has no {name}")
    return dtype


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing the ``.tensor`` attribute used by fp8 serde."""

    tensor: Optional[torch.Tensor]


# =============================================================================
# estimate_serialized_size
# =============================================================================


def test_estimate_serialized_size_single_group() -> None:
    """Estimate is exactly num_elements bytes (1 byte/elem, no margin)."""
    serializer = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    numel = 2 * 4 * 256 * 128
    assert serializer.estimate_serialized_size(layout) == numel


def test_estimate_serialized_size_multi_group() -> None:
    """Multi-group layouts sum element counts across groups."""
    serializer = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4, 8]), torch.Size([16])],
        dtypes=[torch.bfloat16, torch.float16],
    )
    numel = 32 + 16
    assert serializer.estimate_serialized_size(layout) == numel


# =============================================================================
# serialize / deserialize round-trip
# =============================================================================


def test_roundtrip_bfloat16_preserves_structure() -> None:
    """Values survive fp8 round-trip with high correlation."""
    shape = torch.Size([2, 4, 64, 128])
    original = torch.randn(
        shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
    )
    src = _FakeMemoryObj(tensor=original.clone())

    # fp8 = 1 byte/elem; temp buffer is plain uint8.
    temp = _FakeMemoryObj(tensor=torch.zeros(original.numel(), dtype=torch.uint8))

    serializer = Fp8QuantizationSerializer()
    n = serializer.serialize(src, temp, _TEST_KEY)  # type: ignore[arg-type]
    assert n == original.numel()

    # Round-trip: deserialize into a fresh buffer with the original shape.
    recovered = _FakeMemoryObj(tensor=torch.zeros(shape, dtype=torch.bfloat16))
    Fp8QuantizationDeserializer().deserialize(temp, recovered, _TEST_KEY)  # type: ignore[arg-type]

    assert recovered.tensor is not None
    corr = torch.corrcoef(
        torch.stack([recovered.tensor.float().flatten(), original.float().flatten()])
    )[0, 1].item()
    assert corr > 0.99, f"fp8 round-trip correlation too low: {corr:.4f}"


def test_roundtrip_fp8_kv_cache_is_byte_exact() -> None:
    """An fp8 KV cache (uint8 codes, as vLLM stores it) round-trips unchanged."""
    shape = torch.Size([2, 4, 64, 128])
    original = torch.randint(
        0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0)
    )
    src = _FakeMemoryObj(tensor=original.clone())
    temp = _FakeMemoryObj(tensor=torch.zeros(original.numel(), dtype=torch.uint8))

    n = Fp8QuantizationSerializer().serialize(src, temp, _TEST_KEY)  # type: ignore[arg-type]
    assert n == original.numel()

    recovered = _FakeMemoryObj(tensor=torch.zeros(shape, dtype=torch.uint8))
    Fp8QuantizationDeserializer().deserialize(temp, recovered, _TEST_KEY)  # type: ignore[arg-type]

    assert recovered.tensor is not None
    assert torch.equal(recovered.tensor, original)


@pytest.mark.parametrize("fp8_format", _FP8_FORMATS)
@pytest.mark.parametrize("kv_dtype", _ONE_BYTE_KV_DTYPES)
def test_serialize_one_byte_kv_copies_every_byte(
    kv_dtype: str, fp8_format: str
) -> None:
    """A 1-byte KV cache is stored verbatim whatever fp8 format is configured."""
    src = _FakeMemoryObj(tensor=_ALL_BYTES.reshape(16, 16).view(_torch_dtype(kv_dtype)))
    dst = _FakeMemoryObj(tensor=torch.zeros(256, dtype=torch.uint8))

    serializer = Fp8QuantizationSerializer(_torch_dtype(fp8_format))
    n = serializer.serialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]

    assert n == 256
    assert dst.tensor is not None
    assert torch.equal(dst.tensor, _ALL_BYTES)


@pytest.mark.parametrize("fp8_format", _FP8_FORMATS)
@pytest.mark.parametrize("kv_dtype", _ONE_BYTE_KV_DTYPES)
def test_deserialize_one_byte_kv_restores_every_byte(
    kv_dtype: str, fp8_format: str
) -> None:
    """Stored bytes land in a 1-byte KV cache unchanged, bit for bit."""
    src = _FakeMemoryObj(tensor=_ALL_BYTES.clone())
    dst = _FakeMemoryObj(
        tensor=torch.zeros(16, 16, dtype=torch.uint8).view(_torch_dtype(kv_dtype))
    )

    deserializer = Fp8QuantizationDeserializer(_torch_dtype(fp8_format))
    deserializer.deserialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]

    assert dst.tensor is not None
    assert torch.equal(dst.tensor.view(torch.uint8).flatten(), _ALL_BYTES)


@pytest.mark.parametrize("fp8_format", _FP8_FORMATS)
@pytest.mark.parametrize("kv_dtype", _WIDE_KV_DTYPES)
def test_serialize_wide_kv_quantizes_to_fp8(kv_dtype: str, fp8_format: str) -> None:
    """fp16/bf16/fp32 KV caches are still cast to the configured fp8 format."""
    src = _FakeMemoryObj(
        tensor=torch.tensor(_QUANT_VALUES, dtype=_torch_dtype(kv_dtype))
    )
    dst = _FakeMemoryObj(tensor=torch.zeros(len(_QUANT_VALUES), dtype=torch.uint8))

    serializer = Fp8QuantizationSerializer(_torch_dtype(fp8_format))
    n = serializer.serialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]

    assert n == len(_QUANT_VALUES)
    assert dst.tensor is not None
    assert dst.tensor.tolist() == _QUANT_BYTES[fp8_format]


@pytest.mark.parametrize("fp8_format", _FP8_FORMATS)
@pytest.mark.parametrize("kv_dtype", _WIDE_KV_DTYPES)
def test_deserialize_wide_kv_dequantizes_from_fp8(
    kv_dtype: str, fp8_format: str
) -> None:
    """fp8 bytes are cast back to the fp16/bf16/fp32 destination's values."""
    src = _FakeMemoryObj(
        tensor=torch.tensor(_QUANT_BYTES[fp8_format], dtype=torch.uint8)
    )
    dst = _FakeMemoryObj(
        tensor=torch.zeros(len(_QUANT_VALUES), dtype=_torch_dtype(kv_dtype))
    )

    deserializer = Fp8QuantizationDeserializer(_torch_dtype(fp8_format))
    deserializer.deserialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]

    # Compare bits so -0.0 must come back as -0.0.
    expected = torch.tensor(_QUANT_VALUES, dtype=_torch_dtype(kv_dtype))
    assert dst.tensor is not None
    assert torch.equal(dst.tensor.view(torch.uint8), expected.view(torch.uint8))


def test_serialize_raises_on_missing_tensor() -> None:
    """A MemoryObj without ``.tensor`` is rejected rather than silently no-op'd."""
    serializer = Fp8QuantizationSerializer()
    src = _FakeMemoryObj(tensor=None)
    dst = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        serializer.serialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]


def test_deserialize_raises_on_missing_tensor() -> None:
    deserializer = Fp8QuantizationDeserializer()
    src = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    dst = _FakeMemoryObj(tensor=None)
    with pytest.raises(ValueError):
        deserializer.deserialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]
