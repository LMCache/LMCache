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


# =============================================================================
# exact bytes and values
# =============================================================================

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_KV_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# Zeros, rounding ties, subnormals, each format's max finite value and just
# past it, overflow, inf and nan, so every cast branch is exercised.
_SPECIAL_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, 1.0625, 1.1875, 2.0**-6, 2.0**-9, 2.0**-10,
    2.0**-16, 2.0**-17, 1e-30, 240.0, 448.0, 464.0, 480.0, 57344.0, 61440.0,
    65504.0, 1e30, float("inf"), float("-inf"), float("nan"),
]  # fmt: skip


def _kv_values(dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
    """Return a tensor of ``shape`` that starts with every special value."""
    values = torch.randn(shape, generator=torch.Generator().manual_seed(0)) * 300
    special = torch.tensor(_SPECIAL_VALUES)
    flat = values.view(-1)
    flat[: special.numel()] = special
    flat[special.numel() : 2 * special.numel()] = -special
    return values.to(dtype)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """View ``t`` as integers so NaN payloads and signed zeros compare exactly."""
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.contiguous().view(int_dtype)


@pytest.mark.parametrize("fp8_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("kv_dtype", _KV_DTYPES)
def test_serialize_writes_fp8_cast_bytes(
    kv_dtype: torch.dtype, fp8_dtype: torch.dtype
) -> None:
    """dst holds the bytes of ``src.to(fp8_dtype)``; bytes past them are untouched."""
    shape = torch.Size([2, 3, 16, 8])
    original = _kv_values(kv_dtype, shape)
    expected = original.to(fp8_dtype).view(torch.uint8).flatten()
    n = original.numel()
    sentinel = 0xAB
    dst = _FakeMemoryObj(tensor=torch.full((n + 7,), sentinel, dtype=torch.uint8))

    written = Fp8QuantizationSerializer(fp8_dtype).serialize(
        _FakeMemoryObj(tensor=original.clone()),  # type: ignore[arg-type]
        dst,  # type: ignore[arg-type]
        _TEST_KEY,
    )

    assert written == n
    assert dst.tensor is not None
    assert torch.equal(dst.tensor[:n], expected)
    assert torch.equal(dst.tensor[n:], torch.full((7,), sentinel, dtype=torch.uint8))


# Hand-derived from the OCP fp8 bit layouts: (value, stored byte). Stored L2
# objects must keep decoding the same way across releases.
_KNOWN_CODES = {
    torch.float8_e4m3fn: [
        (1.0, 0x38), (-1.0, 0xB8), (448.0, 0x7E), (-0.0, 0x80),
        (2.0**-9, 0x01), (0.0, 0x00),
    ],
    torch.float8_e5m2: [
        (1.0, 0x3C), (57344.0, 0x7B), (float("inf"), 0x7C),
        (float("-inf"), 0xFC), (2.0**-16, 0x01), (-0.0, 0x80),
    ],
}  # fmt: skip


@pytest.mark.parametrize("fp8_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("kv_dtype", _KV_DTYPES)
def test_known_values_map_to_known_bytes(
    kv_dtype: torch.dtype, fp8_dtype: torch.dtype
) -> None:
    """Exactly representable values store as fixed bytes and decode back unchanged."""
    values = torch.tensor([v for v, _ in _KNOWN_CODES[fp8_dtype]]).to(kv_dtype)
    codes = torch.tensor([c for _, c in _KNOWN_CODES[fp8_dtype]], dtype=torch.uint8)
    stored = _FakeMemoryObj(tensor=torch.zeros(len(codes), dtype=torch.uint8))
    decoded = _FakeMemoryObj(tensor=torch.zeros(len(codes), dtype=kv_dtype))

    Fp8QuantizationSerializer(fp8_dtype).serialize(
        _FakeMemoryObj(tensor=values),  # type: ignore[arg-type]
        stored,  # type: ignore[arg-type]
        _TEST_KEY,
    )
    Fp8QuantizationDeserializer(fp8_dtype).deserialize(
        stored,  # type: ignore[arg-type]
        decoded,  # type: ignore[arg-type]
        _TEST_KEY,
    )

    assert stored.tensor is not None and decoded.tensor is not None
    assert torch.equal(stored.tensor, codes)
    assert torch.equal(_bits(decoded.tensor), _bits(values))


def test_serialize_non_contiguous_src_uses_row_major_order() -> None:
    """A strided src is written in its logical row-major element order."""
    original = _kv_values(torch.bfloat16, torch.Size([8, 4, 16])).transpose(0, 2)
    assert not original.is_contiguous()
    expected = original.contiguous().to(torch.float8_e4m3fn).view(torch.uint8)
    dst = _FakeMemoryObj(tensor=torch.zeros(original.numel(), dtype=torch.uint8))

    Fp8QuantizationSerializer().serialize(
        _FakeMemoryObj(tensor=original),  # type: ignore[arg-type]
        dst,  # type: ignore[arg-type]
        _TEST_KEY,
    )

    assert dst.tensor is not None
    assert torch.equal(dst.tensor, expected.flatten())


def test_serialize_raises_when_dst_too_small() -> None:
    """A dst smaller than one byte per element is an error, not a truncated write."""
    src = _FakeMemoryObj(tensor=torch.ones(16, dtype=torch.bfloat16))
    dst = _FakeMemoryObj(tensor=torch.zeros(15, dtype=torch.uint8))
    with pytest.raises(RuntimeError):
        Fp8QuantizationSerializer().serialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]


@pytest.mark.parametrize("fp8_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("kv_dtype", _KV_DTYPES)
def test_deserialize_casts_every_fp8_code(
    kv_dtype: torch.dtype, fp8_dtype: torch.dtype
) -> None:
    """Each fp8 code decodes to ``code.to(kv_dtype)``; extra src bytes are ignored."""
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    payload = codes.repeat(4)
    src = _FakeMemoryObj(
        tensor=torch.cat([payload, torch.full((9,), 0x7F, dtype=torch.uint8)])
    )
    shape = torch.Size([4, 16, 16])
    expected = payload.view(fp8_dtype).to(kv_dtype).reshape(shape)
    dst = _FakeMemoryObj(tensor=torch.zeros(shape, dtype=kv_dtype))

    Fp8QuantizationDeserializer(fp8_dtype).deserialize(
        src,  # type: ignore[arg-type]
        dst,  # type: ignore[arg-type]
        _TEST_KEY,
    )

    assert dst.tensor is not None
    assert torch.equal(_bits(dst.tensor), _bits(expected))
