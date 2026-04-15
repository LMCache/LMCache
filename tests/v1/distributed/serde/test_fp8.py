# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Fp8QuantizationSerializer / Fp8QuantizationDeserializer.

These tests use BytesBufferMemoryObj and TensorMemoryObj directly so they
do not need an L1Manager or GPU; they verify the pure transform logic.
"""

# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.fp8 import (
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing the ``.tensor`` attribute used by fp8 serde."""

    tensor: Optional[torch.Tensor]


# =============================================================================
# estimate_serialized_size
# =============================================================================


def test_estimate_serialized_size_single_group() -> None:
    """Estimate includes 1.5x margin over exact fp8 size (1 byte/elem)."""
    ser = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    numel = 2 * 4 * 256 * 128
    estimated = ser.estimate_serialized_size(layout)
    # Must be >= actual fp8 bytes (numel) and include the 1.5x margin
    assert estimated == int(numel * 1.5)
    assert estimated >= numel


def test_estimate_serialized_size_multi_group() -> None:
    """Multi-group layouts sum element counts across groups, then apply margin."""
    ser = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4, 8]), torch.Size([16])],
        dtypes=[torch.bfloat16, torch.float16],
    )
    numel = 32 + 16
    estimated = ser.estimate_serialized_size(layout)
    assert estimated == int(numel * 1.5)
    assert estimated >= numel


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

    ser = Fp8QuantizationSerializer()
    n = ser.serialize(src, temp)
    assert n == original.numel()

    # Round-trip: deserialize into a fresh buffer with the original shape.
    recovered = _FakeMemoryObj(tensor=torch.zeros(shape, dtype=torch.bfloat16))
    Fp8QuantizationDeserializer().deserialize(temp, recovered)

    corr = torch.corrcoef(
        torch.stack([recovered.tensor.float().flatten(), original.float().flatten()])
    )[0, 1].item()
    assert corr > 0.99, f"fp8 round-trip correlation too low: {corr:.4f}"


def test_serialize_raises_on_missing_tensor() -> None:
    """A MemoryObj without ``.tensor`` is rejected rather than silently no-op'd."""
    ser = Fp8QuantizationSerializer()
    src = _FakeMemoryObj(tensor=None)
    dst = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        ser.serialize(src, dst)


def test_deserialize_raises_on_missing_tensor() -> None:
    deser = Fp8QuantizationDeserializer()
    src = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    dst = _FakeMemoryObj(tensor=None)
    with pytest.raises(ValueError):
        deser.deserialize(src, dst)
