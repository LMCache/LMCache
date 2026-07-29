# SPDX-License-Identifier: Apache-2.0
"""Tests for MP CacheGen distributed serde."""

# Standard
from dataclasses import dataclass
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.cachegen import (
    CacheGenMpDeserializer,
    CacheGenMpSerializer,
    cachegen_layout_to_mp_tensor,
    mp_layout_to_cachegen_tensor,
)
from lmcache.v1.memory_management import MemoryObj

_GPU_AVAILABLE = torch_device_type == "cuda" or torch_device_type == "xpu"


@dataclass
class _FakeMemoryObj:
    """Small MemoryObj test double exposing tensor and byte access."""

    tensor: torch.Tensor

    def get_size(self) -> int:
        """Return the tensor size in bytes."""
        return int(self.tensor.numel() * self.tensor.element_size())

    @property
    def byte_array(self) -> memoryview:
        """Return a byte view of the CPU tensor contents."""
        return memoryview(self.tensor.cpu().numpy())


def _obj(tensor: torch.Tensor) -> MemoryObj:
    """Cast a fake object into the MemoryObj protocol used by serde tests."""
    return cast(MemoryObj, _FakeMemoryObj(tensor=tensor))


def _kv_tensor(num_tokens: int, device: str) -> torch.Tensor:
    """Create a CacheGen-shaped KV tensor for tests."""
    torch.manual_seed(123)
    return torch.rand(
        32,
        2,
        num_tokens,
        8,
        128,
        dtype=torch.bfloat16,
        device=device,
    )


def _mp_kv_tensor(num_tokens: int) -> torch.Tensor:
    """Create an MP-shaped KV tensor for tests."""
    return torch.arange(
        2 * 32 * num_tokens * 8 * 128,
        dtype=torch.float32,
    ).reshape(2, 32, num_tokens, 8 * 128)


def test_estimate_serialized_size_includes_cachegen_metadata_overhead() -> None:
    """The CacheGen estimate includes metadata overhead beyond raw KV bytes."""
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=16,
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([32, 2, 16, 8, 128])],
        dtypes=[torch.bfloat16],
    )
    raw_bytes = (
        32 * 2 * 16 * 8 * 128 * torch.empty((), dtype=torch.bfloat16).element_size()
    )
    estimate = serializer.estimate_serialized_size(layout)

    assert estimate >= raw_bytes
    assert estimate > raw_bytes


def test_estimate_serialized_size_handles_mp_layout_with_metadata_overhead() -> None:
    """MP layout estimates include CacheGen overhead after shape conversion."""
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=16,
        num_heads=8,
        head_size=128,
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 32, 16, 8 * 128])],
        dtypes=[torch.bfloat16],
    )
    raw_bytes = 2 * 32 * 16 * 8 * 128 * torch.empty((), dtype=torch.bfloat16).itemsize

    estimate = serializer.estimate_serialized_size(layout)

    assert estimate >= raw_bytes
    assert estimate > raw_bytes


def test_estimate_serialized_size_rejects_mp_hidden_dim_mismatch() -> None:
    """MP layout estimation rejects inconsistent head metadata."""
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=16,
        num_heads=4,
        head_size=128,
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 32, 16, 8 * 128])],
        dtypes=[torch.bfloat16],
    )

    with pytest.raises(ValueError, match="hidden_dim"):
        serializer.estimate_serialized_size(layout)


def test_serialize_rejects_missing_tensor() -> None:
    """Serialization requires a source tensor."""
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=16,
    )
    bad = cast(MemoryObj, _FakeMemoryObj(tensor=None))  # type: ignore[arg-type]
    dst = _obj(torch.zeros(1024, dtype=torch.uint8))
    with pytest.raises(ValueError, match="src.tensor"):
        serializer.serialize(bad, dst)


def test_mp_layout_to_cachegen_tensor_converts_dimension_order() -> None:
    """MP [2, L, T, D] tensors convert to CacheGen [L, 2, T, H, HS]."""
    src_tensor = _mp_kv_tensor(16)
    expected = src_tensor.view(2, 32, 16, 8, 128).permute(1, 0, 2, 3, 4)

    got = mp_layout_to_cachegen_tensor(
        src_tensor,
        num_heads=8,
        head_size=128,
        chunk_size=16,
    )

    assert torch.equal(got, expected)


def test_cachegen_layout_to_mp_tensor_converts_dimension_order() -> None:
    """CacheGen [L, 2, T, H, HS] tensors convert to MP [2, L, T, D]."""
    mp_tensor = _mp_kv_tensor(16)
    cachegen_tensor = mp_tensor.view(2, 32, 16, 8, 128).permute(1, 0, 2, 3, 4)

    got = cachegen_layout_to_mp_tensor(
        cachegen_tensor,
        mp_tensor.shape,
        num_heads=8,
        head_size=128,
    )

    assert torch.equal(got, mp_tensor)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="No GPU backend for CacheGen kernels")
def test_cachegen_mp_serde_round_trip_cpu_l1_shape_and_dtype() -> None:
    """CacheGen MP serde supports CPU L1 tensors around GPU kernels."""
    chunk_size = 16
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=chunk_size,
    )
    deserializer = CacheGenMpDeserializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
    )
    src_tensor = _kv_tensor(chunk_size, "cpu")
    src = _obj(src_tensor)
    layout = MemoryLayoutDesc(shapes=[src_tensor.shape], dtypes=[src_tensor.dtype])
    dst = _obj(
        torch.zeros(serializer.estimate_serialized_size(layout), dtype=torch.uint8)
    )

    n = serializer.serialize(src, dst)
    dst_tensor = dst.tensor
    assert dst_tensor is not None
    assert 0 < n <= dst_tensor.numel()

    encoded = _obj(dst_tensor[:n].clone())
    out = _obj(torch.zeros_like(src_tensor))
    deserializer.deserialize(encoded, out)
    out_tensor = out.tensor
    assert out_tensor is not None
    assert out_tensor.shape == src_tensor.shape
    assert out_tensor.dtype == torch.bfloat16
    assert out_tensor.mean() != 0


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="No GPU backend for CacheGen kernels")
def test_cachegen_mp_serde_round_trip_mp_layout_cpu_l1_shape_and_dtype() -> None:
    """CacheGen MP serde preserves MP destination shape around GPU kernels."""
    chunk_size = 16
    serializer = CacheGenMpSerializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=chunk_size,
        num_heads=8,
        head_size=128,
    )
    deserializer = CacheGenMpDeserializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
        num_heads=8,
        head_size=128,
    )
    src_tensor = _mp_kv_tensor(chunk_size).to(torch.bfloat16)
    src = _obj(src_tensor)
    layout = MemoryLayoutDesc(shapes=[src_tensor.shape], dtypes=[src_tensor.dtype])
    dst = _obj(
        torch.zeros(serializer.estimate_serialized_size(layout), dtype=torch.uint8)
    )

    n = serializer.serialize(src, dst)
    dst_tensor = dst.tensor
    assert dst_tensor is not None
    assert 0 < n <= dst_tensor.numel()

    encoded = _obj(dst_tensor[:n].clone())
    out = _obj(torch.zeros_like(src_tensor))
    deserializer.deserialize(encoded, out)
    out_tensor = out.tensor
    assert out_tensor is not None
    assert out_tensor.shape == src_tensor.shape
    assert out_tensor.dtype == torch.bfloat16
    assert out_tensor.mean() != 0


def test_mp_cachegen_layout_helpers_round_trip() -> None:
    """MP layout survives conversion to CacheGen layout and back."""
    src_tensor = _mp_kv_tensor(16)
    cachegen_tensor = mp_layout_to_cachegen_tensor(
        src_tensor,
        num_heads=8,
        head_size=128,
        chunk_size=16,
    )
    out_tensor = cachegen_layout_to_mp_tensor(
        cachegen_tensor,
        src_tensor.shape,
        num_heads=8,
        head_size=128,
    )

    assert torch.equal(out_tensor, src_tensor)


def test_deserialize_rejects_configured_dtype_mismatch() -> None:
    """Deserialize validates the configured dtype against the destination."""
    deserializer = CacheGenMpDeserializer(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        chunk_size=16,
        dtype=torch.float16,
    )
    src = _obj(torch.zeros(1, dtype=torch.uint8))
    dst = _obj(torch.zeros(32, 2, 16, 8, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="configured dtype"):
        deserializer.deserialize(src, dst)
