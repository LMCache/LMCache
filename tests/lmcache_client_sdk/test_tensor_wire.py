# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the inline tensor wire format."""

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache_client_sdk.tensor_wire import decode_tensors, encode_tensors

SHAPE = torch.Size((2, 2, 256, 256))


def test_encode_decode_roundtrip() -> None:
    """Check that encoded tensors decode back to the same values."""
    tensors = [
        torch.arange(2 * 2 * 256 * 256, dtype=torch.float16).view(SHAPE),
        torch.zeros(SHAPE, dtype=torch.float16),
    ]
    payload = encode_tensors(tensors)
    decoded = decode_tensors(
        payload,
        expected_shape=SHAPE,
        expected_dtype=torch.float16,
        expected_count=2,
    )
    assert len(decoded) == 2
    assert torch.equal(decoded[0], tensors[0])
    assert torch.equal(decoded[1], tensors[1])


def test_encode_rejects_gpu_tensor() -> None:
    """Check that a non-CPU tensor is rejected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    gpu = torch.zeros(SHAPE, device="cuda")
    with pytest.raises(ValueError):
        encode_tensors([gpu])


def test_decode_rejects_shape_mismatch() -> None:
    """Check that a shape mismatch raises a ValueError."""
    tensors = [torch.zeros(SHAPE, dtype=torch.float16)]
    payload = encode_tensors(tensors)
    with pytest.raises(ValueError):
        decode_tensors(
            payload,
            expected_shape=torch.Size((3, 2, 256, 256)),
            expected_dtype=torch.float16,
            expected_count=1,
        )


def test_decode_rejects_count_mismatch() -> None:
    """Check that a count mismatch raises a ValueError."""
    tensors = [torch.zeros(SHAPE, dtype=torch.float16)]
    payload = encode_tensors(tensors)
    with pytest.raises(ValueError):
        decode_tensors(
            payload,
            expected_shape=SHAPE,
            expected_dtype=torch.float16,
            expected_count=2,
        )


def test_decode_rejects_dtype_mismatch() -> None:
    """Check that a dtype mismatch raises a ValueError."""
    tensors = [torch.zeros(SHAPE, dtype=torch.float16)]
    payload = encode_tensors(tensors)
    with pytest.raises(ValueError):
        decode_tensors(
            payload,
            expected_shape=SHAPE,
            expected_dtype=torch.float32,
            expected_count=1,
        )


def test_decode_rejects_garbage_payload() -> None:
    """Check that a non-envelope payload raises a DecodeError."""
    with pytest.raises(msgspec.DecodeError):
        decode_tensors(
            b"not-an-envelope",
            expected_shape=SHAPE,
            expected_dtype=torch.float16,
            expected_count=1,
        )
