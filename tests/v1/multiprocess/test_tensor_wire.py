# SPDX-License-Identifier: Apache-2.0
"""Tests for the safe inline engine-driven tensor wire format."""

# Standard
import gc
import warnings

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    TensorEnvelope,
    TensorFrame,
    decode_tensors,
    encode_tensors,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tensor_wire_round_trip(dtype: torch.dtype) -> None:
    """Round trip typed tensors, including a non-contiguous input."""
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6).T.to(dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded = decode_tensors(
            encode_tensors([tensor]),
            expected_shape=tensor.shape,
            expected_dtype=dtype,
            expected_count=1,
        )

    assert len(decoded) == 1
    assert torch.equal(decoded[0], tensor)
    assert decoded[0].is_contiguous()


def test_tensor_wire_decoded_tensor_keeps_payload_alive() -> None:
    """Keep zero-copy tensor storage valid after the encoded payload is released."""

    def decode() -> list[torch.Tensor]:
        tensor = torch.arange(8, dtype=torch.float32)
        payload = encode_tensors([tensor])
        return decode_tensors(
            payload,
            expected_shape=tensor.shape,
            expected_dtype=tensor.dtype,
            expected_count=1,
        )

    decoded = decode()
    gc.collect()

    assert torch.equal(decoded[0], torch.arange(8, dtype=torch.float32))


def test_tensor_wire_rejects_unknown_fields() -> None:
    """Reject fields outside the versioned typed schema."""
    payload = msgspec.msgpack.encode(
        {
            "version": 1,
            "tensors": [{"dtype": "float32", "shape": [1], "data": b"\0" * 4}],
            "unexpected": True,
        }
    )

    with pytest.raises(msgspec.ValidationError):
        decode_tensors(
            payload,
            expected_shape=torch.Size([1]),
            expected_dtype=torch.float32,
            expected_count=1,
        )


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (TensorFrame("float16", (2, 2), memoryview(b"\0" * 8)), "dtype mismatch"),
        (TensorFrame("float32", (4,), memoryview(b"\0" * 16)), "shape mismatch"),
        (
            TensorFrame("float32", (2, 2), memoryview(b"\0" * 12)),
            "byte length mismatch",
        ),
    ],
)
def test_tensor_wire_rejects_mismatched_metadata(
    frame: TensorFrame, match: str
) -> None:
    """Reject dtype, shape, and byte lengths that differ from the layout."""
    payload = msgspec.msgpack.encode(TensorEnvelope(version=1, tensors=[frame]))

    with pytest.raises(ValueError, match=match):
        decode_tensors(
            payload,
            expected_shape=torch.Size([2, 2]),
            expected_dtype=torch.float32,
            expected_count=1,
        )


def test_tensor_wire_rejects_dtype_outside_shared_allowlist() -> None:
    """Reject dtype names that are not accepted by the shared IPC wire schema."""
    payload = msgspec.msgpack.encode(
        TensorEnvelope(
            version=1,
            tensors=[TensorFrame("os.system", (2, 2), memoryview(b"\0" * 16))],
        )
    )

    with pytest.raises(ValueError, match="Unsupported inline tensor dtype"):
        decode_tensors(
            payload,
            expected_shape=torch.Size([2, 2]),
            expected_dtype=torch.float32,
            expected_count=1,
        )


def test_tensor_wire_rejects_mismatched_count() -> None:
    """Reject tensor counts that differ from the request's object count."""
    payload = encode_tensors([torch.zeros(2, 2)])

    with pytest.raises(ValueError, match="count mismatch"):
        decode_tensors(
            payload,
            expected_shape=torch.Size([2, 2]),
            expected_dtype=torch.float32,
            expected_count=2,
        )
