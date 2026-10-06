# SPDX-License-Identifier: Apache-2.0
"""Reject invalid section boundaries even when the total and CRC agree."""

# Standard
import struct

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_codec import (
    CorruptEncodedKVError,
    EncodedKV,
    UnsupportedConfigError,
    deserialize_header,
    serialize_header,
)

pytestmark = pytest.mark.no_shared_allocator


@pytest.mark.parametrize("lengths", [(-1, 5, 0), (5, -1, 0), (5, 0, -1)])
def test_serialize_rejects_negative_sections(lengths: tuple[int, int, int]) -> None:
    """An equal total length cannot make a negative section valid."""
    enc = EncodedKV(
        k_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        k_payload_len=lengths[0],
        v_payload_len=lengths[1],
        scale_payload_len=lengths[2],
        payload=b"1234",
    )
    with pytest.raises(UnsupportedConfigError, match="non-negative"):
        serialize_header(enc)


@pytest.mark.parametrize("lengths", [(-1, 5, 0), (5, -1, 0), (5, 0, -1)])
def test_v1_decode_rejects_negative_sections(lengths: tuple[int, int, int]) -> None:
    """Validate each V1 section before slicing the CRC-protected payload."""
    enc = EncodedKV(
        k_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        k_payload_len=4,
        payload=b"1234",
    )
    blob = bytearray(serialize_header(enc) + bytes(enc.payload))
    # V1's fixed metadata is an eight-byte magic, six uint16s, and
    # seven int64s. An empty scale_shape puts section lengths next.
    lengths_offset = struct.calcsize("<8sHHHHHHqqqqqqq")
    struct.pack_into("<qqq", blob, lengths_offset, *lengths)
    with pytest.raises(CorruptEncodedKVError, match="negative payload length"):
        deserialize_header(bytes(blob))
