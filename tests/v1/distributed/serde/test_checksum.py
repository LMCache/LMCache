# SPDX-License-Identifier: Apache-2.0
"""Tests for the checksum (corruption-detection) serde.

These use a byte-buffer stand-in exposing ``.byte_array`` so they do not need
an L1Manager or GPU; they verify the pure transform + factory wiring through
the public interface. Mirrors ``test_aesgcm.py``'s structure.
"""

# Standard
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)
from lmcache.v1.distributed.serde.checksum import (
    ChecksumDeserializer,
    ChecksumSerializer,
)

# Public wire contract: [1B version][8B chunk_fingerprint][8B xxh3_64][payload].
_FRAME_OVERHEAD = 1 + 8 + 8


class _ByteBuf:
    """Minimal MemoryObj stand-in exposing a mutable ``byte_array``.

    Backed by a ctypes ``c_ubyte`` array so ``byte_array`` has the same ``"<B"``
    memoryview format as the real ``MemoryObj``; a plain ``bytearray`` is format
    ``"B"`` and would hide format-specific bugs (e.g. slice-assign failures).
    """

    def __init__(self, data: bytes) -> None:
        self._arr = (ctypes.c_ubyte * len(data)).from_buffer_copy(bytes(data))

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._arr)

    @property
    def buf(self) -> bytes:
        return bytes(self._arr)


def _key(chunk_hash: bytes = b"\x11" * 32) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_hash, model_name="m", kv_rank=0)


def _serde() -> tuple[ChecksumSerializer, ChecksumDeserializer]:
    return ChecksumSerializer(), ChecksumDeserializer()


def _frame(serializer: ChecksumSerializer, plaintext: bytes, key: ObjectKey) -> bytes:
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([len(plaintext)])], dtypes=[torch.uint8]
    )
    dst = _ByteBuf(bytearray(serializer.estimate_serialized_size(layout)))
    n = serializer.serialize(_ByteBuf(bytearray(plaintext)), dst, key)  # type: ignore[arg-type]
    return bytes(dst.buf[:n])


# =============================================================================
# estimate_serialized_size
# =============================================================================


def test_estimate_is_plaintext_plus_frame_overhead():
    serializer, _ = _serde()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])], dtypes=[torch.bfloat16]
    )
    plaintext = 2 * 4 * 256 * 128 * 2  # bfloat16 = 2 bytes/elem
    assert serializer.estimate_serialized_size(layout) == plaintext + _FRAME_OVERHEAD


def test_estimate_multi_group():
    serializer, _ = _serde()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4, 8]), torch.Size([16])],
        dtypes=[torch.float16, torch.uint8],
    )
    plaintext = 32 * 2 + 16 * 1
    assert serializer.estimate_serialized_size(layout) == plaintext + _FRAME_OVERHEAD


# =============================================================================
# round-trip
# =============================================================================


def test_roundtrip_recovers_plaintext():
    serializer, deserializer = _serde()
    key = _key()
    plaintext = bytes(range(256)) * 8
    frame = _frame(serializer, plaintext, key)
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == plaintext


def test_roundtrip_with_over_allocated_load_temp():
    """The load temp may be larger than the stored frame; deserialize must
    derive the payload length from dst, not the padded src buffer."""
    serializer, deserializer = _serde()
    key = _key()
    plaintext = b"\xab" * 512
    frame = _frame(serializer, plaintext, key)
    padded = _ByteBuf(bytearray(frame) + bytearray(64))  # trailing garbage
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(padded, dst, key)
    assert bytes(dst.buf) == plaintext


def test_empty_payload_roundtrips():
    serializer, deserializer = _serde()
    key = _key()
    frame = _frame(serializer, b"", key)
    assert len(frame) == _FRAME_OVERHEAD
    dst = _ByteBuf(bytearray(0))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == b""


# =============================================================================
# corruption / mismatch detection
# =============================================================================


def test_tampered_payload_fails():
    """A single flipped payload byte must be rejected -- the exact scenario
    this serde exists to catch (corrupted/malformed on-disk data)."""
    serializer, deserializer = _serde()
    key = _key()
    frame = bytearray(_frame(serializer, b"payload" * 16, key))
    frame[-1] ^= 0x01  # flip a payload byte
    with pytest.raises(ValueError, match="xxh3_64 mismatch"):
        deserializer.deserialize(_ByteBuf(frame), _ByteBuf(bytearray(112)), key)


def test_chunk_fingerprint_mismatch_fails():
    """A frame for one chunk must be rejected when loaded for another chunk."""
    serializer, deserializer = _serde()
    frame = _frame(serializer, b"secret" * 20, _key(b"\x11" * 32))
    with pytest.raises(ValueError, match="chunk fingerprint mismatch"):
        deserializer.deserialize(
            _ByteBuf(bytearray(frame)), _ByteBuf(bytearray(120)), _key(b"\x22" * 32)
        )


def test_unknown_version_fails():
    _, deserializer = _serde()
    payload_len = 64
    frame = bytearray(b"\x02" + b"\x00" * 16 + b"\x00" * payload_len)
    with pytest.raises(ValueError, match="unknown version"):
        deserializer.deserialize(
            _ByteBuf(frame), _ByteBuf(bytearray(payload_len)), _key()
        )


def test_truncated_frame_rejected():
    _, deserializer = _serde()
    with pytest.raises(ValueError, match="truncated frame"):
        deserializer.deserialize(
            _ByteBuf(bytearray(4)), _ByteBuf(bytearray(64)), _key()
        )


def test_short_chunk_hash_is_zero_padded_consistently():
    """chunk_hash shorter than 8 bytes must still round-trip (zero-padded)."""
    serializer, deserializer = _serde()
    key = _key(chunk_hash=b"\x01\x02\x03")
    plaintext = b"short-hash-payload"
    frame = _frame(serializer, plaintext, key)
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == plaintext


# =============================================================================
# factory / config
# =============================================================================


def test_registered():
    assert "checksum" in get_registered_serde_types()


def test_factory_builds_from_config():
    proc = create_serde_processor(SerdeConfig(type="checksum", kwargs={}))
    assert proc is not None
    proc.close()


def test_factory_respects_max_workers():
    proc = create_serde_processor(
        SerdeConfig(type="checksum", kwargs={"max_workers": 2})
    )
    assert proc is not None
    proc.close()
