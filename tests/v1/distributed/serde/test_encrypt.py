# SPDX-License-Identifier: Apache-2.0
"""Tests for the AES-GCM encryption serde.

These use a byte-buffer stand-in exposing ``.byte_array`` so they do not need
an L1Manager or GPU; they verify the pure transform + factory wiring through
the public interface.
"""

# Standard
from dataclasses import dataclass
import os
import tempfile

# Third Party
from cryptography.exceptions import InvalidTag
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)
from lmcache.v1.distributed.serde.encrypt import (
    DecryptDeserializer,
    EncryptSerializer,
    HkdfKeyProvider,
)

_MASTER = b"unit-test-master-key-material-32b!!"
# Public wire contract: [1B version][12B IV][ciphertext || 16B GCM tag].
_FRAME_OVERHEAD = 1 + 12 + 16


@dataclass
class _ByteBuf:
    """Minimal MemoryObj stand-in exposing a mutable ``byte_array``."""

    buf: bytearray

    @property
    def byte_array(self) -> bytearray:
        return self.buf


def _key(cache_salt: str = "alice") -> ObjectKey:
    return ObjectKey(
        chunk_hash=b"\x11" * 32, model_name="m", kv_rank=0, cache_salt=cache_salt
    )


def _serde(
    cache_salt_key_len: int = 16,
) -> tuple[EncryptSerializer, DecryptDeserializer]:
    provider = HkdfKeyProvider(_MASTER, key_len=cache_salt_key_len)
    return EncryptSerializer(provider), DecryptDeserializer(provider)


def _encrypt(serializer: EncryptSerializer, plaintext: bytes, key: ObjectKey) -> bytes:
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
    frame = _encrypt(serializer, plaintext, key)
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == plaintext


def test_roundtrip_with_over_allocated_load_temp():
    """The load temp may be larger than the stored frame; deserialize must
    derive the ciphertext length from dst, not the padded src buffer."""
    serializer, deserializer = _serde()
    key = _key()
    plaintext = b"\xab" * 512
    frame = _encrypt(serializer, plaintext, key)
    padded = _ByteBuf(bytearray(frame) + bytearray(64))  # trailing garbage
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(padded, dst, key)
    assert bytes(dst.buf) == plaintext


def test_distinct_iv_per_chunk():
    """Same plaintext + key encrypts to different ciphertext (random IV)."""
    serializer, _ = _serde()
    key = _key()
    plaintext = b"\x00" * 128
    assert _encrypt(serializer, plaintext, key) != _encrypt(serializer, plaintext, key)


# =============================================================================
# authentication / isolation
# =============================================================================


def test_wrong_tenant_key_fails():
    """A frame encrypted for one salt cannot be decrypted under another —
    per-tenant isolation at the crypto layer."""
    serializer, deserializer = _serde()
    frame = _encrypt(serializer, b"secret" * 20, _key("alice"))
    with pytest.raises(InvalidTag):
        deserializer.deserialize(
            _ByteBuf(bytearray(frame)), _ByteBuf(bytearray(120)), _key("bob")
        )


def test_tampered_ciphertext_fails():
    serializer, deserializer = _serde()
    key = _key()
    frame = bytearray(_encrypt(serializer, b"payload" * 16, key))
    frame[-1] ^= 0x01  # flip a tag byte
    with pytest.raises(InvalidTag):
        deserializer.deserialize(_ByteBuf(frame), _ByteBuf(bytearray(112)), key)


def test_malformed_frame_rejected():
    _, deserializer = _serde()
    with pytest.raises(ValueError):
        deserializer.deserialize(
            _ByteBuf(bytearray(4)), _ByteBuf(bytearray(64)), _key()
        )


def test_empty_salt_roundtrips():
    """Empty cache_salt (anonymous traffic) is a valid tenant bucket."""
    serializer, deserializer = _serde()
    key = _key(cache_salt="")
    plaintext = b"anon" * 40
    frame = _encrypt(serializer, plaintext, key)
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == plaintext


def test_aes256_roundtrips():
    serializer, deserializer = _serde(cache_salt_key_len=32)
    key = _key()
    plaintext = b"\x42" * 256
    frame = _encrypt(serializer, plaintext, key)
    dst = _ByteBuf(bytearray(len(plaintext)))
    deserializer.deserialize(_ByteBuf(bytearray(frame)), dst, key)
    assert bytes(dst.buf) == plaintext


# =============================================================================
# factory / config
# =============================================================================


def test_registered():
    assert "encrypt" in get_registered_serde_types()


def test_factory_builds_from_config():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(32))
        path = f.name
    try:
        proc = create_serde_processor(
            SerdeConfig(
                type="encrypt",
                kwargs={"key_provider": "hkdf", "master_key_path": path},
            )
        )
        assert proc is not None
    finally:
        os.unlink(path)


def test_factory_missing_master_key_path():
    with pytest.raises(ValueError):
        create_serde_processor(SerdeConfig(type="encrypt", kwargs={}))


def test_factory_bad_aes_bits():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(32))
        path = f.name
    try:
        with pytest.raises(ValueError):
            create_serde_processor(
                SerdeConfig(
                    type="encrypt",
                    kwargs={"master_key_path": path, "aes_bits": 192},
                )
            )
    finally:
        os.unlink(path)


def test_factory_unsupported_provider():
    with pytest.raises(ValueError):
        create_serde_processor(
            SerdeConfig(type="encrypt", kwargs={"key_provider": "keyring"})
        )
