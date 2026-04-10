# SPDX-License-Identifier: Apache-2.0
"""Tests for AsyncLookupMsg serialization with bytes hashes (sha256_cbor support).

The sha256_cbor hash algorithm returns a 32-byte digest (bytes), whereas the
legacy default hash algorithm returns an int.  LookupRequestMsg must accept
both so that async loading works correctly when sha256_cbor is configured.
"""

# Standard
import hashlib

# Third Party
import msgspec

# First Party
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)


class TestLookupRequestMsgWithIntHashes:
    """Backward compatibility: int hashes must still serialize correctly."""

    def test_serialization_roundtrip(self) -> None:
        hashes = [123456789, 987654321, 0, 2**32 - 1]
        offsets = [0, 128, 256, 384]
        msg = LookupRequestMsg(
            lookup_id="test-id-int",
            hashes=hashes,
            offsets=offsets,
        )
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupRequestMsg)
        assert decoded.lookup_id == msg.lookup_id
        assert decoded.hashes == hashes
        assert decoded.offsets == offsets

    def test_describe(self) -> None:
        msg = LookupRequestMsg(lookup_id="abc", hashes=[1, 2, 3], offsets=[0, 1, 2])
        assert "abc" in msg.describe()
        assert "3" in msg.describe()


class TestLookupRequestMsgWithBytesHashes:
    """Core bug fix: sha256_cbor returns bytes; LookupRequestMsg must accept them."""

    @staticmethod
    def _sha256_hash(data: bytes) -> bytes:
        """Return the SHA-256 digest of *data*, mimicking vLLM sha256_cbor output."""
        return hashlib.sha256(data).digest()

    def test_bytes_hash_serialization_roundtrip(self) -> None:
        """LookupRequestMsg must serialize and deserialize bytes hashes unchanged."""
        hashes = [
            self._sha256_hash(b"chunk_0"),
            self._sha256_hash(b"chunk_1"),
            self._sha256_hash(b"chunk_2"),
        ]
        offsets = [0, 128, 256]
        msg = LookupRequestMsg(
            lookup_id="test-id-bytes",
            hashes=hashes,
            offsets=offsets,
        )
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupRequestMsg)
        assert decoded.lookup_id == msg.lookup_id
        assert decoded.hashes == hashes
        assert all(isinstance(h, bytes) for h in decoded.hashes)
        assert decoded.offsets == offsets

    def test_bytes_hash_is_32_bytes(self) -> None:
        """sha256_cbor produces a 32-byte digest that exceeds int64 range."""
        sha256_digest = self._sha256_hash(b"some token chunk")
        assert len(sha256_digest) == 32
        msg = LookupRequestMsg(
            lookup_id="overflow-test",
            hashes=[sha256_digest],
            offsets=[0],
        )
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupRequestMsg)
        assert decoded.hashes[0] == sha256_digest

    def test_mixed_int_and_bytes_hashes(self) -> None:
        """A hash list containing both int and bytes elements must round-trip."""
        hashes = [12345, self._sha256_hash(b"mixed"), 67890]
        offsets = [0, 128, 256]
        msg = LookupRequestMsg(
            lookup_id="mixed-test",
            hashes=hashes,
            offsets=offsets,
        )
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupRequestMsg)
        assert decoded.hashes[0] == 12345
        assert isinstance(decoded.hashes[1], bytes)
        assert decoded.hashes[2] == 67890

    def test_request_configs_preserved(self) -> None:
        """request_configs must survive serialization alongside bytes hashes."""
        hashes = [self._sha256_hash(b"cfg_chunk")]
        msg = LookupRequestMsg(
            lookup_id="cfg-test",
            hashes=hashes,
            offsets=[0],
            request_configs={"key": "value"},
        )
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupRequestMsg)
        assert decoded.request_configs == {"key": "value"}
        assert decoded.hashes == hashes


class TestLookupResponseMsg:
    def test_serialization_roundtrip(self) -> None:
        msg = LookupResponseMsg(lookup_id="resp-id", num_hit_tokens=512)
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupResponseMsg)
        assert decoded.lookup_id == "resp-id"
        assert decoded.num_hit_tokens == 512

    def test_describe(self) -> None:
        msg = LookupResponseMsg(lookup_id="x", num_hit_tokens=256)
        assert "x" in msg.describe()
        assert "256" in msg.describe()


class TestLookupCleanupMsg:
    def test_serialization_roundtrip(self) -> None:
        msg = LookupCleanupMsg(lookup_id="cleanup-id")
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupCleanupMsg)
        assert decoded.lookup_id == "cleanup-id"

    def test_describe(self) -> None:
        msg = LookupCleanupMsg(lookup_id="cl-123")
        assert "cl-123" in msg.describe()
