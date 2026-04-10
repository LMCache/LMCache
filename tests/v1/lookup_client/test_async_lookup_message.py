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
    """原始 int 哈希：向后兼容性验证。"""

    def test_serialization_roundtrip(self):
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

    def test_describe(self):
        msg = LookupRequestMsg(lookup_id="abc", hashes=[1, 2, 3], offsets=[0, 1, 2])
        assert "abc" in msg.describe()
        assert "3" in msg.describe()


class TestLookupRequestMsgWithBytesHashes:
    """sha256_cbor 返回 bytes：核心 bug 修复验证。"""

    @staticmethod
    def _sha256_hash(data: bytes) -> bytes:
        """模拟 vLLM sha256_cbor 哈希函数的输出（32 字节摘要）。"""
        return hashlib.sha256(data).digest()

    def test_bytes_hash_serialization_roundtrip(self):
        """LookupRequestMsg 应能序列化和反序列化 bytes 类型的 hashes。"""
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

    def test_bytes_hash_is_32_bytes(self):
        """sha256_cbor 输出正好 32 字节，超出 int64 范围但应被接受。"""
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

    def test_mixed_int_and_bytes_hashes(self):
        """混合 int 和 bytes 的哈希列表也应被支持。"""
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

    def test_request_configs_preserved(self):
        """带 request_configs 的消息序列化也应正常。"""
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
    def test_serialization_roundtrip(self):
        msg = LookupResponseMsg(lookup_id="resp-id", num_hit_tokens=512)
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupResponseMsg)
        assert decoded.lookup_id == "resp-id"
        assert decoded.num_hit_tokens == 512

    def test_describe(self):
        msg = LookupResponseMsg(lookup_id="x", num_hit_tokens=256)
        assert "x" in msg.describe()
        assert "256" in msg.describe()


class TestLookupCleanupMsg:
    def test_serialization_roundtrip(self):
        msg = LookupCleanupMsg(lookup_id="cleanup-id")
        raw = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(raw, type=LookupCleanupMsg)
        assert decoded.lookup_id == "cleanup-id"

    def test_describe(self):
        msg = LookupCleanupMsg(lookup_id="cl-123")
        assert "cl-123" in msg.describe()
