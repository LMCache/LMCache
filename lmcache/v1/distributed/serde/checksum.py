# SPDX-License-Identifier: Apache-2.0
"""Lightweight corruption-detection serde for L2 storage.

Wraps KV cache bytes with a version byte, a chunk-hash fingerprint, and an
``xxh3_64`` hash of the payload on store, and verifies both on load. Unlike
``aesgcm``, this serde provides no confidentiality and no protection against
a determined adversary who can recompute a matching hash after modifying a
file -- its only job is to catch *unintentional* corruption (bit rot,
truncated writes, misplaced/duplicated files) cheaply, without requiring key
management. See ``docs/design/v1/distributed/serde/checksum.md`` for the
threat model, algorithm choice, and performance comparison with
``aesgcm`` and ``zlib.crc32``.
"""

# Third Party
import torch
import xxhash

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# Wire frame per chunk: [1B version][8B chunk_fingerprint][8B xxh3_64][payload].
_VERSION = 1
_FINGERPRINT_LEN = 8
_HASH_LEN = 8
_HDR_LEN = 1 + _FINGERPRINT_LEN + _HASH_LEN


def _plaintext_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Return the total byte size of the KV data described by ``layout_desc``."""
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n * torch.empty((), dtype=dtype).element_size()
    return total


def _chunk_fingerprint(key: ObjectKey) -> bytes:
    """Derive a fixed-width 8-byte fingerprint from ``key.chunk_hash``.

    This is not a cryptographic binding to the full ``ObjectKey``. It is a
    compact chunk-level guard against a payload landing in another chunk's
    slot (race, misplacement, accidental copy). Hashes longer than 8 bytes
    are truncated and shorter ones are zero-padded.
    """
    chunk_hash = key.chunk_hash
    if len(chunk_hash) >= _FINGERPRINT_LEN:
        return chunk_hash[:_FINGERPRINT_LEN]
    return chunk_hash + b"\x00" * (_FINGERPRINT_LEN - len(chunk_hash))


class ChecksumSerializer(Serializer):
    """Frame KV bytes with a version, chunk-hash fingerprint, and xxh3_64."""

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Frame ``src`` into ``dst`` as ``[version][fingerprint][xxh3_64][payload]``.

        Args:
            src: Source KV bytes (read-locked).
            dst: Destination byte buffer, sized by ``estimate_serialized_size``.
            key: Object key; ``key.chunk_hash`` seeds the chunk fingerprint.

        Returns:
            Number of bytes written to ``dst``.
        """
        payload = bytes(src.byte_array)
        digest = xxhash.xxh3_64(payload).digest()
        # Cast to native "B": MemoryObj.byte_array is a ctypes-backed memoryview
        # with format "<B", which does not support slice assignment.
        out = memoryview(dst.byte_array).cast("B")
        out[0:1] = bytes((_VERSION,))
        out[1 : 1 + _FINGERPRINT_LEN] = _chunk_fingerprint(key)
        out[1 + _FINGERPRINT_LEN : _HDR_LEN] = digest
        out[_HDR_LEN : _HDR_LEN + len(payload)] = payload
        return _HDR_LEN + len(payload)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return exact framed size: plaintext + fixed 17-byte header.

        xxh3_64 does not expand its input, so this is exact rather than an
        upper bound.
        """
        return _plaintext_bytes(layout_desc) + _HDR_LEN


class ChecksumDeserializer(Deserializer):
    """Verify and strip the frame produced by :class:`ChecksumSerializer`."""

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Verify ``src``'s frame and copy its payload into ``dst``.

        The payload length is taken from ``dst``'s size, not ``src``, which
        may be padded larger than the stored frame.

        Args:
            src: Source byte buffer holding the stored frame (may be padded).
            dst: Destination KV buffer (write-locked); its size is the
                payload length.
            key: Object key; ``key.chunk_hash`` must match the chunk fingerprint
                stored in the frame.

        Raises:
            ValueError: If the frame is truncated, has an unknown version,
                was stored for a different chunk (fingerprint mismatch), or
                its xxh3_64 hash does not match the payload (corruption).
        """
        out = memoryview(dst.byte_array).cast("B")
        payload_len = len(out)
        blob = bytes(src.byte_array)
        frame_end = _HDR_LEN + payload_len
        if len(blob) < frame_end:
            raise ValueError("checksum serde: truncated frame")
        if blob[0] != _VERSION:
            raise ValueError(f"checksum serde: unknown version {blob[0]}")

        stored_fingerprint = blob[1 : 1 + _FINGERPRINT_LEN]
        expected_fingerprint = _chunk_fingerprint(key)
        if stored_fingerprint != expected_fingerprint:
            raise ValueError(
                "checksum serde: chunk fingerprint mismatch -- payload "
                "belongs to a different chunk"
            )

        stored_digest = blob[1 + _FINGERPRINT_LEN : _HDR_LEN]
        payload = blob[_HDR_LEN:frame_end]
        computed_digest = xxhash.xxh3_64(payload).digest()
        if computed_digest != stored_digest:
            raise ValueError(
                "checksum serde: xxh3_64 mismatch -- payload corrupted or tampered"
            )

        out[:payload_len] = payload


def _create_checksum_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    """Build the checksum serde from ``SerdeConfig.kwargs``.

    Kwargs:
        max_workers: Serde thread-pool size (default ``1``).
    """
    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        ChecksumSerializer(),
        ChecksumDeserializer(),
        max_workers=max_workers,
    )


register_serde_factory("checksum", _create_checksum_serde)
