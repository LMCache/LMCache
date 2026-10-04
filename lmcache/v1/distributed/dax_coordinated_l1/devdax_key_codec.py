# SPDX-License-Identifier: Apache-2.0
"""Canonical key and layout hashing for DAX-Coordinated L1."""

# Standard
from hashlib import sha256
import struct

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import DaxCoordinatedL1Config
from lmcache.v1.distributed.dax_coordinated_l1.devdax_layout import (
    DevDaxPayloadGeometry,
)

_KEY_SCHEMA_VERSION = 1


def _length_prefixed(value: bytes) -> bytes:
    """Encode bytes with an unsigned big-endian 32-bit length prefix."""
    return struct.pack(">I", len(value)) + value


def canonical_key_bytes(key: ObjectKey, layout_profile_digest: bytes) -> bytes:
    """Encode an ObjectKey with fixed field order and endian.

    Args:
        key: Object identity to encode.
        layout_profile_digest: Exact 32-byte digest of the shared layout.

    Returns:
        Stable canonical bytes shared by every participant.

    Raises:
        ValueError: If the layout digest is not 32 bytes or integer fields do
            not fit the fixed encoding.
    """
    if len(layout_profile_digest) != 32:
        raise ValueError("layout_profile_digest must contain exactly 32 bytes")
    return b"".join(
        (
            struct.pack(">I", _KEY_SCHEMA_VERSION),
            _length_prefixed(key.chunk_hash),
            _length_prefixed(key.model_name.encode("utf-8")),
            struct.pack(">q", key.kv_rank),
            struct.pack(">Q", key.object_group_id),
            _length_prefixed(key.cache_salt.encode("utf-8")),
            layout_profile_digest,
        )
    )


def key_digest(key: ObjectKey, layout_profile_digest: bytes) -> bytes:
    """Return the SHA-256 digest of the canonical DAX-Coordinated L1 key."""
    return sha256(canonical_key_bytes(key, layout_profile_digest)).digest()


def layout_id(layout_desc: MemoryLayoutDesc) -> int:
    """Return a stable 32-bit identifier for a memory layout description."""
    encoded = bytearray()
    encoded.extend(struct.pack(">I", len(layout_desc.shapes)))
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        encoded.extend(struct.pack(">I", len(shape)))
        for dimension in shape:
            encoded.extend(struct.pack(">q", dimension))
        encoded.extend(_length_prefixed(str(dtype).encode("ascii")))
    return int.from_bytes(sha256(encoded).digest()[:4], "big")


def layout_profile_digest(
    config: DaxCoordinatedL1Config,
    geometry: DevDaxPayloadGeometry,
    payload_alignment: int,
    model_layout_digest: bytes,
) -> bytes:
    """Hash the validated shared layout and its required runtime model digest.

    Equal-sized but incompatible KV layouts must not attach to the same arena.
    The byte encoding remains compatible with the model-derived split layout.
    """
    if len(model_layout_digest) != 32:
        raise ValueError("model_layout_digest must contain 32 bytes")
    schema_version = 1
    encoded = struct.pack(">II", schema_version, config.participant_count)
    encoded += struct.pack(">I", len(config.buckets_per_level))
    encoded += b"".join(struct.pack(">Q", count) for count in config.buckets_per_level)
    encoded += struct.pack(
        ">QQQ",
        geometry.payload_slot_bytes,
        geometry.payload_slot_count,
        payload_alignment,
    )
    encoded += _length_prefixed(config.visibility_mode.encode("ascii"))
    encoded += model_layout_digest
    if config.ownership_mode != "equal":
        encoded += _length_prefixed(config.ownership_mode.encode("ascii"))
    return sha256(encoded).digest()
