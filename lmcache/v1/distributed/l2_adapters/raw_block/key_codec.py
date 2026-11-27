# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import hashlib
import urllib.parse

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import decode_storage_key, format_storage_key


@dataclass(frozen=True)
class RawBlockKeySpec:
    """Encoded raw-block key plus stable slot-header identity."""

    encoded: str
    slot_identity: int


def object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey using raw-block's reversible key shape.

    Args:
        key: Object key supplied by the MP storage layer.

    Returns:
        A stable string containing model name, KV rank, object group id,
        chunk hash, optional cache salt and cache namespace.

    """
    return format_storage_key(
        key, "{quoted_model}@{rank}@{group}@{hash}{suffix}", rank_format="#010x"
    )


def decode_object_key(encoded: str) -> ObjectKey:
    """Deserialize an ObjectKey from raw-block's reversible encoding.

    Args:
        encoded: Encoded key string produced by ``object_key_to_string``.

    Returns:
        The reconstructed ``ObjectKey``.

    Raises:
        ValueError: If the encoded string has an unexpected shape or invalid
            hexadecimal chunk hash.
    """
    return decode_storage_key(encoded, model_decoder=urllib.parse.unquote)


def encode_object_key(key: ObjectKey) -> RawBlockKeySpec:
    """Encode an MP ObjectKey for raw-block storage.

    Args:
        key: Object key supplied by the MP storage layer.

    Returns:
        Encoded key and deterministic slot-header identity.
    """
    encoded = object_key_to_string(key)
    return RawBlockKeySpec(
        encoded=encoded,
        slot_identity=object_key_slot_identity(encoded),
    )


def object_key_slot_identity(encoded: str) -> int:
    """Return the stable 64-bit slot identity for an encoded ObjectKey.

    Args:
        encoded: Raw-block ObjectKey string produced by ``object_key_to_string``.

    Returns:
        Unsigned 64-bit identity stored in the raw-block slot header.
    """
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
