# SPDX-License-Identifier: Apache-2.0
"""Flattening domain values into the plain data a capture may hold.

Object keys appear in several sections and are captured by the million,
so they go out as positional tuples: per-key field names cost more than
the fields. Sharing the encoding keeps the sections agreeing on it.
"""

# Standard
from typing import cast

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity, decode_tags
from lmcache.v1.distributed.api import ObjectKey

EncodedKey = tuple[bytes, str, int, int, str] | tuple[bytes, str, int, int, str, str]
"""An :class:`ObjectKey` as ``(chunk_hash, model_name, kv_rank,
object_group_id, cache_salt[, tags_token])``.
The optional token preserves additional matching tags."""


def encode_key(key: ObjectKey) -> EncodedKey:
    """Flatten a key for a capture, fields in declaration order."""
    fields = (
        key.chunk_hash,
        key.model_name,
        key.kv_rank,
        key.object_group_id,
        key.cache_salt,
    )
    token = key.identity.tags_token()
    return (*fields, token) if token else fields


def decode_key(encoded: object) -> ObjectKey:
    """Rebuild a key from :func:`encode_key`.

    Args:
        encoded: An :func:`encode_key` value, as read back from an
            artifact -- a sequence, not necessarily a tuple.

    Returns:
        The key, validated by ``ObjectKey`` itself.

    Raises:
        ValueError: If the sequence has an unsupported field count or a field
            violates an ``ObjectKey`` invariant.
    """
    values = cast("EncodedKey", encoded)
    if len(values) not in (5, 6):
        raise ValueError("invalid coordinator key encoding: expected 5 or 6 fields")
    chunk_hash, model_name, kv_rank, object_group_id, cache_salt = values[:5]
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
            tags=decode_tags(values[5] if len(values) == 6 else ""),
        )
    )
