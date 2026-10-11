# SPDX-License-Identifier: Apache-2.0
"""Flattening domain values into the plain data a capture may hold.

Object keys appear in several sections and are captured by the million,
so they go out as positional tuples: per-key field names cost more than
the fields. Sharing the encoding keeps the sections agreeing on it.
"""

# Standard
from typing import cast

# First Party
from lmcache.v1.distributed.api import ObjectKey

EncodedKey = tuple[bytes, str, int, int, str, str]
"""An :class:`ObjectKey` as ``(chunk_hash, model_name, kv_rank,
object_group_id, cache_salt, cache_format)``."""


def encode_key(key: ObjectKey) -> EncodedKey:
    """Flatten a key for a capture, fields in declaration order."""
    return (
        key.chunk_hash,
        key.model_name,
        key.kv_rank,
        key.object_group_id,
        key.cache_salt,
        key.cache_format,
    )


def decode_key(encoded: object) -> ObjectKey:
    """Rebuild a key from :func:`encode_key`.

    Args:
        encoded: An :func:`encode_key` value, as read back from an
            artifact -- a sequence, not necessarily a tuple.

    Returns:
        The key, validated by ``ObjectKey`` itself.

    Raises:
        ValueError: If a field violates an ``ObjectKey`` invariant.
    """
    return ObjectKey(*cast("EncodedKey", encoded))
