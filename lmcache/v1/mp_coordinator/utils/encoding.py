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

EncodedKey = tuple[bytes, str, int, int, str, tuple[tuple[str, str], ...]]
"""An :class:`ObjectKey` as ``(chunk_hash, model_name, kv_rank,
object_group_id, cache_salt, tags)``."""


def encode_key(key: ObjectKey) -> EncodedKey:
    """Flatten a key for a capture, fields in declaration order."""
    return (
        key.chunk_hash,
        key.model_name,
        key.kv_rank,
        key.object_group_id,
        key.cache_salt,
        key.tags,
    )


def decode_key(encoded: object) -> ObjectKey:
    """Rebuild a key from :func:`encode_key`.

    Args:
        encoded: An :func:`encode_key` value, as read back from an
            artifact -- a sequence, not necessarily a tuple.

    Returns:
        The key, validated by ``ObjectKey`` itself.

    Raises:
        ValueError: If the payload shape or a field violates an ``ObjectKey``
            invariant.
    """
    if not isinstance(encoded, (list, tuple)):
        raise ValueError("captured ObjectKey must be a sequence")
    if len(encoded) == 5:
        # Captures written before request tags existed have no tag field.
        chunk_hash, model_name, kv_rank, object_group_id, cache_salt = encoded
        tags: tuple[tuple[str, str], ...] = ()
    elif len(encoded) == 6:
        chunk_hash, model_name, kv_rank, object_group_id, cache_salt, raw_tags = encoded
        if not isinstance(raw_tags, (list, tuple)):
            raise ValueError("captured ObjectKey tags must be a sequence")
        tag_pairs: list[tuple[str, str]] = []
        for raw_tag in raw_tags:
            if (
                not isinstance(raw_tag, (list, tuple))
                or len(raw_tag) != 2
                or not isinstance(raw_tag[0], str)
                or not isinstance(raw_tag[1], str)
            ):
                raise ValueError(
                    "captured ObjectKey tags must contain string name/value pairs"
                )
            tag_pairs.append((raw_tag[0], raw_tag[1]))
        tags = tuple(tag_pairs)
    else:
        raise ValueError(
            "captured ObjectKey must have 5 legacy fields or 6 current fields"
        )
    return ObjectKey(
        chunk_hash=cast(bytes, chunk_hash),
        model_name=cast(str, model_name),
        kv_rank=cast(int, kv_rank),
        object_group_id=cast(int, object_group_id),
        cache_salt=cast(str, cache_salt),
        tags=tags,
    )
