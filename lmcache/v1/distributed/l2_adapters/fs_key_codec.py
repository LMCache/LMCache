# SPDX-License-Identifier: Apache-2.0
"""Shared reversible filename codec for filesystem L2 adapters."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.distributed.api import ObjectKey

_KEY_SEP = "@"
_PATH_SLASH_REPLACEMENT = "-SEP-"
_FILE_EXT = ".data"


def object_key_to_filename(key: ObjectKey) -> str:
    """Encode an object key as a reversible filesystem-safe filename.

    Args:
        key: Object key to encode.

    Returns:
        A ``.data`` filename containing every identity field of ``key``.
    """
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = (
        f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}{_FILE_EXT}"
    return f"{base}{_FILE_EXT}"


def filename_to_object_key(filename: str) -> ObjectKey | None:
    """Decode a filesystem cache filename into an object key.

    Args:
        filename: Cache file basename to decode.

    Returns:
        The decoded key, or ``None`` when the filename is not a valid
        LMCache data filename.
    """
    if not filename.endswith(_FILE_EXT):
        return None
    stem = filename[: -len(_FILE_EXT)]
    parts = stem.split(_KEY_SEP)
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        return None

    model_name = safe_model.replace(_PATH_SLASH_REPLACEMENT, "/")
    try:
        return ObjectKey(
            chunk_hash=bytes.fromhex(chunk_hash_hex),
            model_name=model_name,
            kv_rank=int(kv_rank_str, 16),
            object_group_id=int(object_group_str, 16),
            cache_salt=cache_salt,
        )
    except ValueError:
        return None
