# SPDX-License-Identifier: Apache-2.0
"""Reversible L2 key names shared by storage adapters."""

# Standard
from collections.abc import Callable
import urllib.parse

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import ObjectKey


def encode_storage_key(key: ObjectKey) -> str:
    """Return the canonical L2 wire name, including every matching field."""
    return format_storage_key(key, "{model}@{rank}@{group}@{hash}{suffix}")


def format_storage_key(
    key: ObjectKey,
    template: str,
    *,
    model_separator: str = "/",
    rank_format: str = "08x",
    model_suffix: str = "",
) -> str:
    """Format an adapter's existing key template without changing its names.

    Args:
        key: Object whose identity determines the name.
        template: Format string accepting ``model``, ``quoted_model``, ``rank``,
            ``group``, ``hash``, ``hash_prefix``, ``salt`` and ``suffix``.
            ``suffix`` includes the salt when nonempty. An empty trailing
            ``@{salt}`` is omitted to preserve existing storage names.
        model_separator: Replacement for slashes in ``model``.
        rank_format: Python integer format for ``rank``.
        model_suffix: Adapter-specific suffix applied to ``model``.

    Returns:
        The formatted storage name.
    """
    if not key.identity.cache_salt and template.endswith("@{salt}"):
        template = template[:-7]
    suffix = f"@{key.identity.cache_salt}" if key.identity.cache_salt else ""
    chunk_hex = key.identity.chunk_hash.hex()
    result = template.format(
        model=key.identity.model_name.replace("/", model_separator) + model_suffix,
        quoted_model=urllib.parse.quote(key.identity.model_name, safe=""),
        rank=format(key.identity.kv_rank, rank_format),
        group=f"{key.identity.object_group_id:x}",
        hash=chunk_hex,
        hash_prefix=chunk_hex[:4],
        salt=key.identity.cache_salt,
        suffix=suffix,
    )
    return result


def storage_shard(key: ObjectKey) -> tuple[str, str]:
    """Return two directory components from the first four hash hex digits."""
    chunk_hex = key.identity.chunk_hash.hex()
    return chunk_hex[:2], chunk_hex[2:4]


def encode_fs_filename(key: ObjectKey) -> str:
    """Return the filename shared by Python and native filesystem adapters."""
    filename = format_storage_key(
        key,
        "{model}@{rank}@{group}@{hash}{suffix}.data",
        model_separator="-SEP-",
        rank_format="#010x",
    )
    return filename


def decode_storage_key(
    encoded: str, *, model_decoder: Callable[[str], str] = str
) -> ObjectKey:
    """Recover an L2 key from its storage name.

    Args:
        encoded: ``model@rank@group@hash[@salt]``.
        model_decoder: Decoder for backend-specific model escaping.

    Returns:
        An identity-equivalent key.

    Raises:
        ValueError: The field count or hexadecimal fields are malformed.
    """
    parts = encoded.split("@")
    if len(parts) == 4:
        parts.append("")
    if len(parts) != 5:
        raise ValueError("invalid ObjectKey encoding: expected 4 or 5 fields")
    model, rank, group, chunk_hash, salt = parts
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=bytes.fromhex(chunk_hash),
            kv_rank=int(rank, 16),
            object_group_id=int(group, 16),
            model_name=model_decoder(model),
            cache_salt=salt,
        )
    )


def decode_fs_filename(filename: str) -> ObjectKey:
    """Recover a Python/native FS key; reject invalid suffixes and fields."""
    if not filename.endswith(".data"):
        raise ValueError("invalid ObjectKey filename extension")
    return decode_storage_key(
        filename[:-5], model_decoder=lambda model: model.replace("-SEP-", "/")
    )
