# SPDX-License-Identifier: Apache-2.0
"""Reversible L2 key names shared by storage adapters."""

# Standard
from collections.abc import Callable, Mapping
import base64
import urllib.parse

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import ObjectKey, validate_identity_tags
from lmcache.v1.utils.frozen_dict import freeze_dict


def encode_tags(tags: Mapping[str, str]) -> str:
    """Return a canonical path-safe tag token."""
    if not tags:
        return ""
    payload = msgspec.json.encode(dict(sorted(tags.items())))
    return "x1-" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_tags(token: str) -> Mapping[str, str]:
    """Decode a tag token, rejecting malformed or noncanonical encodings."""
    if not token:
        return {}
    if not token.startswith("x1-"):
        raise ValueError("unsupported identity tags version")
    encoded = token[3:]
    payload = base64.b64decode(
        encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
    )
    decoded = msgspec.json.decode(payload, type=dict[str, str])
    validate_identity_tags(decoded)
    tags = freeze_dict(decoded, sort_keys=True)
    if encode_tags(tags) != token:
        raise ValueError("non-canonical identity tags token")
    return tags


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
    """Format an adapter's existing key template and append matching tags.

    Args:
        key: Object whose identity determines the name.
        template: Format string accepting ``model``, ``quoted_model``, ``rank``,
            ``group``, ``hash``, ``hash_prefix``, ``salt`` and ``suffix``.
            ``suffix`` includes the salt and tags; templates without it receive
            a trailing tags token. Empty trailing ``@{salt}`` is omitted only
            for untagged keys, preserving their existing storage names.
        model_separator: Replacement for slashes in ``model``.
        rank_format: Python integer format for ``rank``.
        model_suffix: Adapter-specific suffix applied to ``model``.

    Returns:
        The formatted storage name. Tag order does not affect the result.
    """
    token = encode_tags(key.identity.tags)
    if not key.cache_salt and not token and template.endswith("@{salt}"):
        template = template[:-7]
    suffix = f"@{key.cache_salt}" if key.cache_salt else ""
    if token:
        suffix = f"@{key.cache_salt}@{token}"
    chunk_hex = key.chunk_hash.hex()
    result = template.format(
        model=key.model_name.replace("/", model_separator) + model_suffix,
        quoted_model=urllib.parse.quote(key.model_name, safe=""),
        rank=format(key.kv_rank, rank_format),
        group=f"{key.object_group_id:x}",
        hash=chunk_hex,
        hash_prefix=chunk_hex[:4],
        salt=key.cache_salt,
        suffix=suffix,
    )
    if "{suffix}" not in template and token:
        result += f"@{token}"
    return result


def encode_native_storage_keys(keys: list[ObjectKey]) -> list[str]:
    """Reuse a common suffix until the first salt or tag change."""
    if len(keys) < 2:
        return _encode_native_keys_scalar(keys)
    first = keys[0].identity
    second = keys[1].identity
    if second.cache_salt != first.cache_salt or second.tags != first.tags:
        return _encode_native_keys_scalar(keys)
    token = encode_tags(first.tags)
    salt = first.cache_salt
    suffix = f"@{salt}@{token}" if token else (f"@{salt}" if salt else "")
    prefixes = {}
    result = []
    for index, key in enumerate(keys):
        ident = key.identity
        if ident.cache_salt != salt or ident.tags != first.tags:
            result.extend(_encode_native_keys_scalar(keys[index:]))
            break
        fields = (ident.model_name, ident.kv_rank, ident.object_group_id)
        prefix = prefixes.get(fields)
        if prefix is None:
            prefix = (
                f"{ident.model_name}@{ident.kv_rank:08x}@{ident.object_group_id:x}@"
            )
            prefixes[fields] = prefix
        result.append(prefix + ident.chunk_hash.hex() + suffix)
    return result


def _encode_native_keys_scalar(keys: list[ObjectKey]) -> list[str]:
    """Encode the fixed native wire format for arbitrary identities."""
    out = []
    for k in keys:
        identity = k.identity
        token = encode_tags(identity.tags)
        salt = identity.cache_salt
        suffix = f"@{salt}@{token}" if token else (f"@{salt}" if salt else "")
        out.append(
            f"{identity.model_name}@{identity.kv_rank:08x}@{identity.object_group_id:x}@{identity.chunk_hash.hex()}{suffix}"
        )
    return out


def encode_fs_filename(key: ObjectKey) -> str:
    """Return the Python/native FS filename, preserving legacy untagged names."""
    return format_storage_key(
        key,
        "{model}@{rank}@{group}@{hash}{suffix}.data",
        model_separator="-SEP-",
        rank_format="#010x",
    )


def decode_storage_key(
    encoded: str, *, model_decoder: Callable[[str], str] = str
) -> ObjectKey:
    """Recover an L2 key from a legacy or tagged storage name.

    Args:
        encoded: ``model@rank@group@hash[@salt[@tags_token]]``.
        model_decoder: Decoder for backend-specific model escaping.

    Returns:
        An identity-equivalent key.

    Raises:
        ValueError: Fields or the versioned tags token are malformed.
    """
    parts = encoded.split("@")
    token = parts.pop() if len(parts) == 6 else ""
    if len(parts) == 4:
        parts.append("")
    if len(parts) != 5:
        raise ValueError("invalid ObjectKey encoding: expected 4, 5 or 6 fields")
    model, rank, group, chunk_hash, salt = parts
    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk_hash),
        kv_rank=int(rank, 16),
        object_group_id=int(group, 16),
        model_name=model_decoder(model),
        cache_salt=salt,
        tags=decode_tags(token),
    )


def decode_fs_filename(filename: str) -> ObjectKey:
    """Recover a Python/native FS key; reject invalid suffixes and fields."""
    if not filename.endswith(".data"):
        raise ValueError("invalid ObjectKey filename extension")
    return decode_storage_key(
        filename[:-5], model_decoder=lambda model: model.replace("-SEP-", "/")
    )
