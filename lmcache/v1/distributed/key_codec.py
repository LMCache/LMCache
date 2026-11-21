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

# Known tag names are encoded as an index, unknown ones as their name. The table
# is append-only: an index is part of the wire format, so inserting a name in the
# middle would rename every object written by an older engine.
_TAG_NAMES: tuple[str, ...] = ("kv_dtype",)


def _tag_name_item(name: str) -> object:
    """Encode a tag name as its table index, or as itself when unknown."""
    try:
        return _TAG_NAMES.index(name)
    except ValueError:
        return name


def _tag_name_from_item(item: object) -> str:
    """Decode a tag name from its table index or its literal name."""
    if isinstance(item, int):
        try:
            return _TAG_NAMES[item]
        except IndexError:
            raise ValueError(f"unknown identity tag index {item}") from None
    if isinstance(item, str):
        return item
    raise ValueError(f"invalid identity tag name {item!r}")


def encode_tags(tags: Mapping[str, str]) -> str:
    """Return a canonical path-safe tag token.

    The payload is msgpack rather than JSON because every byte lands in an L2
    file name, where the limit is 255 bytes; known names ride as table indices.
    """
    if not tags:
        return ""
    payload = msgspec.msgpack.encode(
        [[_tag_name_item(name), value] for name, value in sorted(tags.items())]
    )
    return "x2-" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _encode_legacy_tags(tags: Mapping[str, str]) -> str:
    """Return the JSON token older engines wrote, for canonicality checks."""
    payload = msgspec.json.encode(dict(sorted(tags.items())))
    return "x1-" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_legacy_tags(encoded: str) -> dict[str, str]:
    """Decode the JSON token older engines wrote."""
    payload = base64.b64decode(
        encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
    )
    try:
        return msgspec.json.decode(payload, type=dict[str, str])
    except Exception as exc:
        # msgspec's error hierarchy differs between versions (DecodeError is a
        # ValueError in some, not in others); this function's contract is
        # ValueError, so normalise rather than leak the dependency's type.
        raise ValueError("invalid identity tags token") from exc


def decode_tags(token: str) -> Mapping[str, str]:
    """Decode a tag token, rejecting malformed or noncanonical encodings."""
    if not token:
        return {}
    if token.startswith("x1-"):
        decoded = _decode_legacy_tags(token[3:])
    elif token.startswith("x2-"):
        encoded = token[3:]
        payload = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        try:
            pairs = msgspec.msgpack.decode(payload)
        except Exception as exc:
            raise ValueError("invalid identity tags token") from exc
        if not isinstance(pairs, list):
            raise ValueError("invalid identity tags token")
        try:
            decoded = {_tag_name_from_item(name): value for name, value in pairs}
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid identity tags token") from exc
    else:
        raise ValueError("unsupported identity tags version")
    if not decoded:
        # An empty mapping is encoded as no token at all, never as one.
        raise ValueError("non-canonical identity tags token")
    validate_identity_tags(decoded)
    tags = freeze_dict(decoded, sort_keys=True)
    # Only the current encoding has a canonical form here: a legacy token was
    # produced by a different encoder and is accepted as written.
    canonical = (
        encode_tags(tags) if token.startswith("x2-") else _encode_legacy_tags(tags)
    )
    if canonical != token:
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
