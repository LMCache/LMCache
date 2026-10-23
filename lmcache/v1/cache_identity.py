# SPDX-License-Identifier: Apache-2.0
"""Object identities and request-config extensions."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NoReturn
import base64

# Third Party
import msgspec


class _FrozenTags(dict[str, str]):
    """An immutable dict snapshot that msgspec can serialize."""

    def _immutable(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("identity tags are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def normalize_identity(
    model_name: str, cache_salt: str, tags: Mapping[str, str]
) -> Mapping[str, str]:
    """Validate common identity fields and snapshot tags in name order."""
    if "@" in model_name:
        raise ValueError("model_name must not contain '@'")
    if any(char in cache_salt for char in "@/\\\x00"):
        raise ValueError("cache_salt must not contain '@', '/', '\\' or NUL")
    if len(cache_salt) > 128:
        raise ValueError("cache_salt exceeds max length 128")
    for name, value in tags.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError("tags require non-empty string names and string values")
    return _FrozenTags(sorted(tags.items()))


def encode_tags(tags: Mapping[str, str]) -> str:
    """Return a canonical path-safe tag token; empty tags preserve legacy names."""
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
    tags = normalize_identity("", "", msgspec.json.decode(payload, type=dict[str, str]))
    if encode_tags(tags) != token:
        raise ValueError("non-canonical identity tags token")
    return tags


def get_request_tags(request_configs: Mapping[str, Any] | None) -> Mapping[str, str]:
    """Extract and canonicalize matching tags from request configuration."""
    return normalize_identity(
        "",
        "",
        {
            name.removeprefix("lmcache.tag."): value
            for name, value in (request_configs or {}).items()
            if name.startswith("lmcache.tag.")
        },
    )


def get_request_xattrs(request_configs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract L1 xattrs from request configuration."""
    return {
        name.removeprefix("lmcache.xattr."): value
        for name, value in (request_configs or {}).items()
        if name.startswith("lmcache.xattr.")
    }


@dataclass(frozen=True)
class ObjectKeyIdentity:
    """Complete matching identity of a stored object."""

    chunk_hash: bytes
    kv_rank: int
    model_name: str
    object_group_id: int = 0
    cache_salt: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tags",
            normalize_identity(self.model_name, self.cache_salt, self.tags),
        )
        if self.object_group_id < 0:
            raise ValueError("object_group_id must be >= 0")

    def __hash__(self) -> int:
        return hash(
            (
                self.chunk_hash,
                self.kv_rank,
                self.object_group_id,
                self.model_name,
                self.cache_salt,
                tuple(self.tags.items()),
            )
        )

    def tags_token(self) -> str:
        """Return the canonical encoding of opaque matching tags."""
        return encode_tags(self.tags)
