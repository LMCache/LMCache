# SPDX-License-Identifier: Apache-2.0
"""Versioned cache identity and compatibility checks.

The token digest and model name already live in LMCache's storage keys.  This
module defines the additional revisions needed to decide whether two KV-cache
representations are safe to reuse.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, cast
import hashlib
import json
import re

CACHE_IDENTITY_CONFIG_PREFIX = "lmcache.cache_identity."
CACHE_IDENTITY_REVISION_TAG = "lmcache.tag.cache_identity_revision"

_CACHE_IDENTITY_CONFIG_ROOT = CACHE_IDENTITY_CONFIG_PREFIX.removesuffix(".")
_REVISION_RE = re.compile(r"v1:[0-9a-f]{64}\Z")
_MAX_VALUE_LENGTH = 256
_HASH_DOMAIN = b"lmcache-cache-identity-v1\x00"


class CacheIdentityError(ValueError):
    """Raised when a cache identity is incomplete or malformed."""


class CacheIdentityMismatchError(CacheIdentityError):
    """Raised when cache identities are not reuse-compatible."""


def _validate_revision_value(
    field_name: str, value: str | None, *, required: bool = False
) -> None:
    """Validate one opaque revision value without normalizing it."""
    if value is None:
        if required:
            raise CacheIdentityError(
                f"cache identity field {field_name!r} must be provided"
            )
        return
    if not isinstance(value, str):
        raise CacheIdentityError(
            f"cache identity field {field_name!r} must be a string"
        )
    if not value or value != value.strip():
        raise CacheIdentityError(
            f"cache identity field {field_name!r} must be non-empty and "
            "must not have surrounding whitespace"
        )
    if len(value) > _MAX_VALUE_LENGTH:
        raise CacheIdentityError(
            f"cache identity field {field_name!r} exceeds "
            f"{_MAX_VALUE_LENGTH} characters"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CacheIdentityError(
            f"cache identity field {field_name!r} contains a control character"
        )


@dataclass(frozen=True, slots=True)
class BaseCacheIdentity:
    """Semantic revisions that determine the KV values for a token prefix."""

    model_revision: str
    tokenizer_revision: str
    weight_revision: str
    adapter_revision: str | None = None

    def __post_init__(self) -> None:
        """Reject missing or malformed semantic revision fields."""
        for item in fields(self):
            _validate_revision_value(
                item.name,
                getattr(self, item.name),
                required=item.name != "adapter_revision",
            )


@dataclass(frozen=True, slots=True)
class CacheRepresentationIdentity:
    """Physical and algorithm revisions required to decode cached KV safely."""

    topology_fingerprint: str
    backend_revision: str
    kv_dtype: str
    quantization_revision: str | None = None
    compression_revision: str | None = None
    drop_algorithm_id: str | None = None
    drop_policy_revision: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed representations and half-specified drop policies."""
        for item in fields(self):
            _validate_revision_value(
                item.name,
                getattr(self, item.name),
                required=item.name
                in {"topology_fingerprint", "backend_revision", "kv_dtype"},
            )
        if (self.drop_algorithm_id is None) != (self.drop_policy_revision is None):
            raise CacheIdentityError(
                "drop_algorithm_id and drop_policy_revision must be set together"
            )

    def mismatched_fields(
        self, other: "CacheRepresentationIdentity"
    ) -> tuple[str, ...]:
        """Return representation fields whose values differ from ``other``.

        Args:
            other: Representation identity to compare against.

        Returns:
            Field names that prevent cache reuse. ``("type",)`` denotes an
            object of the wrong type.
        """
        if not isinstance(other, CacheRepresentationIdentity):
            return ("type",)
        return tuple(
            item.name
            for item in fields(self)
            if getattr(self, item.name) != getattr(other, item.name)
        )

    def is_compatible_with(self, other: "CacheRepresentationIdentity") -> bool:
        """Return whether cached bytes from ``other`` are safe to reuse.

        Args:
            other: Representation identity attached to the cached bytes.

        Returns:
            ``True`` when every representation field matches.
        """
        return not self.mismatched_fields(other)


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Complete semantic and representation identity for cache reuse."""

    base: BaseCacheIdentity
    representation: CacheRepresentationIdentity

    def __post_init__(self) -> None:
        """Reject components that are not the declared identity types."""
        if not isinstance(self.base, BaseCacheIdentity):
            raise CacheIdentityError("base must be a BaseCacheIdentity")
        if not isinstance(self.representation, CacheRepresentationIdentity):
            raise CacheIdentityError(
                "representation must be a CacheRepresentationIdentity"
            )

    @property
    def revision(self) -> str:
        """Return a stable digest suitable for cache-key namespacing.

        Returns:
            A lowercase SHA-256 digest prefixed with the identity schema
            version, currently ``"v1:"``.
        """
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"v1:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, dict[str, str | None]]:
        """Return the canonical JSON-compatible identity payload.

        Returns:
            A nested mapping with ``base`` and ``representation`` sections.
        """
        return {
            "base": asdict(self.base),
            "representation": asdict(self.representation),
        }

    def to_request_configs(self) -> dict[str, str]:
        """Flatten this identity into request-scoped LMCache configuration.

        Returns:
            Required and configured optional identity fields under the
            ``lmcache.cache_identity.`` namespace.
        """
        values = {**asdict(self.base), **asdict(self.representation)}
        return {
            f"{CACHE_IDENTITY_CONFIG_PREFIX}{name}": value
            for name, value in values.items()
            if value is not None
        }

    def mismatched_fields(self, other: "CacheIdentity") -> tuple[str, ...]:
        """Return fully-qualified fields whose values differ from ``other``.

        Args:
            other: Complete cache identity to compare against.

        Returns:
            Field paths that prevent reuse. ``("type",)`` denotes an object
            of the wrong type.
        """
        if not isinstance(other, CacheIdentity):
            return ("type",)
        mismatches = [
            f"base.{item.name}"
            for item in fields(self.base)
            if getattr(self.base, item.name) != getattr(other.base, item.name)
        ]
        mismatches.extend(
            f"representation.{name}"
            for name in self.representation.mismatched_fields(other.representation)
        )
        return tuple(mismatches)

    def is_compatible_with(self, other: "CacheIdentity") -> bool:
        """Return whether a cache written under ``other`` may be reused.

        Args:
            other: Identity attached to the cached representation.

        Returns:
            ``True`` when the semantic and representation fields all match.
        """
        return not self.mismatched_fields(other)

    def require_compatible(self, other: "CacheIdentity") -> None:
        """Require ``other`` to be reuse-compatible with this identity.

        Args:
            other: Identity attached to the cached representation.

        Raises:
            CacheIdentityMismatchError: If any identity field differs.
        """
        mismatches = self.mismatched_fields(other)
        if mismatches:
            raise CacheIdentityMismatchError(
                "cache identity mismatch: " + ", ".join(mismatches)
            )


_BASE_FIELDS = {item.name for item in fields(BaseCacheIdentity)}
_REPRESENTATION_FIELDS = {item.name for item in fields(CacheRepresentationIdentity)}
_ALL_FIELDS = _BASE_FIELDS | _REPRESENTATION_FIELDS
_REQUIRED_FIELDS = {
    "model_revision",
    "tokenizer_revision",
    "weight_revision",
    "topology_fingerprint",
    "backend_revision",
    "kv_dtype",
}


def cache_identity_from_request_configs(
    request_configs: Mapping[str, Any] | None,
) -> CacheIdentity | None:
    """Parse a strict cache identity from request-scoped configuration.

    An identity is optional. Once any structured identity field is present,
    all required fields must be present. The internal digest tag carried by a
    materialized or deserialized key cannot reconstruct the structured
    identity, so this function validates it and returns ``None``.

    Args:
        request_configs: Request-scoped LMCache configuration, or ``None``.

    Returns:
        The parsed identity, or ``None`` when no structured identity exists.

    Raises:
        CacheIdentityError: If the identity is incomplete, malformed, mixed
            with the internal digest tag, or contains an unknown field.
    """
    if not request_configs:
        return None

    has_tag_revision = CACHE_IDENTITY_REVISION_TAG in request_configs
    tag_revision = request_configs.get(CACHE_IDENTITY_REVISION_TAG)
    structured: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in request_configs.items():
        if key == _CACHE_IDENTITY_CONFIG_ROOT:
            unknown.append("<root>")
            continue
        if not isinstance(key, str) or not key.startswith(CACHE_IDENTITY_CONFIG_PREFIX):
            continue
        name = key[len(CACHE_IDENTITY_CONFIG_PREFIX) :]
        if name not in _ALL_FIELDS:
            unknown.append(name)
        else:
            if value is None:
                raise CacheIdentityError(
                    f"cache identity field {name!r} must be a string"
                )
            structured[name] = value

    if unknown:
        raise CacheIdentityError(
            "unknown cache identity field(s): " + ", ".join(sorted(unknown))
        )
    if has_tag_revision:
        if structured:
            raise CacheIdentityError(
                "internal cache identity revision cannot be mixed with "
                "structured identity fields"
            )
        _validate_compact_revision(tag_revision)
        return None
    if not structured:
        return None

    missing = _REQUIRED_FIELDS - structured.keys()
    if missing:
        raise CacheIdentityError(
            "incomplete cache identity; missing: " + ", ".join(sorted(missing))
        )
    base = BaseCacheIdentity(
        model_revision=structured["model_revision"],
        tokenizer_revision=structured["tokenizer_revision"],
        weight_revision=structured["weight_revision"],
        adapter_revision=structured.get("adapter_revision"),
    )
    representation = CacheRepresentationIdentity(
        topology_fingerprint=structured["topology_fingerprint"],
        backend_revision=structured["backend_revision"],
        kv_dtype=structured["kv_dtype"],
        quantization_revision=structured.get("quantization_revision"),
        compression_revision=structured.get("compression_revision"),
        drop_algorithm_id=structured.get("drop_algorithm_id"),
        drop_policy_revision=structured.get("drop_policy_revision"),
    )
    return CacheIdentity(base=base, representation=representation)


def cache_identity_revision(
    request_configs: Mapping[str, Any] | None,
) -> str:
    """Return the validated cache identity revision.

    Args:
        request_configs: Request-scoped LMCache configuration, or ``None``.

    Returns:
        A versioned digest, or ``""`` when no identity is configured.

    Raises:
        CacheIdentityError: If the structured identity or internal digest tag
            is incomplete or malformed.
    """
    if not request_configs:
        return ""
    identity = cache_identity_from_request_configs(request_configs)
    if identity is not None:
        return identity.revision
    revision = request_configs.get(CACHE_IDENTITY_REVISION_TAG, "")
    if revision:
        _validate_compact_revision(revision)
    return cast(str, revision)


def materialize_cache_identity_revision(
    request_configs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Replace structured identity fields with one validated revision tag.

    Token databases call this once per request before constructing many cache
    keys. The compact copy keeps all unrelated request configuration intact
    while avoiding repeated canonical JSON and digest work for every chunk.

    Args:
        request_configs: Request-scoped LMCache configuration, or ``None``.

    Returns:
        Configuration with structured identity fields replaced by one internal
        digest tag. ``None`` remains ``None``.

    Raises:
        CacheIdentityError: If the configured identity is invalid.
    """
    if not request_configs:
        return None if request_configs is None else dict(request_configs)
    revision = cache_identity_revision(request_configs)
    if not revision:
        return (
            request_configs
            if isinstance(request_configs, dict)
            else dict(request_configs)
        )
    if CACHE_IDENTITY_REVISION_TAG in request_configs and not any(
        isinstance(key, str) and key.startswith(CACHE_IDENTITY_CONFIG_PREFIX)
        for key in request_configs
    ):
        return (
            request_configs
            if isinstance(request_configs, dict)
            else dict(request_configs)
        )
    materialized = {
        key: value
        for key, value in request_configs.items()
        if not (isinstance(key, str) and key.startswith(CACHE_IDENTITY_CONFIG_PREFIX))
    }
    materialized[CACHE_IDENTITY_REVISION_TAG] = revision
    return materialized


def namespace_chunk_hash(chunk_hash: bytes, revision: str) -> bytes:
    """Bind a chunk hash to an identity revision.

    Args:
        chunk_hash: Existing token-prefix chunk digest.
        revision: Validated versioned cache identity digest, or ``""``.

    Returns:
        A domain-separated SHA-256 digest. When ``revision`` is empty, returns
        the original ``chunk_hash`` object to preserve legacy keys exactly.

    Raises:
        CacheIdentityError: If a non-empty revision is malformed.
    """
    if not revision:
        return chunk_hash
    _validate_compact_revision(revision)
    return hashlib.sha256(
        _HASH_DOMAIN + revision.encode("ascii") + b"\x00" + chunk_hash
    ).digest()


def namespace_chunk_hashes(chunk_hashes: list[bytes], revision: str) -> list[bytes]:
    """Bind multiple chunk hashes to one validated identity revision.

    This batch form validates and encodes the revision once, which keeps the
    multiprocess expansion cost proportional to chunks rather than object
    groups or tensor-parallel ranks.

    Args:
        chunk_hashes: Existing token-prefix chunk digests.
        revision: Validated versioned cache identity digest, or ``""``.

    Returns:
        Domain-separated SHA-256 digests. When ``revision`` is empty, returns
        the original list to preserve the legacy fast path.

    Raises:
        CacheIdentityError: If a non-empty revision is malformed.
    """
    if not revision:
        return chunk_hashes
    _validate_compact_revision(revision)
    prefix = _HASH_DOMAIN + revision.encode("ascii") + b"\x00"
    return [hashlib.sha256(prefix + chunk_hash).digest() for chunk_hash in chunk_hashes]


def _validate_compact_revision(revision: Any) -> None:
    """Validate the versioned digest accepted on serialized key round-trips."""
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise CacheIdentityError(
            "cache identity revision must have form 'v1:' followed by "
            "64 lowercase hexadecimal characters"
        )
