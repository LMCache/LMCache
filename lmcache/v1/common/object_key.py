# SPDX-License-Identifier: Apache-2.0
"""Keys that identify stored KV objects, independent of either endpoint."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

# First Party
from lmcache.v1.common.frozen_dict import freeze_dict


def validate_identity_tags(tags: Mapping[str, str]) -> None:
    """Validate identity tag names and values."""
    for name, value in tags.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError("tags require non-empty string names and string values")


@dataclass(frozen=True)
class ObjectKeyIdentity:
    """Matching fields shared by local objects and internal wire messages."""

    chunk_hash: bytes
    kv_rank: int
    model_name: str
    object_group_id: int = 0
    cache_salt: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "@" in self.model_name:
            raise ValueError("model_name must not contain '@'")
        if self.object_group_id < 0:
            raise ValueError("object_group_id must be >= 0")
        if any(char in self.cache_salt for char in "@/\\\x00"):
            raise ValueError("cache_salt must not contain '@', '/', '\\' or NUL")
        if len(self.cache_salt) > 128:
            raise ValueError("cache_salt exceeds max length 128")
        validate_identity_tags(self.tags)
        object.__setattr__(self, "tags", freeze_dict(self.tags, sort_keys=True))


@dataclass(frozen=True, init=False)
class ObjectKey:
    """
    The unique identifier for an object in the distributed storage manager
    """

    identity: ObjectKeyIdentity

    extra_attributes: dict[str, object] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Request-scoped metadata exposed to L2 adapters.

    This field is excluded from equality, hashing, and ``repr``.
    Upstream modules may enrich it before dispatch.
    L1 managers and L2 adapters should not mutate it.
    """

    def __init__(
        self,
        chunk_hash: bytes | None = None,
        model_name: str | None = None,
        kv_rank: int | None = None,
        object_group_id: int | None = None,
        cache_salt: str | None = None,
        *,
        identity: ObjectKeyIdentity | None = None,
        extra_attributes: dict[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Build a key from its identity or the existing flat constructor.

        Explicit flat fields override the supplied identity. This also retains
        ``dataclasses.replace(key, object_group_id=...)`` and other flat field
        replacements used by existing consumers.

        Args:
            chunk_hash: Content hash; required without ``identity``.
            model_name: Model name; required without ``identity``.
            kv_rank: Packed KV rank; required without ``identity``.
            object_group_id: Object group; defaults to 0 without an identity.
            cache_salt: Isolation salt; defaults to empty without an identity.
            identity: Immutable identity providing fields not supplied explicitly.
            extra_attributes: Request-scoped metadata exposed to L2 adapters,
                shallow-copied so the key can be enriched independently.
            tags: Additional matching dimensions for the flat constructor.

        Raises:
            TypeError: A required flat field is missing without ``identity``.
            ValueError: An identity field violates its invariant.
        """
        if identity is None:
            if chunk_hash is None or model_name is None or kv_rank is None:
                raise TypeError("chunk_hash, model_name and kv_rank are required")
            identity = ObjectKeyIdentity(
                chunk_hash=chunk_hash,
                model_name=model_name,
                kv_rank=kv_rank,
                object_group_id=object_group_id if object_group_id is not None else 0,
                cache_salt=cache_salt if cache_salt is not None else "",
                tags=tags if tags is not None else {},
            )
        elif (
            chunk_hash is not None
            or model_name is not None
            or kv_rank is not None
            or object_group_id is not None
            or cache_salt is not None
            or tags is not None
        ):
            identity = replace(
                identity,
                chunk_hash=chunk_hash
                if chunk_hash is not None
                else identity.chunk_hash,
                model_name=model_name
                if model_name is not None
                else identity.model_name,
                kv_rank=kv_rank if kv_rank is not None else identity.kv_rank,
                object_group_id=(
                    object_group_id
                    if object_group_id is not None
                    else identity.object_group_id
                ),
                cache_salt=cache_salt
                if cache_salt is not None
                else identity.cache_salt,
                tags=tags if tags is not None else identity.tags,
            )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self,
            "extra_attributes",
            dict(extra_attributes) if extra_attributes is not None else {},
        )

    @property
    def chunk_hash(self) -> bytes:
        """Read chunk_hash from the identity."""
        return self.identity.chunk_hash

    @property
    def model_name(self) -> str:
        """Read model_name from the identity."""
        return self.identity.model_name

    @property
    def cache_salt(self) -> str:
        """Read cache_salt from the identity."""
        return self.identity.cache_salt

    @property
    def kv_rank(self) -> int:
        """Read kv_rank from the identity."""
        return self.identity.kv_rank

    @property
    def object_group_id(self) -> int:
        """Read object_group_id from the identity."""
        return self.identity.object_group_id

    def to_encoded_object_key(self) -> "EncodedObjectKey":
        """Return the JSON-safe :class:`EncodedObjectKey` projection."""
        return EncodedObjectKey(
            chunk_hash_hex=self.chunk_hash.hex(),
            model_name=self.model_name,
            kv_rank=self.kv_rank,
            object_group_id=self.object_group_id,
            cache_salt=self.cache_salt,
            tags=self.identity.tags,
        )

    @staticmethod
    def IntHash2Bytes(chunk_hash: int) -> bytes:
        # NOTE: this is only used by tests
        return chunk_hash.to_bytes(4, byteorder="big")

    @staticmethod
    def Bytes2IntHash(chunk_hash: bytes) -> int:
        # NOTE: this is only used by tests
        return int.from_bytes(chunk_hash, byteorder="big") & ((1 << 64) - 1)

    @staticmethod
    def ComputeKVRank(
        world_size: int,
        global_rank: int,
        local_world_size: int,
        local_rank: int,
    ) -> int:
        """
        Compute the kv_rank from world_size and worker_id

        Args:
            world_size (int): The total number of workers (include TP + PP)
            global_rank (int): The global worker id (from 0 to world_size - 1)
            local_world_size (int): The local world size (for local node),
                should NOT be greater than 8
            local_rank (int): The local world rank (for local node)

        Returns:
            The special KV rank (bitmap) used by the objectkey

        Example:
            In the case of TP=4, PP=2, the TP worker 1 on node 1 has:
            - world_size = 8
            - global_rank = 5
            - local_world_size = 4
            - local_rank = 1

            The output KV rank is the bitmap:
            +--head--+
            |00000000|
            |00000000|
            |00000000|
            |00000000| layers
            |00001100|
            |00001100|
            |00001100|
            |00001100|
            +--------+
        """
        # TODO(ApostaC): in the long run, we want to have the above bitmap based
        # representation for asymmetric parallelism (e.g., sharing across different
        # TP/PP settings).
        # For now, let's have a simple implementation that just
        # differentiate between different parallel setups

        # For each number, we use 8-bit, and pack them together
        return (
            (world_size << 24)
            | (global_rank << 16)
            | (local_world_size << 8)
            | local_rank
        )

    @staticmethod
    def WorldSizeFromKVRank(kv_rank: int) -> int:
        """Recover the world size :meth:`ComputeKVRank` packed into a rank.

        Args:
            kv_rank: A ``kv_rank`` produced by :meth:`ComputeKVRank`.

        Returns:
            The parallel setup's world size (TP x PP).
        """
        return (kv_rank >> 24) & 0xFF


@dataclass(frozen=True)
class EncodedObjectKey:
    """JSON-safe wire form of :class:`ObjectKey` — ``chunk_hash`` is
    hex-encoded; other fields are preserved verbatim."""

    chunk_hash_hex: str
    """Hex-encoded ``ObjectKey.chunk_hash``."""

    model_name: str
    kv_rank: int

    object_group_id: int = 0
    """Defaults to ``0`` so pre-``object_group_id`` wire payloads still
    deserialize."""

    cache_salt: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)
    """Canonical matching tags that participate in object identity."""

    def __post_init__(self) -> None:
        validate_identity_tags(self.tags)
        object.__setattr__(self, "tags", freeze_dict(self.tags, sort_keys=True))

    def to_object_key(self) -> ObjectKey:
        """Recover the corresponding :class:`ObjectKey`.

        Raises:
            ValueError: ``chunk_hash_hex`` is not valid hex, or one of
                :class:`ObjectKey`'s field invariants is violated.
        """
        return ObjectKey(
            chunk_hash=bytes.fromhex(self.chunk_hash_hex),
            model_name=self.model_name,
            kv_rank=self.kv_rank,
            object_group_id=self.object_group_id,
            cache_salt=self.cache_salt,
            tags=self.tags,
        )
