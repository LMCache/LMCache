# SPDX-License-Identifier: Apache-2.0
"""Immutable native KV-event identities and ordered stream payloads."""

# Standard
from dataclasses import dataclass

NativeBlockHash = bytes | int
"""Opaque native content identity; integers retain their original signedness."""


@dataclass(frozen=True)
class KVEventProfile:
    """Immutable native identity contract for one model/cache context.

    ``profile_id`` identifies the engine, hash algorithm, seed semantics and
    identity representation/version. ``block_size`` is the native GPU-event
    granularity, which must divide the server's chunk size.
    """

    profile_id: str
    block_size: int

    def __post_init__(self) -> None:
        if not self.profile_id or self.block_size <= 0:
            raise ValueError(
                "native event profile requires an ID and positive block size"
            )


@dataclass(frozen=True)
class EventBinding:
    """Native identities and original tokens for one complete LMCache chunk.

    Hashes and parent come from the engine's native event path. MP never
    reconstructs them. Tuples keep retained replay records independent of
    request cleanup and subsequent binding registrations.
    """

    profile_id: str
    block_hashes: tuple[NativeBlockHash, ...]
    parent_block_hash: NativeBlockHash | None
    token_ids: tuple[int, ...]
    block_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_hashes, tuple) or not isinstance(
            self.token_ids, tuple
        ):
            raise ValueError(
                "native event binding hashes and tokens must be immutable tuples"
            )
        if not self.profile_id or self.block_size <= 0 or not self.block_hashes:
            raise ValueError(
                "native event binding requires a profile and complete blocks"
            )
        if len(self.token_ids) != len(self.block_hashes) * self.block_size:
            raise ValueError("native event blocks and tokens are not aligned")
        hashes = self.block_hashes + (
            () if self.parent_block_hash is None else (self.parent_block_hash,)
        )
        if any(type(value) not in (bytes, int) for value in hashes):
            raise ValueError("native event identities must be bytes or integers")
        if len({type(value) for value in hashes}) != 1:
            raise ValueError("native event identities must use one representation")


# Subscribe only to advertising servers; older versions expose a polling API.
KV_EVENT_CAPABILITY = "kv_event_native_bindings_v1"
KV_EVENT_KIND_STORED = "stored"
KV_EVENT_KIND_REMOVED = "removed"
KV_EVENT_MEDIUM_CPU = "CPU"


@dataclass(frozen=True)
class KVEventRecord:
    """Immutable native CPU transition; removals own exactly the advertised IDs."""

    seq: int
    kind: str
    medium: str
    model_name: str
    block_hashes: tuple[NativeBlockHash, ...]
    parent_block_hash: NativeBlockHash | None
    token_ids: tuple[int, ...]
    block_size: int
    profile_id: str
    world_size: int
    cache_salt: str = ""


@dataclass
class KVEventBatch:
    """One ordered batch. Incarnation changes or lost=True require withdrawal
    of earlier announcements; enabled=False ends the stream. Resume at next_cursor.
    """

    enabled: bool
    incarnation: int
    next_cursor: int
    lost: bool
    events: list[KVEventRecord]
