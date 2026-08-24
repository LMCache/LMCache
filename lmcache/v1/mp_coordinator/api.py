# SPDX-License-Identifier: Apache-2.0
"""Cross-module contract vocabulary for the MP coordinator.

The cache-event types both sides of the event stream speak: emitted by
MP servers, consumed by the coordinator's key directory.
Encoding-level checks (key convertibility,
hex validity) belong to the HTTP envelopes in :mod:`schemas`.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, Tier

# ``token_offset`` value meaning "the emitter did not report a position".
# Distinct from 0, which is a real position (a chunk at the start of its
# sequence): an emitter predating token offsets sends no offset at all, and
# treating that as 0 would place every chunk at the sequence start and
# re-RoPE reused KV from the wrong source position.
UNKNOWN_TOKEN_OFFSET = -1


class CacheEventType(str, Enum):
    """The kind of cache-state change a :class:`CacheEventBatch` reports.

    ``STORE`` commits placements; ``DELETE`` removes them (owners report
    evictions as deletes); ``ACCESS`` refreshes recency without changing
    placement state. ``CONFIG`` carries no placement at all: it declares
    one compartment's configured capacity, so the fleet view has a
    denominator for the bytes the other three report.
    """

    STORE = "store"
    DELETE = "delete"
    ACCESS = "access"
    CONFIG = "config"


@dataclass(frozen=True)
class BlendMatch:
    """One cached chunk found inside a query sequence.

    Shared by the coordinator's blend index, which produces matches, and
    the MP-server client, which consumes them.

    Attributes:
        chunk_hash: The matched chunk's ``ObjectKey.chunk_hash``; the
            same bytes a local ``CBMatchResult.hash`` holds.
        old_st: Its position in the sequence it was stored under
            (re-RoPE source).
        cur_st: Its position in the query (re-RoPE target).
    """

    chunk_hash: bytes
    old_st: int
    cur_st: int


@dataclass(frozen=True)
class CacheEventEntry:
    """One key's worth of change inside a :class:`CacheEventBatch`.

    Attributes:
        key: The object key the change applies to.
        size_bytes: Bytes committed for the key (``store`` only; ``0``
            otherwise).
        token_ids: The chunk's token ids, stamped on ``store`` entries
            (empty when the emitter no longer holds them); the directory
            indexes them by the key's chunk hash.
        token_offset: Token position of the chunk's first token in the
            sequence it was stored under; prefix-chained chunk hashes do
            not reveal it. :data:`UNKNOWN_TOKEN_OFFSET` when unreported.
    """

    key: EncodedObjectKey
    size_bytes: int = 0
    token_ids: list[int] = field(default_factory=list)
    token_offset: int = UNKNOWN_TOKEN_OFFSET

    def __post_init__(self) -> None:
        """Enforce intrinsic invariants.

        Raises:
            ValueError: If ``size_bytes`` is negative, or ``token_offset``
                is negative and not :data:`UNKNOWN_TOKEN_OFFSET`.
        """
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0 (got {self.size_bytes})")
        if self.token_offset < 0 and self.token_offset != UNKNOWN_TOKEN_OFFSET:
            raise ValueError(
                f"token_offset must be >= 0 or UNKNOWN_TOKEN_OFFSET "
                f"({UNKNOWN_TOKEN_OFFSET}), got {self.token_offset}"
            )


@dataclass(frozen=True)
class CacheEventBatch:
    """A batch of same-typed cache events from one MP server.

    Attributes:
        instance_id: The emitter's unique ID (non-empty).
        incarnation: The emitter's restart counter (non-negative). A
            higher value fences off all placements reported by lower
            values of the same ``instance_id``.
        seq: Per-``(instance_id, incarnation)`` monotonic batch counter,
            starting at 1.
        event_type: What happened to every entry in the batch.
        tier: The cache tier the events apply to (``l1`` or ``l2``;
            never ``all``).
        backend: The storage backend within the tier (``"dram"``,
            ``"cxl"``, ``"fs"``, ``"valkey"``, ...). Required non-empty
            for ``store``/``delete`` (it is part of the placement
            identity); empty for ``access``, which only refreshes
            key-level recency and carries no placement identity.
        entries: The affected keys. Always empty for ``config``, which
            declares a compartment rather than reporting placements.
        shared: ``True`` when the backend is a storage domain mounted by
            several instances (e.g. one S3 bucket or CXL pool). ``False``
            (default) marks the storage private to this instance.
        ts: Emitter wall-clock seconds for the batch (``0.0`` if unknown).
        capacity_bytes: ``config`` only. The compartment's configured
            capacity; ``0`` declares no limit.
        capacity_revision: ``config`` only. Which declaration this batch
            belongs to. One declaration spans one batch per compartment,
            all sharing a revision, so the consumer can tell a fresh
            declaration from a continuation and drop compartments the new
            one omits.
    """

    instance_id: str
    incarnation: int
    seq: int
    event_type: CacheEventType
    tier: Tier
    backend: str
    entries: list[CacheEventEntry] = field(default_factory=list)
    shared: bool = False
    ts: float = 0.0
    capacity_bytes: int = 0
    capacity_revision: int = 0

    def __post_init__(self) -> None:
        """Enforce intrinsic invariants.

        Raises:
            ValueError: If ``instance_id`` is empty, ``backend`` is empty
                on a placement-bearing batch (``store``/``delete``) or on
                ``config``, ``incarnation``, ``ts``, ``capacity_bytes`` or
                ``capacity_revision`` is negative, ``seq`` < 1, ``tier`` is
                not a concrete tier (``l1``/``l2``), or a ``config`` batch
                carries entries.
        """
        if not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        if not self.backend and self.event_type != CacheEventType.ACCESS:
            raise ValueError(
                f"backend must be non-empty for {self.event_type.value} batches"
            )
        if self.event_type == CacheEventType.CONFIG and self.entries:
            raise ValueError("config batches declare capacity and carry no entries")
        if self.capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0 (got {self.capacity_bytes})")
        if self.capacity_revision < 0:
            raise ValueError(
                f"capacity_revision must be >= 0 (got {self.capacity_revision})"
            )
        if self.incarnation < 0:
            raise ValueError(f"incarnation must be >= 0 (got {self.incarnation})")
        if self.seq < 1:
            raise ValueError(f"seq must be >= 1 (got {self.seq})")
        if self.tier not in (Tier.L1, Tier.L2):
            raise ValueError(
                f"cache events must target a concrete tier (got {self.tier.value!r})"
            )
        if self.ts < 0.0:
            raise ValueError(f"ts must be >= 0 (got {self.ts})")
