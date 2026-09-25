# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args
import enum

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)

FetchingPolicy = Literal["prefix", "full"]
"""Which found objects a prefetch loads and reports.

``"prefix"`` -- only fetch the prefix hit and discard all non-prefix hits.

``"full"`` -- fetch all of the hit chunks, no matter whether they are in the
prefix or not.
"""

FULL_ATTENTION_WINDOW_CHUNKS = -1
"""``GroupedObjectKeys.sliding_window_size`` value for a full-attention object
group: serving a prefix needs every chunk of it present."""

_VALID_FETCHING_POLICIES = frozenset(get_args(FetchingPolicy))


def _lookup_kv_ranks(ipc_key: "IPCCacheServerKey") -> list[int]:
    """The kv ranks an IPC key addresses, in rank order.

    A key without a ``worker_id`` (a lookup) fans out to every worker of its
    world size; a worker-specific key addresses that worker's shard only.
    """
    if ipc_key.worker_id is None:
        # For look up request, we want to expand to all workers
        # TODO (ApostaC): include local world size/rank info
        # in the future once it's in IPCCacheServerKey
        return [
            ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=worker_id,
            )
            for worker_id in range(ipc_key.world_size)
        ]
    return [
        ObjectKey.ComputeKVRank(
            world_size=ipc_key.world_size,
            global_rank=ipc_key.worker_id,
            local_world_size=ipc_key.world_size,
            local_rank=ipc_key.worker_id,
        )
    ]


class Tier(str, enum.Enum):
    """A cache tier.

    Subclasses ``str`` so it validates from / compares equal to the bare wire
    value (``Tier.L2 == "l2"``) and serializes as that value. ``ALL`` is only
    valid for operations that explicitly support multiple tiers.
    """

    L1 = "l1"
    L2 = "l2"
    ALL = "all"


class L1BackendType(str, enum.Enum):
    """The storage medium backing the L1 tier (a closed set, unlike L2
    backends, which are an open adapter-type registry).

    Subclasses ``str`` so it compares equal to and serializes as the bare
    wire value (``L1BackendType.DRAM == "dram"``).
    """

    DRAM = "dram"
    DEVDAX = "devdax"
    GDS = "gds"


@dataclass(frozen=True)
class ObjectKey:
    """
    The unique identifier for an object in the distributed storage manager
    """

    chunk_hash: bytes
    """ Content hash of this particular chunk """

    model_name: str
    """ Name of the model this chunk belongs to.

    Invariant: must not contain ``@``. The L2 adapters use ``@`` as the
    field separator in serialized keys/filenames and rely on this
    invariant for unambiguous parsing. HuggingFace model IDs use
    alphanumerics + ``/-_.`` so this rejects nothing that appears in
    practice.
    """

    kv_rank: int
    """ The rank that uniquely identifies the slice of the KV cache """

    object_group_id: int = 0
    """ Index of the object group this chunk belongs to. """

    cache_salt: str = ""
    """ Per-user isolation salt. Same content from different users with
    different cache_salt values produces different ObjectKeys, giving
    strict per-user cache isolation. Defaults to empty string, in which
    case serialized keys and filenames match the pre-cache_salt shape
    (no trailing salt field) — no migration is needed for un-salted
    deployments.

    Invariant: must not contain ``@``, ``/``, ``\\``, or NUL. The L2
    adapters use ``@`` as the field separator; ``/`` and ``\\`` are
    filesystem path separators (FS adapter embeds the salt into
    filenames); NUL terminates C strings (C++ connector). Max length
    128 to stay well within ``NAME_MAX`` (255) after the model, rank,
    hash, and extension are added.
    """

    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
        if "@" in self.model_name:
            raise ValueError(
                f"model_name must not contain '@' (got {self.model_name!r})"
            )
        if self.object_group_id < 0:
            raise ValueError(
                f"object_group_id must be >= 0 (got {self.object_group_id})"
            )
        bad = self._SALT_FORBIDDEN_CHARS & set(self.cache_salt)
        if bad:
            raise ValueError(
                f"cache_salt must not contain {bad!r} (got {self.cache_salt!r})"
            )
        if len(self.cache_salt) > self._SALT_MAX_LEN:
            raise ValueError(
                f"cache_salt exceeds max length {self._SALT_MAX_LEN} "
                f"(got {len(self.cache_salt)})"
            )

    def to_encoded_object_key(self) -> "EncodedObjectKey":
        """Return the JSON-safe :class:`EncodedObjectKey` projection."""
        return EncodedObjectKey(
            chunk_hash_hex=self.chunk_hash.hex(),
            model_name=self.model_name,
            kv_rank=self.kv_rank,
            object_group_id=self.object_group_id,
            cache_salt=self.cache_salt,
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
        )


@dataclass(frozen=True)
class ModuleMemoryCapacity:
    """One compartment's configured capacity: an L1 medium or an L2 adapter.

    Keyed on the same ``(tier, backend)`` axis cache events use.

    Attributes:
        tier: ``Tier.L1`` or ``Tier.L2``.
        backend: Medium within the tier (``"dram"``, ``"devdax"``,
            ``"gds"``, or an L2 adapter type such as ``"s3"``).
        capacity_bytes: Configured capacity. ``0`` means undeclared --
            reported as unknown, not as full.
        shared: Set when instances mount this pool, so its capacity must
            not be summed across them.
    """

    tier: "Tier"
    backend: str
    capacity_bytes: int
    shared: bool = False


@dataclass(frozen=True)
class CapacitySnapshot:
    """This server's memory capacities at one point in time.

    Carries no revision: the cache-event subscriber numbers declarations as
    it emits them, on the single event-bus drain thread, so the number and
    the topology it labels cannot come apart.

    Attributes:
        modules: One entry per memory compartment.
    """

    modules: tuple["ModuleMemoryCapacity", ...]


@dataclass(frozen=True)
class KeyEntry:
    """One entry in a :class:`KeyListPage` including the encoded object
    key and its object size."""

    key: EncodedObjectKey
    size_bytes: int


@dataclass(frozen=True)
class KeyListPage:
    """A page of keys returned by ``L2AdapterInterface.list_l2_keys``."""

    entries: tuple[KeyEntry, ...]
    """The keys in the current page."""

    next_page_token: str | None
    """``None`` means this is the last page. Otherwise pass the token
    verbatim to the next call to fetch the next page."""


@dataclass(frozen=True)
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post_init__(self):
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )


GroupKind = Literal["attention", "recurrent", "aux"]
"""Object-group kind label: attention KV, recurrent state pages, or a
connector-private aux group. Derived server-side from
``EngineGroupInfo.extra_object_group_tag``; never sent on the wire."""


@dataclass(frozen=True)
class AttnWindowDesc:
    """Per-object-group cross-chunk attention windows, in LMCache chunks.

    ``num_chunks_in_sw[g]`` is the number of trailing prefix chunks that must
    be present for object group ``g`` to serve a cache hit. ``-1`` means full
    attention (the whole prefix); ``w >= 1`` is a sliding window of ``w``
    chunks.
    """

    num_chunks_in_sw: list[int]

    world_size: int = 1
    """Number of kv_rank shards per chunk (the ``fold_unfold_ranked``
    fan-out): the TP world size for head-sharded models, pipeline stages
    times DCP size for MLA."""

    group_kinds: tuple[GroupKind, ...] = ()
    """Optional per-group kind labels parallel to ``num_chunks_in_sw``.
    Empty when the producer predates kinds (treat every group as
    attention)."""

    _VALID_GROUP_KINDS = frozenset(get_args(GroupKind))

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(
                f"AttnWindowDesc: world_size must be >= 1, got {self.world_size}"
            )
        for w in self.num_chunks_in_sw:
            if w == 0 or w < -1:
                raise ValueError(
                    "AttnWindowDesc: each window must be -1 (full attention) "
                    f"or >= 1 chunk, got {w}"
                )
        if self.group_kinds:
            if len(self.group_kinds) != len(self.num_chunks_in_sw):
                raise ValueError(
                    f"AttnWindowDesc: group_kinds has {len(self.group_kinds)} "
                    f"entries but num_chunks_in_sw has "
                    f"{len(self.num_chunks_in_sw)}"
                )
            bad = set(self.group_kinds) - self._VALID_GROUP_KINDS
            if bad:
                raise ValueError(f"AttnWindowDesc: unknown group kinds {bad!r}")

    @property
    def num_object_groups(self) -> int:
        """Number of object groups this descriptor covers."""
        return len(self.num_chunks_in_sw)

    def is_full_attention(self, object_group_idx: int) -> bool:
        """Whether the object group depends on the entire prefix.

        Args:
            object_group_idx: 0-based object group index.

        Returns:
            True if the group attends to the whole prefix, False if it uses a
            bounded sliding window.
        """
        return self.num_chunks_in_sw[object_group_idx] < 0


DEFAULT_ATTN_WINDOW_DESC = AttnWindowDesc(num_chunks_in_sw=[-1])
"""A single full-attention object group; the default when no per-object-group
windows are supplied."""


class PrefetchLockMode(enum.Enum):
    """Whether a prefetch read-locks the objects it makes resident.

    ``LOCK`` -- the prefetched objects are read-locked until the caller
    releases them.

    ``NO_LOCK`` -- the prefetched objects are left resident and unlocked
    (immediately evictable).
    """

    LOCK = enum.auto()
    NO_LOCK = enum.auto()


@dataclass(frozen=True)
class GroupedObjectKeys:
    """The object keys of one ``(object group, kv rank)`` row of a prefetch.

    Attributes:
        keys: Chunk-ordered object keys in this ``(object group, kv rank)``
            group (one row of the request).
        object_group_id: The object group these keys belong to.
        layout_desc: Memory layout of this object group's objects.
        sliding_window_size: Number of trailing prefix chunks this object group
            needs present to serve a prefix: ``FULL_ATTENTION_WINDOW_CHUNKS``
            (``-1``) for full attention, ``w >= 1`` for a sliding window of
            ``w`` chunks (``1`` for recurrent state).

    Note:
        ``keys[i]`` is the object covering tokens
        ``[i * chunk_size, (i + 1) * chunk_size)`` of the request for this
        object group on this kv rank, so a key's index is its chunk index.
    """

    keys: list[ObjectKey]
    object_group_id: int
    layout_desc: MemoryLayoutDesc
    sliding_window_size: int = FULL_ATTENTION_WINDOW_CHUNKS

    def __post_init__(self) -> None:
        if self.sliding_window_size == 0 or self.sliding_window_size < -1:
            raise ValueError(
                "GroupedObjectKeys: sliding_window_size must be -1 (full attention) "
                f"or >= 1 chunk, got {self.sliding_window_size}"
            )


@dataclass(frozen=True)
class PrefetchTaskSpec:
    """A prefetch request: which objects to make resident in L1, and how.

    Attributes:
        key_groups: One :class:`GroupedObjectKeys` row per ``(object group, kv
            rank)``.
        num_kv_readers: Read locks to take per prefetched object -- one per
            reader that will retrieve it. Ignored under ``NO_LOCK``.
        fetching_policy: See :data:`FetchingPolicy`.
        lock_mode: See :class:`PrefetchLockMode`.

    Note:
        Every key group holds the same number of keys (``group_size``). The
        groups may be listed in any order; the prefetch result is reported
        per group, in the same order as ``key_groups``.
    """

    key_groups: list[GroupedObjectKeys]
    num_kv_readers: int = 1
    fetching_policy: FetchingPolicy = "prefix"
    lock_mode: PrefetchLockMode = PrefetchLockMode.LOCK

    def __post_init__(self) -> None:
        if not self.key_groups:
            raise ValueError("PrefetchTaskSpec: key_groups must not be empty")
        if self.fetching_policy not in _VALID_FETCHING_POLICIES:
            raise ValueError(
                "PrefetchTaskSpec: fetching_policy must be one of "
                f"{sorted(_VALID_FETCHING_POLICIES)}, got {self.fetching_policy!r}"
            )
        if self.num_kv_readers < 1:
            raise ValueError(
                f"PrefetchTaskSpec: num_kv_readers={self.num_kv_readers} "
                "must be >= 1 (total read locks per key)"
            )
        sizes = {len(row.keys) for row in self.key_groups}
        if len(sizes) > 1:
            raise ValueError(
                "PrefetchTaskSpec: every key group must have the same number "
                f"of keys, got {[len(row.keys) for row in self.key_groups]}"
            )

    @property
    def group_size(self) -> int:
        """Number of keys in every key group (the request's chunk count)."""
        return len(self.key_groups[0].keys)

    @property
    def group_layout_descs(self) -> dict[int, MemoryLayoutDesc]:
        """Map each object group id to its memory layout."""
        return {row.object_group_id: row.layout_desc for row in self.key_groups}


@dataclass(frozen=True)
class PrefetchHandle:
    """Opaque handle returned by ``StorageManager.submit_prefetch_task``.

    Carries the bookkeeping needed to later query the prefetch status
    without exposing controller internals.
    """

    prefetch_request_id: int
    """Opaque ID for tracking the request in the prefetch controller; -1
    marks an already-complete empty request."""

    external_request_id: str
    """Request ID from the caller for end-to-end tracing."""

    total_requested_keys: int
    """Total number of keys originally requested."""

    submit_time: float
    """Monotonic timestamp when the prefetch task was submitted."""

    sliding_windows: tuple[int, ...] = ()
    """Sliding-window size of every key group of the request, in group
    order; ``FULL_ATTENTION_WINDOW_CHUNKS`` for a full-attention group."""


def ipc_key_to_object_keys(
    ipc_key: "IPCCacheServerKey",
    chunk_hashes: list[bytes],
    object_group_ids: list[int],
) -> list[list[ObjectKey]]:
    """
    Convert a single IPCCacheServerKey and its chunk hashes to per-object-group
    lists of ObjectKey.

    When the ipc_key's worker_id is None, each chunk hash is exploded into
    multiple ObjectKeys (one per worker in world_size).

    ``cache_salt`` is read directly from ``ipc_key`` so the produced
    ObjectKeys are per-user isolated whenever the sender set a non-empty
    salt. There is intentionally no separate ``cache_salt`` parameter —
    duplicating the source of truth would risk silent isolation bugs
    where a caller passes ``ipc_key`` but forgets the salt.

    Args:
        ipc_key: The IPC key providing model_name, world_size, worker_id,
            and cache_salt.
        chunk_hashes: List of chunk hash bytes, one per chunk.
        object_group_ids: Object group ids to produce keys for.

    Returns:
        list[list[ObjectKey]]: The i-th element is the list of ObjectKeys
        for ``object_group_ids[i]``.
    """
    cache_salt = ipc_key.cache_salt

    # The (chunk_hash, kv_rank) expansion is independent of the object group,
    # so compute it once and reuse it for every group.
    kv_ranks = _lookup_kv_ranks(ipc_key)

    return [
        [
            ObjectKey(
                chunk_hash=chunk_hash,
                model_name=ipc_key.model_name,
                kv_rank=kv_rank,
                object_group_id=object_group_id,
                cache_salt=cache_salt,
            )
            for chunk_hash in chunk_hashes
            for kv_rank in kv_ranks
        ]
        for object_group_id in object_group_ids
    ]


def ipc_key_to_grouped_object_keys(
    ipc_key: "IPCCacheServerKey",
    chunk_hashes: list[bytes],
    object_group_ids: list[int],
    group_layout_descs: dict[int, MemoryLayoutDesc],
    attn_desc: AttnWindowDesc,
) -> list[GroupedObjectKeys]:
    """Expand an IPC key and its chunk hashes into prefetch key rows.

    Produces one :class:`GroupedObjectKeys` row per ``(object group, kv rank)``,
    group-major and rank-minor (all ranks of ``object_group_ids[0]`` first, in
    rank order, then the next group, ...). Every row is chunk-ordered over
    ``chunk_hashes``.

    Args:
        ipc_key: The IPC key providing model_name, world_size, worker_id,
            and cache_salt.
        chunk_hashes: Chunk hash bytes, one per chunk, in token order.
        object_group_ids: Object group ids to produce rows for, in the order
            the rows should appear.
        group_layout_descs: Maps each object group id to its memory layout.
        attn_desc: Registration-wide attention windows; the window of object
            group ``g`` is ``attn_desc.num_chunks_in_sw[g]``.

    Returns:
        ``len(object_group_ids) * num_ranks`` rows, group-major / rank-minor.

    Raises:
        ValueError: If an object group id has no layout in
            ``group_layout_descs`` or no window in ``attn_desc``.

    Note:
        A key without ``worker_id`` fans out to every kv rank of its world
        size; a worker-specific key yields that rank only. ``cache_salt`` is
        taken from ``ipc_key``.
    """
    kv_ranks = _lookup_kv_ranks(ipc_key)
    rows: list[GroupedObjectKeys] = []
    for object_group_id in object_group_ids:
        layout_desc = group_layout_descs.get(object_group_id)
        if layout_desc is None:
            raise ValueError(
                f"ipc_key_to_grouped_object_keys: no layout for object group "
                f"{object_group_id} (have {sorted(group_layout_descs)})"
            )
        if not 0 <= object_group_id < attn_desc.num_object_groups:
            raise ValueError(
                f"ipc_key_to_grouped_object_keys: object group {object_group_id} is "
                f"outside attn_desc's {attn_desc.num_object_groups} groups"
            )
        window = attn_desc.num_chunks_in_sw[object_group_id]
        for kv_rank in kv_ranks:
            rows.append(
                GroupedObjectKeys(
                    keys=[
                        ObjectKey(
                            chunk_hash=chunk_hash,
                            model_name=ipc_key.model_name,
                            kv_rank=kv_rank,
                            object_group_id=object_group_id,
                            cache_salt=ipc_key.cache_salt,
                        )
                        for chunk_hash in chunk_hashes
                    ],
                    object_group_id=object_group_id,
                    layout_desc=layout_desc,
                    sliding_window_size=window,
                )
            )
    return rows
