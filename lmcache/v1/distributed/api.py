# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass
import enum

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)


class TrimPolicy(enum.Enum):
    """How to pick the retained subset of found keys for a prefetch.

    PREFIX retains the longest contiguous run from index 0; SEGMENTED_PREFIX
    keeps the keys that loaded when an L2 hit failed to load into L1 mid-prefix
    (gaps and all); SPARSE retains every found key for an intentional scatter.
    """

    PREFIX = enum.auto()
    SEGMENTED_PREFIX = enum.auto()
    SPARSE = enum.auto()


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


@dataclass(frozen=True)
class PrefetchHandle:
    """Opaque handle returned by ``StorageManager.submit_prefetch_task``.

    Carries the bookkeeping needed to later query lookup / prefetch status
    without exposing controller internals.
    """

    prefetch_request_id: int
    """Opaque ID for tracking L2 prefetch in the controller.
    -1 if no L2 request was submitted."""

    external_request_id: str
    """Request ID from the caller for end-to-end tracing."""

    l1_found_indices: tuple[int, ...]
    """Original-key indices found (read-locked) in L1 at submission time."""

    total_requested_keys: int
    """Total number of keys originally requested (the result-bitmap size)."""

    submit_time: float
    """Monotonic timestamp when the prefetch task was submitted."""

    l2_orig_indices: tuple[int, ...] = ()
    """Original-key index of each key submitted to L2; maps the controller's
    local result bitmap back to original positions."""


def ipc_key_to_object_keys(
    ipc_key: IPCCacheServerKey,
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
    if ipc_key.worker_id is None:
        # For look up request, we want to expand to all workers
        # TODO (ApostaC): include local world size/rank info
        # in the future once it's in IPCCacheServerKey
        kv_ranks = [
            ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=worker_id,
            )
            for worker_id in range(ipc_key.world_size)
        ]
    else:
        kv_ranks = [
            ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=ipc_key.worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=ipc_key.worker_id,
            )
        ]

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
