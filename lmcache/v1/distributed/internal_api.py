# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum

# First Party
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    L1BackendType,
    MemoryLayoutDesc,
    ObjectKey,
)


class TrimPolicy(enum.Enum):
    """How the prefetch controller picks the retained subset of found keys.

    .. deprecated::
        Controller-internal; superseded by ``PrefetchTaskSpec.fetching_policy``.
        Removed with prefetch controller v2.

    PREFIX retains the longest contiguous run from index 0; SEGMENTED_PREFIX
    keeps the keys that loaded when an L2 hit failed to load into L1 mid-prefix
    (gaps and all); SPARSE retains every found key for an intentional scatter.
    """

    PREFIX = enum.auto()
    SEGMENTED_PREFIX = enum.auto()
    SPARSE = enum.auto()


class PrefetchMode(enum.Enum):
    """The intent of a prefetch request, as seen by the prefetch controller.

    .. deprecated::
        Controller-internal; superseded by ``PrefetchTaskSpec.lock_mode``.
        Removed with prefetch controller v2.

    ``LOOKUP`` -- prefetch for an imminent reader: loaded keys are read-locked
    for the requesting workers, and whether they persist or are dropped after
    use follows the configured prefetch policy.

    ``WARM`` -- speculative pre-warm with no imminent reader: loaded keys are
    retained and left unlocked (immediately resident and evictable), so a later
    lookup can hit them.
    """

    LOOKUP = enum.auto()
    WARM = enum.auto()


@dataclass(frozen=True)
class PrefetchRequestSpec:
    """Immutable inputs of a single L2 prefetch request (controller payload).

    .. deprecated::
        Controller-internal; superseded by ``PrefetchTaskSpec``. Removed with
        prefetch controller v2.

    Bundles the caller-supplied arguments that travel together into the
    prefetch controller's submission queue. See
    ``PrefetchController._start_lookup_phase`` for per-field semantics.

    Attributes:
        keys: Object keys to prefetch; order defines the prefix.
        group_layout_descs: Maps object_group_id to that group's memory
            layout for L1 write-buffer allocation; must cover every object
            group that appears in ``keys``.
        num_kv_readers: Total read locks to take per key -- one per
            reader that will retrieve the object.
        policy: Retained-subset policy (see :class:`TrimPolicy`).
        attn_desc: Cross-chunk attention windows for the groups ``keys``
            covers; a caller prefetching a subset of the registration's
            groups must narrow it to that subset (it drives the fold
            stride).
        mode: Prefetch intent (see :class:`PrefetchMode`).
    """

    keys: list[ObjectKey]
    group_layout_descs: dict[int, MemoryLayoutDesc]
    num_kv_readers: int = 1
    policy: TrimPolicy = TrimPolicy.PREFIX
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    mode: PrefetchMode = PrefetchMode.LOOKUP

    def __post_init__(self) -> None:
        if self.num_kv_readers < 1:
            raise ValueError(
                f"PrefetchRequestSpec: num_kv_readers={self.num_kv_readers} "
                "must be >= 1 (total read locks per key)"
            )
        # Every object group that appears in ``keys`` needs a layout; the ids
        # need not be contiguous or start at 0.
        missing = {k.object_group_id for k in self.keys} - set(self.group_layout_descs)
        if missing:
            raise ValueError(
                "PrefetchRequestSpec: group_layout_descs must cover the object "
                f"groups of every key; missing {sorted(missing)}, got "
                f"{sorted(self.group_layout_descs)}"
            )


@dataclass(frozen=True)
class L1MemoryDesc:
    """Describe the contiguous L1 memory arena exposed to external backends.

    Attributes:
        ptr: Base address of the L1 arena.
        size: Final size of the L1 arena in bytes.
        align_bytes: Allocation alignment within the arena.
        stable_registration_size: Stable size of the L1 arena. For a lazy
            allocator, this is a snapshot of the currently pinned prefix and
            does not change as the allocator grows. ``None`` means that no
            stable size is exposed.
    """

    ptr: int
    size: int
    align_bytes: int
    stable_registration_size: int | None = None


@dataclass(frozen=True)
class L1ObjectMeta:
    """Per-object metadata published alongside keys in L1 cache events.

    Attributes:
        size_bytes: Logical byte size of the object.
        backend: The storage medium backing the object.
    """

    size_bytes: int
    backend: L1BackendType


class EventListener(ABC):  # noqa: B024
    pass


# For L1 manager event notifications
class L1ManagerListener(EventListener):
    """
    Listener for L1 manager events
    """

    @abstractmethod
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that new keys have been reserved for read on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully read
        """
        pass

    @abstractmethod
    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been reserved for write on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully written
        """
        pass

    @abstractmethod
    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing
        and reserved for read on L1.

        This will only be trigger by the prefetch operation now.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully
                finished for writing and reserved for read
        """
        # NOTE (ApostaC): may consider renaming this to `on_l1_keys_finish_prefetch`
        # for better clarity
        pass

    @abstractmethod
    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L1.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        pass

    @abstractmethod
    def on_l1_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        pass


class L2AdapterListener(EventListener):
    """Listener for L2 adapter events, analogous to L1ManagerListener."""

    @abstractmethod
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        """
        Notify the listener that keys have been successfully stored in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been stored.
            sizes (list[int]): The byte size of each stored key.
        """
        pass

    @abstractmethod
    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed (lookup hit) in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed.
        """
        pass

    @abstractmethod
    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L2.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted.
        """
        pass


# For Eviction
class EvictionDestination(enum.Enum):
    """
    The destination of evicted objects
    """

    DISCARD = enum.auto()
    """Discard the evicted objects"""

    L2_CACHE = enum.auto()
    """Evict to L2 storage"""


@dataclass(frozen=True)
class EvictionAction:
    """
    An action to be taken for eviction
    """

    destination: EvictionDestination
    """The destination of the evicted object"""

    keys: list[ObjectKey] = field(default_factory=list)
    """The key of the object to be evicted"""


class L2StoreResult(int):
    """Immutable result of a completed L2 store task.

    Encodes both the success flag and bytes transferred in the int
    value: ``>= 0`` means success (value = bytes transferred);
    ``-1`` means failure.

    Args:
        success: Whether the store task succeeded.
        bytes_transferred: Bytes actually written to L2. Must be >= 0.

    Raises:
        ValueError: If ``bytes_transferred`` is negative.
    """

    def __new__(cls, success: bool, bytes_transferred: int) -> "L2StoreResult":
        if bytes_transferred < 0:
            raise ValueError(f"bytes_transferred must be >= 0, got {bytes_transferred}")
        return super().__new__(cls, bytes_transferred if success else -1)

    def is_successful(self) -> bool:
        """Return ``True`` when the store task succeeded."""
        return int(self) >= 0

    def bytes_transferred(self) -> int:
        """Return the number of bytes actually written, or 0 on failure."""
        value = int(self)
        return value if value >= 0 else 0


@dataclass(frozen=True)
class QuotaEntry:
    """Snapshot of a single quota registration."""

    cache_salt: str
    limit_bytes: int
