# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import enum
import hashlib
import json

# First Party
from lmcache.v1.distributed.api import L1BackendType, ObjectKey

CXL_METADATA_KEY = "lmcache.cxl.arena"


@dataclass(frozen=True)
class L1MemoryDesc:
    """
    Describes the L1 memory buffer registered with an external backend (e.g. Nixl).
    """

    ptr: int
    size: int
    align_bytes: int


@dataclass(frozen=True)
class CxlArenaDescriptor:
    """Identify one owner's slab and its current incarnation.

    Args:
        pool_id: Deployment-assigned identity of the shared pool.
        offset: Byte offset of the slab header in the pool device.
        size: Usable payload bytes following the header.
        alignment: Header length and allocation alignment in bytes.
        session_id: Fresh identity each time the owner opens the slab.

    Raises:
        ValueError: If the identity or aligned range is invalid.

    The first ``alignment`` bytes contain an identity fingerprint. Readers
    verify it through their own mapping before using peer payload pointers.
    """

    pool_id: str
    offset: int
    size: int
    alignment: int
    session_id: str

    def __post_init__(self) -> None:
        if not self.pool_id or not self.session_id:
            raise ValueError("CXL pool and session identities must be non-empty")
        if self.alignment < 4096 or self.alignment & (self.alignment - 1):
            raise ValueError("CXL alignment must be a power of two >= 4096")
        if (
            self.offset < 0
            or self.offset % self.alignment
            or self.size <= 0
            or self.size % self.alignment
        ):
            raise ValueError("CXL slab offset and size must be aligned")

    def to_json(self) -> str:
        """Return a canonical JSON string for coordinator registration."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "CxlArenaDescriptor":
        """Decode registration metadata; raise ValueError for invalid fields.

        Args:
            value: JSON produced by :meth:`to_json`.

        Returns:
            The validated slab descriptor.
        """
        try:
            fields = json.loads(value)
            if not isinstance(fields, dict):
                raise ValueError("CXL arena metadata must be an object")
            for name in ("offset", "size", "alignment"):
                if type(fields.get(name)) is not int:
                    raise ValueError(f"CXL {name} must be an integer")
            for name in ("pool_id", "session_id"):
                if not isinstance(fields.get(name), str):
                    raise ValueError(f"CXL {name} must be a string")
            return cls(**fields)
        except TypeError as exc:
            raise ValueError("Invalid CXL arena metadata") from exc

    def fingerprint(self) -> bytes:
        """Return the 32-byte identity expected in the mapped slab header."""
        return hashlib.sha256(self.to_json().encode()).digest()


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
