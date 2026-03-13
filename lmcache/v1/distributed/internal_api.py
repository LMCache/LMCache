# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum

# First Party
from lmcache.v1.distributed.api import ObjectKey


@dataclass(frozen=True)
class L1MemoryDesc:
    """
    Describes the L1 memory buffer registered with an external backend (e.g. Nixl).
    """

    ptr: int
    size: int
    align_bytes: int


class EventListener(ABC):  # noqa: B024
    pass


class StorageManagerListener(EventListener):
    @abstractmethod
    def on_sm_read_prefetched(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        """
        Notify the listener that keys have been reserved for read.

        Args:
            succeeded_keys (list[ObjectKey]): The keys that have been successfully
                reserved
            failed_keys (list[ObjectKey]): The keys that failed to be reserved
        """
        pass

    @abstractmethod
    def on_sm_read_prefetched_finished(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        """
        Notify the listener that read locks have been released.

        Args:
            succeeded_keys (list[ObjectKey]): The keys whose read locks were
                successfully released
            failed_keys (list[ObjectKey]): The keys that failed to release their read
                locks
        """
        pass

    @abstractmethod
    def on_sm_reserved_write(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        """
        Notify the listener that keys have been reserved for write.

        Args:
            succeeded_keys (list[ObjectKey]): The keys that have been successfully
                reserved
            failed_keys (list[ObjectKey]): The keys that failed to be reserved
        """
        pass

    @abstractmethod
    def on_sm_write_finished(
        self,
        succeeded_keys: list[ObjectKey],
        failed_keys: list[ObjectKey],
    ):
        """
        Notify the listener that keys have been finished for writing.

        Args:
            succeeded_keys (list[ObjectKey]): The keys that have been successfully
                written
            failed_keys (list[ObjectKey]): The keys that failed to finish writing
        """
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


class L2ManagerListener(EventListener):
    # Just a placeholder here. Waiting for L2 manager to be finalized.
    @abstractmethod
    def on_l2_lookup_and_lock(self):
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
