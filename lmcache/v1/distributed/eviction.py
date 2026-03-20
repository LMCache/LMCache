# SPDX-License-Identifier: Apache-2.0
"""
Eviction module to determine the what to evict from L1 cache
"""

# Standard
from abc import abstractmethod

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
    L1ManagerListener,
)


class EvictionPolicy(L1ManagerListener):
    """
    Base class for eviction policies

    It implements the L1ManagerListener by calling the abstract methods that
    are to be implemented by subclasses.
    """

    @abstractmethod
    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to register
        """
        pass

    @abstractmethod
    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that new keys have been created.

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        pass

    @abstractmethod
    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been accessed.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        pass

    @abstractmethod
    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been deleted.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        pass

    @abstractmethod
    def get_eviction_actions(self, expected_ratio: float) -> list[EvictionAction]:
        """
        Get the eviction actions to evict objects from L1 cache.

        Args:
            expected_ratio (float): A hint indicating approximately what fraction
                of tracked keys should be evicted. Value should be in range [0.0, 1.0].
                For example, 0.1 means roughly 10% of keys should be evicted.
                This is a hint and the policy may return more or fewer keys.

        Returns:
            list[EvictionAction]: The eviction actions to perform. Each
                action contains the keys and one eviction destination.

        Notes:
            The eviction action may not be successfully executed, or it
            may be executed asynchronously. Therefore, the eviction policy
            should not assume that the objects are evicted immediately, but
            it should use `on_keys_deleted` to know when the objects are actually
            deleted.
        """
        pass

    # L1ManagerListener implementations
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        # No-op
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        self.on_keys_touched(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        # No-op
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        # TODO (ApostaC): we don't differentiate between the created
        # keys and updated keys here. Probably need to fix that by introducing
        # a new callback in L1ManagerListener or adding `mode` argument into
        # on_keys_reserved_write.
        self.on_keys_created(keys)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        self.on_keys_removed(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        self.on_keys_created(keys)
