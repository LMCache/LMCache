# SPDX-License-Identifier: Apache-2.0
"""
Interface for storage controllers

Storage controllers are separate modules/threads that sees the L1 Manager
and can operate on it.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.l1_manager import L1Manager

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager


class StorageControllerInterface(ABC):
    def __init__(
        self,
        storage_manager: "StorageManager",
        l1_manager: L1Manager,
    ):
        self._l1_manager = l1_manager
        self._storage_manager = storage_manager

    def get_storage_manager(self) -> "StorageManager":
        """
        Get the storage manager instance.
        This function will be used by sub classes to access storage manager APIs.

        Returns:
            L1Manager: The storage manager instance.
        """
        return self._storage_manager

    def get_l1_manager(self) -> L1Manager:
        """
        Get the L1 manager instance.
        This function will be used by sub classes to access L1 manager APIs.

        Returns:
            L1Manager: The L1 manager instance.
        """
        return self._l1_manager

    @abstractmethod
    def start(self):
        """
        Start the storage controller.
        This function should be implemented by subclasses to start
        any necessary threads or processes.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the storage controller.
        This function should be implemented by subclasses to stop
        any running threads or processes.
        """
        pass
