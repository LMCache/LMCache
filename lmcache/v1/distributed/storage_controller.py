# SPDX-License-Identifier: Apache-2.0
"""
Interface for storage controllers

Storage controllers are separate modules/threads that sees the L1 Manager
and can operate on it.
"""

# Standard
from abc import ABC, abstractmethod


class StorageControllerInterface(ABC):
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

    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Return whether this controller can still do its work.

        Implementations must answer without taking any lock the controller's
        own loop holds while calling into storage, and without building a
        status snapshot. Readiness probes call this on a fixed interval, so a
        blocking answer would stall the caller for as long as the loop's
        current pass takes.

        Returns:
            True while the controller is running.
        """
        pass

    @abstractmethod
    def report_status(self) -> dict:
        """
        Return a status dict for this controller.

        Must include at least ``is_healthy: bool``, matching
        :meth:`is_healthy`. Unlike :meth:`is_healthy`, this may take locks and
        gather detailed counters, so it is for inspection rather than probes.
        """
        pass
