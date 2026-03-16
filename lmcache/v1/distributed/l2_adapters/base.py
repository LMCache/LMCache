# SPDX-License-Identifier: Apache-2.0
"""
Interface for L2 adapters
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.native_storage_ops import Bitmap

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryObj

L2TaskId = int


class L2AdapterInterface(ABC):
    """
    The abstracted interface for L2 I/O adapters.

    The L2 I/O adapter mainly provides 3 main functionalities with non-blocking
    primitives:
    1. Store: store a batch of memory objects associated with a batch of keys.
    2. Lookup and lock: look up and lock a batch of objects by the given keys.
       will also try to 'lock' the objects to prevent being evicted before
       loading them to L1.
    3. Load: load a batch of objects by the given keys. The load operation is
       not guaranteed to succeed, and the caller should check the return value.
       In most of cases, it should be likely to succeed if the objects are locked.

    Note that the store and the load operation are pre-provided with the data buffer
    (i.e., memory objects), which is managed by the caller (L2 controller). The L2
    adapter is not supposed to manage the lifecycle of the memory objects.

    The non-blocking interface is designed as follows:
    1. Submit task
    2. Query the completed tasks (either pop all the completed tasks or query
       a specific task by its id)
    3. Use event fd to signal the completion of the tasks. The event fd will be
       handled by the caller (L2 controller). Note that the event fd will be
       closed by the `close()` function.


    Error handling:
    1. For store operation, we only provide a coarse-grained error handling, which
       means that we only report the error at the task level. If a store task fails,
       we will report the failure of the whole task, instead of reporting the failure
       of each key-object pair in the task. The caller can choose to retry the failed
       task or not.
    2. For both lookup and load operations, we will return a bitmap indicating the
       success or failure of each key-object pair in the task.

    Thread-safe:
    The L2 adapter is designed to be called by a 2 controller threads (store controller
    and prefetch controller), therefore, it needs to be thread-safe.
    """

    #####################
    # Event Fd Interface
    #####################

    # IMPORTANT: Each of the three event fd methods below MUST return a
    # distinct file descriptor.  The store controller and prefetch controller
    # build fd-to-adapter lookup maps; if any two methods return the same fd
    # (within one adapter or across adapters), poll-based dispatch will
    # silently misroute events.

    @abstractmethod
    def get_store_event_fd(self) -> int:
        """
        Get the event fd for store operation, which will be signaled on the
        completion of the store tasks.

        Returns:
            int: the event fd for store operation.

        Note:
            Must be distinct from the lookup and load event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    @abstractmethod
    def get_lookup_and_lock_event_fd(self) -> int:
        """
        Get the event fd for lookup and lock operation, which will be signaled
        on the completion of the lookup and lock tasks.

        Returns:
            int: the event fd for lookup and lock operation.

        Note:
            Must be distinct from the store and load event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    @abstractmethod
    def get_load_event_fd(self) -> int:
        """
        Get the event fd for load operation, which will be signaled on the completion
        of the load tasks.

        Returns:
            int: the event fd for load operation.

        Note:
            Must be distinct from the store and lookup event fds of this
            adapter, and from the event fds of all other adapters.
        """
        pass

    #####################
    # Store Interface
    #####################

    @abstractmethod
    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a store task to store a batch of memory objects associated with
        a batch of keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be stored.
            objects (list[MemoryObj]): the list of memory objects to be stored.
                The length of the objects list should be the same as the length of
                the keys list.

        Returns:
            L2TaskId: the task id of the submitted store task.
        """
        pass

    @abstractmethod
    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """
        Pop all the completed store tasks with a flag indicating
        whether the task is successful or not.

        Returns:
            dict[L2TaskId, bool]: a dictionary mapping the task id to a boolean flag
            indicating whether the task is successful or not. True means
            successful, and False means failed.
        """
        pass

    #####################
    # Lookup and Lock Interface
    #####################

    @abstractmethod
    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        """
        Submit a lookup and lock task to look up and lock a batch of objects
        by the given keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be looked up and locked.

        Returns:
            L2TaskId: the task id of the submitted lookup and lock task.
        """
        pass

    @abstractmethod
    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """
        Non-blockingly query the result of a lookup and lock task by its task id.
        The result is a bitmap indicating the success or failure of each key-object
        pair in the task.

        For a single task id, this function will ONLY return a non-None value ONCE.
        (Which means this function is not idempotent)

        Args:
            task_id (L2TaskId): the task id of the lookup and lock task.

        Returns:
            Optional[Bitmap]: a bitmap indicating the success or failure of each
            key-object pair in the task. 1 means successful, and 0 means failed.
            None is returned when the lookup and lock task is not completed.
        """
        pass

    @abstractmethod
    def submit_unlock(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Submit an unlock task to unlock a batch of objects by the given keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be unlocked.

        Note:
            This function does not return any task id, meaning that the caller
            assumes the unlock operation will be eventually successful, and will
            NEVER retry.
            Therefore, the implementation MUST make sure that the unlock operation
            is successful (i.e., have error handling and retry mechanism if needed).
        """
        pass

    #####################
    # Load Interface
    ######################

    @abstractmethod
    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a load task to load a batch of objects by the given keys. The load
        operation is not guaranteed to succeed, and the caller should check the
        return value.

        Args:
            keys (list[ObjectKey]): the list of keys to be loaded.
            objects (list[MemoryObj]): the list of memory objects as the load buffer.
                The L2 adapter will write the loaded data to the memory buffer provided
                by the caller. The caller is responsible for managing the lifecycle of
                the memory objects, and should make sure that the memory buffer is valid
                until the load task is completed.
                The length of the objects list should be the same as the length of the
                keys list.

        Returns:
            L2TaskId: the task id of the submitted load task.
        """
        pass

    @abstractmethod
    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """
        Non-blockingly query the result of a load task by its task id. The result
        is a bitmap indicating the success or failure of each key-object pair in
        the task.

        For a single task id, this function will ONLY return a non-None value ONCE.
        (Which means this function is not idempotent)

        Args:
            task_id (L2TaskId): the task id of the load task.

        Returns:
            Optional[Bitmap]: a bitmap indicating the success or failure of each
            key-object pair in the task. 1 means successful, and 0 means failed.
            None is returned when the load task is not completed.
        """
        pass

    #####################
    # Cleanup Interface
    #####################

    @abstractmethod
    def close(self) -> None:
        """
        Close the L2 adapter and release all the resources. After calling this function,
        the L2 adapter should not be used anymore.
        """
        pass

    #####################
    # Status Interface
    #####################

    def report_status(self) -> dict:
        """
        Return a status dict for this adapter.

        Must include at least ``is_healthy: bool``.
        Subclasses should override this with adapter-specific metrics.
        """
        return {
            "is_healthy": True,
            "extra_warning": "report_status is not implemented and runs default impl",
        }
