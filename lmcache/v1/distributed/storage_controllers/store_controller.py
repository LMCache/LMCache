# SPDX-License-Identifier: Apache-2.0
"""
Store Controller: asynchronously copies data from L1 to L2 after writes complete.

The controller runs a background thread with an event-driven loop that:
1. Listens for L1 write-completion events via StoreListener.
2. Submits store tasks to L2 adapters based on StorePolicy decisions.
3. Monitors L2 task completion via event fds.
4. Releases L1 read locks and optionally deletes keys from L1.
"""

# Standard
from dataclasses import dataclass
import os
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StorePolicy,
)

logger = init_logger(__name__)

# Poll timeout in milliseconds for the store loop
STORE_LOOP_POLL_TIMEOUT_MS = 500


# Helper classes (module-level, before main class)


class StoreListener(L1ManagerListener):
    """
    Listener that receives L1 write-completion callbacks and enqueues
    keys for the StoreController's background loop.

    The ``on_keys_write_finished`` callback is invoked inside L1Manager's
    lock, so it must be non-blocking. It appends keys to an internal list
    and signals an eventfd to wake up the controller's select.poll().
    """

    def __init__(self) -> None:
        self._pending_keys: list[ObjectKey] = []
        self._lock = threading.Lock()
        self._event_fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    def get_event_fd(self) -> int:
        """
        Return the eventfd that is signaled when new keys are available.

        Returns:
            int: The eventfd file descriptor.
        """
        return self._event_fd

    def pop_pending_keys(self) -> list[ObjectKey]:
        """
        Pop all pending keys from the queue.

        This is non-blocking and should be called by the StoreController's
        main loop after select.poll() indicates the eventfd is ready.

        Returns:
            list[ObjectKey]: All keys enqueued since the last pop.
        """
        with self._lock:
            keys = self._pending_keys
            self._pending_keys = []
        return keys

    def pending_count(self) -> int:
        """Return the number of pending keys waiting to be processed."""
        with self._lock:
            return len(self._pending_keys)

    # L1ManagerListener implementation

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """
        Enqueue keys and signal the eventfd.

        Called inside L1Manager's lock. Must be fast and must not
        call any L1Manager methods (would deadlock).

        Args:
            keys (list[ObjectKey]): Keys that finished writing.
        """
        with self._lock:
            self._pending_keys.extend(keys)
        os.eventfd_write(self._event_fd, 1)

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def close(self) -> None:
        """Close the eventfd."""
        os.close(self._event_fd)


@dataclass
class InFlightStoreTask:
    """
    Tracks a single submitted L2 store task so the controller can
    release L1 read locks and perform cleanup when it completes.
    """

    adapter_index: int
    """Which L2 adapter this task was submitted to."""

    keys: list[ObjectKey]
    """All keys that were submitted in this store task."""

    read_locked_keys: list[ObjectKey]
    """The subset of keys for which reserve_read succeeded
    (i.e., keys holding an L1 read lock that must be released)."""


# Main class


class StoreController(StorageControllerInterface):
    """
    Asynchronously stores L1 data to L2 adapters after write completion.

    The controller:
    1. Registers a StoreListener with L1Manager to receive
       on_keys_write_finished callbacks.
    2. Runs a background thread with an event-driven loop using
       select.poll() on the listener eventfd and all L2 adapter
       store eventfds.
    3. On new keys: consults StorePolicy to decide targets,
       calls reserve_read to get MemoryObjs, and submits store
       tasks to L2 adapters.
    4. On L2 task completion: releases read locks, optionally
       deletes keys from L1 per policy.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The store policy for deciding targets and deletions.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: StorePolicy,
    ) -> None:
        super().__init__(l1_manager)
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy

        self._listener = StoreListener()
        self.get_l1_manager().register_listener(self._listener)

        # (adapter_index, task_id) -> InFlightStoreTask
        # Composite key is needed because task IDs are only unique
        # within a single adapter, not across adapters.
        self._in_flight_tasks: dict[tuple[int, L2TaskId], InFlightStoreTask] = {}

        # Shadow counter for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0

        # Map eventfd -> adapter index for quick lookup in poll results
        self._efd_to_adapter_index: dict[int, int] = {}
        for i, adapter in enumerate(self._l2_adapters):
            efd = adapter.get_store_event_fd()
            self._efd_to_adapter_index[efd] = i

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._store_loop,
            daemon=True,
        )

    def start(self) -> None:
        """Start the background store loop thread."""
        logger.info("Starting StoreController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop, wait for the thread to join.

        Releases all in-flight read locks on shutdown so that
        L1 objects are not permanently locked.
        """
        self._stop_flag.set()
        # Wake up the poll loop so it can exit promptly
        os.eventfd_write(self._listener.get_event_fd(), 1)
        self._thread.join()
        self._cleanup_in_flight_tasks()
        self._listener.close()

    def report_status(self) -> dict:
        """Return a status dict for the store controller."""
        is_healthy = self._thread.is_alive()
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "pending_keys_count": self._listener.pending_count(),
            "in_flight_task_count": self._status_in_flight_count,
            "num_l2_adapters": len(self._l2_adapters),
        }

    # Private methods

    def _store_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The StoreListener's eventfd (new keys from L1 writes).
        - Each L2 adapter's store eventfd (completed store tasks).

        Exits when the stop flag is set.
        """
        poller = select.poll()

        listener_efd = self._listener.get_event_fd()
        poller.register(listener_efd, select.POLLIN)

        for efd in self._efd_to_adapter_index:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            ready = poller.poll(STORE_LOOP_POLL_TIMEOUT_MS)

            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                # Consume the eventfd value
                try:
                    os.eventfd_read(fd)
                except (OSError, BlockingIOError):
                    pass

                if fd == listener_efd:
                    keys = self._listener.pop_pending_keys()
                    if keys:
                        self._process_new_keys(keys)
                elif fd in self._efd_to_adapter_index:
                    adapter_index = self._efd_to_adapter_index[fd]
                    self._process_completed_tasks(adapter_index)

    def _process_new_keys(self, keys: list[ObjectKey]) -> None:
        """
        Process a batch of newly written keys.

        1. Ask the policy which adapters each key should go to.
        2. For each adapter target, reserve read access on L1 to get
           MemoryObj references (skip keys that fail — best-effort).
        3. Submit store tasks to L2 adapters.
        4. Track in-flight tasks for later cleanup.

        Args:
            keys (list[ObjectKey]): Keys that finished writing to L1.
        """
        plan = self._policy.select_store_targets(keys, self._adapter_descriptors)

        l1_mgr = self.get_l1_manager()

        for adapter_index, target_keys in plan.items():
            if not target_keys:
                continue

            if adapter_index >= len(self._l2_adapters):
                logger.error(
                    "StorePolicy returned invalid adapter index %d "
                    "(only %d adapters available). Skipping.",
                    adapter_index,
                    len(self._l2_adapters),
                )
                continue

            # Reserve read to get MemoryObj references and hold read locks
            read_results = l1_mgr.reserve_read(target_keys)

            successful_keys = []
            successful_objs = []
            for key in target_keys:
                result = read_results.get(key)
                if result is None:
                    continue
                err, obj = result
                if err != L1Error.SUCCESS or obj is None:
                    logger.debug(
                        "Skipping key %s for L2 store (adapter %d): %s",
                        key,
                        adapter_index,
                        err,
                    )
                    continue
                successful_keys.append(key)
                successful_objs.append(obj)

            if not successful_keys:
                continue

            adapter = self._l2_adapters[adapter_index]
            task_id = adapter.submit_store_task(successful_keys, successful_objs)

            self._in_flight_tasks[(adapter_index, task_id)] = InFlightStoreTask(
                adapter_index=adapter_index,
                keys=successful_keys,
                read_locked_keys=list(successful_keys),
            )
            self._status_in_flight_count += 1

            logger.debug(
                "Submitted store task %d to adapter %d with %d keys.",
                task_id,
                adapter_index,
                len(successful_keys),
            )

    def _process_completed_tasks(self, adapter_index: int) -> None:
        """
        Process completed store tasks for a given adapter.

        1. Pop all completed tasks from the adapter.
        2. Release L1 read locks for each task.
        3. If the task succeeded, ask the policy which keys to
           delete from L1 and delete them.
        4. Remove the task from in-flight tracking.

        Args:
            adapter_index (int): Index of the adapter whose eventfd
                was signaled.
        """
        adapter = self._l2_adapters[adapter_index]
        completed = adapter.pop_completed_store_tasks()

        l1_mgr = self.get_l1_manager()

        for task_id, success in completed.items():
            composite_key = (adapter_index, task_id)
            task = self._in_flight_tasks.pop(composite_key, None)
            if task is not None:
                self._status_in_flight_count -= 1
            if task is None:
                logger.warning(
                    "Completed store task %d (adapter %d) not found in tracking.",
                    task_id,
                    adapter_index,
                )
                continue

            # Always release read locks
            l1_mgr.finish_read(task.read_locked_keys)

            if success:
                logger.debug(
                    "L2 store task %d completed: adapter %d, %d keys.",
                    task_id,
                    adapter_index,
                    len(task.keys),
                )
                delete_keys = self._policy.select_l1_deletions(task.keys)
                if delete_keys:
                    l1_mgr.delete(delete_keys)
            else:
                logger.warning(
                    "Store task %d to adapter %d failed for keys: %s",
                    task_id,
                    adapter_index,
                    task.keys,
                )

    def _cleanup_in_flight_tasks(self) -> None:
        """
        Release all held read locks for any in-flight tasks that
        haven't completed. Called during stop().
        """
        l1_mgr = self.get_l1_manager()
        for (adapter_index, task_id), task in self._in_flight_tasks.items():
            logger.warning(
                "Cleaning up in-flight store task %d (adapter %d, %d keys).",
                task_id,
                adapter_index,
                len(task.read_locked_keys),
            )
            l1_mgr.finish_read(task.read_locked_keys)
        self._in_flight_tasks.clear()
