# SPDX-License-Identifier: Apache-2.0
"""
Store Controller: asynchronously copies data from L1 to L2 after writes complete.

The controller runs a background thread with an event-driven loop that:
1. Listens for L1 write-completion events via StoreListener.
2. If serde enabled: submits serialize tasks and waits for serde event fd.
3. Submits store tasks to L2 adapters based on StorePolicy decisions.
4. Monitors L2 task completion via event fds.
5. Releases L1 read locks and optionally deletes keys from L1.
"""

# Standard
from dataclasses import dataclass, field
import os
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.serde import (
    SerdeProcessor,
    SerdeTaskId,
    make_temp_key,
    serialized_layout_desc,
)
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StorePolicy,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

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

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        # No op here because we don't want to trigger store when the
        # objects are prefetched to L1.
        pass

    def close(self) -> None:
        """Close the eventfd."""
        os.close(self._event_fd)


@dataclass
class InFlightSerializeTask:
    """Tracks an in-flight serialization task submitted to a SerdeProcessor."""

    adapter_index: int
    serde_task_id: SerdeTaskId
    keys: list[ObjectKey]
    """Original keys (the logical keys for L2)."""

    read_locked_keys: list[ObjectKey]
    """Original keys holding L1 read locks."""

    temp_keys: list[ObjectKey]
    """Temp buffer keys (write-locked, being serialized into)."""

    temp_objs: list[MemoryObj]
    """Temp buffer MemoryObjs (needed for submit_store_task after serialize)."""


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

    temp_keys: list[ObjectKey] = field(default_factory=list)
    """Temporary serialization buffer keys in L1 (write-locked during L2 I/O).
    Empty list when serde is disabled."""


# Main class


class StoreController(StorageControllerInterface):
    """
    Asynchronously stores L1 data to L2 adapters after write completion.

    The controller:
    1. Registers a StoreListener with L1Manager to receive
       on_keys_write_finished callbacks.
    2. Runs a background thread with an event-driven loop using
       select.poll() on the listener eventfd, serde serialize eventfds,
       and all L2 adapter store eventfds.
    3. On new keys: consults StorePolicy to decide targets,
       calls reserve_read to get MemoryObjs, and either submits
       store tasks directly (no serde) or submits serialize tasks
       to SerdeProcessors first.
    4. On serde completion: releases original read locks, submits
       store tasks to L2 with serialized temp buffers.
    5. On L2 task completion: releases read/temp locks, optionally
       deletes keys from L1 per policy.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The store policy for deciding targets and deletions.
        serde_processors: Optional list of SerdeProcessors, one per adapter.
            None or [None]*N means serde disabled (existing behavior).
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: StorePolicy,
        serde_processors: list[SerdeProcessor | None] | None = None,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy

        if serde_processors is None:
            self._serde_processors: list[SerdeProcessor | None] = [None] * len(
                l2_adapters
            )
        else:
            if len(serde_processors) != len(l2_adapters):
                raise ValueError(
                    f"serde_processors length ({len(serde_processors)}) must "
                    f"match l2_adapters length ({len(l2_adapters)})"
                )
            self._serde_processors = serde_processors

        self._listener = StoreListener()
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()

        # (adapter_index, task_id) -> InFlightStoreTask
        # Composite key is needed because task IDs are only unique
        # within a single adapter, not across adapters.
        self._in_flight_tasks: dict[tuple[int, L2TaskId], InFlightStoreTask] = {}

        # (adapter_index, serde_task_id) -> InFlightSerializeTask
        # Composite key needed because each SerdeProcessor allocates task IDs
        # independently — bare task IDs would collide across adapters.
        self._in_flight_serialize_tasks: dict[
            tuple[int, SerdeTaskId], InFlightSerializeTask
        ] = {}

        # Shadow counter for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0

        # Map eventfd -> adapter index for quick lookup in poll results
        self._store_efd_to_adapter: dict[int, int] = {}
        for i, adapter in enumerate(self._l2_adapters):
            efd = adapter.get_store_event_fd()
            self._store_efd_to_adapter[efd] = i

        # Map serde serialize eventfd -> adapter index
        self._serialize_efd_to_adapter: dict[int, int] = {}
        for i, serde in enumerate(self._serde_processors):
            if serde is not None:
                self._serialize_efd_to_adapter[serde.get_serialize_event_fd()] = i

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
        - Each SerdeProcessor's serialize eventfd (completed serializations).
        - Each L2 adapter's store eventfd (completed store tasks).

        Exits when the stop flag is set.
        """
        poller = select.poll()

        listener_efd = self._listener.get_event_fd()
        poller.register(listener_efd, select.POLLIN)

        for efd in self._serialize_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        for efd in self._store_efd_to_adapter:
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

                try:
                    if fd == listener_efd:
                        keys = self._listener.pop_pending_keys()
                        if keys:
                            self._process_new_keys(keys)
                    elif fd in self._serialize_efd_to_adapter:
                        self._process_serialize_completions(
                            self._serialize_efd_to_adapter[fd]
                        )
                    elif fd in self._store_efd_to_adapter:
                        adapter_index = self._store_efd_to_adapter[fd]
                        self._process_completed_tasks(adapter_index)
                except Exception:
                    logger.exception(
                        "Unexpected error in store loop while processing fd %d",
                        fd,
                    )

    def _process_new_keys(self, keys: list[ObjectKey]) -> None:
        """
        Process a batch of newly written keys.

        1. Ask the policy which adapters each key should go to.
        2. For each adapter target, reserve read access on L1 to get
           MemoryObj references (skip keys that fail — best-effort).
        3. If serde disabled: submit store tasks to L2 immediately.
        4. If serde enabled: allocate temp buffers and submit
           serialize task to SerdeProcessor.

        Args:
            keys (list[ObjectKey]): Keys that finished writing to L1.
        """
        plan = self._policy.select_store_targets(keys, self._adapter_descriptors)

        l1_mgr = self._l1_manager

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

            serde = self._serde_processors[adapter_index]

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

            if serde is not None:
                # Serde enabled: alloc temp, submit async serialize
                self._submit_serialize_task(
                    adapter_index, serde, successful_keys, successful_objs
                )
                continue

            # Serde disabled: submit directly (existing behavior)
            self._submit_store_task(
                adapter_index,
                successful_keys,
                successful_objs,
                read_locked_keys=list(successful_keys),
                temp_keys=[],
            )

    # =========================================================================
    # Serialize phase (serde enabled only)
    # =========================================================================

    def _submit_serialize_task(
        self,
        adapter_index: int,
        serde: SerdeProcessor,
        successful_keys: list[ObjectKey],
        successful_objs: list[MemoryObj],
    ) -> None:
        """Allocate temp buffers and submit async serialize to SerdeProcessor."""
        l1_mgr = self._l1_manager

        temp_keys = [make_temp_key(key, "ser") for key in successful_keys]
        ser_layout = serialized_layout_desc(
            MemoryLayoutDesc(
                shapes=successful_objs[0].get_shapes(),
                dtypes=successful_objs[0].get_dtypes(),
            ),
            serde,
        )
        temp_write_results = l1_mgr.reserve_write(
            keys=temp_keys,
            is_temporary=[True] * len(temp_keys),
            layout_desc=ser_layout,
            mode="new",
        )

        # Filter to keys where temp alloc succeeded
        final_keys = []
        final_read_keys = []
        final_read_objs = []
        final_temp_keys = []
        final_temp_objs = []
        failed_orig_keys_set: set[ObjectKey] = set()

        for orig_key, orig_obj, temp_key in zip(
            successful_keys, successful_objs, temp_keys, strict=True
        ):
            temp_result = temp_write_results.get(temp_key)
            if temp_result is None or temp_result[0] != L1Error.SUCCESS:
                failed_orig_keys_set.add(orig_key)
                continue

            final_keys.append(orig_key)
            final_read_keys.append(orig_key)
            final_read_objs.append(orig_obj)
            final_temp_keys.append(temp_key)
            final_temp_objs.append(temp_result[1])

        # Release read locks for keys whose temp alloc failed
        failed_read_keys = [k for k in successful_keys if k in failed_orig_keys_set]
        if failed_read_keys:
            l1_mgr.finish_read(failed_read_keys)

        # Note: failed temp alloc keys were never added to L1's object store
        # (reserve_write returned an error), so there's nothing to finish_write
        # or delete — they simply don't exist.

        if not final_keys:
            return

        # Submit async serialize to the SerdeProcessor
        serde_task_id = serde.submit_serialize(final_read_objs, final_temp_objs)

        self._in_flight_serialize_tasks[(adapter_index, serde_task_id)] = (
            InFlightSerializeTask(
                adapter_index=adapter_index,
                serde_task_id=serde_task_id,
                keys=final_keys,
                read_locked_keys=final_read_keys,
                temp_keys=final_temp_keys,
                temp_objs=final_temp_objs,
            )
        )

    def _process_serialize_completions(self, adapter_index: int) -> None:
        """Handle completed serialize tasks from a SerdeProcessor.

        On success: release original read locks, submit store to L2.
        On failure: release both read locks and temp buffers.
        """
        serde = self._serde_processors[adapter_index]
        assert serde is not None, (
            f"serialize completion for adapter {adapter_index} but no serde processor"
        )
        l1_mgr = self._l1_manager

        for task in list(self._in_flight_serialize_tasks.values()):
            if task.adapter_index != adapter_index:
                continue

            result = serde.query_serialize_result(task.serde_task_id)
            if result is None:
                continue

            del self._in_flight_serialize_tasks[(adapter_index, task.serde_task_id)]

            # Release original read locks — data is in temp now
            l1_mgr.finish_read(task.read_locked_keys)

            if result:
                logger.debug(
                    "Serialize completed for adapter %d, %d keys — "
                    "submitting serialized data to L2.",
                    task.adapter_index,
                    len(task.keys),
                )
                # Transition temp buffers: write-locked -> read-locked
                # so L2 can safely read from them during store.
                l1_mgr.finish_write_and_reserve_read(task.temp_keys)

                # Submit to L2 with temp buffers (now read-locked)
                self._submit_store_task(
                    task.adapter_index,
                    task.keys,
                    task.temp_objs,
                    read_locked_keys=list(task.temp_keys),
                    temp_keys=[],
                )
            else:
                # Serialize failed — clean up temp buffers
                logger.warning(
                    "Serialize task failed for adapter %d, %d keys",
                    task.adapter_index,
                    len(task.keys),
                )
                if task.temp_keys:
                    l1_mgr.finish_write(task.temp_keys)
                    l1_mgr.delete(task.temp_keys)

    # =========================================================================
    # L2 store phase
    # =========================================================================

    def _submit_store_task(
        self,
        adapter_index: int,
        store_keys: list[ObjectKey],
        store_objs: list[MemoryObj],
        read_locked_keys: list[ObjectKey],
        temp_keys: list[ObjectKey],
    ) -> None:
        """Submit a store task to L2 and track it (shared for both paths)."""
        adapter = self._l2_adapters[adapter_index]
        task_id = adapter.submit_store_task(store_keys, store_objs)

        self._in_flight_tasks[(adapter_index, task_id)] = InFlightStoreTask(
            adapter_index=adapter_index,
            keys=store_keys,
            read_locked_keys=read_locked_keys,
            temp_keys=temp_keys,
        )
        self._status_in_flight_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": adapter_index,
                    "key_count": len(store_keys),
                },
            )
        )

        logger.debug(
            "Submitted store task %d to adapter %d with %d keys.",
            task_id,
            adapter_index,
            len(store_keys),
        )

    def _process_completed_tasks(self, adapter_index: int) -> None:
        """
        Process completed store tasks for a given adapter.

        1. Pop all completed tasks from the adapter.
        2. Release L1 read locks for each task.
        3. Release temp buffer write locks if serde was enabled.
        4. If the task succeeded, ask the policy which keys to
           delete from L1 and delete them.
        5. Remove the task from in-flight tracking.

        Args:
            adapter_index (int): Index of the adapter whose eventfd
                was signaled.
        """
        adapter = self._l2_adapters[adapter_index]
        completed = adapter.pop_completed_store_tasks()

        l1_mgr = self._l1_manager

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

            # Release read locks.
            # No serde: these are the original l1_keys.
            # Serde: these are the temp_keys (transitioned to read-locked
            # after serialization). finish_read auto-deletes them since
            # is_temporary=True.
            l1_mgr.finish_read(task.read_locked_keys)

            # Release temp buffers if serde was enabled
            if task.temp_keys:
                l1_mgr.finish_write(task.temp_keys)
                l1_mgr.delete(task.temp_keys)

            if success:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L2_STORE_COMPLETED,
                        metadata={
                            "adapter_index": adapter_index,
                            "succeeded_count": len(task.keys),
                            "failed_count": 0,
                        },
                    )
                )
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
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L2_STORE_COMPLETED,
                        metadata={
                            "adapter_index": adapter_index,
                            "succeeded_count": 0,
                            "failed_count": len(task.keys),
                        },
                    )
                )
                logger.warning(
                    "Store task %d to adapter %d failed for keys: %s",
                    task_id,
                    adapter_index,
                    task.keys,
                )

    def _cleanup_in_flight_tasks(self) -> None:
        """
        Release all held locks for any in-flight tasks that
        haven't completed. Called during stop().
        """
        l1_mgr = self._l1_manager

        # Clean up in-flight serialize tasks
        for task in self._in_flight_serialize_tasks.values():
            logger.warning(
                "Cleaning up in-flight serialize task (adapter %d, %d keys).",
                task.adapter_index,
                len(task.keys),
            )
            l1_mgr.finish_read(task.read_locked_keys)
            if task.temp_keys:
                l1_mgr.finish_write(task.temp_keys)
                l1_mgr.delete(task.temp_keys)
        self._in_flight_serialize_tasks.clear()

        # Clean up in-flight L2 store tasks
        for (adapter_index, task_id), store_task in self._in_flight_tasks.items():
            logger.warning(
                "Cleaning up in-flight store task %d (adapter %d, %d keys).",
                task_id,
                adapter_index,
                len(store_task.read_locked_keys),
            )
            l1_mgr.finish_read(store_task.read_locked_keys)
            if store_task.temp_keys:
                l1_mgr.finish_write(store_task.temp_keys)
                l1_mgr.delete(store_task.temp_keys)
        self._in_flight_tasks.clear()
