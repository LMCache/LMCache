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
import enum
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


class StorePhase(enum.Enum):
    """Phase of a store request lifecycle."""

    SERIALIZE = enum.auto()
    STORE = enum.auto()


StoreRequestId = int


@dataclass
class InFlightStoreRequest:
    """Tracks a single store request across serialize and L2 store phases.

    Mirrors the unified ``InFlightPrefetchRequest`` pattern: one struct
    with a phase enum instead of separate serialize/store dataclasses.
    """

    request_id: StoreRequestId
    adapter_index: int
    phase: StorePhase

    keys: list[ObjectKey]
    """Original logical keys for L2."""

    read_locked_keys: list[ObjectKey]
    """Keys currently holding L1 read locks.
    SERIALIZE phase: original keys (released when serialize completes).
    STORE phase (no serde): original keys.
    STORE phase (serde): temp keys (transitioned to read-locked after
    serialize; auto-deleted on finish_read since is_temporary=True)."""

    temp_keys: list[ObjectKey] = field(default_factory=list)
    """Temp buffer keys for serde. Empty when serde is disabled."""

    temp_objs: list[MemoryObj] = field(default_factory=list)
    """Temp buffer MemoryObjs. Populated during SERIALIZE phase; used to
    submit the L2 store task after serialize succeeds."""

    serde_task_id: SerdeTaskId | None = None
    """Serialize task ID (SERIALIZE phase only)."""

    l2_task_id: L2TaskId | None = None
    """L2 store task ID (STORE phase only)."""

    l2_store_result: bool | None = None
    """L2 store outcome (True=success, False=failure). Populated by
    ``_drain_l2_store_completions`` when the adapter signals its store
    eventfd. Read by ``_advance_request`` to drive the terminal
    transition. ``None`` while the L2 store is still in flight."""


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
        serde_processors: list[SerdeProcessor | None],
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy

        # Caller must pass a list of the same length as l2_adapters.
        # Individual elements may be None (serde disabled for that adapter).
        if len(serde_processors) != len(l2_adapters):
            raise ValueError(
                f"serde_processors length ({len(serde_processors)}) must "
                f"match l2_adapters length ({len(l2_adapters)})"
            )
        self._serde_processors: list[SerdeProcessor | None] = list(serde_processors)

        self._listener = StoreListener()
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()

        # All in-flight store requests, keyed by request_id.
        self._in_flight_requests: dict[StoreRequestId, InFlightStoreRequest] = {}
        self._next_request_id: StoreRequestId = 0

        # Secondary index: (adapter_index, l2_task_id) → request_id
        # for O(1) lookup when L2 store tasks complete.
        self._l2_task_to_request: dict[tuple[int, L2TaskId], StoreRequestId] = {}

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
        self._cleanup_in_flight_requests()
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

            new_keys: list[ObjectKey] = []
            advance_needed = False
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    os.eventfd_read(fd)
                except (OSError, BlockingIOError):
                    pass

                if fd == listener_efd:
                    try:
                        new_keys.extend(self._listener.pop_pending_keys())
                    except Exception:
                        logger.exception(
                            "Unexpected error popping pending keys from listener"
                        )
                else:
                    # Any adapter eventfd (serialize / store) — actual
                    # dispatch lives in _advance_request.
                    advance_needed = True

            if new_keys:
                try:
                    self._process_new_keys(new_keys)
                except Exception:
                    logger.exception(
                        "Unexpected error processing new keys in store loop"
                    )

            if advance_needed:
                try:
                    self._drain_l2_store_completions()
                except Exception:
                    logger.exception("Unexpected error draining L2 store completions")
                try:
                    for request in list(self._in_flight_requests.values()):
                        self._advance_request(request)
                except Exception:
                    logger.exception(
                        "Unexpected error advancing in-flight store requests"
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
                self._submit_serialize(
                    adapter_index, serde, successful_keys, successful_objs
                )
                continue

            # Serde disabled: create STORE request and submit to L2
            request = InFlightStoreRequest(
                request_id=self._next_request_id,
                adapter_index=adapter_index,
                phase=StorePhase.STORE,
                keys=list(successful_keys),
                read_locked_keys=list(successful_keys),
            )
            self._next_request_id += 1
            self._in_flight_requests[request.request_id] = request
            self._status_in_flight_count += 1
            self._submit_l2_store(request, successful_objs)

    # =========================================================================
    # Serialize phase (serde enabled only)
    # =========================================================================

    def _submit_serialize(
        self,
        adapter_index: int,
        serde: SerdeProcessor,
        successful_keys: list[ObjectKey],
        successful_objs: list[MemoryObj],
    ) -> None:
        """Allocate temp buffers and submit async serialize to SerdeProcessor.

        Args:
            adapter_index: Index of the L2 adapter the data will be stored to.
            serde: SerdeProcessor for that adapter.
            successful_keys: Original ObjectKeys that L1 ``reserve_read`` just
                succeeded for. These hold L1 read locks that must be released
                eventually.
            successful_objs: Corresponding read-locked MemoryObjs (same length
                and order as ``successful_keys``); the source data to serialize.
        """
        l1_mgr = self._l1_manager

        temp_keys = [make_temp_key(key) for key in successful_keys]
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

        # Release read locks for keys whose temp alloc failed.
        # failed_orig_keys_set is a subset of successful_keys, so we can
        # convert directly without filtering.
        if failed_orig_keys_set:
            l1_mgr.finish_read(list(failed_orig_keys_set))

        # Note: failed temp alloc keys were never added to L1's object store
        # (reserve_write returned an error), so there's nothing to finish_write
        # or delete — they simply don't exist.

        if not final_keys:
            return

        # Submit async serialize to the SerdeProcessor.
        # If submit_serialize raises, we must release the read locks and
        # temp buffers that would otherwise be tracked in the in-flight dict.
        try:
            serde_task_id = serde.submit_serialize(final_read_objs, final_temp_objs)
        except Exception:
            logger.exception(
                "Failed to submit serialize task for adapter %d",
                adapter_index,
            )
            l1_mgr.finish_read(final_read_keys)
            l1_mgr.finish_write(final_temp_keys)
            l1_mgr.delete(final_temp_keys)
            return

        request = InFlightStoreRequest(
            request_id=self._next_request_id,
            adapter_index=adapter_index,
            phase=StorePhase.SERIALIZE,
            keys=final_keys,
            read_locked_keys=final_read_keys,
            temp_keys=final_temp_keys,
            temp_objs=final_temp_objs,
            serde_task_id=serde_task_id,
        )
        self._next_request_id += 1
        self._in_flight_requests[request.request_id] = request
        self._status_in_flight_count += 1

    # =========================================================================
    # L2 store phase
    # =========================================================================

    def _submit_l2_store(
        self,
        request: InFlightStoreRequest,
        store_objs: list[MemoryObj],
    ) -> None:
        """Submit the L2 store task for a request in STORE phase.

        The caller must have already registered ``request`` in
        ``_in_flight_requests`` and incremented ``_status_in_flight_count``.
        If ``submit_store_task`` raises, this method releases the request's
        read locks and removes it from tracking so resources do not leak.
        """
        adapter = self._l2_adapters[request.adapter_index]
        try:
            l2_task_id = adapter.submit_store_task(request.keys, store_objs)
        except Exception:
            logger.exception(
                "submit_store_task raised for adapter %d, request %d — aborting",
                request.adapter_index,
                request.request_id,
            )
            # read_locked_keys covers both cases: original keys (non-serde) or
            # temp keys (serde, post finish_write_and_reserve_read → temp
            # buffers auto-delete on finish_read).
            self._l1_manager.finish_read(request.read_locked_keys)
            del self._in_flight_requests[request.request_id]
            self._status_in_flight_count -= 1
            return

        request.l2_task_id = l2_task_id
        self._l2_task_to_request[(request.adapter_index, l2_task_id)] = (
            request.request_id
        )

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": request.adapter_index,
                    "key_count": len(request.keys),
                },
            )
        )

        logger.debug(
            "Submitted store task %d to adapter %d with %d keys.",
            l2_task_id,
            request.adapter_index,
            len(request.keys),
        )

    def _drain_l2_store_completions(self) -> None:
        """Pop completed L2 store tasks from every adapter and deposit
        the outcome on the corresponding in-flight request.

        Done in a separate pass (rather than inside ``_advance_request``)
        because ``pop_completed_store_tasks`` is a batch drain that
        returns all completed tasks at once — calling it per request
        would lose results for other requests sharing the adapter.
        ``_advance_request`` then reads ``request.l2_store_result`` to
        drive the terminal transition.
        """
        for adapter_idx, adapter in enumerate(self._l2_adapters):
            completed = adapter.pop_completed_store_tasks()
            for task_id, success in completed.items():
                composite_key = (adapter_idx, task_id)
                request_id = self._l2_task_to_request.pop(composite_key, None)
                if request_id is None:
                    logger.warning(
                        "Completed store task %d (adapter %d) not found in tracking.",
                        task_id,
                        adapter_idx,
                    )
                    continue
                request = self._in_flight_requests.get(request_id)
                if request is None:
                    logger.warning(
                        "Completed store task %d (adapter %d): request %d missing.",
                        task_id,
                        adapter_idx,
                        request_id,
                    )
                    continue
                request.l2_store_result = success

    def _advance_request(self, request: InFlightStoreRequest) -> None:
        """Single dispatcher for store request state transitions.

        Called for every in-flight request on each eventfd wakeup. Polls
        the request's pending result for its current phase and triggers
        the corresponding transition:

          - ``SERIALIZE``: query the serialize result. On success,
            release original read locks, transition temp buffers to
            read-locked, move to ``STORE``, and submit the L2 store
            task. On failure, release read locks and temp buffers and
            remove the request.
          - ``STORE``: if ``_drain_l2_store_completions`` deposited a
            result, release read locks (temp buffers auto-delete since
            ``is_temporary=True``), publish ``L2_STORE_COMPLETED``,
            apply policy L1 deletions on success, and remove the
            request.

        The poll loop never dispatches to per-event handlers — it
        relays adapter events here, so all per-phase logic lives in
        this method.
        """
        l1_mgr = self._l1_manager

        if request.phase == StorePhase.SERIALIZE:
            assert request.serde_task_id is not None
            serde = self._serde_processors[request.adapter_index]
            assert serde is not None, (
                f"SERIALIZE request for adapter {request.adapter_index} "
                f"but no serde processor"
            )
            result = serde.query_serialize_result(request.serde_task_id)
            if result is None:
                return

            # Release original read locks — data is in temp now
            l1_mgr.finish_read(request.read_locked_keys)

            if result:
                logger.debug(
                    "Serialize completed for adapter %d, %d keys — "
                    "submitting serialized data to L2.",
                    request.adapter_index,
                    len(request.keys),
                )
                # Transition temp buffers: write-locked -> read-locked
                # so L2 can safely read from them during store.
                l1_mgr.finish_write_and_reserve_read(request.temp_keys)

                # Transition to STORE phase
                request.phase = StorePhase.STORE
                request.read_locked_keys = list(request.temp_keys)
                request.serde_task_id = None
                self._submit_l2_store(request, request.temp_objs)
            else:
                logger.warning(
                    "Serialize task failed for adapter %d, %d keys",
                    request.adapter_index,
                    len(request.keys),
                )
                if request.temp_keys:
                    l1_mgr.finish_write(request.temp_keys)
                    l1_mgr.delete(request.temp_keys)
                del self._in_flight_requests[request.request_id]
                self._status_in_flight_count -= 1

        elif request.phase == StorePhase.STORE:
            if request.l2_store_result is None:
                return

            success = request.l2_store_result
            adapter_index = request.adapter_index

            # Release read locks. No serde: original keys. Serde: temp
            # keys (transitioned to read-locked after serialize; auto-
            # delete on finish_read since is_temporary=True).
            l1_mgr.finish_read(request.read_locked_keys)
            del self._in_flight_requests[request.request_id]
            self._status_in_flight_count -= 1

            if success:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L2_STORE_COMPLETED,
                        metadata={
                            "adapter_index": adapter_index,
                            "succeeded_count": len(request.keys),
                            "failed_count": 0,
                        },
                    )
                )
                logger.debug(
                    "L2 store completed: adapter %d, request %d, %d keys.",
                    adapter_index,
                    request.request_id,
                    len(request.keys),
                )
                delete_keys = self._policy.select_l1_deletions(request.keys)
                if delete_keys:
                    l1_mgr.delete(delete_keys)
            else:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L2_STORE_COMPLETED,
                        metadata={
                            "adapter_index": adapter_index,
                            "succeeded_count": 0,
                            "failed_count": len(request.keys),
                        },
                    )
                )
                logger.warning(
                    "Store request %d to adapter %d failed for keys: %s",
                    request.request_id,
                    adapter_index,
                    request.keys,
                )

    def _cleanup_in_flight_requests(self) -> None:
        """
        Release all held locks for any in-flight requests that
        haven't completed. Called during stop().
        """
        l1_mgr = self._l1_manager

        for request in self._in_flight_requests.values():
            logger.warning(
                "Cleaning up in-flight store request %d "
                "(adapter %d, phase %s, %d keys).",
                request.request_id,
                request.adapter_index,
                request.phase.name,
                len(request.keys),
            )
            if request.phase == StorePhase.SERIALIZE:
                # Read locks on originals + write locks on temp buffers
                l1_mgr.finish_read(request.read_locked_keys)
                if request.temp_keys:
                    l1_mgr.finish_write(request.temp_keys)
                    l1_mgr.delete(request.temp_keys)
            elif request.phase == StorePhase.STORE:
                # Read locks on originals (no serde) or temp keys (serde)
                l1_mgr.finish_read(request.read_locked_keys)

        self._in_flight_requests.clear()
        self._l2_task_to_request.clear()
