# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Submits lookup_and_lock tasks to all L2 adapters.
3. Computes a load plan using the PrefetchPolicy, trimmed to the contiguous
   prefix of found keys.
4. Reserves L1 write buffers and submits load tasks to L2 adapters.
5. On load completion, transitions L1 entries from write-locked to read-locked.
6. Reports prefix hit count.
"""

# Standard
from dataclasses import dataclass, field
from typing import Iterable
import enum
import os
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


# HELPER FUNCTIONS
def trim_load_plan_to_first_n_keys(
    load_plan: dict[int, Bitmap],
    num_keys: int,
    n: int,
) -> dict[int, Bitmap]:
    """
    Trim the load plan to only include keys with indices < n.

    For example, if n=3 and the combined load plan has keys
    {0, 1, 3}, the trimmed plan will only include key indices [0, 1]
    and exclude index 3.

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        num_keys: Total number of keys (bitmap size).
        n: Number of keys to include in the trimmed plan (prefix length).

    Returns:
        Trimmed load plan with only key indices < n.

    Note:
        the adapter index will not appear in the return dict if
        it has no keys in the prefix.
    """
    if n <= 0:
        return {}

    trimmed_plan: dict[int, Bitmap] = {}
    mask_bitmap = Bitmap(num_keys, n)
    for adapter_idx, bitmap in load_plan.items():
        new_bitmap = bitmap & mask_bitmap
        if new_bitmap.popcount() == 0:
            continue

        trimmed_plan[adapter_idx] = new_bitmap

    return trimmed_plan


def trim_load_plan_to_prefix(
    load_plan: dict[int, Bitmap],
    num_keys: int,
) -> dict[int, Bitmap]:
    """
    Trim the load plan to the longest contiguous prefix of keys.

    For example, if num_keys=5 and the combined load plan has keys
    {0, 1, 3}, the prefix is 2 (keys 0 and 1), so the trimmed plan
    will only include key indices [0, 1] and exclude index 3.

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        num_keys: Total number of keys in the request.

    Returns:
        Trimmed load plan with only prefix key indices.

    Note:
        the adapter index will not appear in the return dict if
        it has no keys in the prefix.
    """
    merged_plan = Bitmap(num_keys)
    for bitmap in load_plan.values():
        merged_plan = merged_plan | bitmap

    prefix_length = merged_plan.count_leading_ones()
    return trim_load_plan_to_first_n_keys(load_plan, num_keys, prefix_length)


def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge multiple bitmaps with a bitwise OR."""
    if not bitmaps:
        return Bitmap(0)
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

PrefetchRequestId = int


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    phase: PrefetchPhase

    # Lookup phase: adapter_idx -> task_id (removed as results arrive)
    pending_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Lookup phase: adapter_idx -> bitmap (populated as results arrive)
    lookup_results: dict[int, Bitmap] = field(default_factory=dict)

    # Load phase: adapter_idx -> bitmap of key indices to load
    load_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> task_id (removed as results arrive)
    pending_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap (populated as results arrive)
    load_results: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: keys that were write-reserved in L1
    write_reserved_keys: list[ObjectKey] = field(default_factory=list)
    write_reserved_objs: dict[ObjectKey, MemoryObj] = field(default_factory=dict)

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that submits lookup_and_lock to all adapters.
    3. Uses PrefetchPolicy to compute a load plan from lookup results.
    4. Reserves L1 write buffers and submits load tasks to adapters.
    5. On completion, transitions loaded keys to read-locked state.
    6. Reports the number of prefix hits via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
    ) -> None:
        super().__init__(l1_manager)
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy
        # TODO(ApostaC): max_in_flight should not be a static constant.
        # Replace with a dynamic admission controller that monitors L1 memory
        # usage of in-flight prefetch requests. A fixed limit can still blow
        # up L1 memory when individual requests are large.
        self._max_in_flight = max_in_flight

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: list[
            tuple[PrefetchRequestId, list[ObjectKey], MemoryLayoutDesc]
        ] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[
            tuple[PrefetchRequestId, list[ObjectKey], MemoryLayoutDesc]
        ] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Thread-safe results (background -> external)
        self._results_lock = threading.Lock()
        self._completed_results: dict[PrefetchRequestId, int] = {}

        # Map eventfds to adapter indices for quick lookup in poll.
        # Relies on the L2AdapterInterface contract that every adapter
        # returns distinct fds for store/lookup/load, and no two adapters
        # share an fd.  See the docstrings in L2AdapterInterface.
        self._lookup_efd_to_adapter: dict[int, int] = {}
        self._load_efd_to_adapter: dict[int, int] = {}
        for i, adapter in enumerate(self._l2_adapters):
            self._lookup_efd_to_adapter[adapter.get_lookup_and_lock_event_fd()] = i
            self._load_efd_to_adapter[adapter.get_load_event_fd()] = i

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )

    # =========================================================================
    # External API (thread-safe)
    # =========================================================================

    def submit_prefetch_request(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request for the given keys.

        Thread-safe. Can be called from any thread.

        Only the **contiguous prefix** of found keys is loaded from L2.
        If L2 has keys {0, 1, 3, 4} but not key 2, only keys {0, 1} are
        loaded because the gap at index 2 breaks the prefix.  Keys beyond
        the prefix are never transferred, saving I/O bandwidth and L1
        memory.  Use :meth:`query_prefetch_result` to retrieve the number
        of prefix hits once the request completes.

        Args:
            keys: List of object keys to prefetch from L2 into L1.
                The ordering defines the prefix: index 0 is the first key.
            layout_desc: Memory layout for L1 write buffer allocation.

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._submission_queue.append((request_id, keys, layout_desc))
        os.eventfd_write(self._submission_efd, 1)
        return request_id

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the number of prefix hits if the request
        has completed, None if still in progress. Each result can only
        be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits, or None if not yet complete.
        """
        with self._results_lock:
            return self._completed_results.pop(request_id, None)

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._results_lock:
            completed_results_count = len(self._completed_results)
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "max_in_flight": self._max_in_flight,
            "submission_queue_size": submission_queue_size,
            "pending_queue_size": self._status_pending_count,
            "in_flight_request_count": self._status_in_flight_count,
            "lookup_phase_count": self._status_lookup_phase_count,
            "load_phase_count": self._status_load_phase_count,
            "completed_results_count": completed_results_count,
            "num_l2_adapters": len(self._l2_adapters),
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background prefetch loop thread."""
        logger.info("Starting PrefetchController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop and wait for the thread to join.

        Cleans up any in-flight requests (releases L1 write locks,
        L2 locks) before returning.
        """
        self._stop_flag.set()
        os.eventfd_write(self._submission_efd, 1)
        self._thread.join()
        self._cleanup_in_flight_requests()
        os.close(self._submission_efd)

    # =========================================================================
    # Background loop
    # =========================================================================

    def _prefetch_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The submission eventfd (new prefetch requests).
        - Each L2 adapter's lookup eventfd (completed lookups).
        - Each L2 adapter's load eventfd (completed loads).
        """
        poller = select.poll()
        poller.register(self._submission_efd, select.POLLIN)
        for efd in self._lookup_efd_to_adapter:
            poller.register(efd, select.POLLIN)
        for efd in self._load_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            ready = poller.poll(PREFETCH_LOOP_POLL_TIMEOUT_MS)

            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    os.eventfd_read(fd)
                except (OSError, BlockingIOError):
                    pass

                if fd == self._submission_efd:
                    self._drain_submission_queue()
                elif fd in self._lookup_efd_to_adapter:
                    self._process_lookup_completions(self._lookup_efd_to_adapter[fd])
                elif fd in self._load_efd_to_adapter:
                    self._process_load_completions(self._load_efd_to_adapter[fd])

            self._start_pending_requests()

    def _drain_submission_queue(self) -> None:
        """Move items from the thread-safe submission queue to the
        pending queue."""
        with self._submission_lock:
            items = self._submission_queue
            self._submission_queue = []
        self._pending_queue.extend(items)
        self._status_pending_count += len(items)

    def _start_pending_requests(self) -> None:
        """Start pending requests up to the max in-flight limit."""
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, keys, layout_desc = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, keys, layout_desc)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> None:
        """Submit lookup_and_lock to all adapters for a new request."""
        if not self._l2_adapters:
            self._complete_request(request_id, 0)
            return

        pending_lookup_tasks: dict[int, L2TaskId] = {}
        for i, adapter in enumerate(self._l2_adapters):
            task_id = adapter.submit_lookup_and_lock_task(keys)
            pending_lookup_tasks[i] = task_id

        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=keys,
            layout_desc=layout_desc,
            phase=PrefetchPhase.LOOKUP,
            pending_lookup_tasks=pending_lookup_tasks,
        )
        self._in_flight_requests[request_id] = request
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

    def _process_lookup_completions(self, adapter_index: int) -> None:
        """Check all LOOKUP-phase requests for completed lookups from
        this adapter."""
        ready_to_transition: list[InFlightPrefetchRequest] = []

        for request in list(self._in_flight_requests.values()):
            if request.phase != PrefetchPhase.LOOKUP:
                continue
            if adapter_index not in request.pending_lookup_tasks:
                continue

            task_id = request.pending_lookup_tasks[adapter_index]
            result = self._l2_adapters[adapter_index].query_lookup_and_lock_result(
                task_id
            )

            if result is not None:
                request.lookup_results[adapter_index] = result
                del request.pending_lookup_tasks[adapter_index]

                if request.all_lookups_done():
                    ready_to_transition.append(request)

        for request in ready_to_transition:
            self._transition_to_load_phase(request)

    # =========================================================================
    # Load phase
    # =========================================================================

    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Compute load plan, reserve L1 buffers, and submit load tasks."""
        request.phase = PrefetchPhase.PLAN_AND_LOAD
        self._status_lookup_phase_count -= 1
        self._status_load_phase_count += 1

        # Step 1: get load plan from policy
        load_plan = self._policy.select_load_plan(
            request.keys,
            request.lookup_results,
            self._adapter_descriptors,
        )

        # Step 2: trim the load plan to only prefix
        trimmed_plan = trim_load_plan_to_prefix(load_plan, len(request.keys))

        if not trimmed_plan:
            # Nothing to load after trimming to prefix. Unlock all lookup locks
            # and complete with 0 hits.
            self._unlock_all_lookups(request)
            self._complete_request(request.request_id, 0)
            return

        # Step 3: reserve L1 write buffers
        merged_bitmap = merge_bitmaps(trimmed_plan.values(), len(request.keys))
        keys_to_reserve = merged_bitmap.gather(request.keys)
        l1_mgr = self.get_l1_manager()

        write_results = l1_mgr.reserve_write(
            keys=keys_to_reserve,
            is_temporary=[True] * len(keys_to_reserve),
            layout_desc=request.layout_desc,
            mode="new",
        )

        # Step 4: filter to successfully reserved keys
        reserved_key_set: set[ObjectKey] = set()
        for key, (err, mem_obj) in write_results.items():
            if err == L1Error.SUCCESS and mem_obj is not None:
                request.write_reserved_keys.append(key)
                request.write_reserved_objs[key] = mem_obj
                reserved_key_set.add(key)
            else:
                logger.debug(
                    "Prefetch request %d: reserve write failed for %s: %s",
                    request.request_id,
                    key,
                    err,
                )

        # Step 5: recompute load plan excluding failed reservations
        reserved_bitmap = Bitmap(len(request.keys))
        for i, key in enumerate(request.keys):
            if key in reserved_key_set:
                reserved_bitmap.set(i)

        prefix_length = reserved_bitmap.count_leading_ones()
        trimmed_plan = trim_load_plan_to_first_n_keys(
            load_plan, len(request.keys), prefix_length
        )
        request.load_plan = trimmed_plan

        ## Step 6: phase 1 unlock — keys locked in lookup but not in plan
        self._unlock_unneeded_keys(request)

        if not trimmed_plan:
            # Nothing loadable after filtering
            if request.write_reserved_keys:
                l1_mgr.finish_write(request.write_reserved_keys)
                l1_mgr.delete(request.write_reserved_keys)
            self._complete_request(request.request_id, 0)
            return

        ## Step 7: submit load tasks per adapter
        for adapter_idx, bitmap in trimmed_plan.items():
            per_adapter_keys = bitmap.gather(request.keys)
            per_adapter_objs = [
                request.write_reserved_objs[key] for key in per_adapter_keys
            ]
            task_id = self._l2_adapters[adapter_idx].submit_load_task(
                per_adapter_keys, per_adapter_objs
            )
            request.pending_load_tasks[adapter_idx] = task_id

        logger.debug(
            "Prefetch request %d: submitted load tasks to %d adapters for %d keys",
            request.request_id,
            len(trimmed_plan),
            len(reserved_key_set),
        )

    def _process_load_completions(self, adapter_index: int) -> None:
        """Check all PLAN_AND_LOAD-phase requests for completed loads."""
        ready_to_finalize: list[InFlightPrefetchRequest] = []

        for request in list(self._in_flight_requests.values()):
            if request.phase != PrefetchPhase.PLAN_AND_LOAD:
                continue
            if adapter_index not in request.pending_load_tasks:
                continue

            task_id = request.pending_load_tasks[adapter_index]
            result = self._l2_adapters[adapter_index].query_load_result(task_id)

            if result is not None:
                request.load_results[adapter_index] = result
                del request.pending_load_tasks[adapter_index]

                if request.all_loads_done():
                    ready_to_finalize.append(request)

        for request in ready_to_finalize:
            self._finalize_load(request)

    def _finalize_load(self, request: InFlightPrefetchRequest) -> None:
        """
        Finalize a completed load: build result bitmap, transition L1
        state, release non-prefix read locks, and report prefix hits.

        Only prefix keys are submitted for loading, but partial load
        failures can create gaps.  Keys beyond the gap that were
        successfully loaded still need their read locks released.
        """
        num_keys = len(request.keys)

        # Scatter per-adapter local load results into global positions.
        # Each adapter's load bitmap is locally indexed (size == adapter's
        # key count).  The plan bitmap maps local → global indices via
        # get_indices_list().
        result_bitmap = Bitmap(num_keys)
        for adapter_idx, plan_bitmap in request.load_plan.items():
            load_bitmap = request.load_results.get(adapter_idx)
            if load_bitmap is None:
                continue
            plan_indices = plan_bitmap.get_indices_list()
            for global_i in load_bitmap.gather(plan_indices):
                result_bitmap.set(global_i)

        # Separate loaded vs. failed among write-reserved keys
        loaded_keys: list[ObjectKey] = result_bitmap.gather(request.keys)
        loaded_set = set(loaded_keys)
        failed_keys = [k for k in request.write_reserved_keys if k not in loaded_set]

        # Phase 2 unlock: release L2 locks for all keys in the load plan
        self._unlock_all_plan_keys(request)

        l1_mgr = self.get_l1_manager()

        # Transition loaded keys: write-locked -> read-locked
        if loaded_keys:
            l1_mgr.finish_write_and_reserve_read(loaded_keys)

        # Clean up failed keys
        if failed_keys:
            l1_mgr.finish_write(failed_keys)
            l1_mgr.delete(failed_keys)

        # Partial load failures can create gaps in the prefix.
        # Release read locks for loaded keys beyond the prefix.
        prefix_hits = result_bitmap.count_leading_ones()
        prefix_mask = Bitmap(num_keys, prefix_hits)
        non_prefix_loaded_bitmap = result_bitmap & (~prefix_mask)
        non_prefix_loaded = non_prefix_loaded_bitmap.gather(request.keys)
        if non_prefix_loaded:
            l1_mgr.finish_read(non_prefix_loaded)

        self._complete_request(request.request_id, prefix_hits)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _unlock_unneeded_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 1 unlock: keys locked in lookup but not in the load plan."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            plan_bitmap = request.load_plan.get(adapter_idx, Bitmap(len(request.keys)))
            to_unlock_bitmap = lookup_bitmap & (~plan_bitmap)
            unlock_keys = to_unlock_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_plan_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 2 unlock: release L2 locks for all keys in the load plan."""
        for adapter_idx, load_bitmap in request.load_plan.items():
            unlock_keys = load_bitmap.gather(request.keys)
            self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_lookups(self, request: InFlightPrefetchRequest) -> None:
        """Unlock all keys locked during lookup (nothing to load case)."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            unlock_keys = lookup_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _complete_request(
        self, request_id: PrefetchRequestId, prefix_hits: int
    ) -> None:
        """Store the result and remove from in-flight tracking."""
        with self._results_lock:
            self._completed_results[request_id] = prefix_hits
        removed = self._in_flight_requests.pop(request_id, None)
        if removed is not None:
            self._status_in_flight_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug(
            "Prefetch request %d completed: %d prefix hits",
            request_id,
            prefix_hits,
        )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        l1_mgr = self.get_l1_manager()
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write(request.write_reserved_keys)
                    l1_mgr.delete(request.write_reserved_keys)
                self._unlock_all_plan_keys(request)
            elif request.phase == PrefetchPhase.LOOKUP:
                self._unlock_all_lookups(request)
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
