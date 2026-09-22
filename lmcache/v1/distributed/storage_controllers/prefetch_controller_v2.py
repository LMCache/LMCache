# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

TODO: write the description
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import enum
import select
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.adapter_lifecycle import (
    AddAdapterOp,
    RemoveAdapterOp,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

if TYPE_CHECKING:
    # First Party
    pass

logger = init_logger(__name__)

# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

PrefetchRequestId = int

# Helper functions


def _get_prefetch_write_tag(request_id: PrefetchRequestId) -> str:
    """Return the L1 write tag of one prefetch request.

    Args:
        request_id: The prefetch request id.

    Returns:
        The tag for the request's L1 write reservations.

    Note:
        Tags are per request, so concurrent requests loading the same key
        do not contend for its L1 reservation.
    """
    return f"prefetch:{request_id}"


def _copy_rows(bitmaps: list[Bitmap]) -> list[Bitmap]:
    """Return an independent copy of every bitmap in ``bitmaps``.

    Args:
        bitmaps: The bitmaps to copy.

    Returns:
        New bitmaps with the same sizes and bits, sharing no storage with
        the inputs.
    """
    return [Bitmap(bitmap.size()) | bitmap for bitmap in bitmaps]


# Class definitions


class _MapState:
    """Tracks the global bitmap state.

    It maps from L2Adapter/L1Manager index to the global bitmaps of the
    object key groups.
    The object key groups are represented by the list of Bitmaps. Each
    element corresponds to the "row" in the prefetch request (see
    `GroupedObjectKeys`)
    """

    def __init__(self) -> None:
        self._state: dict[int, list[Bitmap]] = {}
        self._num_rows: int | None = None
        self._num_cols: int | None = None

    def _ensure_layout(self, bitmaps: list[Bitmap]) -> None:
        """Ensure that the layout of the bitmaps is consistent across all
        adapters. The number of rows and columns should be the same for all
        adapters.

        Raises:
            ValueError: If the layout of the bitmaps is inconsistent, or
                if the bitmaps are empty.
        """
        num_cols = set(bitmap.size() for bitmap in bitmaps)
        if not num_cols:
            raise ValueError("Bitmaps cannot be empty")

        if len(num_cols) > 1:
            raise ValueError(f"Bitmaps have inconsistent number of columns: {num_cols}")

        if self._num_rows is None:
            self._num_rows = len(bitmaps)
            self._num_cols = num_cols.pop()
        else:
            if len(bitmaps) != self._num_rows:
                raise ValueError(
                    f"Number of rows {len(bitmaps)} does not match "
                    f"expected {self._num_rows}"
                )
            if num_cols.pop() != self._num_cols:
                raise ValueError(
                    f"Number of columns {num_cols} does not match "
                    f"expected {self._num_cols}"
                )

    def _ensure_compatible(self, other: "_MapState") -> None:
        """Ensure that ``other`` can be combined with this map.

        Raises:
            ValueError: If both maps have a layout and the layouts differ.
        """
        if self._num_rows is None or other._num_rows is None:
            return
        if (self._num_rows, self._num_cols) != (other._num_rows, other._num_cols):
            raise ValueError(
                f"Layout ({other._num_rows} rows, {other._num_cols} cols) does "
                f"not match expected ({self._num_rows} rows, "
                f"{self._num_cols} cols)"
            )

    def merge(self) -> list[Bitmap]:
        """Union the rows of every adapter index into a single row list.

        Returns:
            One bitmap per row, where a bit is set if any adapter index has
            it set. An empty map yields an empty list.
        """
        if self._num_cols is None or self._num_rows is None:
            return []

        merged: list[Bitmap] = [Bitmap(self._num_cols) for _ in range(self._num_rows)]
        for bitmaps in self._state.values():
            merged = [m | bitmap for m, bitmap in zip(merged, bitmaps, strict=False)]
        return merged

    def __setitem__(self, adapter_idx: int, bitmaps: list[Bitmap]) -> None:
        self._ensure_layout(bitmaps)
        self._state[adapter_idx] = bitmaps

    def __getitem__(self, adapter_idx: int) -> list[Bitmap]:
        return self._state[adapter_idx]

    def __contains__(self, adapter_idx: int) -> bool:
        """Return whether ``adapter_idx`` has rows stored in this map."""
        return adapter_idx in self._state

    def __add__(self, other: "_MapState") -> "_MapState":
        """Merge two _MapState instances by unioning their bitmaps
        per adapter index.

        If an adapter index exists in one instance but not the other, the
        missing bitmap is treated as an empty bitmap (no keys).

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = _MapState()

        for adapter_idx, bitmaps_l in self._state.items():
            bitmaps_r = other._state.get(adapter_idx, None)
            if bitmaps_r is not None:
                result[adapter_idx] = [
                    b_l | b_r for b_l, b_r in zip(bitmaps_l, bitmaps_r, strict=False)
                ]
            else:
                result[adapter_idx] = _copy_rows(bitmaps_l)

        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx not in self._state:
                result[adapter_idx] = _copy_rows(bitmaps_r)

        return result

    def __iadd__(self, other: "_MapState") -> "_MapState":
        """In-place union of two _MapState instances.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = [
                    b_l | b_r
                    for b_l, b_r in zip(
                        self._state[adapter_idx], bitmaps_r, strict=False
                    )
                ]
            else:
                self[adapter_idx] = _copy_rows(bitmaps_r)
        return self

    def __sub__(self, other: "_MapState") -> "_MapState":
        """Clears the bits in this _MapState that are set in the other _MapState
        and returns a new _MapState instance with the result.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        result = _MapState()

        for adapter_idx, bitmaps_l in self._state.items():
            bitmaps_r = other._state.get(adapter_idx, None)
            if bitmaps_r is not None:
                result[adapter_idx] = [
                    b_l & ~b_r for b_l, b_r in zip(bitmaps_l, bitmaps_r, strict=False)
                ]
            else:
                result[adapter_idx] = _copy_rows(bitmaps_l)

        return result

    def __isub__(self, other: "_MapState") -> "_MapState":
        """In-place clears the bits in this _MapState that are set in the
        other _MapState.

        Raises:
            ValueError: If the two maps have different layouts.
        """
        self._ensure_compatible(other)
        for adapter_idx, bitmaps_r in other._state.items():
            if adapter_idx in self._state:
                self[adapter_idx] = [
                    b_l & ~b_r
                    for b_l, b_r in zip(
                        self._state[adapter_idx], bitmaps_r, strict=False
                    )
                ]
        return self


@dataclass
class PrefetchKeyState:
    """Tracks the states of the keys in a prefetch task"""

    # Keys that read-locked in L1 (i.e. finished prefetch)
    # Mapping from l1 manager index to the global bitmap
    l1_locked_keys: _MapState = field(default_factory=_MapState)

    # Keys that read-locked in L2 (i.e. finished L2 lookup and lock)
    # Mapping from l2 adapter index to the global bitmap
    l2_locked_keys: _MapState = field(default_factory=_MapState)

    # Keys that are planned to be used in L1
    # Mapping from l1 manager index to the global bitmap
    l1_planned_keys: _MapState = field(default_factory=_MapState)

    # Keys that are planned to be loaded from L2
    # Mapping from l2 adapter index to the global bitmap
    l2_planned_keys: _MapState = field(default_factory=_MapState)

    # Keys that are write-locked in L1 (for L2 to load)
    # Mapping from l1 manager index to the global bitmap
    l1_reserved_keys: _MapState = field(default_factory=_MapState)


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[GroupedObjectKeys]
    phase: PrefetchPhase
    num_kv_readers: int = 1

    # TODO: re-implement the inflight prefetch request
    key_states: PrefetchKeyState = field(default_factory=PrefetchKeyState)


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Looks up the keys and lock them in L1 and L2
    - 2.1 If it's an L1-only request, a fast path is taken and the request is
          completed immediately.
    3. Computes the load plan based on the found keys in L1 and L2;
       unlocks L1 keys that fall outside the plan.
    4. Reserves L1 write buffers for the proposed load plan
    5. Replan again based on the L1 write buffer reservation result, and unlock
       keys that fall outside the plan.
    6. Submit the load tasks to L2 adapters, which load the objects into L1.
    7. On completion, loaded keys become read-locked for the retriever.
    8. Reports the L1+L2 retained-key bitmap via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
    """

    # Singleton dispatch for the in-flight load gauges: tests may construct
    # multiple controllers but the OTel SDK only honors the first gauge
    # registration, so the callbacks read from the most recently built
    # instance via ``_gauge_target``.
    _gauges_registered: bool = False
    _gauge_target: "PrefetchController | None" = None

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
    ) -> None:
        # TODO: change it to l1_managers for multiple L1 managers
        self._l1_manager = l1_manager
        self._l2_adapters: dict[int, L2AdapterInterface] = {
            desc.index: adapter
            for desc, adapter in zip(adapter_descriptors, l2_adapters, strict=True)
        }
        self._adapter_descriptors: dict[int, AdapterDescriptor] = {
            desc.index: desc for desc in adapter_descriptors
        }
        self._policy = policy
        # TODO: remove max_in_flight and make it dynamic
        self._max_in_flight = max_in_flight

        # TODO: remove the unused fields

        # the in-flight operations are done.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._adapter_ctrl_efd = create_event_notifier()

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: list[tuple[PrefetchRequestId, PrefetchTaskSpec]] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[tuple[PrefetchRequestId, PrefetchTaskSpec]] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = create_event_notifier()

        # Thread-safe lookup results (background -> external)
        self._lookup_results_lock = threading.Lock()
        self._completed_lookups: dict[PrefetchRequestId, int] = {}

        # Thread-safe prefetch results (background -> external).  The condition
        # variable lets a WAIT_PREFETCH_STATUS handler block until a result is
        # published instead of busy-polling QUERY_PREFETCH_STATUS.
        self._prefetch_results_lock = threading.Lock()
        self._prefetch_results_cv = threading.Condition(self._prefetch_results_lock)
        self._completed_results: dict[PrefetchRequestId, Bitmap] = {}

        # Map eventfds to adapter indices for quick lookup in poll.
        # Relies on the L2AdapterInterface contract that every adapter
        # returns distinct fds for store/lookup/load, and no two adapters
        # share an fd.  See the docstrings in L2AdapterInterface.
        self._lookup_efd_to_adapter: dict[int, int] = {}
        self._load_efd_to_adapter: dict[int, int] = {}
        for adapter_id, adapter in self._l2_adapters.items():
            self._lookup_efd_to_adapter[adapter.get_lookup_and_lock_event_fd()] = (
                adapter_id
            )
            self._load_efd_to_adapter[adapter.get_load_event_fd()] = adapter_id

        self._event_bus = get_event_bus()

        PrefetchController._gauge_target = self
        if not PrefetchController._gauges_registered:
            PrefetchController._gauges_registered = True
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.num_inflight_l2_loads",
                "L2 -> L1 prefetch load tasks currently executing, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_loads_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.inflight_load_memory_usage_bytes",
                "L1 bytes reserved by in-flight L2 -> L1 prefetch loads, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_load_bytes_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.l2_prefetch_adapters",
                (
                    "Count of L2 adapters attached to the prefetch controller, "
                    "tagged by ``state`` (active or draining)."
                ),
                lambda: (
                    PrefetchController._gauge_target.get_adapter_state_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )

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
        spec: PrefetchTaskSpec,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request.

        Thread-safe. Can be called from any thread.

        Args:
            spec: The prefetch request inputs (see :class:`PrefetchTaskSpec`).

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        # TODO: add to submission queue
        # TODO: implement L1-only fast path
        raise NotImplementedError("submit_prefetch_request is not implemented yet")

    def query_lookup_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the keys that are found during the lookup for a specific request.

        Thread-safe. Returns the prefix-hit count if the lookup phase
        has completed, None if still in progress, or the prefetch request
        has already been consumed by query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits from the lookup phase, or None if not yet complete
            or if the request has already been consumed by a previous call to this
            method.

        Note:
            This function does not pop the result. The caller need to make sure to call
            the query_prefetch_result after calling this function, otherwise nobody
            will clean up the completed lookups dictionary, causing memory leak.
        """
        raise NotImplementedError("query_lookup_result is not implemented yet")

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> Bitmap | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the retained-key bitmap if the request
        has completed, None if still in progress. Each result can only
        be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits, or None if not yet complete.

        Note:
            This function will pop the completed lookup results as well.
            Therefore, the caller need to make sure that never call
            query_lookup_result after calling this function, otherwise it will
            get None forever.
        """
        raise NotImplementedError("query_prefetch_result is not implemented yet")

    def wait_prefetch_result(
        self, request_id: PrefetchRequestId, timeout: float
    ) -> bool:
        """
        Block until a prefetch request's result is published, or until timeout.

        Thread-safe. Lets a handler wait for prefetch completion instead of
        busy-polling query_prefetch_result. Does not consume the result; the
        caller still retrieves it via query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.
            timeout: Maximum number of seconds to wait for the result.

        Returns:
            True if the result became available within the timeout, False if
            the wait timed out.
        """
        raise NotImplementedError("wait_prefetch_result is not implemented yet")

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._prefetch_results_lock:
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
            "num_active_adapters": len(self._l2_adapters) - len(self._draining),
            "num_draining_adapters": len(self._draining),
        }

    def get_adapter_state_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        raise NotImplementedError(
            "get_adapter_state_observations is not implemented yet"
        )

    def get_inflight_loads_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        raise NotImplementedError(
            "get_inflight_loads_observations is not implemented yet"
        )

    def get_inflight_load_bytes_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        raise NotImplementedError(
            "get_inflight_load_bytes_observations is not implemented yet"
        )

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
        # TODO: update this if needed
        self._stop_flag.set()
        self._submission_efd.notify()
        self._thread.join()
        self._cleanup_in_flight_requests()
        self._submission_efd.close()
        self._adapter_ctrl_efd.close()

    def add_adapter(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: AdapterDescriptor,
    ) -> None:
        """Blocking function to add a new adapter into the prefetch
        controller with the specified adapter ID and descriptor.

        Args:
            adapter_id: Stable id assigned by the StorageManager.
            adapter: The adapter instance to attach.
            descriptor: The adapter's descriptor (``descriptor.index`` must
                equal ``adapter_id``).

        Raises:
            RuntimeError: If the background loop did not apply the op in
                time (e.g. the loop is not running).
        """
        # TODO: update this if needed
        op = AddAdapterOp(
            adapter_id=adapter_id,
            adapter=adapter,
            descriptor=descriptor,
            done=threading.Event(),
        )
        with self._adapter_ops_lock:
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        if not op.done.wait(timeout=PREFETCH_LOOP_POLL_TIMEOUT_MS / 1000 + 5.0):
            raise RuntimeError(
                f"PrefetchController did not attach adapter {adapter_id} in time"
            )

    def request_remove_adapter(self, adapter_id: int) -> threading.Event:
        """Non-blocking function to request the removal of a L2 adapter
        specified by the adapter ID.

        New lookups stop routing to the adapter immediately; in-flight
        requests are allowed to complete.

        Args:
            adapter_id: Stable id of the adapter to drain.

        Returns:
            An Event signaled when the adapter is fully drained.
        """
        # TODO: add this if needed
        op = RemoveAdapterOp(adapter_id=adapter_id, done=threading.Event())
        with self._adapter_ops_lock:
            self._pending_adapter_ops.append(op)
        self._adapter_ctrl_efd.notify()
        return op.done

    # =========================================================================
    # Background main loop
    # =========================================================================

    def _prefetch_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The submission eventfd (new prefetch requests).
        - Each L2 adapter's lookup eventfd (completed lookups).
        - Each L2 adapter's load eventfd (completed loads).
        """
        # TODO: revisit this function, most of it should be fine, but update if needed
        poller = select.poll()
        submission_fd = self._submission_efd.fileno()
        poller.register(submission_fd, select.POLLIN)
        poller.register(self._adapter_ctrl_efd.fileno(), select.POLLIN)
        for efd in self._lookup_efd_to_adapter:
            poller.register(efd, select.POLLIN)
        for efd in self._load_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            # First, apply runtime add/remove of the L2 adapters.
            self._apply_pending_adapter_ops(poller)

            ready = poller.poll(PREFETCH_LOOP_POLL_TIMEOUT_MS)

            signaled_adapters: dict[PrefetchPhase, set[int]] = {
                phase: set() for phase in PrefetchPhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == submission_fd:
                        self._drain_submission_queue()
                    elif fd in self._lookup_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.LOOKUP].add(
                            self._lookup_efd_to_adapter[fd]
                        )
                    elif fd in self._load_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.PLAN_AND_LOAD].add(
                            self._load_efd_to_adapter[fd]
                        )
                except Exception:
                    logger.exception(
                        "Unexpected error in prefetch loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                for request in list(self._in_flight_requests.values()):
                    try:
                        self._advance_request(request, signaled_adapters)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight prefetch request %d",
                            request.request_id,
                        )

            try:
                self._start_pending_requests()
            except Exception:
                logger.exception(
                    "Unexpected error in prefetch loop while starting pending requests"
                )

            # Finalize any draining adapter no longer have any in-flight
            # requests.
            self._finalize_drained_adapters(poller)

    def _advance_request(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: dict[PrefetchPhase, set[int]],
    ) -> None:
        """State-transition dispatcher by phase: poll signaled adapters for
        the request's current phase via the per-phase helper, then trigger
        the phase transition when done."""
        phase_adapters = signaled_adapters[request.phase]
        if not phase_adapters:
            return
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if request.all_loads_done():
                self._finish_request(request)

    # =========================================================================
    # Dynamic adapter add/remove ops
    # =========================================================================

    def _apply_pending_adapter_ops(self, poller: "select.poll") -> None:
        """Apply queued add/remove ops on the prefetch loop thread."""
        # TODO: update if needed
        with self._adapter_ops_lock:
            ops = self._pending_adapter_ops
            self._pending_adapter_ops = []
        for op in ops:
            if isinstance(op, AddAdapterOp):
                self._l2_adapters[op.adapter_id] = op.adapter
                self._adapter_descriptors[op.adapter_id] = op.descriptor
                lookup_efd = op.adapter.get_lookup_and_lock_event_fd()
                load_efd = op.adapter.get_load_event_fd()
                self._lookup_efd_to_adapter[lookup_efd] = op.adapter_id
                self._load_efd_to_adapter[load_efd] = op.adapter_id
                poller.register(lookup_efd, select.POLLIN)
                poller.register(load_efd, select.POLLIN)
                logger.info("PrefetchController attached adapter %d", op.adapter_id)
                op.done.set()
            elif isinstance(op, RemoveAdapterOp):
                if op.adapter_id not in self._l2_adapters:
                    op.done.set()
                    continue
                # Mark draining; new lookups skip it. The adapter stays
                # registered so in-flight requests can still complete.
                self._draining[op.adapter_id] = op.done
                logger.info(
                    "PrefetchController draining adapter %d (no new lookups routed)",
                    op.adapter_id,
                )

    def _adapter_in_use(self, adapter_id: int) -> bool:
        """True if any in-flight request still references ``adapter_id``."""
        # TODO: update if needed
        for request in self._in_flight_requests.values():
            if (
                adapter_id in request.pending_lookup_tasks
                or adapter_id in request.pending_load_tasks
                or adapter_id in request.load_plan
                or adapter_id in request.lookup_results
            ):
                return True
        return False

    def _finalize_drained_adapters(self, poller: "select.poll") -> None:
        """Detach draining adapters no longer referenced by any request."""
        # TODO: update if needed
        for adapter_id in list(self._draining):
            if self._adapter_in_use(adapter_id):
                continue
            adapter = self._l2_adapters.pop(adapter_id)
            self._adapter_descriptors.pop(adapter_id, None)
            lookup_efd = adapter.get_lookup_and_lock_event_fd()
            load_efd = adapter.get_load_event_fd()
            self._lookup_efd_to_adapter.pop(lookup_efd, None)
            self._load_efd_to_adapter.pop(load_efd, None)
            for efd in (lookup_efd, load_efd):
                try:
                    poller.unregister(efd)
                except (KeyError, OSError):
                    pass
            done = self._draining.pop(adapter_id)
            logger.info("PrefetchController detached adapter %d", adapter_id)
            done.set()

    # =========================================================================
    # Submission phase
    # =========================================================================

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
        # TODO: implement the dynamic in flight request admission
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, spec = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, spec)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        spec: PrefetchTaskSpec,
    ) -> None:
        # TODO: implement this
        raise NotImplementedError

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters."""
        raise NotImplementedError
        # for adapter_idx in list(request.pending_lookup_tasks):
        #    if adapter_idx not in signaled_adapters:
        #        continue
        #    task_id = request.pending_lookup_tasks[adapter_idx]
        #    result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
        #        task_id
        #    )
        #    if result is None:
        #        continue
        #    request.lookup_results[adapter_idx] = result
        #    request.l2_adapter2readlocks[adapter_idx] = result
        #    del request.pending_lookup_tasks[adapter_idx]

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Compute the L1 ∪ L2 load plan, reserve L1 buffers, and submit
        load tasks."""
        raise NotImplementedError("transition_to_load_phase is not implemented yet")

    def _update_lookup_results(
        self, request_id: PrefetchRequestId, prefix_hit_count: int
    ) -> None:
        """Store the prefix-hit count from the lookup phase."""
        with self._lookup_results_lock:
            self._completed_lookups[request_id] = prefix_hit_count

    def _report_lookup_hit(
        self, request: InFlightPrefetchRequest, prefix_hit_count: int
    ) -> None:
        """Store the lookup-phase hit and publish its completion event."""
        request.hit_reported = True
        self._update_lookup_results(request.request_id, prefix_hit_count)
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "prefix_hit_count": prefix_hit_count,
                },
            )
        )

    def _poll_load_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending load results from signaled adapters."""
        for adapter_idx in list(request.pending_load_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_load_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_load_result(task_id)
            if result is None:
                continue
            request.load_results[adapter_idx] = result
            del request.pending_load_tasks[adapter_idx]
            request.load_bytes_by_adapter.pop(adapter_idx, None)

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                    },
                )
            )

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _release_l2_locks(
        self, request: InFlightPrefetchRequest, keep: dict[int, Bitmap]
    ) -> None:
        # TODO: re-implement this function
        # TODO: whether `keep` is needed or not? Maybe this function need a
        # complete rewrite
        pass

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _finish_request(self, request: InFlightPrefetchRequest) -> None:
        """Finish a request and free all unnecessary locks.

        The single completion path, with or without an L2 load (with an
        empty load plan the load steps degenerate to no-ops and every
        write-reserved buffer is returned).

        Workflow:

        1. Collect per-adapter load results; split loaded vs failed keys.
        2. Return every L2 read lock still held.
        3. Loaded keys become read-locked for the retriever (WARM:
           unlocked); failed keys' buffers are deleted.
        4. Fold loaded ∪ locked keys to the final hit length / retained set.
        5. Unlock everything outside the retained set (e.g. out of the
           final sliding window).
        6. Report the hit if no earlier step did, then the retained bitmap.

        End state (sliding-window view; loaded keys in the in L2-hit sw
        segment, L1 locks elsewhere)::

            |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                          ^ L1 hit length               ^ L1+L2 hit
            |     unlock     |   unlock   |     unlock     |   locked   |   unlock   |
        """
        # TODO: re-visit the docstring
        # TODO: reimplement this, also generate the correct events according to
        # the old implementation
        pass

    def _complete_request(self, request_id: PrefetchRequestId, result: Bitmap) -> None:
        """Store the retained-key bitmap and remove from in-flight tracking."""
        # TODO: revisit this function to see if it needs to be updated

        with self._prefetch_results_lock:
            self._completed_results[request_id] = result
            # Wake any WAIT_PREFETCH_STATUS handler blocked on this result.
            self._prefetch_results_cv.notify_all()
        removed = self._in_flight_requests.pop(request_id, None)
        if removed is not None:
            self._status_in_flight_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug(
            "Prefetch request %d completed: %d retained keys",
            request_id,
            result.popcount(),
        )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        # TODO: should be okay, but need to double check this function
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write_and_delete(
                        request.write_reserved_keys,
                        tag=_get_prefetch_write_tag(request.request_id),
                    )
            self._release_l2_locks(request, keep={})
            if request.l1_readlocks.popcount() > 0:
                l1_mgr.finish_read(
                    request.l1_readlocks.gather(request.keys),
                    read_locks=request.num_kv_readers,
                )
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
