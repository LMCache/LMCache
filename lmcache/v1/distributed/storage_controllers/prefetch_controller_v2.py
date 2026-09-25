# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

A prefetch request names its objects as a grid: one row per key group (an
object group on one kv rank) and one column per chunk. The controller locks
what L1 already holds, asks every live L2 adapter what it holds, lets the
prefetch policy decide which cells to serve from L1 and which to load from
L2, reserves L1 staging buffers for the loads, and admits each adapter's
loaded objects into L1 as its load task completes. The result is one bitmap
per key group marking the cells resident in L1 when the request finished.

Every lock the controller holds on behalf of a request is recorded in the
request's key states, and every phase leaves those states consistent, so
releasing a request at any point is a walk over its maps.
"""

# Standard
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import enum
import itertools
import select
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    FetchingPolicy,
    GroupedObjectKeys,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.adapter_lifecycle import (
    AddAdapterOp,
    RemoveAdapterOp,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy_v2 import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    L2AdapterDescriptor,
    MapState,
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
    from lmcache.v1.memory_management import MemoryObj

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


def _gather_keys(
    key_groups: list[GroupedObjectKeys],
    bitmaps: Bitmap2D,
) -> list[ObjectKey]:
    """Gather the keys from a grouped list to a flat list based on the
    given bitmap grid.

    Args:
        key_groups: The grouped object keys.
        bitmaps: The grid indicating which keys to gather. Row i corresponds
            to the key group at index i and is as wide as that group's keys.

    Returns:
        A flat list of object keys that are selected by the bitmaps.

    Raises:
        ValueError: If the length of bitmaps does not match the length of
            key_groups, or if any bitmap's length does not match the number
            of keys in the corresponding key group.
    """
    ret = []
    if len(key_groups) != len(bitmaps):
        raise ValueError(
            f"Length of bitmaps {len(bitmaps)} does not match length of "
            f"key_groups {len(key_groups)}"
        )

    for group, bitmap in zip(key_groups, bitmaps, strict=False):
        if len(bitmap) != len(group.keys):
            raise ValueError(
                f"Bitmap length {len(bitmap)} does not match number of keys "
                f"in group {len(group.keys)}"
            )
        ret.extend(bitmap.gather(group.keys))
    return ret


def _scatter_bitmaps(
    global_indices: Bitmap2D,
    local_bitmap: Bitmap,
) -> Bitmap2D:
    """Scatter the 1s in the local bitmap back to a global bitmap
    based on their origin position in the global indices.

    Args:
        global_indices: The global grid of the keys in the local bitmap.
            The i-th 1 in the global grid (row-major) corresponds to the
            i-th bit in the local bitmap.
        local_bitmap: The local bitmap result to scatter.

    Returns:
        The global bitmap after scattering the 1s from the local bitmap.

    Raises:
        ValueError: If the length of local_bitmap does not match the total
            number of 1s in global_indices.

    Example:
        global bitmap: [[0, 1, 0, 1, 0], [1, 0, 0, 1, 0]] (4 1s in total)
        local bitmap: [1, 0, 1, 0] (4 bits)
        return value: [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]] (0 and 2 are set to 1)
    """
    rows, cols = global_indices.size()
    if rows == 0:
        return Bitmap2D([])

    selected_indices: list[int] = []
    for i, row in enumerate(global_indices):
        offset = i * cols
        selected_indices.extend([i + offset for i in row.get_indices_list()])

    if len(selected_indices) != len(local_bitmap):
        raise ValueError(
            "Length of local_bitmap does not match the total number "
            "of 1s in global_indices"
        )

    indices_to_scatter = local_bitmap.gather(selected_indices)

    ret = Bitmap2D.zeros(rows, cols)
    for idx in indices_to_scatter:
        row = idx // cols
        col = idx % cols
        ret[row].set(col)

    return ret


def _scatter_bitmaps_full_global(
    local_bitmap: Bitmap,
    num_rows: int,
    num_cols: int,
) -> Bitmap2D:
    """Scatter the local bitmap to global assuming the original global bitmap
    is full.

    Args:
        local_bitmap: The local bitmap result to scatter.
        num_rows: The number of rows in the global bitmap.
        num_cols: The number of columns in the global bitmap.

    Returns:
        The global bitmaps after scattering the 1s from the local bitmap.

    Raises:
        ValueError: If the length of local_bitmap does not match the total
        number of bits in the global bitmap (num_rows * num_cols).
    """
    if len(local_bitmap) != num_rows * num_cols:
        raise ValueError(
            "Length of local_bitmap does not match the total number "
            "of bits in the global bitmap"
        )

    ret = Bitmap2D.zeros(num_rows, num_cols)
    for idx in local_bitmap.get_indices_list():
        row = idx // num_cols
        col = idx % num_cols
        ret[row].set(col)

    return ret


def _reserve_l1_cells(
    l1_manager: L1Manager,
    key_groups: list[GroupedObjectKeys],
    cells: Bitmap2D,
    retain: dict[ObjectKey, bool],
    tag: str,
) -> tuple[Bitmap2D, dict[ObjectKey, "MemoryObj"], int]:
    """Reserve a staging buffer in one L1 manager for every cell in ``cells``.

    Each row is reserved with its own key group's layout.

    Args:
        l1_manager: The manager to reserve in.
        key_groups: The request's rows; row ``i`` of ``cells`` selects keys
            from ``key_groups[i]``.
        cells: The cells to reserve.
        retain: Whether each key in ``cells`` stays resident after the
            reader is done.
        tag: The writer tag for the reservations.

    Returns:
        The tuple of (L1 reserve result, reserved objects, failed count).
        The failed count is the number of cells that were not reserved for
        any reason.
    """
    success = cells.zeros_like()
    objs: dict[ObjectKey, "MemoryObj"] = {}
    failed_count = 0
    oom_keys: list[ObjectKey] = []
    contended_keys: list[ObjectKey] = []
    for row_id, (group, row) in enumerate(zip(key_groups, cells, strict=True)):
        cols = row.get_indices_list()
        if not cols:
            continue
        keys = row.gather(group.keys)
        results = l1_manager.reserve_write(
            keys=keys,
            is_temporary=[not retain[key] for key in keys],
            layout_desc=group.layout_desc,
            tag=tag,
        )
        for col, key in zip(cols, keys, strict=True):
            error, mem_obj = results[key]
            if error != L1Error.SUCCESS or mem_obj is None:
                failed_count += 1
                if error == L1Error.OUT_OF_MEMORY:
                    oom_keys.append(key)
                elif error == L1Error.KEY_NOT_WRITABLE:
                    contended_keys.append(key)
                logger.debug(
                    "reserve_write failed for %s in row %d: %s", key, row_id, error
                )
                continue
            success[row_id].set(col)
            objs[key] = mem_obj

    event_bus = get_event_bus()
    if oom_keys:
        logger.warning(
            "%s: %d L1 reservations failed with out-of-memory; "
            "L1 may be under memory pressure",
            tag,
            len(oom_keys),
        )
        event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_oom", "keys": oom_keys},
            )
        )
    if contended_keys:
        # The key became resident in L1 after the lock pass; the caller
        # falls back to the L1-only hit.
        event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_contended", "keys": contended_keys},
            )
        )
    return success, objs, failed_count


def _build_request(
    request_id: PrefetchRequestId, spec: PrefetchTaskSpec
) -> "InFlightPrefetchRequest":
    """Create the in-flight request for ``spec`` in the lookup phase."""
    return InFlightPrefetchRequest(
        request_id=request_id,
        key_groups=spec.key_groups,
        fetching_policy=spec.fetching_policy,
        lock_mode=spec.lock_mode,
        num_kv_readers=spec.num_kv_readers,
    )


# Class definitions


@dataclass
class PrefetchKeyState:
    """Tracks the states of the keys in a prefetch task"""

    # Keys that read-locked in L1 (i.e. finished prefetch)
    # Mapping from l1 manager index to the global bitmap
    l1_locked_keys: MapState = field(default_factory=MapState)

    # Keys that read-locked in L2 (i.e. finished L2 lookup and lock)
    # Mapping from l2 adapter index to the global bitmap
    l2_locked_keys: MapState = field(default_factory=MapState)

    # Keys that are write-locked in L1 (for L2 to load)
    # Mapping from l1 manager index to the global bitmap
    l1_reserved_keys: MapState = field(default_factory=MapState)


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    # Basic (immutable fields)
    request_id: PrefetchRequestId
    key_groups: list[GroupedObjectKeys]

    # Policy related fields (immutable)
    fetching_policy: FetchingPolicy
    lock_mode: PrefetchLockMode
    num_kv_readers: int

    # The locked and reserved keys during the prefetch lifecycle.
    key_states: PrefetchKeyState = field(default_factory=PrefetchKeyState)

    # In flight L2 operations: L2 adapter idx -> L2 task ID
    inflight_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    inflight_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)

    # L2 adapter idx -> L1 bytes reserved for that adapter's in-flight load.
    # Entries are removed as load results arrive.
    load_bytes_by_adapter: dict[int, int] = field(default_factory=dict)

    # private fields
    _flattened_keys: list[ObjectKey] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Populate the flattened key list
        if self.key_groups:
            self._flattened_keys = list(
                itertools.chain.from_iterable(group.keys for group in self.key_groups)
            )
        else:
            raise ValueError("key_groups must be provided and cannot be empty")

    def get_flattened_keys(self) -> list[ObjectKey]:
        """Return a flattened list of all keys in the request"""
        return self._flattened_keys

    @property
    def phase(self) -> PrefetchPhase:
        """The request's phase: lookup while any L2 lookup task is
        outstanding, plan-and-load afterwards."""
        if self.inflight_lookup_tasks:
            return PrefetchPhase.LOOKUP
        return PrefetchPhase.PLAN_AND_LOAD

    def all_lookups_done(self) -> bool:
        """Return whether every submitted L2 lookup task has reported."""
        return not self.inflight_lookup_tasks

    def all_loads_done(self) -> bool:
        """Return whether every submitted L2 load task has reported."""
        return not self.inflight_load_tasks


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
       An L1-only request (``skip_l2`` or no L2 adapter) is served on the
       calling thread and is complete when the call returns.
    2. Read-locks the keys resident in L1 and submits lookup-and-lock tasks
       to every live L2 adapter.
    3. Plans which cells to serve from L1 and which to load from L2, and
       releases every lock outside the plan.
    4. Reserves L1 staging buffers for the L2 plan; on reservation failures
       it drops the affected L2 locks and re-plans on what is reserved.
    5. Submits one load task per L2 adapter.
    6. As each load task completes, admits the loaded objects into L1 as
       read-locked, deletes the buffers of failed loads, and returns that
       adapter's L2 locks.
    7. Folds the L1 locked cells into the final hit, releases the rest, and
       publishes one bitmap per key group via query_prefetch_result.

    Args:
        l1_managers: The L1 manager instances.
        l1_manager_descriptors: Descriptors for each L1 manager (same order).
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan and retention decisions.
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
        l1_managers: list[L1Manager],
        l1_manager_descriptors: list[L1ManagerDescriptor],
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[L2AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
    ) -> None:
        self._l1_managers: dict[int, L1Manager] = {
            desc.index: mgr
            for desc, mgr in zip(l1_manager_descriptors, l1_managers, strict=True)
        }
        self._l1_manager_descriptors: dict[int, L1ManagerDescriptor] = {
            desc.index: desc for desc in l1_manager_descriptors
        }

        self._l2_adapters: dict[int, L2AdapterInterface] = {
            desc.index: adapter
            for desc, adapter in zip(adapter_descriptors, l2_adapters, strict=True)
        }
        self._adapter_descriptors: dict[int, L2AdapterDescriptor] = {
            desc.index: desc for desc in adapter_descriptors
        }
        self._policy = policy

        # TODO: remove max_in_flight and make it dynamic
        self._max_in_flight = max_in_flight
        if len(self._l1_managers) != 1:
            logger.error(
                "PrefetchController supports exactly one L1 manager for now; "
                "got %d. L2 loads land in the first one.",
                len(self._l1_managers),
            )

        # Adapters being removed: adapter id -> event set once detached.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._adapter_ctrl_efd = create_event_notifier()

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: deque[tuple[PrefetchRequestId, PrefetchTaskSpec]] = deque()

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[tuple[PrefetchRequestId, PrefetchTaskSpec]] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = create_event_notifier()

        # Thread-safe prefetch results (background -> external).  The condition
        # variable lets a WAIT_PREFETCH_STATUS handler block until a result is
        # published instead of busy-polling QUERY_PREFETCH_STATUS.
        self._prefetch_results_lock = threading.Lock()
        self._prefetch_results_cv = threading.Condition(self._prefetch_results_lock)
        self._completed_results: dict[PrefetchRequestId, Bitmap2D] = {}

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
        skip_l2: bool = False,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request.

        Thread-safe. Can be called from any thread.

        Args:
            spec: The prefetch request inputs (see :class:`PrefetchTaskSpec`).
            skip_l2: If True, the request will only prefetch from L1 and
                skip the L2 lookup and load phases.

        Returns:
            A request ID for tracking via query_prefetch_result.

        Note:
            When ``skip_l2`` is true, or no L2 adapter is attached, the
            request is served from L1 on the calling thread and its result
            is queryable via query_prefetch_result once this returns.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1

        if skip_l2 or not self._l2_adapters:
            request = _build_request(request_id, spec)
            self._lock_l1_keys(request)
            self._plan_load(request, {})
            self._finish_request(request)
            return request_id

        with self._submission_lock:
            self._submission_queue.append((request_id, spec))
            self._submission_efd.notify()
        return request_id

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> Bitmap2D | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the hit cells, one row per key group, if the
        request has completed, None if still in progress. Each result can
        only be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            One bitmap per key group of the request; a set bit marks a key
            that is resident in L1 (and read-locked under ``LOCK``). None if
            not yet complete or already retrieved.
        """
        with self._prefetch_results_lock:
            return self._completed_results.pop(request_id, None)

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
        with self._prefetch_results_cv:
            return self._prefetch_results_cv.wait_for(
                lambda: request_id in self._completed_results, timeout=timeout
            )

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._prefetch_results_lock:
            completed_results_count = len(self._completed_results)
        # Snapshot the loop-owned table; reading a copy is safe from here.
        in_flight = list(self._in_flight_requests.copy().values())
        lookup_phase_count = sum(
            1 for request in in_flight if request.phase == PrefetchPhase.LOOKUP
        )
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "max_in_flight": self._max_in_flight,
            "submission_queue_size": submission_queue_size,
            "pending_queue_size": len(self._pending_queue),
            "in_flight_request_count": len(in_flight),
            "lookup_phase_count": lookup_phase_count,
            "load_phase_count": len(in_flight) - lookup_phase_count,
            "completed_results_count": completed_results_count,
            "num_l2_adapters": len(self._l2_adapters),
            "num_active_adapters": len(self._l2_adapters) - len(self._draining),
            "num_draining_adapters": len(self._draining),
        }

    def get_adapter_state_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Return ``(count, {"state": ...})`` pairs for active and draining
        adapters.

        Note:
            Reads only ``len()`` of two dicts, which is safe from the OTel
            reader thread without locking.
        """
        num_draining = len(self._draining)
        return [
            (len(self._l2_adapters) - num_draining, {"state": "active"}),
            (num_draining, {"state": "draining"}),
        ]

    def _snapshot_inflight_loads(self) -> dict[int, tuple[int, int]]:
        """Return ``{adapter_idx: (task_count, reserved_bytes)}`` for the L2
        load tasks currently executing.

        Note:
            Uses ``dict.copy()`` snapshots so the OTel reader thread can call
            this concurrently with the prefetch loop without locking.
        """
        counts: dict[int, int] = defaultdict(int)
        bytes_by_adapter: dict[int, int] = defaultdict(int)
        for request in self._in_flight_requests.copy().values():
            for idx, reserved in request.load_bytes_by_adapter.copy().items():
                counts[idx] += 1
                bytes_by_adapter[idx] += reserved
        return {idx: (counts[idx], bytes_by_adapter[idx]) for idx in counts}

    def get_inflight_loads_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Return per-adapter ``(task_count, attributes)`` pairs for the
        in-flight L2 load tasks."""
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, (count, _) in self._snapshot_inflight_loads().items():
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (count, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

    def get_inflight_load_bytes_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Return per-adapter ``(reserved_bytes, attributes)`` pairs for the
        L1 bytes held by in-flight L2 load tasks."""
        observations: list[tuple[int | float, dict[str, object]]] = []
        for idx, (_, reserved_bytes) in self._snapshot_inflight_loads().items():
            desc = self._adapter_descriptors.get(idx)
            if desc is None:
                continue
            observations.append(
                (reserved_bytes, {"l2_name": desc.type_name, "adapter_index": idx})
            )
        return observations

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
        self._submission_efd.notify()
        self._thread.join()
        self._cleanup_in_flight_requests()
        self._submission_efd.close()
        self._adapter_ctrl_efd.close()

    def add_adapter(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: L2AdapterDescriptor,
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
                            "Unexpected error advancing in-flight prefetch request "
                            "%d; aborting it",
                            request.request_id,
                        )
                        self._abort_request(request)

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

        # A request leaves the load phase once no load task is outstanding,
        # whether the plan was empty or every adapter has been admitted.
        if request.phase == PrefetchPhase.PLAN_AND_LOAD and request.all_loads_done():
            self._finish_request(request)
            self._retire_request(request)

    # =========================================================================
    # Dynamic adapter add/remove ops
    # =========================================================================

    def _apply_pending_adapter_ops(self, poller: "select.poll") -> None:
        """Apply queued add/remove ops on the prefetch loop thread."""
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
        """True if any in-flight request still references ``adapter_id``.

        Note:
            Membership in ``l2_locked_keys`` keeps an adapter attached until
            the request has returned its L2 read locks; the entry is removed
            when the adapter's load result is admitted.
        """
        for request in self._in_flight_requests.values():
            if (
                adapter_id in request.inflight_lookup_tasks
                or adapter_id in request.inflight_load_tasks
                or adapter_id in request.key_states.l2_locked_keys
            ):
                return True
        return False

    def _finalize_drained_adapters(self, poller: "select.poll") -> None:
        """Detach draining adapters no longer referenced by any request."""
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

    def _start_pending_requests(self) -> None:
        """Start pending requests up to the max in-flight limit."""
        # TODO: implement the dynamic in flight request admission
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, spec = self._pending_queue.popleft()
            self._start_lookup_phase(request_id, spec)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        spec: PrefetchTaskSpec,
    ) -> None:
        """Create the in-flight prefetch request and start the lookup phase"""
        request = _build_request(request_id, spec)
        flattened_keys = request.get_flattened_keys()
        self._lock_l1_keys(request)

        # Submit lookup requests to L2 adapters that are not draining.
        active_adapters = {
            adapter_idx: adapter
            for adapter_idx, adapter in self._l2_adapters.items()
            if adapter_idx not in self._draining
        }

        # TODO(ApostaC): Here, we submit the full keys to L2 adapters.
        # In the future, we can do `All keys - L1 hit keys` to reduce the
        # L2 lookup overhead.
        # However, for P2P case, this will require the storage manager to
        # support unequal length of key groups. We left this as future refactor
        # TODOs.
        for adapter_idx, adapter in active_adapters.items():
            task_id = adapter.submit_lookup_and_lock_task(
                flattened_keys, spec.group_layout_descs
            )
            request.inflight_lookup_tasks[adapter_idx] = task_id

        # No live L2 adapter: serve from L1 alone, on this thread.
        if not active_adapters:
            self._plan_load(request, {})
            self._finish_request(request)
            return

        # Add the inflight request to the tracking dict and publish the events
        self._in_flight_requests[request_id] = request

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(flattened_keys),
                    "adapter_count": len(request.inflight_lookup_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in flattened_keys),
                },
            )
        )

    def _lock_l1_keys(self, request: InFlightPrefetchRequest) -> None:
        """Read-lock every key of the request that is resident in an L1
        manager and record the hits in ``l1_locked_keys``.

        Args:
            request: The request whose keys are looked up.

        Note:
            Locks are taken regardless of the lock mode; under ``NO_LOCK``
            they are released when the request finishes.
        """
        flattened_keys = request.get_flattened_keys()
        num_rows = len(request.key_groups)
        num_cols = len(request.key_groups[0].keys)
        for l1_idx, l1_manager in self._l1_managers.items():
            result = l1_manager.reserve_read(flattened_keys, request.num_kv_readers)
            res_bitmap = Bitmap(len(flattened_keys))
            for i, key in enumerate(flattened_keys):
                error, _obj = result[key]
                if error != L1Error.SUCCESS:
                    continue
                res_bitmap.set(i)
            request.key_states.l1_locked_keys[l1_idx] = _scatter_bitmaps_full_global(
                res_bitmap, num_rows, num_cols
            )

    def _plan_load(
        self,
        request: InFlightPrefetchRequest,
        active_l2_descs: dict[int, L2AdapterDescriptor],
    ) -> MapState:
        """Plan on the current locked maps, release every lock outside the
        plan, and set the locked maps to the plan.

        Args:
            request: The request to plan; its locked maps are the input.
            active_l2_descs: The L2 adapters the plan may load from. An empty
                dict plans on L1 alone.

        Returns:
            The L2 cells the plan dropped, per adapter, with their L2 locks
            already returned.
        """
        states = request.key_states
        load_plan = self._policy.plan_load(
            key_groups=request.key_groups,
            l1_locked_keys=states.l1_locked_keys,
            l2_locked_keys=states.l2_locked_keys,
            l1_manager_descs=self._l1_manager_descriptors,
            l2_adapter_descs=active_l2_descs,
            fetching_policy=request.fetching_policy,
        )

        l1_dropped = states.l1_locked_keys - load_plan.l1_planned_keys
        l2_dropped = states.l2_locked_keys - load_plan.l2_planned_keys
        for l1_idx, cells in l1_dropped.items():
            keys = _gather_keys(request.key_groups, cells)
            if keys:
                self._l1_managers[l1_idx].finish_read(keys, request.num_kv_readers)
        for l2_idx, cells in l2_dropped.items():
            keys = _gather_keys(request.key_groups, cells)
            if keys:
                self._l2_adapters[l2_idx].submit_unlock(keys)

        states.l1_locked_keys = load_plan.l1_planned_keys.copy()
        states.l2_locked_keys = load_plan.l2_planned_keys.copy()
        return l2_dropped

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters
        and update the inflight lookup status and the lookup result
        (l2 locked key states).

        Args:
            request: The in-flight prefetch request to poll.
            signaled_adapters: Set of adapter indices that have signaled
                completion of lookup tasks.
        """
        num_rows = len(request.key_groups)
        num_cols = len(request.key_groups[0].keys)
        for adapter_idx in list(request.inflight_lookup_tasks):
            if adapter_idx not in signaled_adapters:
                continue

            task_id = request.inflight_lookup_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
                task_id
            )
            if result is None:
                continue

            # Update the l2 locked key states
            # NOTE: since we do full global bitmap for L2 load, we use
            # _scatter_bitmaps_full_global instead of _scatter_bitmaps here.
            # Will be changed in the future after we implemented partial L2
            # lookup.
            l2_found_bitmap = _scatter_bitmaps_full_global(result, num_rows, num_cols)
            request.key_states.l2_locked_keys[adapter_idx] = l2_found_bitmap

            # Remove the completed lookup task from inflight_lookup_tasks
            del request.inflight_lookup_tasks[adapter_idx]

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Plan the loads, reserve their L1 buffers, and submit the load
        tasks.

        On return the locked maps equal the plan, the reserved map equals the
        L2 plan, and one load task is in flight per adapter in it. With an
        empty L2 plan no task is submitted and the request is ready to
        finish.
        """
        states = request.key_states

        # Step 1 and 2: plan on the lookup results and release what the plan
        # leaves out. Draining L2 adapters are skipped.
        active_l2_descs = {
            l2_idx: desc
            for l2_idx, desc in self._adapter_descriptors.items()
            if l2_idx not in self._draining
        }
        self._plan_load(request, active_l2_descs)

        # Step 3: reserve L1 write buffers for the planned L2 keys
        # with L1-L2 affinity.
        tag = _get_prefetch_write_tag(request.request_id)
        l1_reserved_keys = MapState()
        reserved_objs: dict[ObjectKey, "MemoryObj"] = {}
        num_failed_reservations = 0

        cells_by_l1, retain = self._build_l1_allocation_plans(request)
        for l1_idx, cells in cells_by_l1.items():
            success, objs, l1_failed_count = _reserve_l1_cells(
                self._l1_managers[l1_idx], request.key_groups, cells, retain, tag
            )
            l1_reserved_keys[l1_idx] = success
            reserved_objs.update(objs)
            num_failed_reservations += l1_failed_count
        states.l1_reserved_keys = l1_reserved_keys

        # Steps 4 and 5 only matter when some reservation failed; otherwise the
        # L2 plan is fully reserved and the replan would return the same plan.
        if num_failed_reservations > 0:
            # Step 4: drop the L2 locks whose L1 reservation failed.
            l2_locked_keys_new = MapState()
            for l2_idx, bitmap2d in states.l2_locked_keys.items():
                l1_idx = self._get_l2_affinity_manager(l2_idx)
                l2_locked_keys_new[l2_idx] = bitmap2d & l1_reserved_keys[l1_idx]

            l2_keys_to_unlock = states.l2_locked_keys - l2_locked_keys_new
            for l2_idx, bitmap2d in l2_keys_to_unlock.items():
                keys = _gather_keys(request.key_groups, bitmap2d)
                if keys:
                    self._l2_adapters[l2_idx].submit_unlock(keys)
            states.l2_locked_keys = l2_locked_keys_new

            # Step 5: re-plan on what is actually reserved. An L2 cell dropped
            #         by the replan also owns a staging buffer in its affinity
            #         L1 manager, which is deleted here.
            l2_dropped = self._plan_load(request, active_l2_descs)
            for l2_idx, bitmaps in l2_dropped.items():
                keys = _gather_keys(request.key_groups, bitmaps)
                if not keys:
                    continue
                l1_idx = self._get_l2_affinity_manager(l2_idx)
                self._l1_managers[l1_idx].finish_write_and_delete(keys, tag=tag)
                for key in keys:
                    reserved_objs.pop(key, None)
                states.l1_reserved_keys[l1_idx] -= bitmaps

        # Step 6: submit the load tasks to L2 adapters
        self._submit_load_tasks(request, reserved_objs)

    def _submit_load_tasks(
        self,
        request: InFlightPrefetchRequest,
        reserved_objs: dict[ObjectKey, "MemoryObj"],
    ) -> None:
        """Submit one load task per adapter in the request's L2 plan.

        Records the reserved bytes per adapter and publishes one
        ``L2_LOAD_TASK_SUBMITTED`` event per adapter plus one
        ``L2_PREFETCH_LOAD_SUBMITTED`` event for the batch.

        Args:
            request: The in-flight request being loaded.
            reserved_objs: The L1 staging buffer of every planned key.
        """
        plan_keys: list[ObjectKey] = []
        for l2_idx, cells in request.key_states.l2_locked_keys.items():
            keys = _gather_keys(request.key_groups, cells)
            if not keys:
                continue
            objects = [reserved_objs[key] for key in keys]
            task_id = self._l2_adapters[l2_idx].submit_load_task(keys, objects)
            request.inflight_load_tasks[l2_idx] = task_id
            plan_keys.extend(keys)

            # Groups may differ in object size, so sum per object.
            total_bytes = sum(obj.get_size() for obj in objects)
            request.load_bytes_by_adapter[l2_idx] = total_bytes
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": l2_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[l2_idx].type_name,
                        "key_count": len(keys),
                        "total_bytes": total_bytes,
                    },
                )
            )

        if not plan_keys:
            return
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": request.request_id,
                    "key_count": len(plan_keys),
                    "adapter_count": len(request.inflight_load_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in plan_keys),
                },
            )
        )
        logger.debug(
            "Prefetch request %d: submitted load tasks to %d adapters for %d keys",
            request.request_id,
            len(request.inflight_load_tasks),
            len(plan_keys),
        )

    def _build_l1_allocation_plans(
        self, request: InFlightPrefetchRequest
    ) -> tuple[dict[int, Bitmap2D], dict[ObjectKey, bool]]:
        """Group the request's L2 load plan by the L1 manager that will
        receive each adapter's keys, deciding retention along the way.

        Args:
            request: The in-flight request whose L2 plan is grouped.

        Returns:
            A tuple of allocation plan and retention plan. More specifically:
            - L1 Manager index -> the bitmap indicating which key to reserve write
            - Object key -> whether the key need to retain in L1 after L1-L0 retrieve

        Note:
            Under ``NO_LOCK`` every key is retained; otherwise the policy
            decides per (L1 manager, L2 adapter) pair.
        """
        cells_by_l1: dict[int, Bitmap2D] = {}
        retain: dict[ObjectKey, bool] = {}
        for l2_idx, planned in request.key_states.l2_locked_keys.items():
            l1_idx = self._get_l2_affinity_manager(l2_idx)
            keys = _gather_keys(request.key_groups, planned)
            if request.lock_mode == PrefetchLockMode.NO_LOCK:
                retention = [True] * len(keys)
            else:
                retention = self._policy.plan_l1_retention(
                    keys,
                    self._l1_manager_descriptors[l1_idx],
                    self._adapter_descriptors[l2_idx],
                )
            retain.update(zip(keys, retention, strict=True))
            if l1_idx in cells_by_l1:
                cells_by_l1[l1_idx] += planned
            else:
                cells_by_l1[l1_idx] = planned.copy()
        return cells_by_l1, retain

    def _get_l2_affinity_manager(self, l2_adapter_idx: int) -> int:
        """Get the L1 manager index that has affinity with the given L2 adapter index.

        Args:
            l2_adapter_idx: The index of the L2 adapter.

        Returns:
            The index of the L1 manager that has affinity with the given L2 adapter
            index.
        """
        # TODO: right now we only support one L1 manager. Update this after we have
        # multi-tier L1 support
        return next(iter(self._l1_managers))

    def _poll_load_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending load results from signaled adapters and admit each
        finished adapter's load into L1.

        For the finished loads, we will update the L1 object states accordingly.
        - Successfully loaded objects: become read-locked.
        - Failed objects: delete the staging buffers.

        Args:
            request: The in-flight prefetch request to poll.
            signaled_adapters: Set of adapter indices that have signaled
                completion of load tasks.
        """
        states = request.key_states
        tag = _get_prefetch_write_tag(request.request_id)
        for adapter_idx in list(request.inflight_load_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.inflight_load_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_load_result(task_id)
            if result is None:
                continue

            # The load task was submitted with the keys of ``planned`` in
            # row-major order, so ``result`` is a flat bitmap over them.
            planned = states.l2_locked_keys[adapter_idx]
            loaded = _scatter_bitmaps(planned, result)
            failed = planned - loaded
            l1_idx = self._get_l2_affinity_manager(adapter_idx)
            l1_manager = self._l1_managers[l1_idx]

            loaded_keys = _gather_keys(request.key_groups, loaded)
            failed_keys = _gather_keys(request.key_groups, failed)

            # Update L1 key status
            if loaded_keys:
                l1_manager.finish_write_and_reserve_read(
                    loaded_keys, read_locks=request.num_kv_readers, tag=tag
                )
            if failed_keys:
                l1_manager.finish_write_and_delete(failed_keys, tag=tag)

            # Unlock L2
            self._l2_adapters[adapter_idx].submit_unlock(
                _gather_keys(request.key_groups, planned)
            )

            # Update the key states and the inflight load tasks
            if l1_idx in states.l1_locked_keys:
                states.l1_locked_keys[l1_idx] += loaded
            else:
                states.l1_locked_keys[l1_idx] = loaded
            states.l1_reserved_keys[l1_idx] -= planned
            del states.l2_locked_keys[adapter_idx]
            del request.inflight_load_tasks[adapter_idx]
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
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "loaded_count": len(loaded_keys),
                        "failed_count": len(failed_keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in loaded_keys
                        ),
                    },
                )
            )
            if failed_keys:
                # Reserved in L1 for this adapter but absent from its load
                # result. Serde failures will get their own reason once
                # adapters can report them.
                self._event_bus.publish(
                    Event(
                        event_type=EventType.L2_PREFETCH_FAILED,
                        metadata={"reason": "not_found", "keys": failed_keys},
                    )
                )

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _finish_request(self, request: InFlightPrefetchRequest) -> None:
        """Settle the request's L1 locks and publish its result.

        For `prefix` loading, it will do fold-unfold. It will also touch the
        L1 keys so that the eviction module can be updated.

        Args:
            request: The request to finish. Its L2 locked map and reserved
                map hold nothing at this point.

        Note:
            Reads and writes only ``request.key_states`` and the result
            store.
        """
        states = request.key_states
        found = states.l1_locked_keys.merge()
        if len(found) == 0:
            found = Bitmap2D.zeros(
                len(request.key_groups), len(request.key_groups[0].keys)
            )

        windows = [group.sliding_window_size for group in request.key_groups]
        hit_length, retain_rows = fold_unfold_grouped(found.to_list(), windows)
        if request.fetching_policy == "prefix":
            hit_cells = Bitmap2D(retain_rows) & found
        else:
            hit_cells = found

        for l1_idx, locked in states.l1_locked_keys.items():
            if request.lock_mode == PrefetchLockMode.NO_LOCK:
                release = locked
            else:
                release = locked - hit_cells
            release_keys = _gather_keys(request.key_groups, release)
            if release_keys:
                self._l1_managers[l1_idx].finish_read(
                    release_keys, read_locks=request.num_kv_readers
                )

            # Notify the eviction module
            hit_keys = _gather_keys(request.key_groups, locked & hit_cells)
            if hit_keys:
                self._l1_managers[l1_idx].touch_keys(hit_keys)

        # TODO(ApostaC): the lookup hit is no longer reported separately;
        # remove this event and the ``prefetch_lookup_hit`` metric and log
        # handlers that consume it.
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "prefix_hit_count": hit_length,
                },
            )
        )
        self._publish_result(request, hit_cells)
        logger.debug(
            "Prefetch request %d completed: %d hit cells",
            request.request_id,
            hit_cells.popcount(),
        )

    def _publish_result(
        self, request: InFlightPrefetchRequest, hit_cells: Bitmap2D
    ) -> None:
        """Store the request's result and wake any waiter."""
        with self._prefetch_results_lock:
            self._completed_results[request.request_id] = hit_cells
            # Wake any WAIT_PREFETCH_STATUS handler blocked on this result.
            self._prefetch_results_cv.notify_all()

    def _abort_request(self, request: InFlightPrefetchRequest) -> None:
        """Give up on a request after an error: return everything it holds,
        report a miss for every cell, and drop it from the in-flight table."""
        self._release_all_locks(request)
        self._publish_result(
            request,
            Bitmap2D.zeros(len(request.key_groups), len(request.key_groups[0].keys)),
        )
        self._retire_request(request)

    def _retire_request(self, request: InFlightPrefetchRequest) -> None:
        """Remove a finished request from the in-flight table."""
        self._in_flight_requests.pop(request.request_id, None)

    def _release_all_locks(self, request: InFlightPrefetchRequest) -> None:
        """Return every lock and buffer the request's key states record.

        L2 read locks are returned, reserved L1 staging buffers are deleted,
        and L1 read locks are released. Outstanding adapter tasks are left
        to complete on their own.
        """
        states = request.key_states
        tag = _get_prefetch_write_tag(request.request_id)
        for l2_idx, cells in states.l2_locked_keys.items():
            keys = _gather_keys(request.key_groups, cells)
            if keys and l2_idx in self._l2_adapters:
                self._l2_adapters[l2_idx].submit_unlock(keys)
        for l1_idx, cells in states.l1_reserved_keys.items():
            keys = _gather_keys(request.key_groups, cells)
            if keys:
                self._l1_managers[l1_idx].finish_write_and_delete(keys, tag=tag)
        for l1_idx, cells in states.l1_locked_keys.items():
            keys = _gather_keys(request.key_groups, cells)
            if keys:
                self._l1_managers[l1_idx].finish_read(
                    keys, read_locks=request.num_kv_readers
                )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        for request in list(self._in_flight_requests.values()):
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.get_flattened_keys()),
            )
            self._release_all_locks(request)
            self._retire_request(request)
