# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Read-locks L1-resident keys so they extend the hit and cannot be
   evicted mid-request, then submits lookup_and_lock tasks to all L2 adapters.
3. Computes a load plan over the L1 ∪ L2 union, keeping the keys retained by
   the TrimPolicy (PREFIX, SEGMENTED_PREFIX, or SPARSE).
4. Reserves L1 write buffers and submits load tasks to L2 adapters.
5. On load completion, transitions L1 entries from write-locked to read-locked.
6. Reports the retained-key bitmap.

Every key counted toward the hit is lock-held from the moment it is
discovered (L1 read lock or L2 lookup lock) until the request completes —
there is never an observed-but-unlocked instant.

Key intervals, sliding-window (SW) view — for full attention every in-L1
key inside the hit is needed (no out-of-window segments); see
docs/design/v1/distributed/storage_controllers/prefetch_l1_lock_pass.md::

    SW-group keys, chunk order:

    |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                  ^ L1 hit length               ^ L1+L2 hit length

    out of L1-hit sw : in L1, behind the L1 hit's window — never needed again
    in L1-hit sw     : the window that makes the L1 hit servable
    out of L2-hit sw : between the L1 hit and the final (L1+L2) window
    in L2-hit sw     : the final window; ends at the L1+L2 hit length
    remaining        : past the L1+L2 hit

    Drawn in general position: the two windows can touch, overlap, or
    coincide (L2 may extend the hit by less than a window, or not at all).
    Segments may then be empty or overlap; the rightmost applicable
    segment's action wins.

Each step in the load phase repeats this figure with its per-segment
actions aligned below it.
Vocabulary: lock = take an L1 read lock; unlock = return the read lock;
loading = L1 write reservation carrying an L2 lookup lock. Locking never
refreshes eviction recency: _finish_request explicitly touches the
retained keys (L1Manager.touch_keys), the ones the request actually
serves.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Iterable
import enum
import math
import select
import threading
import time

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_ranked
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
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
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


# HELPER FUNCTIONS
def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge bitmaps with a bitwise OR into a ``num_keys``-sized bitmap.

    Always returns a ``num_keys``-sized bitmap (empty input -> all zeros), so
    downstream ``&`` operations never hit a size mismatch.
    """
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


def build_trim_mask(
    found: Bitmap,
    num_keys: int,
    policy: TrimPolicy = TrimPolicy.PREFIX,
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
) -> tuple[int, Bitmap]:
    """Subset of ``found`` to keep (load + read-lock + report); the rest is
    released.

    PREFIX trims at the first gap (leading contiguous run). The non-PREFIX
    policies keep every set bit, gaps included, and differ only in intent:
    SEGMENTED_PREFIX keeps the keys that loaded when an L2 hit fails to load
    into L1 (e.g. OOM) mid-prefix; SPARSE keeps an intentionally scattered set.

    Args:
        found: Bitmap of found keys, over key indices ``0..num_keys-1``.
        num_keys: Total number of requested keys.
        policy: Trim policy to apply (see :class:`TrimPolicy`).
        attn_desc: Cross-chunk attention windows of all object groups, in
            object-group order.

    Returns:
        ``(hit_length, retain_mask)`` — prefix hit in chunks and retained bitmap.

    Raises:
        ValueError: If ``policy`` is not a known :class:`TrimPolicy`.
    """
    stride = attn_desc.num_object_groups * attn_desc.world_size
    if policy is TrimPolicy.PREFIX:
        num_chunks = num_keys // stride
        hit_length, retain = fold_unfold_ranked(
            found,
            num_chunks,
            attn_desc.world_size,
            attn_desc.num_chunks_in_sw,
        )
        return hit_length, retain
    elif policy in (TrimPolicy.SEGMENTED_PREFIX, TrimPolicy.SPARSE):
        hit_chunks = found.count_leading_ones() // stride
        return hit_chunks, found
    raise ValueError(f"Unknown TrimPolicy: {policy!r}")


def trim_load_plan_with_mask(
    load_plan: dict[int, Bitmap],
    mask: Bitmap,
) -> dict[int, Bitmap]:
    """Trim the load plan to the key indices set in ``mask`` (gap-tolerant).

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        mask: Bitmap of key indices to retain.

    Returns:
        Trimmed load plan; adapter indices retaining no keys are dropped.
    """
    trimmed_plan: dict[int, Bitmap] = {}
    for adapter_idx, bitmap in load_plan.items():
        new_bitmap = bitmap & mask
        if new_bitmap.popcount() == 0:
            continue
        trimmed_plan[adapter_idx] = new_bitmap
    return trimmed_plan


# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

PrefetchRequestId = int


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


class CallerState(enum.Enum):
    """Caller-facing lifecycle of a prefetch request's result.

    ``ACTIVE`` until a result is published exactly once, then either
    ``PUBLISHED`` (normal completion) or ``PUBLISHED_TIMEOUT`` (deadline
    fallback). The caller reads one immutable result.
    """

    ACTIVE = enum.auto()
    PUBLISHED = enum.auto()
    PUBLISHED_TIMEOUT = enum.auto()


class ResourceState(enum.Enum):
    """Lifecycle of the locks and buffers a prefetch request holds.

    Normal completion goes ``ACTIVE`` -> ``RETIRED`` in one step. A timeout
    goes ``ACTIVE`` -> ``DRAINING`` (late I/O keeps its buffers and locks) ->
    ``RETIRED`` once that I/O has completed.
    """

    ACTIVE = enum.auto()
    DRAINING = enum.auto()
    RETIRED = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    phase: PrefetchPhase
    num_kv_readers: int = 1
    """Total read locks per key to acquire when transitioning from
    write-locked to read-locked.  Must match the ``num_kv_readers`` of the
    corresponding ``submit_prefetch_task`` spec."""

    policy: TrimPolicy = TrimPolicy.PREFIX
    """Which retained-subset policy to apply (see :class:`TrimPolicy`)."""

    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    """Cross-chunk attention windows of all object groups, in object-group
    order."""
    mode: PrefetchMode = PrefetchMode.LOOKUP
    """The prefetch intent (see :class:`PrefetchMode`).  ``WARM`` forces all
    loaded keys permanent and acquires no read lock; ``LOOKUP`` defers
    retention to the policy and read-locks loaded keys."""

    # Lookup phase: adapter_idx -> task_id (removed as results arrive)
    pending_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Lookup phase: adapter_idx -> bitmap (populated as results arrive)
    lookup_results: dict[int, Bitmap] = field(default_factory=dict)
    # L2 read locks currently held (adapter_idx -> key indices).
    # _release_l2_locks subtracts keys as their locks are returned.
    l2_adapter2readlocks: dict[int, Bitmap] = field(default_factory=dict)
    # True once the prefix hit was stored/published.
    hit_reported: bool = False

    # Load phase: adapter_idx -> bitmap of key indices to load
    load_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> task_id (removed as results arrive)
    pending_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Load phase: adapter_idx -> L1 bytes reserved for that adapter's
    # in-flight load.  Read by the inflight_load_memory_usage_bytes gauge.
    load_bytes_by_adapter: dict[int, int] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap (populated as results arrive)
    load_results: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: keys that were write-reserved in L1
    write_reserved_keys: list[ObjectKey] = field(default_factory=list)
    write_reserved_objs: dict[ObjectKey, "MemoryObj"] = field(default_factory=dict)
    # Key indices found (and read-locked) in L1 when the request starts.
    l1_readlocks: Bitmap = field(default_factory=lambda: Bitmap(0))

    group_layout_descs: dict[int, MemoryLayoutDesc] = field(default_factory=dict)
    """Maps object_group_id to that group's layout (one ``MemoryLayoutDesc``
    describes a single group's MemoryObj). Covers every object group."""

    caller_state: CallerState = CallerState.ACTIVE
    """The caller-facing result lifecycle (see :class:`CallerState`)."""
    resource_state: ResourceState = ResourceState.ACTIVE
    """The lock/buffer lifecycle (see :class:`ResourceState`)."""
    deadline_at: float | None = None
    """Absolute monotonic deadline, or ``None`` when unarmed (feature off, WARM,
    or already published)."""
    published_retained: Bitmap | None = None
    """The key indices published to the caller. Their L1 read locks are the
    caller's; a drain keeps them and releases every other lock."""

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that read-locks L1-resident keys
       and submits lookup_and_lock to all adapters.
    3. Computes the load plan based on the found keys in L1 and L2;
       unlocks L1 keys that fall outside the plan.
    4. Reserves L1 write buffers (all-or-nothing) and submits load tasks.
    5. On completion, loaded keys become read-locked for the retriever.
    6. Reports the L1+L2 hit length via query_lookup_result and the
       retained-key bitmap via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
        l2_load_timeout: Optional monotonic deadline (seconds) for a
            ``LOOKUP``-mode request, covering queueing + L2 lookup + load.
            ``None`` (default) disables the feature and keeps behavior
            unchanged; on expiry the caller gets the trim-policy subset already
            usable and the rest is reported as a miss to recompute.
        clock: Monotonic clock used for deadlines; injectable for tests.
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
        l2_load_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters: dict[int, L2AdapterInterface] = {
            desc.index: adapter
            for desc, adapter in zip(adapter_descriptors, l2_adapters, strict=True)
        }
        self._adapter_descriptors: dict[int, AdapterDescriptor] = {
            desc.index: desc for desc in adapter_descriptors
        }
        self._policy = policy
        self._max_in_flight = max_in_flight

        # Optional monotonic L2 load deadline. ``None`` disables it; when
        # disabled ``_armed`` stays empty and the loop never reads the clock, so
        # the hot path is unchanged. ``_clock`` is injectable for deterministic
        # tests.
        self._l2_load_timeout = l2_load_timeout
        self._clock = clock
        # request_id -> absolute deadline for every armed in-flight request.
        # Bounds the loop's wait and drives the in-flight expiry sweep; an entry
        # is removed the instant its result is published.
        self._armed: dict[PrefetchRequestId, float] = {}
        # Shadow counters (loop thread only), surfaced via report_status.
        self._status_deadline_timeouts: int = 0
        self._status_draining_count: int = 0

        # Adapters that are being drained and will be removed after all
        # the in-flight operations are done.
        self._draining: dict[int, threading.Event] = {}

        # Control-plane queue for runtime add/remove, used by the internal
        # loop thread
        self._adapter_ops_lock = threading.Lock()
        self._pending_adapter_ops: list[AddAdapterOp | RemoveAdapterOp] = []
        self._adapter_ctrl_efd = create_event_notifier()

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        # Each queued item carries its absolute deadline (or None when unarmed)
        # so queueing time counts toward the budget even before admission.
        self._pending_queue: list[
            tuple[PrefetchRequestId, PrefetchRequestSpec, float | None]
        ] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[
            tuple[PrefetchRequestId, PrefetchRequestSpec, float | None]
        ] = []
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
        spec: PrefetchRequestSpec,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request.

        Thread-safe. Can be called from any thread.

        A key counts as found if it is already resident in L1 or any L2
        adapter reports it. In the interval-figure terms of the module
        docstring, the request ends with (sliding-window view)::

            |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                          ^ L1 hit length               ^ L1+L2 hit
            |     unlock     |   locked*  |     unlock     |   locked   |   unlock   |

        locked keys stay read-locked for the retriever until it consumes
        them; keys in L2 but not in L1 inside the final window are loaded
        first. (*) the final window when the L2 load lands, else the L1
        hit's own window.

        The retained subset of found keys is chosen by ``spec.policy`` (see
        :class:`TrimPolicy`).  With the default ``PREFIX`` policy, only the
        **contiguous prefix** of found keys is loaded from L2: if L2 has keys
        {0, 1, 3, 4} but not key 2, only keys {0, 1} are loaded because the gap
        at index 2 breaks the prefix.  Keys outside the retained set are never
        transferred, saving I/O bandwidth and L1 memory.  Use
        :meth:`query_prefetch_result` to retrieve the retained set once the
        request completes.

        Args:
            spec: The prefetch request inputs (see :class:`PrefetchRequestSpec`).

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        # Arm the deadline at submission so it also covers queueing time. Only
        # LOOKUP-mode requests (a caller is waiting) are armed; WARM is
        # speculative with no caller to fall back to. Stamp under the lock so
        # later request ids never get an earlier deadline -- the pending queue
        # then stays monotonic and expiry only needs to inspect its head.
        timeout = self._l2_load_timeout if spec.mode is PrefetchMode.LOOKUP else None
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            deadline_at = None if timeout is None else self._clock() + timeout
            self._submission_queue.append((request_id, spec, deadline_at))
        self._submission_efd.notify()
        return request_id

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
        with self._lookup_results_lock:
            return self._completed_lookups.get(request_id, None)

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
        with self._prefetch_results_lock:
            result = self._completed_results.pop(request_id, None)
        if result is not None:
            with self._lookup_results_lock:
                self._completed_lookups.pop(request_id, None)
        return result

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
                lambda: request_id in self._completed_results, timeout
            )

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
            "deadline_timeout_count": self._status_deadline_timeouts,
            "draining_request_count": self._status_draining_count,
        }

    def get_adapter_state_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """``(count, {"state": ...})`` tuples for the ``lmcache_mp.l2_adapters``
        gauge. ``len()`` reads are GIL-atomic, safe from the OTel thread."""
        num_draining = len(self._draining)
        return [
            (len(self._l2_adapters) - num_draining, {"state": "active"}),
            (num_draining, {"state": "draining"}),
        ]

    def _snapshot_inflight_loads(self) -> dict[int, tuple[int, int]]:
        """``{adapter_idx: (count, reserved_bytes)}`` for in-flight L2 -> L1
        loads, computed via GIL-atomic ``dict.copy()`` snapshots so the
        OTel reader thread can call this concurrently with the prefetch
        loop without locking.
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
        """Per-adapter ``(count, attributes)`` for the
        ``lmcache_mp.num_inflight_l2_loads`` gauge."""
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
        """Per-adapter ``(reserved_bytes, attributes)`` for the
        ``lmcache_mp.inflight_load_memory_usage_bytes`` gauge."""
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

            ready = poller.poll(self._next_poll_timeout_ms())

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

            # Expire due requests AFTER advancing, so any completion signaled in
            # this same wake is fully processed first (completion wins the race);
            # only genuinely-pending adapters remain when expiry computes the
            # fallback. Runs every iteration, cheap and a no-op when disabled.
            try:
                self._expire_due_requests(self._clock())
            except Exception:
                logger.exception(
                    "Unexpected error expiring deadline-due prefetch requests"
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
        """True if any in-flight request still references ``adapter_id``."""
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
            request_id, spec, deadline_at = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, spec, deadline_at)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _lock_l1_keys(
        self,
        keys: list[ObjectKey],
        num_kv_readers: int,
        policy: TrimPolicy,
        attn_desc: AttnWindowDesc,
    ) -> Bitmap:
        """Read-lock the L1-resident keys the request will hold.

        Locks every key readable in L1, then, for windowed ``PREFIX``
        retention, releases the read locks on sliding-window chunks behind
        the L1 hit's own window. Locking does not refresh eviction recency;
        the LRU signal is sent by ``_finish_request`` for the retained keys
        only.

        Args:
            keys: The request's object keys in chunk-major prefix order
                (chunk 0's object groups x kv_ranks, then chunk 1's, ...).
                StorageManager has already capped away its own L1 prefix
                hit, so these keys start past it.
            num_kv_readers: Read locks per locked key.
            policy: The request's retained-subset policy.
            attn_desc: Per-group cross-chunk attention windows.

        Returns:
            Bitmap of the key indices read-locked in L1, after releasing
            out-of-window sliding-window chunks.
        """
        lock_results = self._l1_manager.reserve_read(keys, read_locks=num_kv_readers)
        l1_readlocks = Bitmap(len(keys))
        for i, key in enumerate(keys):
            err, _obj = lock_results[key]
            if err == L1Error.SUCCESS:
                l1_readlocks.set(i)
        if policy is not TrimPolicy.PREFIX:
            return l1_readlocks
        hit_length, l1_window = build_trim_mask(
            l1_readlocks, len(keys), policy, attn_desc
        )
        stride = attn_desc.num_object_groups * attn_desc.world_size
        within_l1_hit = Bitmap(len(keys), hit_length * stride)
        # evictable: locked SW chunks behind the window, within the L1 prefix
        # hit -- won't be read again, so release their locks.
        evictable = l1_readlocks & ~l1_window & within_l1_hit
        if evictable.popcount() > 0:
            self._l1_manager.finish_read(
                evictable.gather(keys), read_locks=num_kv_readers
            )
            l1_readlocks = l1_readlocks & ~evictable
        return l1_readlocks

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        spec: PrefetchRequestSpec,
        deadline_at: float | None = None,
    ) -> None:
        """Read-lock L1-resident keys, then submit lookup_and_lock to all
        live (non-draining) adapters for a new request.

        Args:
            request_id: The request's controller id.
            spec: The prefetch request inputs.
            deadline_at: Absolute monotonic deadline stamped at submission, or
                ``None`` when the feature is off or the request is ``WARM``.
        """
        l1_readlocks = self._lock_l1_keys(
            spec.keys, spec.num_kv_readers, spec.policy, spec.attn_desc
        )
        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=spec.keys,
            phase=PrefetchPhase.LOOKUP,
            num_kv_readers=spec.num_kv_readers,
            policy=spec.policy,
            attn_desc=spec.attn_desc,
            mode=spec.mode,
            group_layout_descs=spec.group_layout_descs,
            l1_readlocks=l1_readlocks,
            deadline_at=deadline_at,
        )

        # Skip adapters being drained so a new request never locks keys on
        # an adapter that is on its way out.
        routing_adapters = {
            adapter_id: adapter
            for adapter_id, adapter in self._l2_adapters.items()
            if adapter_id not in self._draining
        }
        if not routing_adapters:
            # No live L2 adapters: finish with whatever L1 alone serves.
            self._finish_request(request)
            return

        for adapter_id, adapter in routing_adapters.items():
            task_id = adapter.submit_lookup_and_lock_task(
                spec.keys, spec.group_layout_descs
            )
            request.pending_lookup_tasks[adapter_id] = task_id
        self._in_flight_requests[request_id] = request
        if deadline_at is not None:
            self._armed[request_id] = deadline_at
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(spec.keys),
                    "adapter_count": len(request.pending_lookup_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in spec.keys),
                },
            )
        )

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Compute the L1 ∪ L2 load plan, reserve L1 buffers, and submit
        load tasks."""
        request.phase = PrefetchPhase.PLAN_AND_LOAD
        self._status_lookup_phase_count -= 1
        self._status_load_phase_count += 1

        num_keys = len(request.keys)

        # Step 1 — generate what keys should be loaded from L2 to L1.
        # Potential L2 load candidates:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |   candidate    | candidate  | candidate  |
        # Exclude draining adapters so no new load targets them; any keys
        # they locked during lookup fall outside the plan and get unlocked
        # in _release_l2_locks. Keys already served from L1 never need an
        # L2 transfer.
        routing_descriptors = [
            desc
            for adapter_id, desc in self._adapter_descriptors.items()
            if adapter_id not in self._draining
        ]
        load_plan = self._policy.select_load_plan(
            request.keys,
            request.lookup_results,
            routing_descriptors,
        )
        # Keys already in L1 don't need to be loaded from L2.
        load_plan = trim_load_plan_with_mask(load_plan, ~request.l1_readlocks)

        # Step 2 — calculate the L1 + L2 maximum hit length and the
        # retained set.
        union_bitmap = (
            merge_bitmaps(load_plan.values(), num_keys) | request.l1_readlocks
        )
        hit_length, retained = build_trim_mask(
            union_bitmap,
            num_keys,
            request.policy,
            request.attn_desc,
        )
        trimmed_plan = trim_load_plan_with_mask(load_plan, retained)

        if not trimmed_plan:
            # Nothing to load from L2: finish releases all unneeded locks
            # and reports the L1-only hit.
            self._finish_request(request)
            return

        # Unlock the keys based on the following figure.
        # SW keys in L1:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |     unlock     | keep lock* |     unlock     | keep lock  |   unlock   |
        # (*) the L1 hit's own window: kept even when outside the final
        # window, as the fallback promise if the L2 load never lands.
        _l1_hit, l1_fallback_retain = build_trim_mask(
            request.l1_readlocks,
            num_keys,
            request.policy,
            request.attn_desc,
        )
        stale = request.l1_readlocks & (~retained) & (~l1_fallback_retain)
        if stale.popcount() > 0:
            request.l1_readlocks = request.l1_readlocks & (
                retained | l1_fallback_retain
            )
            self._l1_manager.finish_read(
                stale.gather(request.keys), read_locks=request.num_kv_readers
            )

        # Step 3 — reserve L1 write buffers for the plan keys.
        # If any failure (OOM or contention or others) happens,
        # we fall back to the L1-only longest hit (`l1_fallback_retain`).
        # SW keys in L2 (and not in L1):
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        |  loading   |     -      |
        keys_to_reserve = merge_bitmaps(trimmed_plan.values(), num_keys).gather(
            request.keys
        )
        reserved = self._reserve_load_buffers(request, keys_to_reserve)
        if len(reserved) < len(keys_to_reserve):
            self._finish_request(request)
            return
        request.load_plan = trimmed_plan

        # Step 4 — free L2 lookup locks for keys outside the plan.
        # L2 lookup locks:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |      free      |    free    |      free      | keep plan  |    free    |
        self._release_l2_locks(request, keep=request.load_plan)

        # Step 5 — submit loads; report the hit.
        self._submit_load_tasks(request, trimmed_plan)
        self._report_lookup_hit(request, hit_length)

    def _reserve_load_buffers(
        self,
        request: InFlightPrefetchRequest,
        keys_to_reserve: list[ObjectKey],
    ) -> set[ObjectKey]:
        """Reserve L1 write buffers for the keys to load from L2.

        The keys sit in the loading segment — in L2, not in L1, inside the
        final window (keys already in L1 were read-locked by
        ``_lock_l1_keys`` and are not reserved here)::

            |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                          ^ L1 hit length               ^ L1+L2 hit
            |       -        |     -      |       -        |  loading   |     -      |

        Successful reservations are recorded on
        ``request.write_reserved_keys`` / ``request.write_reserved_objs``.
        Failures publish an ``L2_PREFETCH_FAILED`` event; the caller
        abandons the L2 load if any key failed (all-or-nothing).

        Args:
            request: The in-flight request the buffers belong to.
            keys_to_reserve: Keys in the trimmed load plan, in prefix order.

        Returns:
            The subset of ``keys_to_reserve`` that now holds a write buffer.
        """
        # WARM retains every loaded key; LOOKUP follows the configured policy.
        if request.mode is PrefetchMode.WARM:
            retentions = [True] * len(keys_to_reserve)
        else:
            retentions = self._policy.select_l1_retentions(
                keys_to_reserve,
            )
        retention_map = dict(zip(keys_to_reserve, retentions, strict=True))

        # Batch reserve_write by object_group_id so each group uses its own
        # tensor shapes.
        write_results: dict[ObjectKey, tuple[L1Error, MemoryObj | None]] = {}
        by_group = sorted(keys_to_reserve, key=attrgetter("object_group_id"))
        for gid, group_iter in groupby(by_group, key=attrgetter("object_group_id")):
            group_keys = list(group_iter)
            gld = request.group_layout_descs[gid]
            gr = self._l1_manager.reserve_write(
                keys=group_keys,
                is_temporary=[not retention_map[k] for k in group_keys],
                layout_desc=gld,
                mode="new",
            )
            write_results.update(gr)

        reserved: set[ObjectKey] = set()
        oom_keys: list[ObjectKey] = []
        contended_keys: list[ObjectKey] = []
        for key, (err, mem_obj) in write_results.items():
            if err == L1Error.SUCCESS and mem_obj is not None:
                request.write_reserved_keys.append(key)
                request.write_reserved_objs[key] = mem_obj
                reserved.add(key)
                continue
            if err == L1Error.OUT_OF_MEMORY:
                oom_keys.append(key)
            elif err == L1Error.KEY_NOT_WRITABLE:
                contended_keys.append(key)
            logger.debug(
                "Prefetch request %d: reserve write failed for %s: %s",
                request.request_id,
                key,
                err,
            )

        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l2_prefetch", "keys": oom_keys},
                )
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "l1_oom", "keys": oom_keys},
                )
            )
        if contended_keys:
            # The key was write-locked by a concurrent request after the L1
            # lock pass; the caller falls back to the L1-only hit.
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "l1_contended", "keys": contended_keys},
                )
            )
        return reserved

    def _submit_load_tasks(
        self,
        request: InFlightPrefetchRequest,
        trimmed_plan: dict[int, Bitmap],
    ) -> None:
        """Submit one load task per adapter in the final trimmed plan.

        Every plan key must hold a write buffer in
        ``request.write_reserved_objs`` (guaranteed by the caller's
        reserved-bitmap trim). Publishes ``L2_LOAD_TASK_SUBMITTED`` per
        adapter and one ``L2_PREFETCH_LOAD_SUBMITTED`` for the batch.

        Args:
            request: The in-flight request being loaded.
            trimmed_plan: Final load plan (adapter index -> key indices).
        """
        plan_keys: list[ObjectKey] = []
        for adapter_idx, bitmap in trimmed_plan.items():
            per_adapter_keys = bitmap.gather(request.keys)
            per_adapter_objs = [
                request.write_reserved_objs[key] for key in per_adapter_keys
            ]
            task_id = self._l2_adapters[adapter_idx].submit_load_task(
                per_adapter_keys, per_adapter_objs
            )
            request.pending_load_tasks[adapter_idx] = task_id
            plan_keys.extend(per_adapter_keys)
            # Per-adapter byte accounting for L2_LOAD_TASK_* throughput
            # events.  Sum individual sizes (groups may differ in size).
            total_bytes = sum(obj.get_size() for obj in per_adapter_objs)
            request.load_bytes_by_adapter[adapter_idx] = total_bytes

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                        "key_count": len(per_adapter_keys),
                        "total_bytes": total_bytes,
                    },
                )
            )

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": request.request_id,
                    "key_count": len(plan_keys),
                    "adapter_count": len(trimmed_plan),
                    "key_count_per_salt": Counter(k.cache_salt for k in plan_keys),
                },
            )
        )
        logger.debug(
            "Prefetch request %d: submitted load tasks to %d adapters for %d keys",
            request.request_id,
            len(trimmed_plan),
            len(plan_keys),
        )

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
        draining = request.resource_state is ResourceState.DRAINING
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if draining:
                # A timed-out lookup plans no load; drain once the late lookups
                # (each carrying an L2 lock) have all reported.
                if request.all_lookups_done():
                    self._finish_drain(request)
            elif request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if draining:
                if request.all_loads_done():
                    self._finish_drain(request)
            elif request.all_loads_done():
                self._finish_request(request)

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters."""
        for adapter_idx in list(request.pending_lookup_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_lookup_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
                task_id
            )
            if result is None:
                continue
            request.lookup_results[adapter_idx] = result
            request.l2_adapter2readlocks[adapter_idx] = result
            del request.pending_lookup_tasks[adapter_idx]

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
        num_keys = len(request.keys)

        result_bitmap = self._scatter_load_results(request)
        loaded_set = set(result_bitmap.gather(request.keys))

        # Return every L2 read lock still held.
        self._release_l2_locks(request, keep={})

        l1_mgr = self._l1_manager

        # Finalize every write-reserved buffer. LOOKUP read-locks the loaded keys
        # for the retriever (the fold below then releases those outside the
        # retained set); WARM makes them resident and unlocked. Failed loads
        # delete their buffer.
        # SW keys in L2 (and not in L1):
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        |load→locked |     -      |
        # SW keys in L1:
        # |     unlock     |   unlock   |     unlock     |   locked   |   unlock   |
        readlock = set() if request.mode is PrefetchMode.WARM else loaded_set
        loaded_keys, failed_keys = self._finalize_write_reserved(
            request, list(request.write_reserved_keys), loaded_set, readlock
        )

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "loaded_count": len(loaded_keys),
                    "failed_count": len(failed_keys),
                    "key_count_per_salt": Counter(k.cache_salt for k in loaded_keys),
                },
            )
        )

        # L2 prefetch-failure anomaly reporting: keys were reserved in L1
        # (expected to load from L2) but did not appear in the load bitmap.
        # Classified as ``not_found`` — the serde_failure reason will be
        # added once the serde PR lands and adapters can distinguish
        # deserialization errors from missing objects.
        if failed_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "not_found", "keys": failed_keys},
                )
            )

        # Include keys served from L1 (read-locked when the request started)
        # so the fold sees all object groups.
        result_bitmap = result_bitmap | request.l1_readlocks

        # Release read locks for any key outside the retained set (partial
        # load failures can create gaps). WARM has no retriever to release
        # the retained keys later, so it releases everything it holds.
        hit_length, retained = build_trim_mask(
            result_bitmap,
            num_keys,
            request.policy,
            request.attn_desc,
        )
        if request.mode is PrefetchMode.WARM:
            if request.l1_readlocks.popcount() > 0:
                l1_mgr.finish_read(
                    request.l1_readlocks.gather(request.keys),
                    read_locks=request.num_kv_readers,
                )
                request.l1_readlocks = Bitmap(num_keys)
        else:
            released = (result_bitmap & (~retained)).gather(request.keys)
            if released:
                l1_mgr.finish_read(released, read_locks=request.num_kv_readers)

        # LRU: the retained keys are the ones this request actually serves;
        # touch them (locking/unlocking never refreshes recency).
        retained_keys = retained.gather(request.keys)
        if retained_keys:
            l1_mgr.touch_keys(retained_keys)

        # Publish the retained bitmap and retire in one step (normal completion).
        # _publish_result_once reports the lookup hit unless the load path
        # already did at submit time (so the engine never waits on the load).
        self._publish_result_once(request, retained, hit_length, timed_out=False)
        self._retire_request_once(request)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _release_l2_locks(
        self, request: InFlightPrefetchRequest, keep: dict[int, Bitmap]
    ) -> None:
        """Release L2 locks in ``request.l2_adapter2readlocks``, except for
        keys in ``keep``.

        Released keys are subtracted from ``request.l2_adapter2readlocks``,
        so repeated calls never double-unlock.

        Args:
            request: The in-flight request whose L2 locks to release.
            keep: Adapter index -> key indices whose locks stay held (the
                load plan); pass ``{}`` to release everything still held.
        """
        num_keys = len(request.keys)
        for adapter_idx, held in list(request.l2_adapter2readlocks.items()):
            keep_bitmap = keep.get(adapter_idx, Bitmap(num_keys))
            unlock_keys = (held & (~keep_bitmap)).gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)
            remaining = held & keep_bitmap
            if remaining.popcount() == 0:
                del request.l2_adapter2readlocks[adapter_idx]
            else:
                request.l2_adapter2readlocks[adapter_idx] = remaining

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _scatter_load_results(self, request: InFlightPrefetchRequest) -> Bitmap:
        """Scatter each adapter's local load bitmap into global key positions.

        Each adapter's load bitmap is locally indexed (size == that adapter's
        key count); the plan bitmap maps local -> global indices via
        ``get_indices_list()``.

        Args:
            request: The in-flight request whose ``load_results`` to scatter.

        Returns:
            A ``len(request.keys)``-sized bitmap of the globally-indexed keys
            that have finished loading so far.
        """
        result_bitmap = Bitmap(len(request.keys))
        for adapter_idx, plan_bitmap in request.load_plan.items():
            load_bitmap = request.load_results.get(adapter_idx)
            if load_bitmap is None:
                continue
            plan_indices = plan_bitmap.get_indices_list()
            for global_i in load_bitmap.gather(plan_indices):
                result_bitmap.set(global_i)
        return result_bitmap

    def _finalize_write_reserved(
        self,
        request: InFlightPrefetchRequest,
        keys: list[ObjectKey],
        loaded: set[ObjectKey],
        readlock: set[ObjectKey],
    ) -> tuple[list[ObjectKey], list[ObjectKey]]:
        """Finalize a set of write-reserved buffers and drop them from tracking.

        The single place that resolves a reserved L1 write buffer, shared by
        normal completion, the load-phase deadline finalize, and the drain: a
        key that loaded and is in ``readlock`` becomes read-locked for the
        retriever; one that loaded but is not in ``readlock`` becomes
        resident-but-unlocked (reusable by a later request); one that did not
        load has its buffer deleted. All ``keys`` are removed from
        ``write_reserved_keys``/``write_reserved_objs`` so no later step touches
        them again.

        Args:
            request: The in-flight request owning the reservations.
            keys: The write-reserved keys to finalize (a subset of
                ``request.write_reserved_keys``).
            loaded: The keys whose load succeeded.
            readlock: The loaded keys to read-lock for the retriever; loaded
                keys outside it are made resident and unlocked.

        Returns:
            ``(loaded_keys, failed_keys)`` — the finalized loaded and deleted
            keys, for the caller's completion events.
        """
        to_readlock = [k for k in keys if k in loaded and k in readlock]
        to_resident = [k for k in keys if k in loaded and k not in readlock]
        failed_keys = [k for k in keys if k not in loaded]
        if to_readlock:
            self._l1_manager.finish_write_and_reserve_read(
                to_readlock, read_locks=request.num_kv_readers
            )
        if to_resident:
            self._l1_manager.finish_write(to_resident)
        if failed_keys:
            self._l1_manager.finish_write_and_delete(failed_keys)
        if keys:
            finalized = set(keys)
            request.write_reserved_keys = [
                k for k in request.write_reserved_keys if k not in finalized
            ]
            for key in keys:
                request.write_reserved_objs.pop(key, None)
        return to_readlock + to_resident, failed_keys

    def _publish_result_once(
        self,
        request: InFlightPrefetchRequest,
        result: Bitmap,
        hit_length: int,
        *,
        timed_out: bool,
    ) -> None:
        """Publish the caller-facing result exactly once and disarm the deadline.

        The caller lifecycle's single terminal transition (success or timeout
        fallback). Idempotent: a second call for the same request (e.g. a late
        completion after a timeout publish) is a no-op, so the caller sees one
        immutable result. Does NOT retire the request -- resource retirement is
        a separate lifecycle handled by :meth:`_retire_request_once`.

        Args:
            request: The in-flight request being published.
            result: The retained-key bitmap the caller receives.
            hit_length: Chunk-level prefix hit to report as the lookup hit.
            timed_out: Whether this publish is a timeout fallback.
        """
        if request.caller_state is not CallerState.ACTIVE:
            return
        request.caller_state = (
            CallerState.PUBLISHED_TIMEOUT if timed_out else CallerState.PUBLISHED
        )
        self._armed.pop(request.request_id, None)
        # Report the lookup hit unless the load path already did at submit time.
        # A timeout fallback may report retained < the hit reported at submit;
        # that is the same shrink the existing partial-load-failure path
        # produces, and the caller sizes its retrieve from this published
        # bitmap (query_prefetch_status), never from the earlier hit.
        if not request.hit_reported:
            self._report_lookup_hit(request, hit_length)
        with self._prefetch_results_lock:
            self._completed_results[request.request_id] = result
            # Wake any WAIT_PREFETCH_STATUS handler blocked on this result.
            self._prefetch_results_cv.notify_all()
        logger.debug(
            "Prefetch request %d %s: %d retained keys",
            request.request_id,
            "timed out (fallback published)" if timed_out else "completed",
            result.popcount(),
        )

    def _retire_request_once(self, request: InFlightPrefetchRequest) -> None:
        """Remove a request from in-flight tracking exactly once.

        The resource lifecycle's terminal transition to ``RETIRED``. Idempotent.
        On the timeout path this runs only after every late completion has
        drained; on the normal path right after :meth:`_publish_result_once`.

        Args:
            request: The request whose tracking entry to remove.
        """
        if request.resource_state is ResourceState.RETIRED:
            return
        was_draining = request.resource_state is ResourceState.DRAINING
        request.resource_state = ResourceState.RETIRED
        self._armed.pop(request.request_id, None)
        removed = self._in_flight_requests.pop(request.request_id, None)
        if removed is not None:
            self._status_in_flight_count -= 1
            if was_draining:
                self._status_draining_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug("Prefetch request %d retired", request.request_id)

    def _next_poll_timeout_ms(self) -> int:
        """Poll timeout bounded by the nearest deadline, in-flight or queued.

        Returns the constant poll timeout when the feature is off or nothing is
        armed (no clock read on that path). Otherwise the wait is
        ``min(poll, nearest_deadline - now)``, rounded up so a sub-millisecond
        remainder does not busy-spin at ``poll(0)``. The queue is deadline-
        monotonic (stamped under the submission lock), so only its head can beat
        the in-flight deadlines.
        """
        if self._l2_load_timeout is None:
            return PREFETCH_LOOP_POLL_TIMEOUT_MS
        nearest: float | None = min(self._armed.values()) if self._armed else None
        if self._pending_queue:
            head = self._pending_queue[0][2]
            if head is not None and (nearest is None or head < nearest):
                nearest = head
        if nearest is None:
            return PREFETCH_LOOP_POLL_TIMEOUT_MS
        remaining_ms = (nearest - self._clock()) * 1000.0
        if remaining_ms <= 0.0:
            return 0
        return min(PREFETCH_LOOP_POLL_TIMEOUT_MS, math.ceil(remaining_ms))

    def _expire_due_requests(self, now: float) -> None:
        """Fire the deadline for every request whose budget has elapsed.

        In-flight requests enter drain-only (publish the usable subset, keep
        late I/O draining); still-queued requests publish their L1-only subset
        and are dropped (they hold no L2 resources). A no-op when disabled.

        Args:
            now: The current monotonic time.
        """
        if self._l2_load_timeout is None:
            return
        for request_id, deadline in list(self._armed.items()):
            if deadline > now:
                continue
            request = self._in_flight_requests.get(request_id)
            if request is None or request.caller_state is not CallerState.ACTIVE:
                # Retired or already published in this same iteration; disarm.
                self._armed.pop(request_id, None)
                continue
            self._enter_drain_only(request, now)
        # Expire queued (not yet admitted) requests. Armed deadlines are
        # monotonic along the queue: walk from the front, keep unarmed (WARM)
        # entries, collect the due ones, and stop at the first armed entry not
        # yet due (index ``cut``); the tail from ``cut`` is untouched.
        keep: list[tuple[PrefetchRequestId, PrefetchRequestSpec, float | None]] = []
        expired: list[tuple[PrefetchRequestId, PrefetchRequestSpec, float]] = []
        cut = len(self._pending_queue)
        for i, (request_id, spec, queued_deadline) in enumerate(self._pending_queue):
            if queued_deadline is None:
                keep.append((request_id, spec, queued_deadline))
            elif queued_deadline <= now:
                expired.append((request_id, spec, queued_deadline))
            else:
                cut = i
                break
        if not expired:
            return
        self._pending_queue = keep + self._pending_queue[cut:]
        for request_id, spec, deadline in expired:
            self._status_pending_count -= 1
            self._expire_queued_request(request_id, spec, deadline)

    def _expire_queued_request(
        self,
        request_id: PrefetchRequestId,
        spec: PrefetchRequestSpec,
        deadline_at: float,
    ) -> None:
        """Publish the L1-only fallback for a request that timed out while queued.

        A queued request never reached L2, so its budget is spent entirely on
        an L2 that never ran. It still does the cheap synchronous L1 lock + trim
        (skipping only L2) so an L1-resident hit is not discarded as a miss, then
        publishes that subset through the single publish path. The caller owns
        the published L1 read locks and releases them via free_lookup_locks;
        there is nothing left to drain. It emits a ``LOOKUP_SUBMITTED`` (to zero
        L2 adapters) so the ``LOOKUP_COMPLETED`` the publish emits still pairs.

        Args:
            request_id: The queued request's controller id.
            spec: The queued request's inputs.
            deadline_at: The request's absolute deadline (for the timeout event).
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(spec.keys),
                    "adapter_count": 0,
                    "key_count_per_salt": Counter(k.cache_salt for k in spec.keys),
                },
            )
        )
        l1_readlocks = self._lock_l1_keys(
            spec.keys, spec.num_kv_readers, spec.policy, spec.attn_desc
        )
        hit_length, retained = build_trim_mask(
            l1_readlocks, len(spec.keys), spec.policy, spec.attn_desc
        )
        release = (l1_readlocks & (~retained)).gather(spec.keys)
        if release:
            self._l1_manager.finish_read(release, read_locks=spec.num_kv_readers)
        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=spec.keys,
            phase=PrefetchPhase.LOOKUP,
            num_kv_readers=spec.num_kv_readers,
            policy=spec.policy,
            attn_desc=spec.attn_desc,
            mode=spec.mode,
            group_layout_descs=spec.group_layout_descs,
            l1_readlocks=l1_readlocks & retained,
            published_retained=retained,
        )
        self._status_deadline_timeouts += 1
        self._emit_deadline_timeout(
            "queued",
            request_id,
            spec.attn_desc,
            len(spec.keys),
            hit_length,
            deadline_at,
        )
        self._publish_result_once(request, retained, hit_length, timed_out=True)

    def _emit_deadline_timeout(
        self,
        phase: str,
        request_id: PrefetchRequestId,
        attn_desc: AttnWindowDesc,
        num_keys: int,
        retained_chunks: int,
        deadline_at: float | None,
    ) -> None:
        """Publish one ``L2_PREFETCH_DEADLINE`` event for a timed-out request.

        Low-cardinality only: ``phase`` (queued/lookup/load), the configured
        budget, the elapsed time since the request entered the controller, and
        the retained vs missed (recompute) chunk counts. No request-id label.
        """
        stride = attn_desc.num_object_groups * attn_desc.world_size
        total_chunks = num_keys // stride if stride else 0
        budget = self._l2_load_timeout
        elapsed = (
            self._clock() - (deadline_at - budget)
            if deadline_at is not None and budget is not None
            else budget
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_DEADLINE,
                metadata={
                    "request_id": request_id,
                    "phase": phase,
                    "budget_seconds": budget,
                    "elapsed_seconds": elapsed,
                    "retained_chunks": retained_chunks,
                    "missed_chunks": max(0, total_chunks - retained_chunks),
                },
            )
        )

    def _enter_drain_only(self, request: InFlightPrefetchRequest, now: float) -> None:
        """Publish the timeout fallback and switch the request to drain-only.

        The usable subset is the trim policy applied to what is safely available
        *now*: L1 read-locked hits, plus (in the load phase) the loads already
        completed and finalized. Pending adapters keep their buffers and L2
        locks -- nothing they still own is freed here. Late completions drain via
        :meth:`_finish_drain`.

        Args:
            request: The in-flight request whose deadline fired.
            now: The current monotonic time (for observability only).
        """
        num_keys = len(request.keys)
        if request.phase == PrefetchPhase.LOOKUP:
            # Only finalized L1 hits are usable; L2 lookups that returned but
            # never loaded into L1 do not count.
            usable = request.l1_readlocks
        else:
            usable = self._scatter_load_results(request) | request.l1_readlocks
        hit_length, retained = build_trim_mask(
            usable, num_keys, request.policy, request.attn_desc
        )
        if request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._finalize_completed_loads_at_deadline(request, retained)
        # Release the L1 read locks outside the published set now, not at the end
        # of the drain: pending adapters do not use them, so holding them would
        # pin L1 entries against eviction for a whole slow-L2 drain. Only the
        # pending tasks' buffers and L2 locks must outlive the publish.
        stale = (request.l1_readlocks & (~retained)).gather(request.keys)
        if stale:
            self._l1_manager.finish_read(stale, read_locks=request.num_kv_readers)
            request.l1_readlocks = request.l1_readlocks & retained
        request.published_retained = retained
        request.resource_state = ResourceState.DRAINING
        self._status_deadline_timeouts += 1
        self._status_draining_count += 1
        phase = "lookup" if request.phase == PrefetchPhase.LOOKUP else "load"
        self._emit_deadline_timeout(
            phase,
            request.request_id,
            request.attn_desc,
            num_keys,
            hit_length,
            request.deadline_at,
        )
        self._publish_result_once(request, retained, hit_length, timed_out=True)

    def _finalize_completed_loads_at_deadline(
        self, request: InFlightPrefetchRequest, retained: Bitmap
    ) -> None:
        """Finalize loads already completed when a load-phase deadline fires.

        Completed keys inside the published ``retained`` set become read-locked
        for the caller; completed keys outside it become resident-but-unlocked
        (reusable by a later request). Both are removed from the write-reservation
        bookkeeping so the later drain touches only pending-adapter buffers.
        Pending adapters are not referenced here, so their buffers stay reserved.

        Args:
            request: The timed-out in-flight request (load phase).
            retained: The published retained-key bitmap.
        """
        completed = self._scatter_load_results(request)
        completed_keys = completed.gather(request.keys)
        if not completed_keys:
            return
        completed_set = set(completed_keys)
        retained_keys = set(retained.gather(request.keys))
        # Completed loads inside the published set serve the caller (read-lock);
        # completed loads outside it become resident-unlocked.
        self._finalize_write_reserved(
            request, completed_keys, completed_set, retained_keys
        )

    def _finish_drain(self, request: InFlightPrefetchRequest) -> None:
        """Drain a timed-out request once its late I/O has all completed.

        Finalizes any still-reserved buffers (loaded -> resident-but-unlocked so
        a later same-key request can hit them; failed -> deleted), returns every
        remaining L2 lock, and retires the request. The L1 read locks outside the
        published set were already released at publish time, and the published
        ones belong to the caller. Never re-publishes and never re-serves the
        (already-answered) caller.

        Args:
            request: The drain-only request whose late I/O has finished.
        """
        if request.write_reserved_keys:
            completed = self._scatter_load_results(request)
            loaded_set = set(completed.gather(request.keys))
            # No retriever remains, so nothing is read-locked (empty readlock):
            # loaded -> resident-unlocked, failed -> deleted.
            self._finalize_write_reserved(
                request, list(request.write_reserved_keys), loaded_set, set()
            )
        self._release_l2_locks(request, keep={})
        self._retire_request_once(request)

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown.

        Runs after the loop thread has joined. ``write_reserved_keys`` holds only
        buffers not yet finalized (a drained request already dropped its
        completed ones), so deleting them frees exactly the pending reservations.
        A draining request's remaining L1 read locks are the published set the
        caller owns and will release itself, so they are left held here to avoid
        a double release; an unpublished request's locks are ours to return.
        """
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write_and_delete(request.write_reserved_keys)
            self._release_l2_locks(request, keep={})
            caller_owns_locks = request.resource_state is ResourceState.DRAINING
            if not caller_owns_locks and request.l1_readlocks.popcount() > 0:
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
        self._armed.clear()
        self._status_in_flight_count = 0
        self._status_draining_count = 0
        self._status_lookup_phase_count = 0
        self._status_load_phase_count = 0
