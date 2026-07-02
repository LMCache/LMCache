# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Pins L1-resident keys (read locks) so they extend the hit and cannot be
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
docs/design/v1/distributed/storage_controllers/prefetch_l1_pin_pass.md::

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
Vocabulary: pin = take an L1 read lock; skip = never lock (stays evictable);
loading = L1 write reservation carrying an L2 lookup lock; unpin = return
the read lock.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from typing import Iterable
import enum
import select
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
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
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

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
        windows = attn_desc.num_chunks_in_sw
        # Benchmarking flag: treat every group as full-attention.
        if attn_desc.force_retrieve_full_kv:
            windows = [-1] * attn_desc.num_object_groups
        hit_length, retain = fold_unfold_ranked(
            found,
            num_chunks,
            attn_desc.world_size,
            windows,
        )
        return hit_length, retain
    if policy in (TrimPolicy.SEGMENTED_PREFIX, TrimPolicy.SPARSE):
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


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    phase: PrefetchPhase
    extra_count: int = 0
    """Extra read locks per key (on top of the default 1) to acquire when
    transitioning from write-locked to read-locked.  Must match the
    ``extra_count`` used in the corresponding ``submit_prefetch_task`` call."""

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
    # L2 lookup locks currently held (adapter_idx -> key indices). Mirrors
    # lookup_results on arrival; _release_l2_locks subtracts as locks are
    # returned, so releasing is idempotent from any state.
    l2_locked: dict[int, Bitmap] = field(default_factory=dict)
    # True once the prefix hit was stored/published; _finish_request reports
    # the final fold's hit iff no earlier step already did.
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
    write_reserved_objs: dict[ObjectKey, MemoryObj] = field(default_factory=dict)
    # L1 pin pass, set when the request starts (before L2 lookup):
    # key indices served from L1 — pinned (read-locked) in LOOKUP mode, a
    # point-in-time peek in WARM mode. Pins outside the final retained set
    # are released in _finish_request.
    l1_bitmap: Bitmap = field(default_factory=lambda: Bitmap(0))
    # Keys backing l1_bitmap with read locks (always empty in WARM mode).
    l1_pinned_keys: set[ObjectKey] = field(default_factory=set)

    group_layout_descs: dict[int, MemoryLayoutDesc] | None = None
    """Maps object_group_id to that group's layout. Each object group is a
    separate keyed L1 allocation, so each needs its own descriptor (one
    ``MemoryLayoutDesc`` describes a single group's MemoryObj). ``None`` when
    all keys share ``layout_desc`` (a single / merged object group)."""

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that pins L1-resident keys (read locks)
       and submits lookup_and_lock to all adapters.
    3. Uses PrefetchPolicy to compute a load plan from the L1 ∪ L2 union of
       lookup results; keys served from L1 are never transferred.
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
        self._pending_queue: list[
            tuple[
                PrefetchRequestId,
                list[ObjectKey],
                MemoryLayoutDesc,
                int,
                TrimPolicy,
                AttnWindowDesc,
                dict[int, MemoryLayoutDesc] | None,
                PrefetchMode,
            ]
        ] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[
            tuple[
                PrefetchRequestId,
                list[ObjectKey],
                MemoryLayoutDesc,
                int,
                TrimPolicy,
                AttnWindowDesc,
                dict[int, MemoryLayoutDesc] | None,
                PrefetchMode,
            ]
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
                "lmcache_mp.l2_adapters",
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
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        policy: TrimPolicy = TrimPolicy.PREFIX,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        group_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        mode: PrefetchMode = PrefetchMode.LOOKUP,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request for the given keys.

        Thread-safe. Can be called from any thread.

        A key counts as found if it is already resident in L1 or any L2
        adapter reports it; the retained subset of found keys is chosen by
        ``policy`` (see :class:`TrimPolicy`).  With the default ``PREFIX``
        policy, only the **contiguous prefix** of found keys is retained: if
        the caches hold keys {0, 1, 3, 4} but not key 2, only keys {0, 1}
        are retained because the gap at index 2 breaks the prefix.  Retained
        keys missing from L1 are loaded from L2; keys outside the retained
        set are never transferred, saving I/O bandwidth and L1 memory.  Use
        :meth:`query_prefetch_result` to retrieve the retained set once the
        request completes.

        Args:
            keys: List of object keys to prefetch from L2 into L1.
                The ordering defines the prefix: index 0 is the first key.
            layout_desc: Memory layout for L1 write buffer allocation.
            extra_count: Extra read locks per key (on top of the default 1)
                to acquire when transitioning loaded keys from write-locked
                to read-locked.  Must match the ``extra_count`` used in the
                corresponding ``submit_prefetch_task`` call so that all TP
                workers can each consume one read lock.
            policy: Which retained-subset policy to apply (see
                :class:`TrimPolicy`).  Defaults to ``PREFIX``.
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order.
            group_layout_descs: Maps object_group_id to that group's layout
                (each group is a separate keyed allocation, possibly with
                different tensor shapes); ``None`` when all share ``layout_desc``.
            mode: The prefetch intent (see :class:`PrefetchMode`).  ``WARM``
                forces every loaded key permanent and acquires no read lock;
                ``LOOKUP`` defers retention to the configured
                :class:`PrefetchPolicy` and read-locks loaded keys.

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._submission_queue.append(
                (
                    request_id,
                    keys,
                    layout_desc,
                    extra_count,
                    policy,
                    attn_desc,
                    group_layout_descs,
                    mode,
                )
            )
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
            (
                request_id,
                keys,
                layout_desc,
                extra_count,
                policy,
                attn_desc,
                group_layout_descs,
                mode,
            ) = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(
                request_id,
                keys,
                layout_desc,
                extra_count,
                policy,
                attn_desc,
                group_layout_descs=group_layout_descs,
                mode=mode,
            )

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _pin_l1_keys(
        self,
        keys: list[ObjectKey],
        extra_count: int,
        policy: TrimPolicy,
        attn_desc: AttnWindowDesc,
        mode: PrefetchMode,
    ) -> tuple[Bitmap, set[ObjectKey]]:
        """Peek L1, then pin (read-lock) the servable subset (LOOKUP only)::

            SW keys in L1:

            |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                          ^ L1 hit length               ^ (unknown yet)
            |      skip      |    pin     |  pin if in L1  |pin if in L1|pin if in L1|

        A lock-free peek estimates the L1 hit length. The pin pass then
        read-locks the window of that hit and every L1-resident key beyond
        it — the final window position is unknown until L2 results arrive.

        Keys out of the L1-hit window are never locked nor touched (L2
        results only extend the hit rightward, so they can never enter the
        final retained set); their eviction recency stays untouched across
        multi-round conversations.

        Args:
            keys: The request's object keys, in prefix order.
            extra_count: Extra read locks per pinned key.
            policy: Trim policy for the L1-only fold.
            attn_desc: Cross-chunk attention windows of all object groups.
            mode: WARM returns the peek with no locks taken.

        Returns:
            ``(l1_bitmap, l1_pinned_keys)`` — key indices served by L1 and
            the read-locked keys backing them (empty in WARM mode, where the
            bitmap is the unlocked point-in-time peek).
        """
        num_keys = len(keys)
        peek = self._l1_manager.peek_keys(keys)
        peek_bitmap = Bitmap(num_keys)
        for i, present in enumerate(peek):
            if present:
                peek_bitmap.set(i)

        if mode is PrefetchMode.WARM or peek_bitmap.popcount() == 0:
            return peek_bitmap, set()

        l1_hit, l1_retain = build_trim_mask(peek_bitmap, num_keys, policy, attn_desc)
        stride = attn_desc.num_object_groups * attn_desc.world_size
        l1_hit_mask = Bitmap(num_keys, l1_hit * stride)
        pin_bitmap = l1_retain | (peek_bitmap & (~l1_hit_mask))

        keys_to_pin = pin_bitmap.gather(keys)
        if not keys_to_pin:
            return Bitmap(num_keys), set()

        pin_results = self._l1_manager.reserve_read(
            keys_to_pin, extra_count=extra_count
        )
        # The fold input is the PIN RESULTS, never the peek: a key evicted
        # between peek and pin simply drops out here. pin_bitmap's set
        # indices are exactly keys_to_pin's positions, so zip them instead
        # of re-scanning all keys.
        l1_bitmap = Bitmap(num_keys)
        l1_pinned_keys: set[ObjectKey] = set()
        pin_indices = pin_bitmap.get_indices_list()
        for idx, key in zip(pin_indices, keys_to_pin, strict=True):
            err, _obj = pin_results[key]
            if err == L1Error.SUCCESS:
                l1_bitmap.set(idx)
                l1_pinned_keys.add(key)
        return l1_bitmap, l1_pinned_keys

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        policy: TrimPolicy = TrimPolicy.PREFIX,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        group_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        mode: PrefetchMode = PrefetchMode.LOOKUP,
    ) -> None:
        """Pin L1-resident keys, then submit lookup_and_lock to all live
        (non-draining) adapters for a new request."""
        l1_bitmap, l1_pinned_keys = self._pin_l1_keys(
            keys, extra_count, policy, attn_desc, mode
        )
        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=keys,
            layout_desc=layout_desc,
            phase=PrefetchPhase.LOOKUP,
            extra_count=extra_count,
            policy=policy,
            attn_desc=attn_desc,
            mode=mode,
            group_layout_descs=group_layout_descs,
            l1_bitmap=l1_bitmap,
            l1_pinned_keys=l1_pinned_keys,
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
            task_id = adapter.submit_lookup_and_lock_task(keys, layout_desc)
            request.pending_lookup_tasks[adapter_id] = task_id
        self._in_flight_requests[request_id] = request
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(keys),
                    "adapter_count": len(request.pending_lookup_tasks),
                    "key_count_per_salt": Counter(k.cache_salt for k in keys),
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
        load_plan = trim_load_plan_with_mask(load_plan, ~request.l1_bitmap)

        # Step 2 — the union fold sets the L1+L2 hit length; pins outside
        # the retained set are released at finish.
        # SW keys in L1:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |      skip      |unpin@finish|  unpin@finish  |  keep pin  |unpin@finish|
        # Trim to the retained subset of the L1 ∪ L2 union, so
        # L1-resident keys extend the hit even where L2 lacks them.
        union_bitmap = merge_bitmaps(load_plan.values(), num_keys) | request.l1_bitmap
        hit_length, retained = build_trim_mask(
            union_bitmap,
            num_keys,
            request.policy,
            request.attn_desc,
        )
        trimmed_plan = trim_load_plan_with_mask(load_plan, retained)

        if not trimmed_plan:
            # Nothing to load from L2: L1 alone serves the retained set (or
            # there is no hit at all). Finish reconciles all locks and
            # reports the hit.
            self._finish_request(request)
            return

        # Step 3 — reserve L1 write buffers for the plan keys; any
        # reservation failure drops its key.
        # SW keys in L2 (and not in L1):
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        |  loading*  |     -      |
        # (*) dropped on OOM or contention.
        plan_bitmap = merge_bitmaps(trimmed_plan.values(), num_keys)
        keys_to_reserve = plan_bitmap.gather(request.keys)
        reserved = self._reserve_load_buffers(request, keys_to_reserve)
        reserved_bitmap = Bitmap(num_keys)
        plan_indices = plan_bitmap.get_indices_list()
        for idx, key in zip(plan_indices, keys_to_reserve, strict=True):
            if key in reserved:
                reserved_bitmap.set(idx)

        # Step 4 — drops move the hit length left: re-fold so the reported
        # hit matches (same row as step 2, against the smaller hit).
        if len(reserved) < len(keys_to_reserve):
            hit_length, retained = build_trim_mask(
                reserved_bitmap | request.l1_bitmap,
                num_keys,
                request.policy,
                request.attn_desc,
            )

        # Step 5 — plan = loading keys inside the final hit.
        # SW plan:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        | load these |     -      |
        # The plan covers only keys holding a write buffer (keys
        # served from L1 need no transfer; unreserved keys have nowhere to
        # load into).
        trimmed_plan = trim_load_plan_with_mask(
            trimmed_plan, reserved_bitmap & retained
        )
        request.load_plan = trimmed_plan

        if not trimmed_plan:
            # Nothing left to load after reservation failures; finish
            # returns the reserved buffers and reconciles all locks.
            self._finish_request(request)
            return

        # Step 6 — free L2 lookup locks for keys outside the plan.
        # L2 lookup locks:
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |      free      |    free    |      free      | keep plan  |    free    |
        self._release_l2_locks(request, keep=request.load_plan)

        # Step 7 — submit loads; report the hit.
        # SW keys in L2 (and not in L1):
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        |  loading   |     -      |
        self._submit_load_tasks(request, trimmed_plan)
        self._report_lookup_hit(request, hit_length)

    def _reserve_load_buffers(
        self,
        request: InFlightPrefetchRequest,
        keys_to_reserve: list[ObjectKey],
    ) -> set[ObjectKey]:
        """Reserve L1 write buffers for the keys to load from L2.

        Successful reservations are recorded on
        ``request.write_reserved_keys`` / ``request.write_reserved_objs``.
        Any failure drops its key (with an ``L2_PREFETCH_FAILED`` event) —
        no lock is ever acquired through a failure result, so a concurrent
        eviction between manager calls cannot invalidate the reported hit.

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

        # When per-group layouts are available, batch reserve_write by
        # object_group_id so each group uses its own tensor shapes.
        if request.group_layout_descs:
            # Chunk-major order interleaves group ids, so sort before
            # groupby (stable: within-group prefix order is preserved).
            write_results: dict[ObjectKey, tuple[L1Error, MemoryObj | None]] = {}
            by_group = sorted(keys_to_reserve, key=attrgetter("object_group_id"))
            for gid, group_iter in groupby(by_group, key=attrgetter("object_group_id")):
                group_keys = list(group_iter)
                gld = request.group_layout_descs.get(gid, request.layout_desc)
                gr = self._l1_manager.reserve_write(
                    keys=group_keys,
                    is_temporary=[not retention_map[k] for k in group_keys],
                    layout_desc=gld,
                    mode="new",
                )
                write_results.update(gr)
        else:
            write_results = self._l1_manager.reserve_write(
                keys=keys_to_reserve,
                is_temporary=[not r for r in retentions],
                layout_desc=request.layout_desc,
                mode="new",
            )

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
            # The key appeared in L1 after the pin pass (e.g. a
            # concurrent request is loading it). Dropped rather than
            # promoted; a later request finds it through its L1 pin pass.
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
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if request.all_loads_done():
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
            request.l2_locked[adapter_idx] = result
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
        """
        Finish a request — the single completion path, with or without an
        L2 load: build the result bitmap, transition loaded keys, return
        failed/unused write buffers, release every read lock outside the
        final retained set, and report the retained-key bitmap.

        Callable with an empty load plan (pure L1 hit, no hit at all, or
        all reservations dropped): the load-specific steps degenerate to
        no-ops and every write-reserved buffer is returned as failed.
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

        # Release every L2 lookup lock still held (idempotent: anything
        # already released at step 6 was subtracted from the tracked state).
        self._release_l2_locks(request, keep={})

        l1_mgr = self._l1_manager

        # Finish — failed loads delete their buffer; the fold below decides
        # the final retained set and every read lock outside it is released.
        # SW keys in L2 (and not in L1):
        # |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
        #                               ^ L1 hit length               ^ L1+L2 hit length
        # |       -        |     -      |       -        |load→pinned |     -      |
        # SW keys in L1:
        # |      skip      |   unpin    |     unpin      |   pinned   |   unpin    |
        if loaded_keys:
            if request.mode is PrefetchMode.WARM:
                # Warm: make ready, pin nothing.
                l1_mgr.finish_write(loaded_keys)
            else:
                # write-locked -> read-locked; extra_count so each TP worker
                # gets its own read lock.
                l1_mgr.finish_write_and_reserve_read(
                    loaded_keys, extra_count=request.extra_count
                )

        # Clean up failed keys
        if failed_keys:
            l1_mgr.finish_write(failed_keys)
            l1_mgr.delete(failed_keys)

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

        # Include keys served from L1 (pinned when the request started) so
        # the fold sees all object groups.
        result_bitmap = result_bitmap | request.l1_bitmap

        # Release read locks for any key outside the retained set (partial
        # load failures can create gaps). WARM requests hold no read locks.
        hit_length, retained = build_trim_mask(
            result_bitmap,
            num_keys,
            request.policy,
            request.attn_desc,
        )
        if request.mode is not PrefetchMode.WARM:
            released = (result_bitmap & (~retained)).gather(request.keys)
            if released:
                l1_mgr.finish_read(released, extra_count=request.extra_count)

        # No-load finishes reach here without a reported hit; the load path
        # already reported at submit time (so the engine never waits on the
        # load) and is not re-reported.
        if not request.hit_reported:
            self._report_lookup_hit(request, hit_length)

        self._complete_request(request.request_id, retained)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _release_l2_locks(
        self, request: InFlightPrefetchRequest, keep: dict[int, Bitmap]
    ) -> None:
        """Release held L2 lookup locks, except the key indices in ``keep``.

        Operates on the tracked lock state (``request.l2_locked``) and
        subtracts what it releases, so any release sequence is idempotent —
        every path can converge on ``_finish_request`` without
        double-unlocking.

        Args:
            request: The in-flight request whose L2 locks to release.
            keep: Adapter index -> key indices whose locks stay held (the
                load plan); pass ``{}`` to release everything still held.
        """
        num_keys = len(request.keys)
        for adapter_idx, held in list(request.l2_locked.items()):
            keep_bitmap = keep.get(adapter_idx, Bitmap(num_keys))
            unlock_keys = (held & (~keep_bitmap)).gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)
            remaining = held & keep_bitmap
            if remaining.popcount() == 0:
                del request.l2_locked[adapter_idx]
            else:
                request.l2_locked[adapter_idx] = remaining

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _complete_request(self, request_id: PrefetchRequestId, result: Bitmap) -> None:
        """Store the retained-key bitmap and remove from in-flight tracking."""
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
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write(request.write_reserved_keys)
                    l1_mgr.delete(request.write_reserved_keys)
            self._release_l2_locks(request, keep={})
            if request.l1_pinned_keys:
                l1_mgr.finish_read(
                    list(request.l1_pinned_keys),
                    extra_count=request.extra_count,
                )
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
