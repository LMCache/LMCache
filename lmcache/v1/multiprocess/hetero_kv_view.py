# SPDX-License-Identifier: Apache-2.0
"""A view of the storage manager that serves heterogeneous engines.

Engines that share a cache need not store KV the same way: today they may
differ in TP size. :class:`HeteroKVView` has the storage manager's prefetch
interface, so a caller asks for objects in its own form and gets a result as
if they were stored that way, while the storage manager stays unaware of the
difference. How one form is translated into another (which keys to ask for
instead, how to convert a hit) is a plan; :mod:`tp_reshard` provides the TP
plan.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from typing import cast
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    PrefetchHandle,
    PrefetchLockMode,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.tp_reshard import HeteroTPPlan, HeteroTPPlanner

logger = init_logger(__name__)

# Concurrent heterogeneous prefetches. Each holds a thread while it waits for
# its prefetches, so this bounds the prefetches in flight, not CPU use.
_LOOKUP_WORKERS = 32


class HeteroKVView:
    """The storage manager's prefetch interface, across heterogeneous engines.

    Same contract as :meth:`StorageManager.submit_prefetch_task`,
    :meth:`StorageManager.query_prefetch_status` and
    :meth:`StorageManager.wait_prefetch_status`, so a caller swaps it in for
    the storage manager without other changes. A prefetch that has a plan
    (see :meth:`plan`) finishes in the background: once its own prefetch is
    done, the chunks it missed are prefetched in the source form, converted
    into L1 in the spec's form and appended, so its result reads as if the
    spec had hit them itself. Any other prefetch goes straight to the storage
    manager.

    Args:
        storage_manager: Storage manager serving the prefetches and holding
            the converted objects.
        registry: Layout registry holding the KV head count engines report.
        source_world_size: TP size other engines of the same model store
            with; ``None`` makes this view a plain pass-through.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        registry: LayoutDescRegistry,
        source_world_size: int | None,
    ) -> None:
        self._sm = storage_manager
        self._tp = HeteroTPPlanner(storage_manager, registry, source_world_size)
        # Threads start on first use, so a server that never needs a plan
        # pays nothing.
        self._lookups = ThreadPoolExecutor(
            max_workers=_LOOKUP_WORKERS, thread_name_prefix="hetero-kv-lookup"
        )
        # Background prefetches by request id; each owns its handle's result.
        self._running: dict[int, Future[PrefetchResult]] = {}
        self._running_lock = threading.Lock()

    def plan(self, spec: PrefetchTaskSpec) -> HeteroTPPlan | None:
        """Build the plan that completes a prefetch's hit, or ``None``.

        Only a locked prefix prefetch of at least one chunk can have one: the
        merge appends converted chunks to a prefix hit and leaves them
        read-locked like it. The TP plan then decides; see
        :meth:`HeteroTPPlanner.plan`.

        Args:
            spec: The prefetch's spec.

        Returns:
            The plan, or ``None`` if the prefetch is served as is.
        """
        if (
            spec.fetching_policy != "prefix"
            or spec.lock_mode != PrefetchLockMode.LOCK
            or spec.group_size == 0
        ):
            return None
        return self._tp.plan(spec)

    def submit_prefetch_task(
        self, spec: PrefetchTaskSpec, external_request_id: str = ""
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously, completing the hit across TP.

        Args:
            spec: The request (see :class:`PrefetchTaskSpec`).
            external_request_id: Caller id for end-to-end log tracing.

        Returns:
            Handle to query or wait on through this view. A qualifying
            prefetch's result belongs to the view: querying the handle on
            the storage manager directly would steal it.
        """
        plan = self.plan(spec)
        handle = self._sm.submit_prefetch_task(spec, external_request_id)
        if plan is not None:
            with self._running_lock:
                self._running[handle.prefetch_request_id] = self._lookups.submit(
                    self._complete, plan, handle
                )
        return handle

    def query_prefetch_status(self, handle: PrefetchHandle) -> PrefetchResult | None:
        """Query the status of a prefetch submitted through this view.

        Args:
            handle: The prefetch.

        Returns:
            The prefetch's result once it has finished, with every hit chunk
            read-locked; None while it is still in progress. A qualifying
            prefetch is in progress until its cross-TP prefetch and reshard
            finish.

        Note:
            Each result is returned once; later calls for the same handle
            return None.
        """
        with self._running_lock:
            future = self._running.get(handle.prefetch_request_id)
            if future is None:
                return self._sm.query_prefetch_status(handle)
            if not future.done():
                return None
            del self._running[handle.prefetch_request_id]
        return future.result()

    def wait_prefetch_status(self, handle: PrefetchHandle, timeout: float) -> bool:
        """Block until a prefetch submitted through this view has a result.

        Args:
            handle: The prefetch.
            timeout: Maximum number of seconds to wait.

        Returns:
            True if a result is available within the timeout, False if the
            wait timed out.
        """
        with self._running_lock:
            future = self._running.get(handle.prefetch_request_id)
        if future is None:
            return self._sm.wait_prefetch_status(handle, timeout)
        return bool(wait_futures([future], timeout).done)

    def close(self) -> None:
        """Stop the lookup and reshard threads after every submitted one finishes.

        Must run while the storage manager still serves prefetches: a running
        lookup waits for its own. See :meth:`L1Resharder.close`.
        """
        self._lookups.shutdown(wait=True)
        self._tp.close()

    def _complete(self, plan: HeteroTPPlan, handle: PrefetchHandle) -> PrefetchResult:
        """Wait for the spec's own prefetch, then fill its miss from the source TP.

        Args:
            plan: The prefetch's cross-TP plan.
            handle: Handle of the spec's own prefetch.

        Returns:
            Its result, with the resharded chunks appended; it never raises.
        """
        result = self._wait(handle)
        found, _ = fold_unfold_grouped(result.hit_cells, handle.sliding_windows)
        if found == plan.num_chunks:
            return result
        try:
            return self._complete_from_source(plan, found, result, handle)
        except Exception:
            # The spec's own locks are still held, so its hit stays a valid
            # answer.
            logger.exception(
                "Cross-TP prefetch failed for request %s; answering with the "
                "same-TP hit",
                handle.external_request_id,
            )
            return result

    def _complete_from_source(
        self,
        plan: HeteroTPPlan,
        start: int,
        first: PrefetchResult,
        handle: PrefetchHandle,
    ) -> PrefetchResult:
        """Prefetch chunks ``start`` on at the source TP, reshard and merge.

        Args:
            plan: The prefetch's cross-TP plan.
            start: Chunks the spec's own prefetch hit.
            first: The spec's own prefetch result.
            handle: Handle of the spec's own prefetch.

        Returns:
            ``first`` with the resharded chunks appended.

        Raises:
            Exception: A failure to start the reshard propagates, after the
                source locks are released.
        """
        request_id = handle.external_request_id
        spec = plan.cross_tp_spec(start)
        cross_handle = self._sm.submit_prefetch_task(spec, request_id)
        result = self._wait(cross_handle)
        found, _ = fold_unfold_grouped(result.hit_cells, cross_handle.sliding_windows)
        if found == 0:
            return first
        try:
            reshard = plan.convert(found, result, request_id)
        except Exception:
            # The source objects are this call's to release until a reshard
            # owns them.
            self._sm.finish_read_prefetched(
                [k for row in spec.key_groups for k in row.keys[:found]],
                read_locks=spec.num_kv_readers,
                l1_owners=result.l1_owners,
            )
            raise
        return plan.merge(first, reshard.result())

    def _wait(self, handle: PrefetchHandle) -> PrefetchResult:
        """Block until a storage manager prefetch finishes and take its result.

        Args:
            handle: The prefetch.

        Returns:
            Its result.
        """
        # A prefetch always publishes a result; the timeout only bounds each
        # wait on the condition variable.
        while not self._sm.wait_prefetch_status(handle, 1.0):
            pass
        return cast(PrefetchResult, self._sm.query_prefetch_status(handle))
