# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass
from functools import partial
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocol import RequestType

logger = init_logger(__name__)

# Sentinel address for keys that were not found (or fell past the L1 prefix).
_INVALID_ADDRESS = TransferChannelAddress(offset=-1, size=0)


@dataclass
class _P2PLookupJob:
    handle: PrefetchHandle
    """ The handle returned by the storage manager """

    keys: list[ObjectKey]
    """ The object keys submitted for this lookup, in request order """


class P2PController:
    """Handles lookup/prefetch/unlock requests from peer P2P adapters

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._next_task_id = 0
        self._jobs: dict[int, _P2PLookupJob] = {}
        self._job_lock = threading.Lock()
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            List of handler specs for lookup-related request types.
        """
        return [
            HandlerSpec(
                RequestType.P2P_LOOKUP_AND_LOCK,
                self.p2p_lookup_and_lock,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.P2P_QUERY_LOOKUP_RESULTS,
                self.p2p_query_lookup_results,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.P2P_UNLOCK_OBJECTS,
                self.p2p_unlock_objects,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        return {
            "active_p2p_lookup_jobs": self._active_job_count(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""
        pass

    # -----------------------------------------------------------------
    # RPC Handlers
    # -----------------------------------------------------------------

    def p2p_lookup_and_lock(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> int:
        """Submit a lookup and lock.

        After L2 prefetch is enabled, the found chunks will be feteched
        from L2 to L1.

        Args:
            keys: the list of object keys to look up and lock.
            layout_desc: memory layout description of the objects.

        Returns:
            A unique task id (int) for querying the lookup status later
        """
        with self._job_lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        # NOTE: skip_l2=True -- only objects already resident in L1 are locked.
        handle = self._ctx.storage_manager.submit_prefetch_task(
            keys,
            layout_desc,
            external_request_id=f"p2p-{task_id}",
            skip_l2=True,
        )

        with self._job_lock:
            self._jobs[task_id] = _P2PLookupJob(handle=handle, keys=keys)

        logger.debug(
            "P2P lookup submitted: task_id=%d, %d keys, %d L1 prefix hits",
            task_id,
            len(keys),
            len(handle.l1_found_indices),
        )
        return task_id

    def p2p_query_lookup_results(
        self,
        task_id: int,
    ) -> list[TransferChannelAddress] | None:
        """Query the results of the lookup request specified by the task ID.

        Returning a list of TransferChannelAddress objects indicates when the
        lookup is completed. None indicates the lookup has not completed yet.

        The returned list will always have the same length as the number of
        keys submitted in the corresponding p2p_lookup_and_lock call. For
        objects that is not found, the corresponding TransferChannelAddress
        will have an invalid offset (negative value).

        Args:
            task_id: The unique task ID returned by p2p_lookup_and_lock.

        Returns:
            A list of TransferChannelAddress objects if the lookup is complete,
            or None if the lookup is still in progress or the result has been
            queried. (Exactly once request)
        """
        with self._job_lock:
            job = self._jobs.get(task_id)
        if job is None:
            logger.warning(
                "P2P lookup job %d not found (already consumed or invalid)",
                task_id,
            )
            return None

        found = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if found is None:
            # Still in progress (only possible once L2 prefetch is enabled).
            return None

        addresses = self._build_addresses(job, found.count_leading_ones())

        with self._job_lock:
            self._jobs.pop(task_id, None)
        return addresses

    def p2p_unlock_objects(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """Unlock the specified object keys.

        Args:
            keys: the list of object keys to unlock.
        """
        if not keys:
            return
        self._ctx.storage_manager.finish_read_prefetched(keys)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_addresses(
        self,
        job: _P2PLookupJob,
        hit_count: int,
    ) -> list[TransferChannelAddress]:
        """Build the per-key transfer addresses for a completed lookup.

        The first ``hit_count`` keys form the locked L1 prefix; their addresses
        are read via ``unsafe_read``. Every remaining key gets an invalid
        address.
        """
        addresses = [_INVALID_ADDRESS] * len(job.keys)
        if hit_count == 0:
            return addresses

        found_keys = job.keys[:hit_count]
        good_keys, good_objs = self._ctx.storage_manager.unsafe_read(found_keys)
        obj_by_key = dict(zip(good_keys, good_objs, strict=True))

        for i, key in enumerate(found_keys):
            obj = obj_by_key.get(key)
            if obj is None:
                # Locked but unreadable (e.g. evicted under a race); leave it
                # marked invalid so the peer skips it.
                continue
            addresses[i] = TransferChannelAddress(
                offset=obj.shm_offset,
                size=obj.shm_byte_length,
            )
        return addresses

    def _active_job_count(self) -> int:
        """Return the number of active P2P lookup jobs (thread-safe)."""
        with self._job_lock:
            return len(self._jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for P2P controller metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_p2p_lookup_jobs",
            "Number of active P2P lookup jobs",
            self._active_job_count,
        )
