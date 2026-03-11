# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.error import L1Error, strerror
from lmcache.v1.distributed.internal_api import StorageManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers import (
    EvictionController,
    PrefetchController,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.logger.storage_manager_stats_logger import (
    StorageManagerStatsLogger,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PrefetchHandle:
    request_id: int
    """Opaque ID for tracking L2 prefetch in the controller. -1 if no L2 request."""

    l1_prefix_hit_count: int
    """Number of leading keys already in L1 at submission time."""

    total_requested_keys: int
    """Total number of keys originally requested."""

    submit_time: float
    """Monotonic timestamp when the prefetch task was submitted."""


class StorageManager:
    def __init__(self, config: StorageManagerConfig):
        self._l1_manager = L1Manager(config.l1_manager_config)
        self._registered_listeners: list[StorageManagerListener] = []

        # Eviction controller
        self._eviction_controller = EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.start()

        # L2 adapters and store controller
        l1_memory_desc = self._l1_manager.get_l1_memory_desc()
        self._l2_adapters: list[L2AdapterInterface] = [
            create_l2_adapter(ac, l1_memory_desc)
            for ac in config.l2_adapter_config.adapters
        ]

        adapter_descriptors = [
            AdapterDescriptor(index=i, config=ac)
            for i, ac in enumerate(config.l2_adapter_config.adapters)
        ]

        self._store_controller = StoreController(
            l1_manager=self._l1_manager,
            l2_adapters=self._l2_adapters,
            adapter_descriptors=adapter_descriptors,
            policy=DefaultStorePolicy(),
        )
        self._store_controller.start()

        # Prefetch controller
        self._prefetch_controller = PrefetchController(
            l1_manager=self._l1_manager,
            l2_adapters=self._l2_adapters,
            adapter_descriptors=adapter_descriptors,
            policy=DefaultPrefetchPolicy(),
        )
        self._prefetch_controller.start()

        # Self-register observability logger
        sm_stats_logger = StorageManagerStatsLogger()
        self.register_listener(sm_stats_logger)
        get_prometheus_controller().register_logger(sm_stats_logger)

    def register_listener(self, listener: StorageManagerListener) -> None:
        """Register a listener for StorageManager events.

        Args:
            listener: The listener to register.
        """
        self._registered_listeners.append(listener)

    # External APIs for serving engine integration code to call
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        """
        Reserve the object for writing into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Description of the memory layout
                for the objects to be reserved.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            dict[ObjectKey, MemoryObj]: A dictionary mapping object keys to their
                reserved memory objects. Note that not all requested keys could be
                reserved (e.g., out of memory or write conflict)
        """
        reserve_result = self._l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False] * len(keys),
            layout_desc=layout_desc,
            mode=mode,
        )

        result = {k: m for k, (e, m) in reserve_result.items() if m is not None}
        successful_keys = list(result.keys())
        failed_keys = [k for k, (e, m) in reserve_result.items() if m is None]
        for listener in self._registered_listeners:
            listener.on_sm_reserved_write(successful_keys, failed_keys)
        return result

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        finish_result = self._l1_manager.finish_write(keys)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        for listener in self._registered_listeners:
            listener.on_sm_write_finished(successful_keys, failed_keys)

        # TODO: global key states update

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        """
        Read the memory objects from L1 storage that has been prefetched beforehand.
        Yielding an optional list of memory objects corresponding to the requested
        keys. If any the object is not found in L1, None is yielded.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for reading.

        Returns:
            Iterator[list[MemoryObj] | None]: An iterator yielding an optional list of
                memory objects corresponding to the requested keys.

        Note:
            If any object is not found in L1 storage, None is yielded. In this case,
            this function will release release the read lock of all successfully read
            memory objects when exiting the context.

            If the caller raised exception during the processing of the yielded memory
            objects, this function will ensure that the read locks will be decreased.
        """
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        bad_keys: list[ObjectKey] = []
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                bad_keys.append(k)
                all_good = False
                continue

            good_keys.append(k)
            good_objs.append(o)

        successfully_yielded = False

        try:
            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception:
            logger.exception(
                "Exception occurred while processing read prefetched results",
            )
            raise
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._l1_manager.finish_read(good_keys)
                for listener in self._registered_listeners:
                    listener.on_sm_read_prefetched_finished(good_keys, bad_keys)

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            extra_count: Extra read locks to release per key
                (on top of the default 1).
        """
        finish_result = self._l1_manager.finish_read(keys, extra_count=extra_count)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        for listener in self._registered_listeners:
            listener.on_sm_read_prefetched_finished(successful_keys, failed_keys)

    def submit_prefetch_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            keys: Object keys to prefetch.
            layout_desc: Memory layout description.
            extra_count: Extra workers (on top of the default
                1) that will independently retrieve the same
                key.  Total locks = 1 + extra_count.

        Returns:
            PrefetchHandle to track the task.
        """
        # NOTE: now we only have L1, so the prefetch is essentially checking how many
        # objects are already in L1, and adding read locks to them.

        l1_read_result = self._l1_manager.reserve_read(keys, extra_count=extra_count)
        hit_count = 0
        for key in keys:
            entry = l1_read_result.get(key, None)
            if entry is None:
                break

            err, obj = entry
            if err != L1Error.SUCCESS:
                break

            hit_count += 1

        # NOTE: For L1, there will be cases that "object in the middle" is not found.
        # In this case, we need to `finish_read` for the latter objects so that
        # there won't be dangling read locks.
        skipped_keys = []
        for key in keys[hit_count:]:
            if key in l1_read_result and l1_read_result[key][1] is not None:
                # this key is actually reserved, need to release the read lock
                skipped_keys.append(key)

        if skipped_keys:
            self._l1_manager.finish_read(skipped_keys, extra_count=extra_count)

        for listener in self._registered_listeners:
            listener.on_sm_read_prefetched(keys[:hit_count], keys[hit_count:])

        # Submit remaining keys to L2 prefetch controller
        remaining_keys = keys[hit_count:]
        request_id = -1
        if remaining_keys and self._l2_adapters:
            request_id = self._prefetch_controller.submit_prefetch_request(
                remaining_keys,
                layout_desc,
                extra_count=extra_count,
            )

        submit_time = time.monotonic()
        logger.debug(
            "Prefetch request submitted: %d total keys, "
            "%d L1 prefix hits, %d remaining for L2 (request_id=%d)",
            len(keys),
            hit_count,
            len(remaining_keys),
            request_id,
        )

        return PrefetchHandle(
            request_id=request_id,
            l1_prefix_hit_count=hit_count,
            total_requested_keys=len(keys),
            submit_time=submit_time,
        )

    def query_prefetch_status(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            the number of prefix hit chunks if the prefetch is done, None if
            it's still in progress.
        """
        l2_result: int = 0

        # Have L2 request, need to check the result from prefetch controller
        if handle.request_id != -1:
            l2_r = self._prefetch_controller.query_prefetch_result(handle.request_id)

            if l2_r is None:
                return None
            l2_result = l2_r  # Just to make linter happy

        total_hits = handle.l1_prefix_hit_count + l2_result
        elapsed_ms = (time.monotonic() - handle.submit_time) * 1000

        if total_hits > 0:
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d prefix hits (%d L1, %d L2) in %.1f ms "
                "(request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                handle.l1_prefix_hit_count,
                l2_result,
                elapsed_ms,
                handle.request_id,
            )
        return total_hits

    # =========================================================================
    # Synchronous lookup / load API (for SYNC_LOOKUP protocol)
    # =========================================================================

    def synchronous_lookup_and_lock(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
    ) -> tuple[int, list[ObjectKey], dict[int, Bitmap] | None]:
        """Synchronous prefix lookup across L1 and L2.

        Checks L1 first (immediate), then synchronously queries L2
        adapters (blocking until all respond).  L2 objects that are
        found are pinned (pin_count incremented) to prevent eviction
        before the subsequent load.

        Does **not** perform L2-to-L1 data movement.

        Args:
            keys: Object keys to look up.
            layout_desc: Memory layout descriptor (passed through for
                later use by the load phase).
            extra_count: Extra MLA reader locks for L1.

        Returns:
            Tuple of:
            - total prefix hit count (L1 + L2)
            - remaining keys not found in L1 (subset of *keys*)
            - L2 lookup results (dict[adapter_idx, Bitmap]) or None if
              no L2 adapters or no remaining keys
        """
        # -- L1 prefix scan (same logic as submit_prefetch_task) ---------------
        l1_read_result = self._l1_manager.reserve_read(keys, extra_count=extra_count)
        l1_hit_count = 0
        for key in keys:
            entry = l1_read_result.get(key, None)
            if entry is None:
                break
            err, obj = entry
            if err != L1Error.SUCCESS:
                break
            l1_hit_count += 1

        # Release non-prefix L1 read locks
        skipped_keys = []
        for key in keys[l1_hit_count:]:
            if key in l1_read_result and l1_read_result[key][1] is not None:
                skipped_keys.append(key)
        if skipped_keys:
            self._l1_manager.finish_read(skipped_keys, extra_count=extra_count)

        for listener in self._registered_listeners:
            listener.on_sm_read_prefetched(keys[:l1_hit_count], keys[l1_hit_count:])

        # -- L2 synchronous lookup (blocking) ----------------------------------
        remaining_keys = keys[l1_hit_count:]
        l2_lookup_results: dict[int, Bitmap] | None = None
        l2_prefix_hits = 0

        if remaining_keys and self._l2_adapters:
            l2_lookup_results = self._prefetch_controller.synchronous_lookup(
                remaining_keys,
            )
            if l2_lookup_results:
                # Merge all adapter bitmaps and count contiguous prefix
                merged = Bitmap(len(remaining_keys))
                for bitmap in l2_lookup_results.values():
                    merged = merged | bitmap
                l2_prefix_hits = merged.count_leading_ones()

        total_hits = l1_hit_count + l2_prefix_hits

        if total_hits > 0:
            logger.info(
                "Synchronous lookup completed (L1+L2): "
                "%d/%d prefix hits (%d L1, %d L2)",
                total_hits,
                len(keys),
                l1_hit_count,
                l2_prefix_hits,
            )

        return total_hits, remaining_keys, l2_lookup_results

    def execute_prefetch_load(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        lookup_results: dict[int, Bitmap],
    ) -> int:
        """Execute L2-to-L1 data movement for previously looked-up keys.

        Called from the RETRIEVE path after a prior
        :meth:`synchronous_lookup_and_lock`.  The L2 objects must still
        be pinned from the lookup.  Pins are released during the load
        phase (phase-1 and phase-2 unlocks).

        Args:
            keys: The remaining keys not in L1 (from
                ``synchronous_lookup_and_lock``).
            layout_desc: Memory layout for L1 buffer allocation.
            lookup_results: L2 lookup results from
                ``synchronous_lookup_and_lock``.

        Returns:
            Number of prefix hits loaded into L1 (with read locks held).
        """
        return self._prefetch_controller.execute_load_phase(
            keys, layout_desc, lookup_results,
        )

    def unlock_l2_lookups(
        self,
        keys: list[ObjectKey],
        lookup_results: dict[int, Bitmap],
    ) -> None:
        """Release L2 pins without loading.

        Use when a request is cancelled after
        :meth:`synchronous_lookup_and_lock` but before
        :meth:`execute_prefetch_load`.

        Args:
            keys: The remaining keys (from
                ``synchronous_lookup_and_lock``).
            lookup_results: L2 lookup results (from
                ``synchronous_lookup_and_lock``).
        """
        self._prefetch_controller.unlock_lookup_results(keys, lookup_results)

    def clear(self, force: bool = False):
        """
        Clear data in the storage manager.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        self._l1_manager.clear(force=force)

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._prefetch_controller.stop()
        self._store_controller.stop()
        self._eviction_controller.stop()

        for adapter in self._l2_adapters:
            adapter.close()

        self._l1_manager.close()

    def report_status(self) -> dict:
        """Return a status dict aggregating all sub-component statuses."""
        l1 = self._l1_manager.report_status()
        store = self._store_controller.report_status()
        prefetch = self._prefetch_controller.report_status()
        eviction = self._eviction_controller.report_status()
        adapters = [a.report_status() for a in self._l2_adapters]
        children = [l1, store, prefetch, eviction] + adapters
        return {
            "is_healthy": all(c["is_healthy"] for c in children),
            "l1_manager": l1,
            "store_controller": store,
            "prefetch_controller": prefetch,
            "eviction_controller": eviction,
            "l2_adapters": adapters,
            "num_l2_adapters": len(self._l2_adapters),
        }

    # Functions for debugging and testing
    def memcheck(self) -> bool:
        """
        Perform memory check for all storage tiers.

        Returns:
            True if memory is consistent, False otherwise.
        """
        return self._l1_manager.memcheck()
