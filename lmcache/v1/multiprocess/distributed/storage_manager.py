# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.distributed.config import StorageManagerConfig
from lmcache.v1.multiprocess.distributed.error import L1Error, strerror
from lmcache.v1.multiprocess.distributed.l1_manager import L1Manager
from lmcache.v1.multiprocess.distributed.storage_controllers import (
    EvictionController,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PrefetchHandle:
    _prefix_hit_chunks: int
    """ how many chunks are hit in the prefix of the requested keys """


class StorageManager:
    def __init__(self, config: StorageManagerConfig):
        self._l1_manager = L1Manager(config.l1_manager_config)

        # Eviction controller
        self._eviction_controller = EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.start()

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

        return {k: m for k, (e, m) in reserve_result.items() if m is not None}

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        self._l1_manager.finish_write(keys)

        # TODO: global key states update
        # TODO: trigger L2 controller

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
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                all_good = False
                continue

            good_keys.append(k)
            good_objs.append(o)

        successfully_yielded = False

        try:
            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception as e:
            logger.warning(
                "Exception occurred while processing read prefetched results: %s",
                str(e),
            )
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._l1_manager.finish_read(good_keys)

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish reading of the prefetched objects, releasing their read locks.

        Args:
            keys (list[ObjectKey]): List of object keys that have been read.
        """
        self._l1_manager.finish_read(keys)

    def submit_prefetch_task(
        self,
        keys: list[ObjectKey],
    ) -> PrefetchHandle:
        """
        Prefetch the objects into L1 memory asynchronously. The prefetched object
        will be added with read locks.

        Args:
            keys (list[ObjectKey]): List of object keys to prefetch.

        Returns:
            PrefetchHandle: A handle to track the prefetch task.
        """
        # NOTE: now we only have L1, so the prefetch is essentially checking how many
        # objects are already in L1, and adding read locks to them.

        l1_read_result = self._l1_manager.reserve_read(keys)
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
            self._l1_manager.finish_read(skipped_keys)

        return PrefetchHandle(_prefix_hit_chunks=hit_count)

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
        return handle._prefix_hit_chunks

    def clear(self):
        """
        Clear all data in the storage manager.
        """
        self._l1_manager.clear()

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._eviction_controller.stop()
        self._l1_manager.close()

    # Functions for debugging and testing
    def memcheck(self) -> None:
        """
        Perform memory check for all storage tiers.
        """
        self._l1_manager.memcheck()
