# SPDX-License-Identifier: Apache-2.0
# NOTE: this file will be moved and the class implementation
# will be largely refactored in the future.
# Standard
from collections import OrderedDict
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import compress
from typing import Any, Generic, Iterator, TypeVar, Union
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MixedMemoryAllocator
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy

logger = init_logger(__name__)

ReserveHandle = int
ReserveResult = tuple[ReserveHandle, dict[IPCCacheEngineKey, MemoryObj]]


class MemoryExhaustedError(Exception):
    """Raised when the memory allocation cannot be completed due to
    insufficient memory.
    """

    pass


# TODO: move this to a separate module
LockKey = TypeVar("LockKey", bound=Hashable)


@dataclass
class LockMetadata:
    count: int
    expire_time: float


class LockManager(Generic[LockKey]):
    """
    A thread-safe class to manage the "locked" keys so that they won't get
    evicted.

    Motivation:
        Usually, lookup and retrieval won’t happen at the same time.
        Therefore, LMCache should make sure that the KV cache that is being
        "looked up” is guaranteed to be retrieved (with a TTL, to prevent
        the memory leak).

    Semantics:
        1. A "locked" memory object in LMCache cannot be evicted until it’s
        fully unlocked

        2. The lock can be accumulated, which means we can lock a memory object
        twice, which will need 2 "unlocks" to fully make it evictable.
        The main motivation is that multiple instances may look up the same
        object.

        3. A lock can expire. Every time an object is "locked", the TTL of the
        lock will be refreshed. Once the TTL expires, the object will become
        fully unlocked, no matter how many "locks" are accumulated.
    """

    DEFAULT_TTL = 300  # 5 minutes

    def __init__(self):
        self._locks: dict[LockKey, LockMetadata] = {}
        self._mutex = threading.Lock()

    def lock(self, key: LockKey) -> None:
        """
        Lock the given key. If the key is already locked, increase the lock
        count and refresh the expire time.

        Args:
            key: the key to lock
        """
        curr_time = time.time()
        with self._mutex:
            if meta := self._locks.get(key):
                meta.count += 1
                meta.expire_time = curr_time + self.DEFAULT_TTL
            else:
                self._locks[key] = LockMetadata(
                    count=1,
                    expire_time=curr_time + self.DEFAULT_TTL,
                )

    def unlock(self, key: LockKey) -> None:
        """
        Unlock the given key. If the key is locked multiple times, decrease
        the lock count. If the lock count reaches zero, remove the lock.

        Args:
            key: the key to unlock

        Note:
            If the key is never locked, this function won't do anything.
        """
        with self._mutex:
            if meta := self._locks.get(key):
                meta.count -= 1
                if meta.count <= 0:
                    del self._locks[key]

    def is_locked(self, key: LockKey) -> bool:
        """
        Query whether the given key is locked.

        Args:
            key: the key to query

        Returns:
            bool: True if the key is locked, False otherwise
        """
        curr_time = time.time()
        with self._mutex:
            if meta := self._locks.get(key):
                if meta.expire_time < curr_time:
                    # Lock expired
                    del self._locks[key]
                    return False

                # Still locked
                return True

            # Not found in the lock dict
            return False


ObjDict = OrderedDict[IPCCacheEngineKey, Any]


class LRUCachePolicyWithLock(LRUCachePolicy[IPCCacheEngineKey]):
    """
    An LRU cache policy that considers the lock status of the keys.
    Locked keys cannot be evicted.
    """

    def __init__(self, lock_manager: LockManager[IPCCacheEngineKey]):
        super().__init__()
        self._lock_manager = lock_manager

    def get_evict_candidates(
        self,
        cache_dict: ObjDict,
        num_candidates: int = 1,
    ) -> list[IPCCacheEngineKey]:
        """
        Overriding the LRUCachePolicy's `get_evict_candidates` method.

        Get the evict candidates from the cache dict, considering the lock
        status of the keys.

        Args:
            cache_dict: the cache dict to get candidates from
            num_candidates: the number of candidates to get

        Returns:
            list[IPCCacheEngineKey]: the list of evict candidates
        """
        evict_keys = []

        def _cannot_evict(key: IPCCacheEngineKey, obj: MemoryObj) -> bool:
            return self._lock_manager.is_locked(key) or not obj.can_evict

        for key, cache in cache_dict.items():
            if _cannot_evict(key, cache):
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys


class MPStorageManager:
    def __init__(self, cpu_buffer_size: float):
        """
        Args:
            cpu_buffer_size: the total size (in GB) of CPU memory buffer
                to be used for storage
        """
        # Lock manager for locking memory objects
        # TODO: have separate lock manager for different storage backends
        # in the future
        self._obj_lock_manager = LockManager[IPCCacheEngineKey]()

        # Allocator for CPU memory (note: this will be moved to storage backend
        # implementation in the future)
        size_in_bytes = int(cpu_buffer_size * (1 << 30))  # Convert GB to bytes
        self._memory_allocator = MixedMemoryAllocator(size_in_bytes)
        self._allocator_lock = threading.Lock()

        # Reserved memory objects
        self._reserved_memory_object_pools: dict[
            ReserveHandle, dict[IPCCacheEngineKey, MemoryObj]
        ] = {}
        self._reserved_keys: set[IPCCacheEngineKey] = set()
        self._reserve_handle = 0
        self._reserve_handle_lock = threading.Lock()

        # Committed memory objects, with LRU policy
        self._cache_policy = LRUCachePolicyWithLock(self._obj_lock_manager)
        self._commited_memory_objects: OrderedDict[IPCCacheEngineKey, MemoryObj] = (
            self._cache_policy.init_mutable_mapping()
        )

        # The lock for reserved buffer and committed buffer
        self._buffer_lock = threading.Lock()

        # NOTE: we should make sure the order of lock acquisition is:
        # 1. allocator lock
        # 2. buffer lock
        # To avoid potential deadlock

    def _allocate_new_reserve_handle(self) -> ReserveHandle:
        """Allocate a new reserve handle in a thread-safe manner."""
        with self._reserve_handle_lock:
            handle = self._reserve_handle
            self._reserve_handle += 1
        return handle

    def _has_key(self, key: IPCCacheEngineKey) -> bool:
        """Check whether the given key already exists in the storage manager.
        Both reserved and committed keys will be considered.

        Not thread-safe, should be protected by the buffer lock
        """
        if key in self._reserved_keys:
            return True
        if key in self._commited_memory_objects:
            return True
        return False

    @_lmcache_nvtx_annotate
    def reserve(
        self,
        keys: list[IPCCacheEngineKey],
        shape: Union[torch.Size, tuple[int, ...]],
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> ReserveResult:
        """Allocate the memory objects to store the given keys in the storage
        manager. If some keys already exist in the storage manager (no matter
        whether it is reserved or committed), the allocation will be skipped.
        It will return a list of boolean to identify which key is "skipped".

        Args:
            keys: the list of keys corresponding to the storage

        Returns:
            ReserveHandle: a special handle to represent this reservation.
                Will be used in "commit".
            dict[IPCCacheEngineKey, MemoryObj]: a dictionary mapping from
                reserved keys to the allocated memory objects.

        Raises:
            MemoryExhaustedError: if the allocation cannot be completed

        Note:
            This function should be thread-safe
        """

        def _confirm_reserve_objects(
            keys: list[IPCCacheEngineKey],
            mask: list[bool],
            objects: list[MemoryObj],
            handle: ReserveHandle,
        ) -> dict[IPCCacheEngineKey, MemoryObj]:
            """Helper function to confirm the reserved objects.
            Will put the reserved objects dictionary into the "reserved pool"

            Args:
                keys: the list of keys
                mask: the list of boolean mask indicating which key is reserved.
                    Should have the same length as keys.
                objects: the list of allocated memory objects.

            Returns:
                dict[IPCCacheEngineKey, MemoryObj]: a dictionary mapping from
                    reserved keys to the allocated memory objects.

            Note:
                (Specific to the current impl) This function will try to acquire
                the buffer lock to put the reserved objects into the pool.
            """
            reserved_dict = {
                k: v for k, v in zip(compress(keys, mask), objects, strict=False)
            }
            with self._buffer_lock:
                self._reserved_memory_object_pools[handle] = reserved_dict
            return reserved_dict

        # Compute number of keys to allocate
        handle = self._allocate_new_reserve_handle()

        num_objects_to_allocate = 0
        ret_mask: list[bool] = []
        for key in keys:
            # NOTE: we do fine-grained locking here since we want to
            # make sure multiple threads can reserve a part of the keys
            # if they have the identical keys submitted at the same time
            with self._buffer_lock:
                if self._has_key(key):
                    ret_mask.append(False)
                else:
                    ret_mask.append(True)
                    num_objects_to_allocate += 1
                    self._reserved_keys.add(key)

        if num_objects_to_allocate == 0:
            # No allocation needed
            with self._buffer_lock:
                self._reserved_memory_object_pools[handle] = {}
            return handle, {}

        # Allocate memory objects
        with self._allocator_lock:
            objects = self._memory_allocator.batched_allocate(
                shape, dtype, num_objects_to_allocate, fmt
            )

        if objects is not None:
            return handle, _confirm_reserve_objects(keys, ret_mask, objects, handle)

        # Failed to allocate, try to evict once
        # NOTE: we are doing very aggressive eviction here: every time
        # we will try to evict num_objects_to_allocate objects and try
        # to allocate again, until we cannot evict any more objects.
        # NOTE: we cannot directly recycle the allocated objects in
        # multi-process mode, because there could be multiple different
        # models connecting to the same storage manager
        with self._allocator_lock, self._buffer_lock:
            while objects is None:
                candidates = self._cache_policy.get_evict_candidates(
                    self._commited_memory_objects,
                    num_objects_to_allocate,
                )

                # If the candidates are not enough, break
                if not candidates:
                    break

                for key in candidates:
                    obj = self._commited_memory_objects.pop(key)
                    obj.ref_count_down()

                logger.info(
                    "Recycled %d committed memory objects to free up space.",
                    len(candidates),
                )

                # Try to allocate again
                objects = self._memory_allocator.batched_allocate(
                    shape, dtype, num_objects_to_allocate, fmt
                )

        if objects is not None:
            return handle, _confirm_reserve_objects(keys, ret_mask, objects, handle)

        raise MemoryExhaustedError(
            f"Memory allocation for {num_objects_to_allocate} objects "
            "failed due to insufficient memory."
        )

    def commit(
        self,
        reserve_handle: ReserveHandle,
    ) -> None:
        """Mark the reserved memory objects as "ready to be used/retrieved".

        Args:
            reserve_handle: the handle returned from the "reserve" function.

        Raises:
            RuntimeError: if the reserve handle is invalid.
        """
        with self._buffer_lock:
            reserved_dict = self._reserved_memory_object_pools.pop(reserve_handle, None)
            if reserved_dict is None:
                raise RuntimeError(f"Invalid reserve handle: {reserve_handle}")
            self._commited_memory_objects.update(reserved_dict)
            # NOTE: we have a potential issue here: the order of keys in
            # reserved_dict is not guaranteed. Also, it does not work for
            # chunked prefill.
            # That said, the order of store is not that important, because
            # the ordering will become correct once the keys are retrieved.
            # If the keys are not being retrieved at all, they will be evicted
            # soon anyway.
            for key in reversed(reserved_dict.keys()):
                self._cache_policy.update_on_put(key)
                self._reserved_keys.remove(key)

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> int:
        """Lookup the and lock memory objects for the given keys.

        Args:
            keys: the list of keys to lookup

        Returns:
            int: the total number of found keys (prefix matching)
        """
        # TODO: implement LOCK mechanism
        found_count = 0
        with self._buffer_lock:
            for key in keys:
                if key in self._commited_memory_objects:
                    found_count += 1
                    self._obj_lock_manager.lock(key)
                else:
                    break
        return found_count

    @_lmcache_nvtx_annotate
    @contextmanager
    def retrieve(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> Iterator[list[MemoryObj]]:
        """Retrieve the memory objects for the given keys.
        The memory objects should be locked before retrieval.
        It will unlock the memory objects after retrieval.

        Args:
            keys: the list of keys to retrieve

        Returns:
            list[MemoryObj]: the list of memory objects corresponding to
                the input keys. It requires all keys to be found.

        Raises:
            RuntimeError if there are one or more memory objects that are
                not found.
        """

        # NOTE: this function is implemented as a context manager. This
        # gives us more flexibility when we have to wait for objects from
        # the L2 memory. Also, it's easier to manage the locking/unlocking,
        # and the ref-counting of the memory objects.
        def _touch_and_get_object(key):
            """
            Raises:
                KeyError: if the key is not found
            """
            obj = self._commited_memory_objects[key]
            self._cache_policy.update_on_hit(key, self._commited_memory_objects)
            return obj

        with self._buffer_lock:
            try:
                objs = [_touch_and_get_object(key) for key in keys]
            except KeyError as e:
                raise RuntimeError(f"Key not found: {e.args[0]}") from e

        try:
            yield objs
        finally:
            # NOTE: unlock is being separated to another function because
            # it should be a callback after the retrieve cuda kernel is
            # done.
            # That said, we still keep the context manager here fore the
            # potential future use.
            pass

    @_lmcache_nvtx_annotate
    def on_retrieve_finished(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> None:
        """Callback function to be called after the retrieve operation is
        finished. It will unlock the memory objects for the given keys.

        Args:
            keys: the list of keys to unlock
        """
        for key in keys:
            self._obj_lock_manager.unlock(key)

    def prefetch(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> None:
        """Prefetch the memory objects for the given keys into L1 memory.

        Args:
            keys: the list of keys to prefetch
        """
        raise NotImplementedError

    def close(self):
        """
        Release the resources held by the storage manager.
        """
        self._memory_allocator.close()

    def memcheck(self):
        """
        Check the memory usage of the storage manager.
        """
        with self._allocator_lock:
            return self._memory_allocator.memcheck()

    def clear(self):
        """
        Clear all the memory objects in the storage manager.
        """
        # obj.ref_count_down may change the allocator state,
        # so we need to acquire the allocator lock
        with self._allocator_lock, self._buffer_lock:
            for key, obj in self._commited_memory_objects.items():
                obj.ref_count_down()
            logger.info(
                "Cleared %d committed memory objects.",
                len(self._commited_memory_objects),
            )
            self._commited_memory_objects.clear()

            for handle, reserved_list in self._reserved_memory_object_pools.items():
                for key, obj in reserved_list.items():
                    obj.ref_count_down()
            logger.info(
                "Cleared %d reserved memory objects pools.",
                len(self._reserved_memory_object_pools),
            )
            self._reserved_memory_object_pools.clear()
            self._reserved_keys.clear()
