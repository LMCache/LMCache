# SPDX-License-Identifier: Apache-2.0
# NOTE: this file will be moved and the class implementation
# will be largely refactored in the future.
# Standard
from collections import OrderedDict
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import compress
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, TypeVar, Union
import asyncio
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MixedMemoryAllocator
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy

if TYPE_CHECKING:
    # First Party
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

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
    def __init__(
        self,
        cpu_buffer_size: float,
        config: Optional["LMCacheEngineConfig"] = None,
        metadata: Optional["LMCacheEngineMetadata"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Args:
            cpu_buffer_size: the total size (in GB) of CPU memory buffer
                to be used for storage
            config: Optional LMCache config for L2 storage plugins
            metadata: Optional metadata for L2 storage operations
            loop: Optional asyncio event loop for async L2 operations
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

        # Optional L2 storage backends (raw block, disk, etc.)
        self._config = config
        self._metadata = metadata
        self._loop = loop
        self._l2_backends: OrderedDict[str, "StorageBackendInterface"] = OrderedDict()
        self._l2_index: set[IPCCacheEngineKey] = set()
        self._l2_index_lock = threading.Lock()

        # Initialize L2 backends if config is provided
        if config is not None and metadata is not None:
            self._init_l2_backends()

    def _init_l2_backends(self) -> None:
        """Initialize L2 storage backends from config."""
        if self._config is None or self._metadata is None:
            return

        # First Party
        from lmcache.v1.storage_backend import CreateStorageBackends
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

        try:
            assert self._loop is not None
            backends = CreateStorageBackends(
                config=self._config,
                metadata=self._metadata,
                loop=self._loop,
                dst_device="cpu",
                lmcache_worker=None,
            )

            # Separate L2 backends (exclude LocalCPUBackend)
            for name, backend in backends.items():
                if not isinstance(backend, LocalCPUBackend):
                    self._l2_backends[name] = backend

            if self._l2_backends:
                logger.info(
                    "MPStorageManager: L2 backends initialized: %s",
                    list(self._l2_backends.keys()),
                )
        except Exception as e:
            logger.warning("Failed to initialize L2 backends: %s", e)

    def _ipc_key_to_cache_key(
        self,
        ipc_key: IPCCacheEngineKey,
    ) -> CacheEngineKey:
        """Convert IPCCacheEngineKey to CacheEngineKey for storage backends."""
        chunk_hash_int = int.from_bytes(
            ipc_key.chunk_hash, byteorder="big", signed=True
        )
        fmt = self._metadata.fmt if self._metadata else "vllm"
        dtype = self._metadata.kv_dtype if self._metadata else torch.bfloat16
        return CacheEngineKey(
            fmt=fmt,
            model_name=ipc_key.model_name,
            world_size=ipc_key.world_size,
            worker_id=ipc_key.worker_id,
            chunk_hash=chunk_hash_int,
            dtype=dtype,
        )

    def _evict_to_l2(self, key: IPCCacheEngineKey, obj: MemoryObj) -> None:
        """Evict a memory object to L2 storage."""
        if not self._l2_backends:
            return

        cache_key = self._ipc_key_to_cache_key(key)

        for name, backend in self._l2_backends.items():
            try:
                backend.batched_submit_put_task([cache_key], [obj])
                with self._l2_index_lock:
                    self._l2_index.add(key)
                break
            except Exception as e:
                logger.warning("Failed to evict to L2 backend %s: %s", name, e)

    def _store_to_l2_async(
        self, key_obj_dict: dict[IPCCacheEngineKey, MemoryObj]
    ) -> None:
        """Asynchronously store committed objects to L2."""
        if not self._l2_backends:
            return

        cache_keys = []
        objs = []

        for ipc_key, obj in key_obj_dict.items():
            cache_key = self._ipc_key_to_cache_key(ipc_key)
            cache_keys.append(cache_key)
            objs.append(obj)

        for name, backend in self._l2_backends.items():
            try:
                backend.batched_submit_put_task(cache_keys, objs)
                with self._l2_index_lock:
                    for ipc_key in key_obj_dict.keys():
                        self._l2_index.add(ipc_key)
                break
            except Exception as e:
                logger.warning("Failed to store to L2 backend %s: %s", name, e)

    def _lookup_in_l2(self, keys: list[IPCCacheEngineKey]) -> int:
        """Check how many consecutive keys exist in L2."""
        if not self._l2_backends:
            return 0

        cache_keys = [self._ipc_key_to_cache_key(k) for k in keys]

        for name, backend in self._l2_backends.items():
            try:
                hit_count = backend.batched_contains(cache_keys)
                if hit_count > 0:
                    return hit_count
            except Exception as e:
                logger.warning("L2 lookup failed for %s: %s", name, e)

        return 0

    def _load_from_l2(self, keys: list[IPCCacheEngineKey]) -> list[Optional[MemoryObj]]:
        """Load memory objects from L2 storage."""
        if not self._l2_backends:
            return [None] * len(keys)

        cache_keys = [self._ipc_key_to_cache_key(k) for k in keys]

        for name, backend in self._l2_backends.items():
            try:
                objs = backend.batched_get_blocking(cache_keys)
                if objs and any(o is not None for o in objs):
                    logger.info("Loaded %d objects from L2 backend %s", len(objs), name)
                    return objs
            except Exception as e:
                logger.warning("L2 load failed for %s: %s", name, e)

        return [None] * len(keys)

    def has_l2_storage(self) -> bool:
        """Check if L2 storage is configured."""
        return len(self._l2_backends) > 0

    def get_l2_backend_names(self) -> list[str]:
        """Get names of configured L2 backends."""
        return list(self._l2_backends.keys())

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
                    # Evict to L2 before freeing (if L2 is configured)
                    self._evict_to_l2(key, obj)
                    obj.ref_count_down()

                logger.info(
                    "Recycled %d committed memory objects to free up space%s.",
                    len(candidates),
                    " (evicted to L2)" if self._l2_backends else "",
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

        # Async store to L2 for persistence (if L2 is configured)
        if self._l2_backends and reserved_dict:
            self._store_to_l2_async(reserved_dict)

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
        # Check L1 (CPU memory) first
        found_count = 0
        with self._buffer_lock:
            for key in keys:
                if key in self._commited_memory_objects:
                    found_count += 1
                    self._obj_lock_manager.lock(key)
                else:
                    break

        if found_count == len(keys):
            return found_count

        # Check L2 for remaining keys (if L2 is configured)
        if self._l2_backends:
            remaining_keys = keys[found_count:]
            l2_hits = self._lookup_in_l2(remaining_keys)

            if l2_hits > 0:
                # Load from L2 to L1 and lock
                keys_to_load = remaining_keys[:l2_hits]
                loaded = self._load_from_l2(keys_to_load)

                with self._buffer_lock:
                    for key, obj in zip(keys_to_load, loaded, strict=False):
                        if obj is not None:
                            self._commited_memory_objects[key] = obj
                            self._obj_lock_manager.lock(key)
                            found_count += 1
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
        # Close L2 backends first
        for name, backend in self._l2_backends.items():
            try:
                backend.close()
                logger.info("Closed L2 backend: %s", name)
            except Exception as e:
                logger.warning("Error closing L2 backend %s: %s", name, e)

        # Close L1 memory allocator
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

        # Clear L2 index
        with self._l2_index_lock:
            self._l2_index.clear()
