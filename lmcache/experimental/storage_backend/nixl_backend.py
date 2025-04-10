# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
from concurrent.futures import Future
from typing import Optional

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (MemoryObj,
                                                    MemoryObjMetadata,
                                                    TensorMemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.connector.nixl_connector import (
    NixlChannel, NixlConfig, NixlObserverInterface)
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class RecvObjPool:

    def __init__(self, enable_gc: bool):
        self.lock = threading.Lock()
        self._data: dict[CacheEngineKey, MemoryObj] = {}
        self._cnt: dict[CacheEngineKey, int] = {}

        # TODO: Remove the hard-code
        # HACK: have a recycle threshold to avoid the memory leak
        self._recent_added_keys: list[CacheEngineKey] = []
        self._recent_add_threshold = 50  # Keep recent 20 keys
        self._recycle_threshold = 200

        self._enable_gc = enable_gc
        if not self._enable_gc:
            logger.warning("GC for receiver is disabled, may lead to memory "
                           "leak in non-testing environment")

    def _gc(self):
        if not self._enable_gc:
            return
        logger.warning("In GC!")
        st = time.perf_counter()
        freed_size = 0
        current_keys = set(self._data.keys())
        recent_keys = set(self._recent_added_keys)
        keys_to_evict = current_keys - recent_keys
        for key in keys_to_evict:
            freed_size += self._data[key].get_size()
            self._data.pop(key)
            self._cnt.pop(key)
        ed = time.perf_counter()
        logger.warning("GC in %.4f msec, released %.2f GB memory",
                       (ed - st) * 1000, freed_size / 1024 / 1024 / 1024)

    def add(self, key: CacheEngineKey, obj: MemoryObj):
        with self.lock:
            # TODO: Get rid of this
            self._recent_added_keys.append(key)
            self._recent_added_keys = \
                    self._recent_added_keys[-self._recent_add_threshold:]

            if key in self._data:
                self._cnt[key] += 1
            else:
                self._data[key] = obj
                self._cnt[key] = 1

    def remove(self, key: CacheEngineKey):
        with self.lock:
            if key in self._cnt:
                self._cnt[key] -= 1
                if self._cnt[key] == 0:
                    self._data.pop(key)
                    self._cnt.pop(key)

    def contains(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if len(self._data) >= self._recycle_threshold:
                self._gc()

            return key in self._data

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            return self._data.get(key, None)


class BasicNixlObserver(NixlObserverInterface):
    """
    Basic implementation of the NixlObserverInterface to handle 
    events from NixlChannel.
    """

    def __init__(self, obj_pool: RecvObjPool):
        """
        Initialize the BasicNixlObserver.
        """
        self.obj_pool = obj_pool

    def __call__(self,
                 keys: list[CacheEngineKey],
                 objs: list[MemoryObj],
                 is_view: bool = True):
        """Blocking function to process the received objects
        
        Args:
          keys: the CacheEngineKeys
          objs: the list of MemoryObj
          is_view: whether the memory objects are the view of the underlying 
            transfer buffer  (i.e., whether it will be overwrite by next 
            transfer)
        """
        clone_time = 0.0
        add_time = 0.0
        for key, value in zip(keys, objs):
            assert value.tensor is not None, \
                    "The tensor in the MemoryObj is None."
            if is_view:
                st = time.perf_counter()
                copied_obj = TensorMemoryObj(value.tensor.clone(),
                                             value.metadata)
                ed = time.perf_counter()
                self.obj_pool.add(key, copied_obj)
                ed2 = time.perf_counter()
                clone_time += (ed - st) * 1000
                add_time += (ed2 - ed) * 1000
            else:
                self.obj_pool.add(key, value)
        logger.debug(
            "Nixl Observer: clone time: %.4f msec, "
            "Add time: %.4f msec for %d objects", clone_time, add_time,
            len(keys))


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the 
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(self, nixl_config: NixlConfig):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        self._obj_pool = RecvObjPool(nixl_config.enable_gc)
        #self._data: dict[CacheEngineKey, MemoryObj] = {}
        #self._data_lock = threading.Lock()

        self._nixl_channel = NixlChannel(nixl_config)

        self._nixl_observer = BasicNixlObserver(self._obj_pool)

        self._nixl_channel.register_receive_observer(
            observer=self._nixl_observer)

        self._registered_keys: list[CacheEngineKey] = []
        self._registered_metadatas: list[MemoryObjMetadata] = []
        self._num_payload_added = 0

    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the storage backend.
        
        :param key: The key to check
        :return: True if the key exists, False otherwise
        """
        return self._obj_pool.contains(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.
        
        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        return False

    def register_put_tasks(
        self,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
    ) -> None:
        """
        Register the put tasks to the backend.
        """
        if len(self._registered_keys) > 0:
            raise RuntimeError("The backend has already registered put tasks.")

        self._registered_keys = keys
        self._registered_metadatas = metadatas
        self._nixl_channel.prepare_send(keys=keys, metadatas=metadatas)

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        Put the MemoryObj into the storage backend and send it to the receiver
        in a blocking way.

        :param key: The key of the MemoryObj.
        :param obj: The MemoryObj to be stored.
        
        :return: a future object

        :note: Right now, the 'key' is not used and it assumes that the memory 
        object has the same order as the keys passed in `register_put_tasks`.
        """
        if len(self._registered_keys) == 0:
            raise RuntimeError("The backend has not registered put tasks.")

        assert self._registered_keys[self._num_payload_added] == key, \
            f"The key {key} is not the same as the registered key "\
            f"{self._registered_keys[self._num_payload_added]}."

        assert \
            self._registered_metadatas[self._num_payload_added] \
            == obj.metadata, \
            f"The {obj.metadata} is not the same as the registered metadata "\
            f"{self._registered_metadatas[self._num_payload_added]}."

        #self._nixl_channel.send([key], [obj])
        self._nixl_channel.add_payload(obj)
        self._num_payload_added += 1
        return None

    def flush_put_tasks(self) -> None:
        """
        Flush the registered tasks 
        """
        assert len(self._registered_keys) > 0, \
            "The backend has not registered put tasks."
        assert self._num_payload_added == len(self._registered_keys), \
            "The number of payloads added is not equal to the number of" \
            "registered keys."

        self._nixl_channel.finish_send()
        self._registered_keys = []
        self._registered_metadatas = []
        self._num_payload_added = 0

    def submit_put_tasks(self, keys: list[CacheEngineKey],
                         objs: list[MemoryObj]) -> Optional[Future]:
        """
        Put the MemoryObj into the storage backend and send it to the 
        receiver in a blocking way.

        :param keys: The keys of the MemoryObj.
        :param objs: The MemoryObj to be stored.

        :return: a future object
        """
        self._nixl_channel.send(keys, objs)
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.
        
        :param key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        return self._obj_pool.get(key)

    def remove(self, key: CacheEngineKey) -> None:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        return self._obj_pool.remove(key)

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()

    @staticmethod
    def CreateNixlBackend(config: LMCacheEngineConfig,
                          metadata: LMCacheEngineMetadata) -> "NixlBackend":
        """
        Create a Nixl backend with the given configuration.

        :param nixl_config: The Nixl configuration.
        :param dst_device: The device where the data is stored.
        
        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        backend = NixlBackend(nixl_config)
        return backend
