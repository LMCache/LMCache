import torch
import threading
import time
from concurrent.futures import Future
from typing import Optional, Dict
from dataclasses import dataclass
import zmq
import enum
import pickle
from nixl._api import nixl_agent

from lmcache.experimental.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.experimental.memory_management import MemoryObj, HostMemoryAllocator, MemoryObjMetadata, TensorMemoryObj
from lmcache.experimental.storage_backend.connector.nixl_connector import NixlChannel, NixlConfig, NixlRole, NixlObserverInterface
from lmcache.utils import CacheEngineKey

from lmcache.logging import init_logger

logger = init_logger(__name__)


class BasicNixlObserver(NixlObserverInterface):
    """
    Basic implementation of the NixlObserverInterface to handle events from NixlChannel.
    """
    def __init__(self, 
                 target_dict: dict[CacheEngineKey, MemoryObj],
                 target_dict_lock: threading.Lock):
        """
        Initialize the observer with the backend reference.
        
        :param backend: The NixlBackend instance to interact with.
        """
        self.target_dict = target_dict
        self.target_dict_lock = target_dict_lock

    def __call__(
            self, 
            keys: list[CacheEngineKey],
            objs: list[MemoryObj],
            is_view: bool = True):
        """Blocking function to process the received objects
        
        Args:
          keys: the CacheEngineKeys
          objs: the list of MemoryObj
          is_view: whether the memory objects are the view of the underlying transfer buffer 
            (i.e., whether it will be overwrite by next transfer)
        """
        with self.target_dict_lock:
            logger.debug(f"Received {len(keys)} keys and {len(objs)} objects.")
            for key, value in zip(keys, objs):
                if key in self.target_dict:
                    continue
                if is_view:
                    copied_obj = TensorMemoryObj(
                        value.tensor.clone(),
                        value.metadata)
                    self.target_dict[key] = copied_obj
                else:
                    # if not a view, we can store the original object directly
                    self.target_dict[key] = value


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the implementation.

    At the sender side, it will never save anything but directly write the data to
    the receiver side.
    """

    def __init__(self, nixl_config: NixlConfig):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        self._data = {}
        self._data_lock = threading.Lock()

        self._nixl_channel = NixlChannel(nixl_config)

        self._nixl_observer = BasicNixlObserver(
            target_dict=self._data,
            target_dict_lock=self._data_lock
        )

        self._nixl_channel.register_receive_observer(
            observer=self._nixl_observer
        )


    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the storage backend.
        
        :param key: The key to check
        :return: True if the key exists, False otherwise
        """
        return key in self._data

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.
        
        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        return False

    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param key: The key of the MemoryObj.
        :param obj: The MemoryObj to be stored.
        
        :return: a future object
        """
        self._nixl_channel.send([key], [obj])
        return None

    def submit_put_tasks(self, keys: list[CacheEngineKey], objs: list[MemoryObj]) -> Optional[Future]:
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
        with self._data_lock:
            if key in self._data:
                # Return a copy of the object to avoid mutation of the original object
                return self._data[key]
            else:
                # Key does not exist in the storage
                logger.warning(f"Key {key} not found in Nixl backend.")
                return None

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()

