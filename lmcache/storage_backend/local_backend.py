import os
import queue
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.config import LMCacheEngineConfig
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.evictor import DummyEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata, KVCache,
                           _lmcache_nvtx_annotate)

logger = init_logger(__name__)


class LocalBackendEndSignal:
    pass


class LMCLocalBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu
    memory.
    """

    def __init__(self, config: LMCacheEngineConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__()

        self.chunk_size = config.chunk_size
        self.config = config
        self.dict: OrderedDict[CacheEngineKey, torch.Tensor] = OrderedDict()
        self.device = config.local_device

        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # FIXME(Jiayi): `use_pin_memory` and `dst_device` should be configged
        # dynamically
        self.use_pin_memory = False
        logger.info(f"Using pinned cpu memory: {self.use_pin_memory}")

        self.dst_device = "cuda"
        # self.async_put_flag = False
        # self.put_events = {}

        # TODO (Jiayi): The storage size and caching
        self.evictor = DummyEvictor()

        self.put_stream = torch.cuda.Stream()

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """
        self.dict.pop(key)

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        while True:
            # TODO: dirty fix to downgrade the priority of the put worker
            time.sleep(0.01)
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            with torch.cuda.stream(self.put_stream):
                self.put_nonblocking(key, value)

    def put_nonblocking(self, key, kv_chunk):
        # TODO(Jiayi): torch.cuda.synchronize() needs to be removed
        # to enable actual async put
        # torch.cuda.synchronize() may disturb inference engine
        if self.use_pin_memory:
            kv_chunk_local = kv_chunk.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            kv_chunk_local = kv_chunk.to(self.device, non_blocking=True)
        self.update_lock.acquire()

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, kv_chunk_local)
        if put_status == PutStatus.ILLEGAL:
            self.update_lock.release()
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        # Store new chunk
        self.dict[key] = kv_chunk_local
        self.update_lock.release()

    def put_blocking(self, key, kv_chunk):
        if self.use_pin_memory:
            kv_chunk_local = kv_chunk.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            kv_chunk_local = kv_chunk.to(self.device)

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, kv_chunk_local)

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return

        # Evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        # Store new chunk
        self.dict[key] = kv_chunk_local

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            self.put_queue.put((key, kv_chunk))

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        self.update_lock.acquire()
        kv_chunk = self.dict.get(key, None)

        # Update cache recency
        if kv_chunk is not None:
            self.evictor.update_on_get(key, self.dict)
            kv_chunk = kv_chunk.to(self.dst_device)
        self.update_lock.release()
        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local backend")

    def __del__(self):
        self.close()


# TODO(Jiayi): need to optimize disk saving/loading
# current impl. with "safetensors" might not be efficient
# but it is better than "torch.save/load"

# TODO(Jiayi): need to support prefetch for disk


class LMCLocalDiskBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(self, config: LMCacheEngineConfig):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__()

        self.chunk_size = config.chunk_size
        self.config = config
        self.dict: OrderedDict[CacheEngineKey,
                               DiskCacheMetadata] = OrderedDict()
        self.path = config.local_device

        assert self.path is not None, ("Need to specify local path if when "
                                       "using LMCLocalDiskBackend")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # TODO(Jiayi): the following async put code is repeated in all backends
        # Please consider use a parent class that can be inherited by all
        # (local) backends
        # This should be also be helpful for more flexible hierarchical backends
        # For async put
        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.put_thread = threading.Thread(target=self.put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # TODO (Jiayi): please remove this hard code
        self.dst_device = "cuda"

        self.evictor = DummyEvictor()

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.to_string().replace("/", "-") + ".pt"

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """

        self.update_lock.acquire()
        path = self.dict[key].path
        self.dict.pop(key)
        self.update_lock.release()

        os.remove(path)

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        put_stream = torch.cuda.Stream()
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            with torch.cuda.stream(put_stream):
                self.put_blocking(key, value)

    def put_blocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
    ) -> None:
        path = self._key_to_path(key)
        logger.debug(f"Saving cache to {path}")
        # The following order matters of `save_file` and `update dictionary`
        # matters

        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, kv_chunk)

        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return

        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)

        save_file({"kv_chunk": kv_chunk}, path)
        self.update_lock.acquire()
        self.dict[key] = DiskCacheMetadata(path,
                                           self.evictor.get_size(kv_chunk))
        self.update_lock.release()

    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if blocking:
            self.put_blocking(key, kv_chunk)
        else:
            self.put_queue.put((key, kv_chunk))

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[KVCache]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        self.update_lock.acquire()
        if key not in self.dict:
            self.update_lock.release()
            return None

        path = self.dict[key].path
        self.evictor.update_on_get(key, self.dict)

        with safe_open(path, framework="pt",
                       device=self.dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
        self.update_lock.release()
        return kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local disk backend")

    def __del__(self):
        self.close()
