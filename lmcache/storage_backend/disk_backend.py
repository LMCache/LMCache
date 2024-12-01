import os
import queue
import threading
import time
import xmlrpc.client
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import copy

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lmcache.config import LMCacheEngineConfig, LMCacheMemPoolMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.evictor import DummyEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.storage_backend.mem_pool import (KVObj, LocalCPUBufferPool,
                                              LocalCPUPool, LocalGPUPool,
                                              LocalPool)
from lmcache.utils import (CacheEngineKey, KVCache, LMCKeyManagerKey,
                           _lmcache_nvtx_annotate)

logger = init_logger(__name__)


class LocalBackendEndSignal:
    pass


# TODO(Jiayi): need to optimize disk saving/loading
# current impl. with "safetensors" might not be efficient
# but it is better than "torch.save/load"

# TODO(Jiayi): need to support prefetch for disk
@_lmcache_nvtx_annotate
@torch.inference_mode()
def save_disk(
    path: str,
    kv_chunk: torch.Tensor,
    key:str
):
    # print("SAVE_DISK"+str(path))
    save_file({"kv_chunk": kv_chunk.contiguous()}, path)
    return key

class LMCDiskBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(self,
                 config: LMCacheEngineConfig,
                 metadata: LMCacheMemPoolMetadata,
                 dst_device: str = "cuda"):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
                configuration
        """
        super().__init__(dst_device)

        assert config.disk_url is not None, (
            "Need to specify local path if when "
            "using LMCLocalDiskBackend")
        self.proxy = xmlrpc.client.ServerProxy(config.disk_url)
        Info: Dict[str, Any] = self.proxy.Info()  # type: ignore

        self.remote_fmt = Info["fmt"]
        self.remote_dtype = Info["dtype"]
        self.remote_chunk_size = Info["chunk_size"]
        self.remote_serde = Info["serde"]

        self.write_key_buffer:int = 10
        self.remain_writing:int=0

        self.chunk_size = config.chunk_size

        # TODO(Jiayi): the following async put code is repeated in all backends
        # Please consider use a parent class that can be inherited by all
        # (local) backends
        # This should be also be helpful for more flexible hierarchical backends
        # For async put
        self.put_queue: queue.Queue[
            Union[Tuple[CacheEngineKey, torch.Tensor],
                  LocalBackendEndSignal]] = queue.Queue()
        self.start_query_pool: List[str]=[]
        self.end_query_pool: List[str]=[]
        self.put_thread = threading.Thread(target=self.batched_put_worker, args=())
        self.put_thread.start()
        self.update_lock = threading.Lock()

        # TODO(Jiayi): The storage size and caching policy for both
        # evictor and mpool need to be configured dynamically
        self.evictor = DummyEvictor()
        # NOTE(Jiayi): This mbufferpool should be smaller than the actual
        # cpu backend but big enough to avoid stalls in save
        # TODO(Jiayi): share the buffer if both cpu and disk backend are enabled
        self.cpu_mbufferpool = LocalCPUBufferPool(metadata)

        self.proc_pool_executor = ProcessPoolExecutor(max_workers=4)

    def _key_transform(self, key: CacheEngineKey) -> str:
        return LMCKeyManagerKey(key.model_name, key.world_size, key.worker_id,
                                key.chunk_hash).to_string()

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
        return self.proxy.contains(self._key_transform(key)) == "YES"

    @_lmcache_nvtx_annotate
    def put_worker(self, ):
        while True:
            item = self.put_queue.get()
            if isinstance(item, LocalBackendEndSignal):
                break
            key, value = item
            # print(self._key_transform(key),
            #                        self.evictor.get_size(value),
            #                        0)
            self.update_lock.acquire()
            # print("xmlrpc:put in put_worker")
            path: str = self.proxy.put(self._key_transform(key),
                                   self.evictor.get_size(value),
                                   0)
            self.update_lock.release()

            path=copy.deepcopy(path)
            self.put_nonblocking(key, value,path)

    @_lmcache_nvtx_annotate
    def batched_put_worker(self, ):
        while True:
            if self.put_queue.empty():
                time.sleep(0.1)
                continue
            batch_size = min(self.put_queue.qsize(),self.write_key_buffer)
            self.start_query_pool=[]
            key_list=[]
            size_list=[]
            for i in range(batch_size):
                item = self.put_queue.get()
                if isinstance(item, LocalBackendEndSignal):
                    break
                self.start_query_pool.append(item)
                key_list.append(self._key_transform(item[0]))
                size_list.append(self.evictor.get_size(item[1]))

            # print("xmlrpc:put in batched_put_worker")
            paths: str = self.proxy.batched_put(key_list, size_list, 0)

            for i in range(batch_size):
                item = self.start_query_pool[i]
                if isinstance(item, LocalBackendEndSignal):
                    break
                key, value = item
                path=copy.deepcopy(paths[i])
                self.put_nonblocking(key, value,path)
    
    def save_end_callback(self,future: Future):
        # print(future)
        try:
            key = future.result()  # This will raise an exception if the Future failed.
            # print("save_end_callback success:", key)
        except Exception as e:
            logger.error(f"Exception in save_end_callback: {e}")
        finally:
            # print(key)
            logger.debug(f"Saving cache {key} finished.")
            self.remain_writing=self.remain_writing-1
            self.end_query_pool.append(key)
            if len(self.end_query_pool) > self.write_key_buffer or self.remain_writing==0:
                self.update_lock.acquire()
                self.proxy.batched_put(self.end_query_pool, 0, 1)
                self.update_lock.release()
                self.end_query_pool=[]

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_nonblocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        path: str,
    ) -> None:
        # Abort put if cache too big
        # print(path)
        # print("Point1")
        if path == "":
            return

        kv_obj = None
        # print("Point2")
        # Allocate the kv chunk
        while kv_obj is None:
            self.update_lock.acquire()
            kv_obj = self.cpu_mbufferpool.allocate(kv_chunk)
            self.update_lock.release()
            if kv_obj is None:
                # TODO(Jiayi): Please tune the sleep time for better performance
                time.sleep(0.01)

        put_stream = torch.cuda.Stream()
        put_stream.wait_stream(torch.cuda.default_stream(kv_chunk.device))
        with torch.cuda.stream(put_stream):
            kv_obj.data.copy_(kv_chunk, non_blocking=True)
            kv_chunk.record_stream(put_stream)
        put_stream.synchronize()
        self.remain_writing=self.remain_writing+1
        # print("!!!",save_disk,path,self._key_transform(key))
        try:
            import pickle
            pickle.dumps(kv_obj.data)
            pickle.dumps(path)
            # print("kv_chunk is serializable.")
        except Exception as e:
            print(f"Serialization error for kv_chunk: {e}")
        future = self.proc_pool_executor.submit(save_disk, path,
                                                kv_obj.data,self._key_transform(key))
        # if not future.done():
        #     time.sleep(0.1)
        # print(future)
        # result = future.result()
        # print(result)
        # print("Point3")
        future.add_done_callback(self.save_end_callback)

        self.update_lock.acquire()
        self.cpu_mbufferpool.free(kv_obj)
        self.update_lock.release()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def put_blocking(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
    ) -> None:
        self.update_lock.acquire()
        print(self._key_transform(key), self.evictor.get_size(kv_chunk), 0)
        print("xmlrpc:put in put_blocking")
        path: str = self.proxy.put(self._key_transform(key),
                                   self.evictor.get_size(kv_chunk),
                                   0)  # type: ignore

        print(path)

        if path == "":
            return
        logger.debug(f"Saving cache to {path}")

        save_file({"kv_chunk": kv_chunk}, path)
        self.update_lock.release()
        print("xmlrpc:put in put_blocking1")
        
        self.proxy.put(self._key_transform(key), 0, 1)

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
    ) -> Optional[KVCache]:  # type: ignore
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        print(self._key_transform(key))
        
        print("xmlrpc:get in get")
        value: Dict[str, Any] = self.proxy.get(
            self._key_transform(key))  # type: ignore
        print(value)

        # still writing
        while value['status'] == 1:
            time.sleep(0.1)
            value = self.proxy.get(self._key_transform(key))  # type: ignore

        if value['status'] == 0:
            return None

        self.update_lock.acquire()
        with safe_open(value['path'], framework="pt",
                       device=self.dst_device) as f:  # type: ignore
            kv_chunk = f.get_tensor("kv_chunk")
        self.update_lock.release()
        return kv_chunk

    def batched_contains(
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
        return self.proxy.contains(self._key_transform(key)) == "YES"

    def batched_get(
        self,
        keys: Iterable[CacheEngineKey],
    ) -> Iterable[Optional[torch.Tensor]]:
        """
        Retrieve the kv cache chunks by the given keys in a batched manner

        
        :param keys: the iterator of keys of the token chunks, including prefix 
                hash and format

        :return: the iterator of kv cache of the token chunks, in the format
            of a big tensor and None if the key is not found
        """
        logger.info("Using default batched implementation of the get() method")
        keys_str = [self._key_transform(key) for key in keys]
        time0 = time.time()
        print("xmlrpc:get in batched_get")
        paths: List[str] = self.proxy.batched_get(keys_str)  # type: ignore
        print("Time taken for batched_get", time.time() - time0)
        for path in paths:
            if path == "":
                yield None
            else:
                with safe_open(path, framework="pt",
                               device=self.dst_device) as f:  # type: ignore
                    kv_chunk = f.get_tensor("kv_chunk")
                yield kv_chunk

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(LocalBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker in local disk backend")
        self.proc_pool_executor.shutdown()

    def __del__(self):
        self.close()
