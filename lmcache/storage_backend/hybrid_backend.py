from typing import Tuple, Optional, Iterator, List
import re
import abc
import io
import torch
import redis
import time
import pickle
import queue
import threading
from multiprocessing import Process, Queue

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.remote_backend import LMCRemoteBackend, LMCPipelinedRemoteBackend
from lmcache.storage_backend.local_backend import LMCLocalBackend
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector
from lmcache.utils import _lmcache_nvtx_annotate, CacheEngineKey

logger = init_logger(__name__)

# FIXME(Jiayi): Put the following worker function(s) into class
@_lmcache_nvtx_annotate
def put_worker(
    queue,
):
    while True:
        item = queue.get()
        key, value, local_store, remote_store = item
        #local_store.put(key, value)
        remote_store.put(key, value)
        
class LMCHybridBackend(LMCBackendInterface):
    """
    A hybrid backend that uses both local and remote backend to store and retrieve data.
    It implements write-through and read-through caching.
    """

    # TODO: LRU eviction policy
    # TODO: async write and read from/to remote backend

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.local_store = LMCLocalBackend(config)
        self.remote_store = LMCRemoteBackend(config, metadata)
        
        
        # Initialize put thread queue
        logger.debug(f"Jiayi: Initializign put thread queue")
        self.put_queue = queue.Queue()
        num_thread = 1 #FIXME(Jiayi): currently the thread num is set to 1
        self.put_threads = [
        threading.Thread(
                target=put_worker, args=(self.put_queue,)
            ) for i in range(num_thread)
        ]
        for t in self.put_threads:
            t.start()
        
        '''
        # Initialize put process queue
        logger.debug(f"Jiayi: Initializign put proc queue")
        torch.multiprocessing.set_start_method('spawn')
        self.put_queue = Queue()
        num_procs = 1 #FIXME(Jiayi): currently the proc num is set to 1
        self.put_procs = [
           Process(
               target=put_worker, args=(self.put_queue,)
           ) for i in range(num_procs)
        ]
        for p in self.put_procs:
            p.start()
        '''
        
        # Jiayi: Prefetch is disabled for now
        # self.prefetch(metadata)

           
    
    def contains(
            self,
            key: Tuple[str, str],
        ) -> bool:
        return self.local_store.contains(key) or self.remote_store.contains(key)

    def put(
            self,
            key: Tuple[str, str],
            value: torch.Tensor,
        ):
        self.local_store.put(key, value)
        self.remote_store.put(key, value)
    
    def put_async(
            self,
            key: Tuple[str, str],
            value: torch.Tensor,
        ):
        #self.local_store.put(key, value)
        self.put_queue.put_nowait((key, value, self.local_store, self.remote_store))

    # TODO(Jiayi): This prefetch can also be async
    def prefetch(
        self,
        metadata
    ):
        keys = self.remote_store.list()
        nfetched = 0
        logger.info("Found %d keys in remote backend", len(keys))
        logger.debug(f"Metadata is {metadata}")
        start = time.perf_counter()
        for key in keys:
            if key.model_name != metadata.model_name or \
                    key.worker_id != metadata.worker_id or \
                    key.world_size != metadata.world_size:
                continue

            retrived_data = self.remote_store.get(key)
            if retrived_data is not None:
                self.local_store.put(key, retrived_data)
                nfetched += 1

        end = time.perf_counter()

        logger.info("Pre-fetched %d keys from remote backend, used %.2f sec", nfetched, end - start)
        
    @_lmcache_nvtx_annotate
    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[torch.Tensor]:
        value = self.local_store.get(key)
        if value is None:
            logger.info("Jiayi: Local cache miss, using remote cache")
            value = self.remote_store.get(key)
            if value is not None:
                logger.info("Jiayi: Remote cache hit, filling local cache")
                self.local_store.put(key, value)
        else:
            logger.info("Jiayi: Local cache hit")
        return value
    
    


class LMCPipelinedHybridBackend(LMCHybridBackend):
    """
    A pipelined hybrid backend that uses both local and remote backend to store and retrieve data.
    It implements write-through and read-through caching.
    """

    # TODO: LRU eviction policy
    # TODO: async write and read from/to remote backend

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.local_store = LMCLocalBackend(config)
        self.remote_store = LMCPipelinedRemoteBackend(config, metadata)
        
        
        # Initialize put thread queue
        logger.debug(f"Jiayi: Initializing put thread queue")
        self.put_queue = queue.Queue()
        num_thread = 1 #FIXME(Jiayi): currently the thread num is set to 1
        self.put_threads = [
        threading.Thread(
                target=put_worker, args=(self.put_queue,)
            ) for _ in range(num_thread)
        ]
        for t in self.put_threads:
            t.start()
        
        '''
        # Initialize put process queue
        logger.debug(f"Jiayi: Initializign put proc queue")
        torch.multiprocessing.set_start_method('spawn')
        self.put_queue = Queue()
        num_procs = 1 #FIXME(Jiayi): currently the proc num is set to 1
        self.put_procs = [
           Process(
               target=put_worker, args=(self.put_queue,)
           ) for i in range(num_procs)
        ]
        for p in self.put_procs:
            p.start()
        '''

    @_lmcache_nvtx_annotate
    def batched_get(
        self,
        keys: Iterator[CacheEngineKey],
    ):
        logger.info("Using pipelined batched implementation of the get() method")
        keys_copy = list(keys)
        fetched_kvs = [None] * len(keys_copy)
        self.get_all_entry(keys, fetched_kvs)
        for fetched_kv in fetched_kvs:
            yield fetched_kv

    @_lmcache_nvtx_annotate
    def get_all_entry(
        self,
        keys: List[CacheEngineKey],
        fetched_kvs: List[Optional[torch.Tensor]],
    ):
        # Retrieve from local cache
        for idx, key in enumerate(keys):
            value = self.local_store.get(key) 
            fetched_kvs[idx] = value  
        
        # Retrieve from remote cache 
        self.remote_store.get_all(keys, fetched_kvs)