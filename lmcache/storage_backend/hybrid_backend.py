from typing import Tuple, Optional
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
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

# FIXME(Jiayi): Put the following worker function(s) into class
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
        
        '''
        # prefetch
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
        '''
           
    
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
        # TODO: considering async write to remote backend
        self.remote_store.put(key, value)
    
    def put_async(
            self,
            key: Tuple[str, str],
            value: torch.Tensor,
        ):
        #self.local_store.put(key, value)
        self.put_queue.put_nowait((key, value, self.local_store, self.remote_store))


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
    
    
    def batched_get_pipeline(
        self,
        keys,
        fetched_kvs,
    ):
        # Retrieve from local cache
        #remote_indices = []
        idx = 0
        logger.debug(f"start retrieving local cache: {fetched_kvs[0] is None}")
        keys_copy = []
        for key in keys:
            #logger.debug(f"local store idx, key: {(idx, key)}")
            
            value = self.local_store.get(key) 
            fetched_kvs[idx] = value 
            #logger.debug(f"local chunk is None: {value is None}")
            idx += 1
            keys_copy.append(key) #FIXME(Jiayi): This is a hack
        
        logger.debug(f"First chunk in local cache is None: {fetched_kvs[0] is None}")
        
        # Retrieve from remote cache 
        self.remote_store.get_all_pipeline(keys_copy, fetched_kvs)

