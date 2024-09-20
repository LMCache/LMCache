from typing import Tuple, Optional, Iterator
import io
import torch
import threading
import queue

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector
from lmcache.storage_backend.serde import CreateSerde
from lmcache.utils import CacheEngineKey
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

# FIXME(Jiayi): Put the following worker function(s) into class
# FIXME(Jiayi): Needs to consider concurrent setting (private queue?)


class RemoteBackendEndSignal:
    pass
        
class LMCRemoteBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the remote server.
    """


    def __init__(
            self, 
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()
        self.existing_keys = set()
        self.put_thread = None
        self.connection = None
        self.connection = CreateConnector(config.remote_url)
        s, d = CreateSerde(config.remote_serde, config, metadata)
        self.serializer = s
        self.deserializer = d

        # For async put
        self.put_queue = queue.Queue()
        self.put_thread = threading.Thread(
                target=self.put_worker, args=()
            ) 
        self.put_thread.start()
        
        # FIXME(Jiayi): please remove this hard code
        self.dst_device = "cuda"

    @_lmcache_nvtx_annotate
    def put_worker(
            self,
    ):
        #put_stream = torch.cuda.Stream()
        while True:
            item = self.put_queue.get()
            if isinstance(item, RemoteBackendEndSignal):
                break
            key, value = item
            #with torch.cuda.stream(put_stream):
            self.put_blocking(key, value)

    def _combine_key(
            self,
            key: CacheEngineKey,
        ) -> str:
        """
        Convert the tuple key to a single key
        """
        return key.to_string()

    def _split_key(
            self,
            key: str,
        ) -> CacheEngineKey:
        """
        Split the single key to a tuple key
        """
        return CacheEngineKey.from_string(key)

    def list(self):
        """
        list the remote keys (and also update the 'cached' existing keys set)
        """
        keys = self.connection.list()
        for key in keys:
            self.existing_keys.add(self._split_key(key))
        return [self._split_key(key) for key in keys]

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
        if key in self.existing_keys:
            return True
        else:
            flag = self.connection.exists(self._combine_key(key))
            if flag:
                self.existing_keys.add(key)
            return flag

    def put_blocking(
            self,
            key: CacheEngineKey,
            kv_chunk: torch.Tensor,
        ) -> None:
        bs = self.serializer.to_bytes(kv_chunk)
        self.connection.set(self._combine_key(key), bs)
        self.existing_keys.add(key)


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
            kv_chunk: the kv cache of the token chunk, in a single big tensor
            blocking: whether to block until the put is done

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
        Retrive the KV cache chunk (in a single big tensor) by the given key
        """
        if not self.contains(key):
            return None

        bs = self.connection.get(self._combine_key(key))
        if bs is None or len(bs) == 0:
            return None

        return self.deserializer.from_bytes(bs).to(self.dst_device)

    def close(self):
        if self.put_thread is not None and self.put_thread.is_alive():
            self.put_queue.put(RemoteBackendEndSignal())
            self.put_thread.join()
            logger.info("Closed the put worker")

        if self.connection is not None:
            self.connection.close()

    def __del__(self):
        self.close()

class LMCPipelinedRemoteBackend(LMCRemoteBackend):
    """
    Implements the pipelined get functionality for the remote backend.
    """

    def __init__(
            self, 
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__(config, metadata)

        self.existing_keys = set()
        self.network_thread = None
        self.deserialize_thread = None

        #Initialize network get thread queue
        logger.debug(f"Initializing network thread queue")
        self.network_queue = queue.Queue()
        self.network_thread = threading.Thread(
               target=self.network_worker, args=()
           ) 
        self.network_thread.start()
        
        #Initialize network get thread queue
        logger.debug(f"Initializing deserial thread queue")
        self.deserialize_queue = queue.Queue()
        self.deserialize_thread = threading.Thread(
               target=self.deserialize_worker, args=()
           ) 
        self.deserialize_thread.start()

        self.result_list = []

    @_lmcache_nvtx_annotate
    def network_worker(
            self,
        ):
        while True:
            item = self.network_queue.get()
            if isinstance(item, RemoteBackendEndSignal):
                break

            idx, key = item
            if self.contains(key):
                data = self.connection.get(self._combine_key(key))
                self.deserialize_queue.put_nowait((idx, data))

            self.network_queue.task_done()
            
    @_lmcache_nvtx_annotate
    def deserialize_worker(
            self,
        ):
        while True:
            item = self.deserialize_queue.get()
            if isinstance(item, RemoteBackendEndSignal):
                break

            idx, data = item
            if data is not None:
               result = self.deserializer.from_bytes(data).to(self.dst_device)
            else:
               result = None
            self.result_list.append(result)
            self.deserialize_queue.task_done()

    @_lmcache_nvtx_annotate
    def batched_get(
        self,
        keys: Iterator[CacheEngineKey],
    ) -> Iterator[Optional[torch.Tensor]]:
        self.result_list = []
        for idx, key in enumerate(keys):
            self.network_queue.put_nowait((idx, key))
        self.network_queue.join()
        self.deserialize_queue.join()
        return self.result_list
    
    def close(self):
        super().close()

        if self.network_thread is not None and self.network_thread.is_alive():
            self.network_queue.put(RemoteBackendEndSignal())
            self.network_thread.join()
            logger.info("Closed the network worker")

        if self.deserialize_thread is not None and self.deserialize_thread.is_alive():
            self.deserialize_queue.put(RemoteBackendEndSignal())
            self.deserialize_thread.join()
            logger.info("Closed the deserialize worker")
        

    def __del__(self):
        self.close()
    
