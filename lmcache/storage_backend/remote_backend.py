from typing import Tuple, Optional, Iterator
import io
import torch
import threading
import queue

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.storage_backend.connector import CreateConnector
from lmcache.storage_backend.serde import TorchSerializer, TorchDeserializer, CacheGenSerializer, CacheGenDeserializer, CreateSerde
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class LMCRemoteBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the Redis.
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
        self.connection = CreateConnector(config.remote_url)
        s, d = CreateSerde("cachegen", config, metadata)
        self.serializer = s
        self.deserializer = d
        #self.serializer = TorchSerializer()
        #self.deserializer = TorchDeserializer()

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

    def put(
            self, 
            key: CacheEngineKey,
            kv_chunk: torch.Tensor,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in a single big tensor

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        bs = self.serializer.to_bytes(kv_chunk)
        self.connection.set(self._combine_key(key), bs)

        self.existing_keys.add(key)

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

        return self.deserializer.from_bytes(bs)


# TODO: this class is WIP. DO NOT USE IT UNTIL IT FINISHED!
class LMCPipelinedRemoteBackend(LMCRemoteBackend):
    """
    The backend with pipelined serialization, deserialization, remote put and remote get.
    It derives from the LMCRemoteBackend and implements the pipelined serialization, deserialization, remote put and remote get.
    It uses the default un-pipelined single get() and put() methods.
    """

    class EndSignal:
        pass

    def __init__(self, config: LMCacheEngineConfig):
        super().__init__(config)
        self.serializer_queue = queue.Queue()
        self.deserializer_queue = queue.Queue()
        self.network_put_queue = queue.Queue()
        self.network_get_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Threads for each component
        self.serializer_thread = threading.Thread(target=self.serializer_worker)
        self.serializer_thread.daemon = True
        self.serializer_thread.start()

        self.deserializer_thread = threading.Thread(target=self.deserializer_worker)
        self.deserializer_thread.daemon = True
        self.deserializer_thread.start()

        self.network_put_thread = threading.Thread(target=self.network_put_worker)
        self.network_put_thread.daemon = True
        self.network_put_thread.start()

        self.network_get_thread = threading.Thread(target=self.network_get_worker)
        self.network_get_thread.daemon = True
        self.network_get_thread.start()

    def serializer_worker(self):
        while True:
            item = self.serializer_queue.get()
            match item:
                case self.EndSignal():
                    self.serializer_queue.task_done()
                    break

                case (key, kv_chunk):
                    serialized_data = self.serializer.to_bytes(kv_chunk)
                    self.network_put_queue.put((key, serialized_data))
                    self.serializer_queue.task_done()

    def deserializer_worker(self):
        while True:
            item = self.deserializer_queue.get()
            match item:
                case self.EndSignal():
                    self.deserializer_queue.task_done()
                    break

                case (key, data) if data is not None and len(data) > 0:
                    tensor = self.deserializer.from_bytes(data) 
                    self.result_queue.put(tensor)
                    self.deserializer_queue.task_done()

                case (_, _):
                    # If the data is empty, put None to the result queue
                    self.result_queue.put(None)
                    self.deserializer_queue.task_done()

    def network_put_worker(self):
        while True:
            item = self.network_put_queue.get()
            match item:
                case self.EndSignal():
                    self.network_put_queue.task_done()
                    break

                case (key, serialized_data):
                    self.connection.set(self._combine_key(key), serialized_data)
                    self.network_put_queue.task_done()

    def network_get_worker(self):
        while True:
            key = self.network_get_queue.get()
            match key:
                case self.EndSignal():
                    self.network_get_queue.task_done()
                    break

                case key:
                    if not self.contains(key):
                        self.deserializer_queue.put((key, None))
                    else:
                        data = self.connection.get(self._combine_key(key))
                        self.deserializer_queue.put((key, data))
                    self.network_get_queue.task_done()
                    
    def batched_put(self, keys_and_chunks: Iterator[Tuple[CacheEngineKey, torch.Tensor]]) -> None:
        count = 0
        for key, kv_chunk in keys_and_chunks:
            self.serializer_queue.put((key, kv_chunk))
            count += 1

        # Wait for all tasks to complete
        self.serializer_queue.join()
        return count

    def batched_get(self, keys: Iterator[CacheEngineKey]) -> Iterator[Optional[torch.Tensor]]:
        count = 0
        for key in keys:
            self.network_get_queue.put(key)
            count += 1

        # TODO: test what will happen if the consumer does not consume all the results
        for _ in range(count):
            tensor = self.result_queue.get()
            yield tensor

    def close(self):
        """
        Gracefully shutdown the threads in the backend
        """
        self.network_get_queue.put(self.EndSignal())
        self.network_put_queue.put(self.EndSignal())
        self.serializer_queue.put(self.EndSignal())
        self.deserializer_queue.put(self.EndSignal())

        self.network_get_thread.join()
        self.network_put_thread.join()
        self.serializer_thread.join()
        self.deserializer.join()

