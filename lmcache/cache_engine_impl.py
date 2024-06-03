import torch
import redis
import io
import re
import time
import os
import hashlib
from typing import Tuple, List, Union, Iterator, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(format='\033[33m%(levelname)s LMCache: \033[0m%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

@dataclass
class LMCacheEngineConfig:
    chunk_size: int 
    backend: str
    persist_path: str

    def from_defaults(
            chunk_size: int = 256,
            backend: str = "cuda",
            persist_path: str = None
        ) -> 'LMCacheEngineConfig':
        return LMCacheEngineConfig(chunk_size, backend, persist_path)

class LMCacheInterface:
    def put(
            self,
            key: Tuple[str, str],
            kv_chunk: KVCache,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        raise NotImplementedError

    def contains(
            self,
            key: Tuple[str, str],
        ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        raise NotImplementedError

    def persist(self):
        """
        Temporary function of persisting, should be removed in the future
        """
        raise NotImplementedError

class LMLocalCahe(LMCacheInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu memory.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        # TODO: remove persist_path in the future
        self.chunk_size = config.chunk_size 
        self.persist_path = config.persist_path
        self.backend = config.backend
        self.config = config
        self.dict = {}
        if self.persist_path is not None and os.path.isfile(self.persist_path):
            logger.info(f"Found persisted file at {self.persist_path}, loading it right now...")
            self.dict, loaded_config = torch.load(self.persist_path)
            if loaded_config != self.config:
                raise RuntimeError(f"Loaded configuration {loaded_config} does not match the current configuration {self.config}")
            logger.info(f"Loaded {len(self.dict)} chunks")

    def contains(
            self, 
            key: Tuple[str, str]
        ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def put(
            self, 
            key: Tuple[str, str],
            kv_chunk: KVCache
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        self.dict[key] = kv_chunk


    def get(
            self,
            key: Tuple[str, str]
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        return self.dict.get(key, None)

    def persist(self):
        """
        Temporary function of persisting
        """
        if self.persist_path is not None:
            torch.save((self.dict, self.config), self.persist_path)
            logger.info(f"Persisted the cache to {self.persist_path}. {os.path.getsize(self.persist_path) / 1e6} MBytes in total")
        else:
            raise RuntimeError("Persist path not found, please set self.persist_path")

class LMRedisCache(LMCacheInterface):
    """
    Cache engine for storing the KV cache of the tokens in the Redis.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()
        self.existing_keys = set()
        host, port = self._parse_cfg(config)
        self.connection = redis.Redis(host=host, port=port)

    def _parse_cfg(
            self,
            config: LMCacheEngineConfig
        ) -> Tuple[str, int]:
        """
        Parse the config to the redis connection parameters (host and port)
        Returns:
            Tuple of hostname and port number
        """
        m = re.match(r"redis://(.*):(\d+)", config.backend)
        if m is None:
            raise ValueError(f"Invalid redis backend {config.backend}")
        return m.group(1), int(m.group(2))

    def _to_redis_key(
            self,
            key: Tuple[str, str],
        ) -> str:
        """
        Convert the key to the redis key
        """
        return ":".join(key)

    def _from_redis_key(
            self,
            key: str,
        ) -> Tuple[str, str]:
        """
        Convert the redis key to the key
        """
        return tuple(key.split(":"))

    def _init_scan(self):
        """
        Scan the redis keys and initialize the existing_keys
        """
        pass

    def contains(
            self, 
            key: Tuple[str, str]
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
            flag = self.connection.exists(self._to_redis_key(key))
            if flag:
                self.existing_keys.add(key)
            return flag

    def put(
            self, 
            key: Tuple[str, str],
            kv_chunk: KVCache
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        with io.BytesIO() as f:
            torch.save(kv_chunk, f)
            f.seek(0)
            start = time.perf_counter()
            string = f.read()
            logging.info("Storing %.2f MBytes redis", len(string) / 1e6)

            self.connection.set(self._to_redis_key(key), string)
            end = time.perf_counter()
            logging.info("Storing %.2f MBytes data to redis in %.2f ms", len(string) / 1e6, (end - start) * 1e3)
        self.existing_keys.add(key)

    def get(
            self,
            key: Tuple[str, str],
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key
        """
        if not self.contains(key):
            return None

        with io.BytesIO() as f:
            start = time.perf_counter()
            data = self.connection.get(self._to_redis_key(key))
            end = time.perf_counter()
            logging.info("Retrieved %.2f MBytes data from redis in %.2f ms", len(data) / 1e6, (end - start) * 1e3)
            if data is None:
                return None
            f.write(data)
            f.seek(0)
            return torch.load(f)

    def persist(self):
        """
        Temporary function of persisting
        """
        logging.warn("Persisting is not supported in Redis cache engine")
