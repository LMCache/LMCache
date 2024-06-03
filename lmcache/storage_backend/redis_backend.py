from typing import Tuple, Optional
import re
import io
import torch
import redis
import time

from lmcache.types import KVCache
from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger

logger = init_logger(__name__)

class LMCRedisBackend(LMCBackendInterface):
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
        m = re.match(r"redis://(.*):(\d+)", config.remote_url)
        if m is None:
            raise ValueError(f"Invalid redis backend {config.remote_url}")
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
            logger.info("Storing %.2f MBytes redis", len(string) / 1e6)

            self.connection.set(self._to_redis_key(key), string)
            end = time.perf_counter()
            logger.info("Storing %.2f MBytes data to redis in %.2f ms", len(string) / 1e6, (end - start) * 1e3)
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
            logger.info("Retrieved %.2f MBytes data from redis in %.2f ms", len(data) / 1e6, (end - start) * 1e3)
            if data is None:
                return None
            f.write(data)
            f.seek(0)
            return torch.load(f)

    def persist(self):
        """
        Temporary function of persisting
        """
        logger.warn("Persisting is not supported in Redis cache engine")
