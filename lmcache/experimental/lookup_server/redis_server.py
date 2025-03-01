import inspect
from typing import List, Optional, Tuple

import redis

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server.abstract_server import \
    LookupServerInterface  # noqa: E501
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


# TODO (Jiayi): Batching is needed for Redis lookup server.
class RedisLookupServer(LookupServerInterface):

    def __init__(self, config: LMCacheEngineConfig):
        self.distributed_url = config.distributed_url
        assert self.distributed_url is not None

        self.url = config.lookup_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        self.connection = redis.Redis(host=host,
                                      port=port,
                                      decode_responses=True)
        logger.info(f"Connected to Redis lookup server at {host}:{port}")
        #decode_responses=False)

    def lookup(self, key: CacheEngineKey) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        logger.debug("Call to lookup in lookup server")
        url = self.connection.get(key.to_string())
        logger.debug(f"KV cache lives on {url}")
        assert not inspect.isawaitable(url)
        if url is None:
            return None
        host, port = url.split(":")
        return host, int(port)

    def insert(self, key: CacheEngineKey):
        """
        Perform insert in the lookup server.
        """
        assert self.distributed_url is not None
        logger.debug("Call to insert in lookup server")
        self.connection.set(key.to_string(), self.distributed_url)

    def remove(self, key: CacheEngineKey):
        """
        Perform remove in the lookup server.
        """
        logger.debug("Call to remove in lookup server")
        self.connection.delete(key.to_string())

    def batched_remove(self, keys: List[CacheEngineKey]):
        """
        Perform batched remove in the lookup server.
        """
        logger.debug("Call to batched remove in lookup server")
        # TODO(Jiayi): We might need to cache the `str_keys` for performance.
        str_keys = [key.to_string() for key in keys]
        self.connection.delete(*str_keys)
