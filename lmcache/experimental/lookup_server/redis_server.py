from typing import Optional, Tuple

import redis 
import inspect

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server.abstract_server import LookupServerInterface # noqa: E501
from lmcache.utils import CacheEngineKey

class RedisLookupServer(LookupServerInterface):
    def __init__(self, config: LMCacheEngineConfig):
        self.url = config.lookup_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)
        
        self.connection = redis.Redis(host=host,
                                      port=port,)
                                      #decode_responses=False)
    
    def lookup(
        self, 
        key: CacheEngineKey
    ) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        url = self.connection.get(key.to_string())
        assert not inspect.isawaitable(url)
        if url is None:
            return None
        host, port = url.split(":")
        return host, int(port)
    
    def insert(self, key: CacheEngineKey):
        """
        Perform insert in the lookup server.
        """
        assert self.url is not None
        self.connection.set(key.to_string(), self.url)
        
    

    def remove(self, key: CacheEngineKey):
        """
        Perform remove in the lookup server.
        """
        self.connection.delete(key.to_string())