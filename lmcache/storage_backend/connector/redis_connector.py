import redis
import os
from typing import Optional, List, Tuple, Union
from lmcache.storage_backend.connector.base_connector import RemoteConnector

from lmcache.logging import init_logger
logger = init_logger(__name__)

class RedisConnector(RemoteConnector):
    """
    The remote url should start with "redis://" and only have one host-port pair
    """
    def __init__(self, host: str, port: int):
        self.connection = redis.Redis(host=host, port=port)

    def exists(self, key: str) -> bool:
        return self.connection.exists(key)

    def get(self, key: str) -> Optional[bytes]:
        return self.connection.get(key)

    def set(self, key: str, obj: bytes) -> None:
        self.connection.set(key, obj)

    def list(self):
        cursor = 0
        all_keys = []
    
        while True:
            cursor, keys = self.connection.scan(cursor=cursor, match='*')
            all_keys.extend(keys)
            if cursor == 0:
                break
    
        return [key.decode('utf-8') for key in all_keys]

    def close(self):
        self.connection.close()

class RedisSentinelConnector(RemoteConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://" and separated by commas.
    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = 'REDIS_TIMEOUT'
    ENV_REDIS_SERVICE_NAME = 'REDIS_SERVICE_NAME'

    def __init__(
            self, 
            hosts_and_ports: List[Tuple[str, Union[str, int]]]
        ):

        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is not found, using default value 'mymaster'")
                service_name = "mymaster"
            case value:
                service_name = value

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        
        self.sentinel = redis.Sentinel(hosts_and_ports, timeout)
        self.master = self.sentinel.master_for(service_name, socket_timeout = timeout)
        self.slave = self.sentinel.slave_for(service_name, socket_timeout = timeout)

    def exists(self, key: str) -> bool:
        return self.slave.exists(key)

    def get(self, key: str) -> Optional[bytes]:
        return self.slave.get(key)

    def set(self, key: str, obj: bytes) -> None:
        self.master.set(key, obj)

    def list(self):
        cursor = 0
        all_keys = []
    
        while True:
            cursor, keys = self.slave.scan(cursor=cursor, match='*')
            all_keys.extend(keys)
            if cursor == 0:
                break
    
        return [key.decode('utf-8') for key in all_keys]

    def close(self):
        self.master.close()
        self.slave.close()
