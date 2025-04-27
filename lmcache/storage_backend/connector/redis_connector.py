import inspect
import os
from typing import List, Optional, Tuple, Union

import redis

from lmcache.logging import init_logger
from lmcache.storage_backend.connector.base_connector import \
    RemoteBytesConnector

logger = init_logger(__name__)


class RedisConnector(RemoteBytesConnector):
    """
    The remote url should start with "redis://" and only have one host-port pair
    """

    def __init__(self, host: str, port: int):
        self.connection = redis.Redis(host=host, port=port)

    def exists(self, key: str) -> bool:
        return bool(self.connection.exists(key))

    def get(self, key: str) -> Optional[bytes]:
        result = self.connection.get(key)

        # assert that result is not a co-routine
        assert not inspect.isawaitable(result)

        return result if result is None else bytes(result)

    def set(self, key: str, obj: bytes) -> None:  # type: ignore[override]
        self.connection.set(key, obj)

    def list(self):
        cursor = 0
        all_keys: List[bytes] = []

        while True:
            ret: Tuple[int, List[bytes]] = self.connection.scan(
                cursor=cursor, match="*")  # type: ignore
            cursor, keys = ret
            all_keys.extend(keys)
            if cursor == 0:
                break

        return [key.decode("utf-8") for key in all_keys]

    def close(self):
        self.connection.close()


class RedisSentinelConnector(RemoteBytesConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://"
    and separated by commas.

    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = "REDIS_TIMEOUT"
    ENV_REDIS_SERVICE_NAME = "REDIS_SERVICE_NAME"

    def __init__(self, hosts_and_ports: List[Tuple[str, Union[str, int]]]):
        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is not"
                    f"found, using default value 'mymaster'")
                service_name = "mymaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1.0  # Ensure float for consistency before casting
            case value:
                try:
                    timeout = float(value)
                except ValueError:
                    logger.warning(
                        (f"Invalid value for {self.ENV_REDIS_TIMEOUT}: {value}. "
                         f"Using default 1.0")
                    )
                    timeout = 1.0

        # Ensure hosts_and_ports has the correct type for redis.Sentinel
        sentinel_nodes: List[Tuple[str,
                                   int]] = [(host, int(port))
                                            for host, port in hosts_and_ports]

        # Cast timeout to int for Sentinel
        self.sentinel = redis.Sentinel(sentinel_nodes,
                                       socket_timeout=int(timeout))
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout)  # master_for can take float
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout)  # slave_for can take float

    def exists(self, key: str) -> bool:
        # redis-py exists returns int (number of keys found), cast to bool
        return bool(self.slave.exists(key))

    def get(self, key: str) -> Optional[bytes]:
        result = self.slave.get(key)
        # redis-py get returns bytes or None, ensure it's bytes if not None
        return result if result is None else bytes(result)

    def set(self, key: str, obj: bytes) -> None:  # type: ignore[override]
        self.master.set(key, obj)

    def list(self):
        cursor = 0
        all_keys: List[bytes] = []

        while True:
            ret: Tuple[int, List[bytes]] = self.slave.scan(
                cursor=cursor, match="*")  # type: ignore
            cursor, keys = ret
            all_keys.extend(keys)
            if cursor == 0:
                break

        return [key.decode("utf-8") for key in all_keys]

    def close(self):
        # Ensure connections are closed if they were successfully established
        if hasattr(self, 'master') and self.master:
            self.master.close()  # type: ignore
        if hasattr(self, 'slave') and self.slave:
            self.slave.close()  # type: ignore
        # Sentinel itself doesn't have a direct close method in older redis-py,
        # closing master/slave connections is usually sufficient.
