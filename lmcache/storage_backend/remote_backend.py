from typing import Tuple, Optional
import re
import abc
import io
import torch
import redis
import time
import pickle

from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger

logger = init_logger(__name__)

# TODO: unit test for the remote connector

class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """
    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a string

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> Optional[str]:
        """
        Get the objects (bytes) of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The object (bytes) of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, key: str, obj: str) -> None:
        """
        Send the objects (bytes) with the corresponding key to the remote server

        Input:
            key: the key of the corresponding object
            obj: the object (bytes) of the corresponding key
        """
        raise NotImplementedError


class RedisConnector(RemoteConnector):
    def __init__(self, host: str, port: int):
        self.connection = redis.Redis(host=host, port=port)

    def exists(self, key: str) -> bool:
        return self.connection.exists(key)

    def get(self, key: str) -> Optional[str]:
        return self.connection.get(key)

    def set(self, key: str, obj: str) -> None:
        self.connection.set(key, obj)


class ObjectStoreClient:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

    def send_all(self, data):
        self.client_socket.sendall(len(data).to_bytes(4, 'big') + data)

    def put(self, key, value):
        data = pickle.dumps(('put', key, value))
        self.send_all(data)
        response = self.client_socket.recv(1024)
        print(response.decode())

    def get(self, key):
        data = pickle.dumps(('get', key, None))
        self.send_all(data)
        header = self.client_socket.recv(4)
        length = int.from_bytes(header, 'big')
        response = self.client_socket.recv(length)
        return pickle.loads(response) if response != b'NONE' else None

    def exists(self, key):
        data = pickle.dumps(('exists', key, None))
        self.send_all(data)
        response = self.client_socket.recv(1024)
        return response == b'YES'


# TODO: torch.distributed connector


class LMCRemoteBackend(LMCBackendInterface):
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
        remote_type, host, port = self._parse_cfg(config)
        match remote_type:
            case "redis":
                self.connection = RedisConnector(host, port)
            case _:
                raise ValueError(f"Unsupported remote type {remote_type}")


    def _parse_cfg(
            self,
            config: LMCacheEngineConfig
        ) -> Tuple[str, int]:
        """
        Parse the config to the redis connection parameters (host and port)
        Returns:
            Tuple of hostname and port number
        """
        m = re.match(r"(.*)://(.*):(\d+)", config.remote_url)
        if m is None:
            raise ValueError(f"Invalid remote url {config.remote_url}")
        return m.group(1), m.group(2), int(m.group(3))

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
        ) -> Optional[torch.Tensor]:
        """
        Retrive the KV cache chunk (in a single big tensor) by the given key
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
