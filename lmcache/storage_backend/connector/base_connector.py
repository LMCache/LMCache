import abc
import time
from typing import Optional, List

from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.logging import init_logger

logger = init_logger(__name__)

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
    def get(self, key: str) -> Optional[bytes]:
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
    def set(self, key: str, obj: bytes) -> None:
        """
        Send the objects (bytes) with the corresponding key to the remote server

        Input:
            key: the key of the corresponding object
            obj: the object (bytes) of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

class RemoteConnectorDebugWrapper(RemoteConnector):
    def __init__(self, connector: RemoteConnector):
        self.connector = connector

    def exists(self, key: str) -> bool:
        return self.connector.exists(key)

    @_lmcache_nvtx_annotate
    def get(self, key: str) -> Optional[bytes]:
        start = time.perf_counter()
        ret = self.connector.get(key)
        end = time.perf_counter()

        if ret is None or len(ret) == 0:
            logger.debug("Didn't get any data from the remote backend, key is {key}")
            return None

        logger.debug("Get %.2f MBytes data from the remote backend takes %.2f ms", 
                    len(ret) / 1e6, 
                    (end - start) * 1e3
                )
        return ret

    def set(self, key: str, obj: bytes) -> None:
        start = time.perf_counter()
        self.connector.set(key, obj)
        end = time.perf_counter()
        logger.debug("Put %.2f MBytes data to the remote backend takes %.2f ms", 
                    len(obj) / 1e6, 
                    (end - start) * 1e3
                )

    def list(self) -> List[str]:
        return self.connector.list()

    def close(self) -> None:
        return self.connector.close()
