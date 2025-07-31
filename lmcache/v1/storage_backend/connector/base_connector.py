# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import abc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """

    @abc.abstractmethod
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a string

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Get the memory_obj of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The memory_obj of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Send the memory_obj with the corresponding key directly
        to the remote server. Will decrease the ref count after
        send finishes.

        Input:
            key: the CacheEngine key
            memory_obj: the memory_obj of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        """
        Close remote server

        """
        raise NotImplementedError

    def support_ping(self) -> bool:
        """
        Check if the connector supports ping operation

        Returns:
            True if ping is supported, False otherwise
        """
        return False

    async def ping(self) -> int:
        """
        Ping the remote server

        Returns:
            The error code, 0 means success
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
