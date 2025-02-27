import abc
from typing import Optional

from lmcache.experimental.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey


class DistributedServerInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    async def handle_get(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Handle get from the peer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def issue_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Perform get from the peer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        """
        Start the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Close the server.
        """
        raise NotImplementedError
