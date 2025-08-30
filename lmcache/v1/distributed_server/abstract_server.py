# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import abc

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import ClientMetaMessage


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
    async def handle_put(
        self,
        meta: ClientMetaMessage,
        reader,
        writer,
    ) -> bool:
        """
        Handle put from the peer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def batched_issue_put(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        dst_url: str,
        dst_location: Optional[str] = None,
    ) -> bool:
        """
        Perform batched put to the peer.
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
