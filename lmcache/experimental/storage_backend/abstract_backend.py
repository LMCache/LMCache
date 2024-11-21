import abc
from typing import Callable

from lmcache.experimental.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey


class StorageWorkerInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    async def put_task(self, key: CacheEngineKey, obj: MemoryObj) -> None:
        """An async function to put the MemoryObj into the storage backend.
        It should free the memory object after finish putting the object.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_task(
        self,
        key: CacheEngineKey,
        callback: Callable[[
            MemoryObj,
        ], None],
    ) -> None:
        """An async function to get the MemoryObj from the storage backend.
        Will call the callback with the MemoryObj when finished.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param Callable[MemoryObj, None] callback: The callback function to 
            be called with the MemoryObj.
        """
        raise NotImplementedError
