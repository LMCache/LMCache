import asyncio
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import List, Optional

import torch

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LocalCPUBackend(StorageBackendInterface):
    def __init__(self):
        pass

    def contains(self, key: CacheEngineKey) -> bool:
        pass

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        pass

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        pass

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        pass

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        pass

    def close(self) -> None:
        pass


