import asyncio
from collections import OrderedDict

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import MemoryAllocatorInterface
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.local_disk_backend import \
    LocalDiskBackend
from lmcache.experimental.storage_backend.remote_backend import RemoteBackend
from lmcache.logging import init_logger

logger = init_logger(__name__)


def CreateStorageBackends(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda") -> OrderedDict[str, StorageBackendInterface]:

    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] =\
        OrderedDict()

    # TODO(Jiayi): The hierarchy is fixed for now
    if config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(config, loop, memory_allocator,
                                              dst_device)
        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        remote_backend = RemoteBackend(config, loop, memory_allocator,
                                       dst_device)
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    # TODO(Jiayi): Please support other backends
    config.enable_blending = False
    assert config.enable_blending is False, \
        "blending is not supported for now"

    return storage_backends
