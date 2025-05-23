# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import MemoryAllocatorInterface
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.local_cpu_backend import \
    LocalCPUBackend
from lmcache.experimental.storage_backend.local_disk_backend import \
    LocalDiskBackend
from lmcache.experimental.storage_backend.remote_backend import RemoteBackend
from lmcache.logging import init_logger

if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    lookup_server: Optional[LookupServerInterface] = None,
    layerwise: bool = False,
) -> OrderedDict[str, StorageBackendInterface]:

    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] =\
        OrderedDict()

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    local_cpu_backend = LocalCPUBackend(config, memory_allocator,
                                        lookup_server, lmcache_worker,
                                        layerwise)
    backend_name = str(local_cpu_backend)
    storage_backends[backend_name] = local_cpu_backend

    if config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(config, loop, local_cpu_backend,
                                              dst_device, lmcache_worker,
                                              lookup_server)
        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
        remote_backend = RemoteBackend(config, metadata, loop,
                                       local_cpu_backend, dst_device,
                                       lookup_server)
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    # TODO(Jiayi): Please support blending
    config.enable_blending = False
    assert config.enable_blending is False, \
        "blending is not supported for now"

    return storage_backends
