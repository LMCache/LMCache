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

# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import asyncio

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryAllocatorInterface
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.weka_gds_backend import WekaGdsBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    lookup_server: Optional[LookupServerInterface] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    # Replace 'cuda' with 'cuda:<device id>'
    if dst_device == "cuda":
        dst_device = f"cuda:{torch.cuda.current_device()}"

    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()

    if config.enable_nixl:
        # First Party
        from lmcache.v1.storage_backend.nixl_backend import NixlBackend

        storage_backends["NixlBackend"] = NixlBackend.CreateNixlBackend(
            config, metadata
        )
        assert config.nixl_buffer_device is not None
        return storage_backends

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    local_cpu_backend = LocalCPUBackend(
        config,
        memory_allocator,
        lookup_server,
        lmcache_worker,
    )
    backend_name = str(local_cpu_backend)
    storage_backends[backend_name] = local_cpu_backend

    if config.local_disk and config.max_local_disk_size > 0:
        local_disk_backend = LocalDiskBackend(
            config,
            loop,
            local_cpu_backend,
            dst_device,
            lmcache_worker,
            lookup_server,
        )
        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.weka_path is not None:
        weka_backend = WekaGdsBackend(config, loop, memory_allocator, dst_device)
        # TODO(Serapheim): there's a chance we don't want the local
        # CPU cache in front of ours. Let's experiment and potentially
        # change that in the future.
        storage_backends[str(weka_backend)] = weka_backend
    if config.gds_path is not None:
        gds_backend = GdsBackend(config, loop, memory_allocator, dst_device)
        storage_backends[str(gds_backend)] = gds_backend
    if config.remote_url is not None:
        remote_backend = RemoteBackend(
            config, metadata, loop, local_cpu_backend, dst_device, lookup_server
        )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    return storage_backends
