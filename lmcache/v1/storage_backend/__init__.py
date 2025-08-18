# SPDX-License-Identifier: Apache-2.0
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
        if config.enable_xpyd:
            # First Party
            from lmcache.v1.storage_backend.nixl_backend_v3 import NixlBackend

            storage_backends["NixlBackend"] = NixlBackend.CreateNixlBackend(
                config, metadata, memory_allocator
            )
        else:
            # First Party
            from lmcache.v1.storage_backend.nixl_backend import NixlBackend

            storage_backends["NixlBackend"] = NixlBackend.CreateNixlBackend(
                config, metadata
            )
            assert config.nixl_buffer_device is not None

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    if config.enable_nixl and not config.local_cpu:
        pass
    else:
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

    # Only wrap if audit is enabled in config
    if config.extra_config is not None and config.extra_config.get(
        "audit_backend_enabled", False
    ):
        # First Party
        from lmcache.v1.storage_backend.audit_backend import AuditBackend

        # Conditionally wrap backends with audit logging if enabled in config
        audited_backends = OrderedDict()
        for name, backend in storage_backends.items():
            # Wrap each normal backend with AuditBackend
            if not isinstance(backend, LocalCPUBackend):
                audited_backend = AuditBackend(backend)
                audited_backends[name] = audited_backend
                logger.info(f"Wrapped {name} with AuditBackend")
            else:
                audited_backends[name] = backend
                logger.info(f"Do not wrap {name} as it is a LocalCPUBackend")
        return audited_backends
    else:
        # If audit is not enabled, use the original backends
        return storage_backends
