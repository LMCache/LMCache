# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import asyncio
import importlib  # Added for dynamic import

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    NixlCPUMemoryAllocator,
    PagedTensorMemoryAllocator,
)
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


def create_dynamic_backends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    local_cpu_backend: LocalCPUBackend,
    dst_device: str,
    lookup_server: Optional[LookupServerInterface],
    storage_backends: OrderedDict[str, StorageBackendInterface],
) -> None:
    """
    Dynamically create backends based on configuration.

    Looks for backend configurations in config.extra_config and instantiates
    them using the specified module and class names.
    """
    if not config.extra_config:
        return

    # Get the list of allowed external backends if configured
    allowed_backends = (
        set(config.external_backends) if config.external_backends else set()
    )

    for backend_name in allowed_backends:
        try:
            module_path = config.extra_config.get(
                f"external_backend.{backend_name}.module_path"
            )
            class_name = config.extra_config.get(
                f"external_backend.{backend_name}.class_name"
            )

            if not module_path or not class_name:
                logger.warning(
                    f"Backend {backend_name} missing module_path or class_name"
                )
                continue

            # Dynamically import the module
            module = importlib.import_module(module_path)
            # Get the class from the module
            backend_class = getattr(module, class_name)

            # Create the backend instance
            backend_instance = backend_class(
                config,
                metadata,
                loop,
                memory_allocator,
                local_cpu_backend,
                dst_device,
                lookup_server,
            )

            # Add to storage backends
            storage_backends[backend_name] = backend_instance
            logger.info(f"Created dynamic backend: {backend_name}")

        except Exception as e:
            logger.error(f"Failed to create backend {backend_name}: {str(e)}")


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

    extra_config = config.extra_config
    enable_nixl_storage = extra_config is not None and extra_config.get(
        "enable_nixl_storage"
    )

    if config.enable_nixl:
        if config.enable_xpyd:
            # First Party
            from lmcache.v1.storage_backend.nixl_backend_v3 import (
                NixlBackend as NixlBackendV3,
            )

            assert isinstance(memory_allocator, NixlCPUMemoryAllocator)
            storage_backends["NixlBackend"] = NixlBackendV3.CreateNixlBackend(
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

    if enable_nixl_storage:
        # First Party
        from lmcache.v1.storage_backend.nixl_storage_backend import (
            NixlStorageBackend,
        )

        if not isinstance(memory_allocator, PagedTensorMemoryAllocator):
            raise TypeError(
                f"Expected PagedTensorMemoryAllocator,"
                f" but got {type(memory_allocator).__name__}"
            )

        storage_backends["NixlStorageBackend"] = (
            NixlStorageBackend.CreateNixlStorageBackend(
                config, loop, metadata, memory_allocator
            )
        )

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

    # Create dynamic backends from configuration
    create_dynamic_backends(
        config,
        metadata,
        loop,
        memory_allocator,
        local_cpu_backend,
        dst_device,
        lookup_server,
        storage_backends,
    )

    # Only wrap if audit is enabled in config
    if config.extra_config is not None and config.extra_config.get(
        "audit_backend_enabled", False
    ):
        # First Party
        from lmcache.v1.storage_backend.audit_backend import AuditBackend

        # Conditionally wrap backends with audit logging if enabled in config
        audited_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
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
