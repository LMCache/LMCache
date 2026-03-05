# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, AbstractSet, Optional
import asyncio
import importlib  # Added for dynamic import

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.p2p_backend import P2PBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def is_cuda_worker(metadata: LMCacheMetadata) -> bool:
    """
    Check if the current role is worker and CUDA is available.

    Args:
        metadata: The LMCache engine metadata.

    Returns:
        True if the worker is not a scheduler and CUDA is available.
    """
    return metadata.role != "scheduler" and torch.cuda.is_available()


def storage_plugin_launcher(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: Optional[LocalCPUBackend],
    dst_device: str,
    storage_backends: OrderedDict[str, StorageBackendInterface],
) -> None:
    """
    Loads custom storage backends based on configuration.

    Looks for backend configurations in config.extra_config and instantiates
    them using the specified module and class names.
    """
    # Get the list of allowed external backends if configured
    storage_plugins = set(config.storage_plugins) if config.storage_plugins else set()
    if storage_plugins and not config.extra_config:
        logger.warning(
            "storage_plugins=%s is set but extra_config is empty; "
            "plugin settings must be provided under extra_config, e.g. "
            "extra_config.storage_plugin.<name>.module_path/class_name",
            sorted(storage_plugins),
        )
        return
    if not config.extra_config:
        return

    for storage_plugin in storage_plugins:
        try:
            module_path = config.extra_config.get(
                f"storage_plugin.{storage_plugin}.module_path"
            )
            class_name = config.extra_config.get(
                f"storage_plugin.{storage_plugin}.class_name"
            )

            if not module_path or not class_name:
                logger.warning(
                    f"Backend {storage_plugin} missing module_path or class_name"
                )
                continue

            logger.warning(
                "The 'memory_allocator' argument is deprecated and will "
                "be ignored. Storage backends now manage their own memory "
                "allocators since PR "
                "https://github.com/LMCache/LMCache/pull/1578"
            )

            # Dynamically import the module
            module = importlib.import_module(module_path)
            # Get the class from the module
            backend_class = getattr(module, class_name)

            # Create the backend instance
            backend_instance = backend_class(
                config=config,
                dst_device=dst_device,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=loop,
            )

            # Add to storage backends
            storage_backends[storage_plugin] = backend_instance
            logger.info(f"Created dynamic backend: {storage_plugin}")

        except Exception as e:
            logger.error(f"Failed to create backend {storage_plugin}: {str(e)}")


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    loop: asyncio.AbstractEventLoop,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,
    skip_backends: Optional[AbstractSet[str]] = None,
    existing_backends: Optional[OrderedDict[str, StorageBackendInterface]] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    if is_cuda_worker(metadata):
        dst_device = f"cuda:{torch.cuda.current_device()}"
    elif dst_device == "xpu":
        dst_device = f"xpu:{torch.xpu.current_device()}"
    else:
        dst_device = "cpu"
    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
    _skip = skip_backends or set()

    extra_config = config.extra_config
    enable_nixl_storage = extra_config is not None and extra_config.get(
        "enable_nixl_storage"
    )

    if config.enable_pd and "PDBackend" not in _skip:
        # First Party
        from lmcache.v1.storage_backend.pd_backend import PDBackend

        storage_backends["PDBackend"] = PDBackend(config, metadata)

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    # Reuse existing LocalCPUBackend when available so that
    # dependent backends (disk, remote, p2p, …) keep working.
    local_cpu_backend: Optional[LocalCPUBackend] = None
    if existing_backends and "LocalCPUBackend" in existing_backends:
        _existing_cpu = existing_backends["LocalCPUBackend"]
        if isinstance(_existing_cpu, LocalCPUBackend):
            local_cpu_backend = _existing_cpu

    if metadata.role == "scheduler":
        # For scheduler role, local_cpu_backend is None
        pass
    elif not config.enable_pd or config.local_cpu:
        if "LocalCPUBackend" in _skip:
            pass  # Skipped — already exists
        elif config.max_local_cpu_size > 0:
            local_cpu_backend = LocalCPUBackend(
                config,
                metadata,
                dst_device,
                lmcache_worker,
            )
            backend_name = str(local_cpu_backend)
            storage_backends[backend_name] = local_cpu_backend
        else:
            logger.info("No cpu memory is allocated as max_local_cpu_size <= 0")

    if config.enable_p2p and "P2PBackend" not in _skip:
        assert local_cpu_backend is not None
        assert lmcache_worker is not None
        p2p_backend = P2PBackend(
            config,
            metadata,
            loop,
            local_cpu_backend,
            lmcache_worker,
        )
        backend_name = str(p2p_backend)
        storage_backends[backend_name] = p2p_backend

    if enable_nixl_storage and "NixlStorageBackend" not in _skip:
        # First Party
        from lmcache.v1.storage_backend.nixl_storage_backend import (
            NixlStorageBackend,
        )

        storage_backends["NixlStorageBackend"] = (
            NixlStorageBackend.CreateNixlStorageBackend(config, loop, metadata)
        )

    if (
        config.local_disk
        and config.max_local_disk_size > 0
        and "LocalDiskBackend" not in _skip
    ):
        assert local_cpu_backend is not None
        local_disk_backend = LocalDiskBackend(
            config,
            loop,
            local_cpu_backend,
            dst_device,
            lmcache_worker,
            metadata,
        )

        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.gds_path is not None and "GdsBackend" not in _skip:
        gds_backend = GdsBackend(
            config,
            metadata,
            loop,
            dst_device,
        )
        storage_backends[str(gds_backend)] = gds_backend

    if config.remote_url is not None and "RemoteBackend" not in _skip:
        assert local_cpu_backend is not None, (
            "Remote backend requires local CPU backend as a buffer."
            "Please turn on local cpu backend with max_local_cpu_size > 0"
        )
        remote_backend = RemoteBackend(
            config,
            metadata,
            loop,
            local_cpu_backend,
            dst_device,
        )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    if not config.enable_pd or config.local_cpu:
        # Load storage backends from configuration
        storage_plugin_launcher(
            config,
            metadata,
            loop,
            local_cpu_backend,
            dst_device,
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
