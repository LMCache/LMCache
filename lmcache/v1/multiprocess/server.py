# SPDX-License-Identifier: Apache-2.0
"""MPCacheEngine compositor and unified cache server entry point."""

# Standard
import argparse
import shutil
import sys
import time

# Third Party
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import (
    EngineModule,
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.non_gpu_transfer import NonGPUTransferModule
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)

logger = init_logger(__name__)


class MPCacheEngine:
    """Compositor that assembles pluggable engine modules.

    Holds the shared :class:`MPCacheEngineContext` and a list of
    :class:`EngineModule` instances.  Provides aggregated
    ``report_status()`` and ``close()`` across all modules.

    Args:
        context: The shared engine context.
        modules: List of engine modules to compose.
    """

    def __init__(
        self,
        context: MPCacheEngineContext,
        modules: list[EngineModule],
    ) -> None:
        self._context = context
        self._modules = modules

    @property
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context."""
        return self._context

    def report_status(self) -> dict:
        """Return an aggregated status dict from all modules.

        Returns:
            Combined status from the storage manager, engine metadata,
            and each module's ``report_status()`` output.
        """
        sm = self._context.storage_manager.report_status()
        status: dict = {
            "is_healthy": sm["is_healthy"],
            "engine_type": self.__class__.__name__,
            "chunk_size": self._context.chunk_size,
            "hash_algorithm": self._context.token_hasher.hash_algorithm_name,
            "active_sessions": self._context.session_manager.active_count(),
            "storage_manager": sm,
        }
        for module in self._modules:
            status.update(module.report_status())
        return status

    def close(self) -> None:
        """Close all modules and release shared resources."""
        for module in self._modules:
            module.close()
        self._context.storage_manager.close()
        logger.info("MPCacheEngine closed")

    # HTTP-layer passthroughs lost in the engine refactor.

    @property
    def storage_manager(self) -> StorageManager:
        """Used by ``/quota/*``."""
        return self._context.storage_manager

    @property
    def gpu_contexts(self) -> dict[int, GPUCacheContext] | None:
        """Used by ``/kvcache/check``; unwraps :class:`GPUContextEntry`."""
        for module in self._modules:
            if isinstance(module, GPUTransferModule):
                return {i: e.cache_context for i, e in module.cache_contexts.items()}
        return None

    def clear(self) -> None:
        """Used by ``/clear-cache``; delegates to :class:`ManagementModule`."""
        for module in self._modules:
            if isinstance(module, ManagementModule):
                module.clear()
                return
        raise RuntimeError("MPCacheEngine.clear: no ManagementModule registered")


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    """Register a handler with the message queue server.

    Args:
        server: The message queue server.
        request_type: The request type to handle.
        handler_function: The handler callable.
    """
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def _build_modules(
    ctx: MPCacheEngineContext,
    mp_config: MPServerConfig,
) -> list[EngineModule]:
    """Assemble the list of engine modules based on configuration.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which modules to load.

    Returns:
        List of initialized engine modules.

    Raises:
        ValueError: If blend engine is requested with supported_transfer_mode="non_gpu".
    """
    modules: list[EngineModule] = [
        LookupModule(ctx),
        ManagementModule(ctx),
    ]

    if mp_config.supported_transfer_mode == "gpu":
        modules.append(GPUTransferModule(ctx))
    elif mp_config.supported_transfer_mode == "non_gpu":
        modules.append(NonGPUTransferModule(ctx))
    elif mp_config.supported_transfer_mode == "auto":
        modules.append(GPUTransferModule(ctx))
        modules.append(NonGPUTransferModule(ctx))
    else:
        raise ValueError(
            f"Unsupported supported_transfer_mode '{mp_config.supported_transfer_mode}'"
        )

    logger.info("Supported transfer mode: %s", mp_config.supported_transfer_mode)

    if mp_config.engine_type == "blend_legacy":
        if mp_config.supported_transfer_mode == "non_gpu":
            raise ValueError(
                "Legacy blend engine requires supported_transfer_mode to be "
                f"'gpu' or 'auto', got '{mp_config.supported_transfer_mode}'"
            )
        # First Party
        from lmcache.v1.multiprocess.modules.blend import BlendModule

        modules.append(BlendModule(ctx))

    # "blend" selects CacheBlend V3 (the current implementation).
    if mp_config.engine_type == "blend":
        if mp_config.supported_transfer_mode == "non_gpu":
            raise ValueError(
                "blend (V3) engine requires supported_transfer_mode 'gpu' or "
                f"'auto', got '{mp_config.supported_transfer_mode}'"
            )
        # First Party
        from lmcache.v1.multiprocess.modules.blend_v3 import BlendV3Module

        gpu_transfer = next(m for m in modules if isinstance(m, GPUTransferModule))
        lookup_module = next(m for m in modules if isinstance(m, LookupModule))
        modules.append(BlendV3Module(ctx, gpu_transfer, lookup_module))

    return modules


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
) -> tuple[MessageQueueServer, MPCacheEngine] | None:
    """Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server.
        storage_manager_config: Configuration for the storage manager.
        obs_config: Configuration for the observability stack.
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive.
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server in a background thread.  Set to
            ``False`` when an external HTTP framework already serves
            ``/metrics`` to avoid port conflicts or redundant servers.

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine).
        If return_engine is False: None (blocks until interrupted).
    """
    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # For non-GPU transfer: apply shm_name from mp_config and verify capacity
    if mp_config.supported_transfer_mode != "gpu":
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        if mp_config.shm_name is not None:
            mem_cfg.shm_name = mp_config.shm_name
        if mem_cfg.shm_name and sys.platform.startswith("linux"):
            logger.info("Checking if shm capacity is larger than L1 request")
            try:
                free_bytes = shutil.disk_usage("/dev/shm").free
                if free_bytes < mem_cfg.size_in_bytes:
                    logger.warning(
                        "Insufficient /dev/shm capacity: need %d bytes, have %d bytes. "
                        "Disabling SHM, falling back to pickle.",
                        mem_cfg.size_in_bytes,
                        free_bytes,
                    )
                    mem_cfg.shm_name = ""
            except OSError:
                logger.warning(
                    "Cannot verify /dev/shm capacity; disabling SHM.",
                    exc_info=True,
                )
                mem_cfg.shm_name = ""

    ctx = MPCacheEngineContext(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
    )

    modules = _build_modules(ctx, mp_config)
    engine = MPCacheEngine(ctx, modules)

    zmq_context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=zmq_context,
    )

    all_specs: list[HandlerSpec] = []
    for module in modules:
        all_specs.extend(module.get_handlers())

    for spec in all_specs:
        add_handler_helper(server, spec.request_type, spec.handler)

    affinity_types = [
        s.request_type for s in all_specs if s.pool == ThreadPoolType.AFFINITY
    ]
    normal_types = [
        s.request_type for s in all_specs if s.pool == ThreadPoolType.NORMAL
    ]
    if affinity_types:
        server.add_affinity_thread_pool(
            affinity_types, max_workers=mp_config.max_gpu_workers
        )
    if normal_types:
        server.add_normal_thread_pool(
            normal_types, max_workers=mp_config.max_cpu_workers
        )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )

    if not hasattr(torch_dev, "init"):
        logger.warning(
            "Backend '%s' does not support init(), skipping device init",
            torch_device_type,
        )
    else:
        torch_dev.init()
    server.start()

    logger.info("LMCache cache server is running...")

    if return_engine:
        return server, engine

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        event_bus.stop()
        server.close()
        engine.close()
    return None


def parse_args():
    """Parse command line arguments for the cache server.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
