# SPDX-License-Identifier: Apache-2.0
"""MPCacheServer compositor and unified cache server entry point."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import argparse
import shutil
import signal
import sys
import time

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.usage_telemetry.l1_usage import InitializeL1Usage
from lmcache.usage_telemetry.l2_usage import InitializeL2ConnectorUsage
from lmcache.usage_telemetry.mp import InitializeMPUsageContext
from lmcache.usage_telemetry.mp_continuous import InitializeMPContinuousUsage
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
    resolve_grpc_metrics_enabled,
)
from lmcache.v1.mp_observability.gc_monitor import (
    init_gc_monitor,
    shutdown_gc_monitor,
)
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    DEFAULT_COORDINATOR_CONFIG,
    CoordinatorConfig,
    MPServerConfig,
    add_coordinator_args,
    add_mp_server_args,
    parse_args_to_coordinator_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
    discover_modules,
    order_modules,
)
from lmcache.v1.multiprocess.ext_server_module import (
    TransportServiceRegistrar,
    build_server_module_router,
    load_server_module_components,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.ipc_policy import set_isolated_ipc

logger = init_logger(__name__)


class MPCacheServer:
    """Compositor that assembles pluggable engine modules.

    Holds the shared :class:`MPCacheServerContext` and a list of
    :class:`EngineModule` instances.  Provides aggregated
    ``report_status()`` and ``close()`` across all modules.

    Args:
        context: The shared engine context.
        modules: List of engine modules to compose.
    """

    def __init__(
        self,
        context: MPCacheServerContext,
        modules: list[EngineModule],
    ) -> None:
        self._context = context
        self._modules = modules

    @property
    def context(self) -> MPCacheServerContext:
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
        self._context.close()
        logger.info("MPCacheServer closed")

    # HTTP-layer passthroughs lost in the engine refactor.

    @property
    def storage_manager(self) -> StorageManager:
        """Used by ``/quota/*``."""
        return self._context.storage_manager

    @property
    def cache_contexts(self) -> dict[int, BaseCacheContext] | None:
        """Used by ``/cache/checksums``; unwraps :class:`ContextEntry`."""
        for module in self._modules:
            if isinstance(module, LMCacheDrivenTransferModule):
                return {
                    i: e.cache_context
                    for i, e in module.context_entries_snapshot().items()
                }
        return None

    def clear(self, force: bool = False) -> None:
        """Used by ``/cache/clear``; delegates to :class:`ManagementModule`."""
        for module in self._modules:
            if isinstance(module, ManagementModule):
                module.clear(force=force)
                return
        raise RuntimeError("MPCacheServer.clear: no ManagementModule registered")


@dataclass(frozen=True)
class ServerBuildComponents:
    """Modules and transport services composed for one MP server instance."""

    modules: list[EngineModule]
    grpc_service_registrars: tuple[TransportServiceRegistrar, ...] = ()
    zmq_service_registrars: tuple[TransportServiceRegistrar, ...] = ()


def _build_module(
    module_cls: type[DiscoverableModule],
    build_ctx: ModuleBuildContext,
) -> None:
    """Run one module factory and register whatever it returns."""
    module = module_cls.create(build_ctx)
    if module is not None:
        build_ctx.register(module)


def _build_modules(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> list[EngineModule]:
    """Assemble only engine modules based on configuration."""
    return _build_server_components(ctx, mp_config, coordinator_config).modules


def _build_server_components(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> ServerBuildComponents:
    """Assemble the modules and transport registrars for one server.

    Modules are discovered by scanning ``lmcache.v1.multiprocess.modules``
    and built in dependency order (each module after the siblings named in
    its ``module_dependencies``). Close order is the reverse of build order,
    so a module is torn down before the dependencies it holds. ``deferred``
    modules are built after the out-of-tree ``--server-module`` plugins
    because they consume plugin contributions.
    """
    build_ctx = ModuleBuildContext(ctx, mp_config, coordinator_config)
    module_classes = discover_modules()
    known_names = {cls.module_name for cls in module_classes}

    for module_cls in order_modules(
        [cls for cls in module_classes if not cls.deferred], known_names
    ):
        _build_module(module_cls, build_ctx)

    plugin_components = load_server_module_components(
        mp_config.server_modules,
        server_context=ctx,
        mp_config=mp_config,
        coordinator_config=coordinator_config,
        built_modules=build_ctx.built,
    )
    plugin_modules = list(plugin_components.modules)
    for module in plugin_modules:
        build_ctx.add_liveness_target(module)

    for module_cls in order_modules(
        [cls for cls in module_classes if cls.deferred], known_names
    ):
        _build_module(module_cls, build_ctx)

    modules = [*build_ctx.built, *plugin_modules]
    plugin_router = build_server_module_router(ctx, plugin_modules)
    if plugin_router is not None:
        modules.append(plugin_router)

    logger.info(
        "Composed %d engine modules: %s",
        len(modules),
        ", ".join(sorted(build_ctx.module_names)),
    )
    return ServerBuildComponents(
        modules=modules,
        grpc_service_registrars=tuple(plugin_components.grpc_service_registrars),
        zmq_service_registrars=tuple(plugin_components.zmq_service_registrars),
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
    coordinator_config: CoordinatorConfig = DEFAULT_COORDINATOR_CONFIG,
) -> tuple[RequestServer, MPCacheServer] | None:
    """Run the LMCache cache server with the selected request transport.

    Args:
        mp_config: Configuration for the multiprocess server.
        storage_manager_config: Configuration for the storage manager.
        obs_config: Configuration for the observability stack.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive.
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server in a background thread.  Set to
            ``False`` when an external HTTP framework already serves
            ``/metrics`` to avoid port conflicts or redundant servers.

    Returns:
        If return_engine is True: tuple of (request server, MPCacheServer).
        If return_engine is False: None (blocks until interrupted).
    """
    # Before any event IPC backend is resolved (KV-cache registration), so
    # the setting is observed by every resolver in this process.
    set_isolated_ipc(mp_config.isolated_ipc)

    # mp_config.instance_id is this server's single source of identity (set via
    # --instance-id, else a random UUID v4). Project it onto the OTel
    # service.instance.id unless observability set that attribute explicitly, so
    # metrics/traces and coordinator membership all key on the same id.
    if obs_config.service_instance_id is None:
        obs_config.service_instance_id = mp_config.instance_id
    if obs_config.grpc_metrics_enabled is None:
        obs_config.grpc_metrics_enabled = resolve_grpc_metrics_enabled(
            obs_config.grpc_metrics_enabled,
            mp_config.transport,
        )

    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    init_gc_monitor(obs_config.gc_monitor)

    maybe_initialize_trace_recorder(
        event_bus, obs_config, storage_manager_config, instance_id=mp_config.instance_id
    )

    # When the engine-driven path is loaded (auto or engine_driven):
    # apply shm_name from mp_config and verify capacity.
    if (
        mp_config.supported_transfer_mode != "lmcache_driven"
        and len(storage_manager_config.l1_manager_configs) == 1
    ):
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

    # blend engine: full per-chunk SWA KV (blended chunks reuse at arbitrary
    # positions). full_sw_kv widens attention groups only; recurrent groups
    # keep their one-block restore window, so a blend server also serves
    # stock hybrid clients.
    is_blend = mp_config.engine_type == "blend"

    ctx = MPCacheServerContext(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
        null_block_id=mp_config.null_block_id,
        separate_object_groups=mp_config.separate_object_groups,
        full_sw_kv=is_blend,
        session_ttl_seconds=mp_config.session_ttl_seconds,
    )

    components = _build_server_components(ctx, mp_config, coordinator_config)
    engine = MPCacheServer(ctx, components.modules)

    InitializeMPUsageContext(mp_config, storage_manager_config)
    InitializeMPContinuousUsage(event_bus, mp_config.chunk_size)
    InitializeL2ConnectorUsage(event_bus, ctx.storage_manager)
    InitializeL1Usage(event_bus, ctx.storage_manager)

    transport = mp_config.transport
    server: RequestServer = create_request_server(
        components.modules,
        mp_config,
        grpc_service_registrars=components.grpc_service_registrars,
        zmq_service_registrars=components.zmq_service_registrars,
    )

    logger.info(
        "LMCache %s cache server is running on %s:%d",
        transport,
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
    finally:
        shutdown_gc_monitor()
    return None


def parse_args():
    """Parse command line arguments for the cache server.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="LMCache Cache Server (without HTTP)")
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    add_coordinator_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    coordinator_config = parse_args_to_coordinator_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
        coordinator_config=coordinator_config,
    )
