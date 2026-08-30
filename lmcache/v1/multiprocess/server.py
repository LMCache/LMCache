# SPDX-License-Identifier: Apache-2.0
"""MPCacheServer compositor and unified cache server entry point."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
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
from lmcache.v1.multiprocess.mq import MultiprocessGrpcServer
from lmcache.v1.multiprocess.service import InstanceLivenessTarget
from lmcache.v1.multiprocess.services.engine_driven_transfer import (
    EngineDrivenTransferService,
)
from lmcache.v1.multiprocess.services.experimental import (
    EXPERIMENTAL_TRANSFER,
    TRANSFER_QUERY,
)
from lmcache.v1.multiprocess.services.experimental.qstore import QStoreService
from lmcache.v1.multiprocess.services.lmcache_driven_transfer import (
    LMCacheDrivenTransferService,
)
from lmcache.v1.multiprocess.services.lookup import EngineLookupService
from lmcache.v1.multiprocess.services.management import ManagementService
from lmcache.v1.multiprocess.services.p2p_controller import P2PController
from lmcache.v1.multiprocess.services.rpc_services import (
    BlendServiceImpl,
    ControllerServiceImpl,
    DebugServiceImpl,
    EngineServiceImpl,
    ObservabilityServiceImpl,
    P2PServiceImpl,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.services.blend import BlendService


class _StatusReporter(Protocol):
    """Public status surface exposed by server-side implementation objects."""

    def report_status(self) -> dict:
        """Return service-specific status information."""
        ...


class _Closeable(Protocol):
    """Public close surface exposed by server-side implementation objects."""

    def close(self) -> None:
        """Release resources owned by this object."""
        ...


@dataclass(frozen=True)
class _BuiltRpcServices:
    """Concrete gRPC services plus lifecycle objects for the cache server."""

    engine_service: EngineServiceImpl
    controller_service: ControllerServiceImpl
    debug_service: DebugServiceImpl
    observability_service: ObservabilityServiceImpl
    p2p_service: P2PServiceImpl
    blend_service: BlendServiceImpl | None
    management: ManagementService
    lmcache_driven_transfer: LMCacheDrivenTransferService | None
    status_reporters: Sequence[_StatusReporter]
    closeables: Sequence[_Closeable]


class MPCacheServer:
    """Compositor for the cache server's shared context and lifecycle.

    Args:
        context: The shared engine context.
        status_reporters: Objects contributing status fields.
        closeables: Objects that must release resources before context close.
        management: Management implementation used by HTTP passthroughs.
        lmcache_driven_transfer: Optional LMCache-driven transfer backend used
            by HTTP cache checksum inspection.
    """

    def __init__(
        self,
        context: MPCacheServerContext,
        *,
        status_reporters: Sequence[_StatusReporter],
        closeables: Sequence[_Closeable],
        management: ManagementService,
        lmcache_driven_transfer: LMCacheDrivenTransferService | None,
    ) -> None:
        self._context = context
        self._status_reporters = tuple(status_reporters)
        self._closeables = tuple(closeables)
        self._management = management
        self._lmcache_driven_transfer = lmcache_driven_transfer

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._context

    def report_status(self) -> dict:
        """Return an aggregated status dict from all services.

        Returns:
            Combined status from the storage manager, engine metadata,
            and each service's ``report_status()`` output.
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
        for reporter in self._status_reporters:
            status.update(reporter.report_status())
        return status

    def close(self) -> None:
        """Close implementation objects and release shared resources."""
        for closeable in self._closeables:
            closeable.close()
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
        if self._lmcache_driven_transfer is None:
            return None
        return {
            i: e.cache_context
            for i, e in self._lmcache_driven_transfer.context_entries_snapshot().items()
        }

    def clear(self) -> None:
        """Used by ``/cache/clear``."""
        self._management.clear()


def _build_rpc_services(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> _BuiltRpcServices:
    """Build concrete gRPC service implementations based on configuration.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which implementations to
            load.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.

    Returns:
        Concrete gRPC services plus lifecycle objects.

    Raises:
        ValueError: If blend engine is requested with
        supported_transfer_mode="engine_driven".
    """
    lookup_service = EngineLookupService(ctx)
    p2p_controller = P2PController(
        ctx,
        mp_config.p2p_config,
        coordinator_config,
        mp_config.instance_id,
    )

    # Build transfer and blend implementations first so ManagementService can
    # receive the liveness targets the reaper scans.
    lmcache_driven_transfer: LMCacheDrivenTransferService | None = None
    engine_driven_transfer: EngineDrivenTransferService | None = None
    if mp_config.supported_transfer_mode == "lmcache_driven":
        lmcache_driven_transfer = LMCacheDrivenTransferService(ctx)
    elif mp_config.supported_transfer_mode == "engine_driven":
        engine_driven_transfer = EngineDrivenTransferService(ctx)
    elif mp_config.supported_transfer_mode == "auto":
        lmcache_driven_transfer = LMCacheDrivenTransferService(ctx)
        engine_driven_transfer = EngineDrivenTransferService(ctx)
    else:
        raise ValueError(
            f"Unsupported supported_transfer_mode '{mp_config.supported_transfer_mode}'"
        )

    logger.info("Supported transfer mode: %s", mp_config.supported_transfer_mode)

    liveness_targets: list[InstanceLivenessTarget] = []
    if lmcache_driven_transfer is not None:
        liveness_targets.append(lmcache_driven_transfer)
    if engine_driven_transfer is not None:
        liveness_targets.append(engine_driven_transfer)

    blend: BlendService | None = None
    if mp_config.engine_type == "blend":
        if mp_config.supported_transfer_mode == "engine_driven":
            raise ValueError(
                "blend engine requires supported_transfer_mode "
                f"'lmcache_driven' or 'auto', got "
                f"'{mp_config.supported_transfer_mode}'"
            )
        # First Party
        from lmcache.v1.mp_coordinator.blend_client import (
            BlendCoordinatorClient,
        )
        from lmcache.v1.multiprocess.services.blend import BlendService

        if lmcache_driven_transfer is None:
            raise ValueError("blend engine requires LMCache-driven transfer support")
        # Opt-in: enabled when a coordinator URL is configured (flag or
        # LMCACHE_COORDINATOR_URL, resolved at config parsing); otherwise
        # None and the blend service matches purely locally.
        #
        # Fleet matching also needs cache-event reporting on: the blend
        # index it queries is built from that stream.
        if coordinator_config.url and not coordinator_config.event_reporting:
            logger.warning(
                "Coordinator URL is set but cache-event reporting is off, so "
                "the coordinator has no cache state to match against: fleet "
                "CacheBlend matching is disabled and blend will match "
                "locally only. Pass --coordinator-event-reporting (or set "
                "LMCACHE_COORDINATOR_EVENT_REPORTING=true) to enable it."
            )
        coordinator = BlendCoordinatorClient.maybe_create(
            coordinator_config.url if coordinator_config.event_reporting else "",
            timeout=coordinator_config.blend_timeout,
            match_concurrency=coordinator_config.blend_match_concurrency,
        )
        blend = BlendService(
            ctx,
            lmcache_driven_transfer,
            coordinator=coordinator,
            enable_segmented_prefix=mp_config.enable_segmented_prefix,
            enable_dedup_content=mp_config.enable_dedup_content,
        )
        # The blend service mirrors per-instance CB rope state, so the reaper must
        # notify it via drop_instance_state when an instance is reaped.
        liveness_targets.append(blend)

    # Experimental intermediate tensor transfer services.
    enabled_features = set(mp_config.enable)
    experimental_transfer: list[str] = []
    qstore: QStoreService | None = None
    for enabled_feature in enabled_features:
        if enabled_feature not in EXPERIMENTAL_TRANSFER:
            raise ValueError(
                f"Unknown --enable experimental service '{enabled_feature}'."
            )
        if lmcache_driven_transfer is None:
            raise ValueError(
                f"Experimental service '{enabled_feature}' requires "
                "supported_transfer_mode='lmcache_driven' or 'auto'."
            )
        if enabled_feature == TRANSFER_QUERY:
            qstore = QStoreService(ctx)
            liveness_targets.append(qstore)
        else:
            raise ValueError(f"Unsupported experimental service '{enabled_feature}'.")
        experimental_transfer.append(enabled_feature)

    management = ManagementService(
        ctx,
        liveness_targets=liveness_targets,
        worker_reap_timeout_seconds=mp_config.worker_reap_timeout_seconds,
        worker_registration_grace_seconds=mp_config.worker_registration_grace_seconds,
        experimental_transfer=experimental_transfer,
    )

    engine_service = EngineServiceImpl(
        lookup_service,
        lmcache_driven_transfer=lmcache_driven_transfer,
        engine_driven_transfer=engine_driven_transfer,
        qstore=qstore,
        blend=blend,
    )
    controller_service = ControllerServiceImpl(management)
    debug_service = DebugServiceImpl(management)
    observability_service = ObservabilityServiceImpl(management)
    p2p_service = P2PServiceImpl(p2p_controller)
    blend_service = BlendServiceImpl(blend) if blend is not None else None

    status_reporters: list[_StatusReporter] = [
        lookup_service,
        p2p_controller,
        management,
    ]
    closeables: list[_Closeable] = [
        # Stop the reaper before transfer/blend services release their state.
        management,
        lookup_service,
        p2p_controller,
    ]
    for item in (
        lmcache_driven_transfer,
        engine_driven_transfer,
        qstore,
        blend,
    ):
        if item is not None:
            status_reporters.append(item)
            closeables.append(item)

    return _BuiltRpcServices(
        engine_service=engine_service,
        controller_service=controller_service,
        debug_service=debug_service,
        observability_service=observability_service,
        p2p_service=p2p_service,
        blend_service=blend_service,
        management=management,
        lmcache_driven_transfer=lmcache_driven_transfer,
        status_reporters=status_reporters,
        closeables=closeables,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
    coordinator_config: CoordinatorConfig = DEFAULT_COORDINATOR_CONFIG,
) -> tuple[MultiprocessGrpcServer, MPCacheServer] | None:
    """Run the LMCache cache server with typed gRPC services.

    Args:
        mp_config: Configuration for the gRPC multiprocess server.
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
        If return_engine is True: tuple of (MultiprocessGrpcServer, MPCacheServer).
        If return_engine is False: None (blocks until interrupted).
    """
    # mp_config.instance_id is this server's single source of identity (set via
    # --instance-id, else a random UUID v4). Project it onto the OTel
    # service.instance.id unless observability set that attribute explicitly, so
    # metrics/traces and coordinator membership all key on the same id.
    if obs_config.service_instance_id is None:
        obs_config.service_instance_id = mp_config.instance_id

    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    init_gc_monitor(obs_config.gc_monitor)

    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # When the engine-driven path is loaded (auto or engine_driven):
    # apply shm_name from mp_config and verify capacity.
    if mp_config.supported_transfer_mode != "lmcache_driven":
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
        separate_object_groups=mp_config.separate_object_groups,
        full_sw_kv=is_blend,
    )

    rpc_services = _build_rpc_services(ctx, mp_config, coordinator_config)
    engine = MPCacheServer(
        ctx,
        status_reporters=rpc_services.status_reporters,
        closeables=rpc_services.closeables,
        management=rpc_services.management,
        lmcache_driven_transfer=rpc_services.lmcache_driven_transfer,
    )

    InitializeMPUsageContext(mp_config, storage_manager_config)
    InitializeMPContinuousUsage(event_bus, mp_config.chunk_size)
    InitializeL2ConnectorUsage(event_bus, ctx.storage_manager)
    InitializeL1Usage(event_bus, ctx.storage_manager)

    # gRPC is now the only supported mp-mode transport; ``host`` may
    # optionally carry the ``grpc://`` scheme for readability but is not
    # required (``MultiprocessGrpcServer`` also accepts a bare host:port).
    host = mp_config.host
    bind_prefix = host if "://" in host else "grpc://" + host
    bind_url = bind_prefix + ":" + str(mp_config.port)
    server = MultiprocessGrpcServer(bind_url=bind_url)
    server.add_service("EngineService", rpc_services.engine_service)
    server.add_service("ControllerService", rpc_services.controller_service)
    server.add_service("DebugService", rpc_services.debug_service)
    server.add_service("ObservabilityService", rpc_services.observability_service)
    server.add_service("P2PService", rpc_services.p2p_service)
    if rpc_services.blend_service is not None:
        server.add_service("BlendService", rpc_services.blend_service)
    server.assign_thread_pools(
        max_cpu_workers=mp_config.max_cpu_workers,
        max_gpu_workers=mp_config.max_gpu_workers,
    )

    logger.info(
        "LMCache cache server is running on %s",
        bind_url,
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
    parser = argparse.ArgumentParser(
        description="LMCache gRPC Cache Server (without HTTP)"
    )
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
