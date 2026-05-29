# SPDX-License-Identifier: Apache-2.0
"""MPCacheEngine compositor and unified cache server entry point."""

# Standard
from typing import TypeVar
import argparse
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

# Type variable for ``MPCacheEngine._find_module``: the lookup is invariant
# in the requested module class, so callers receive the precise subclass
# back rather than a bare ``EngineModule`` whose protocol-level surface
# would not include the methods (``clear``, ``gpu_contexts``) the facade
# delegates to.
ModuleT = TypeVar("ModuleT", bound=EngineModule)


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

    # ------------------------------------------------------------------
    # HTTP-API compatibility facade
    #
    # The MP HTTP endpoints (``/clear-cache``, ``/quota``,
    # ``/kvcache/check``, ``/status`` and friends in
    # ``lmcache/v1/multiprocess/http_apis/``) treat ``MPCacheEngine`` as
    # a stable surface and reach for ``engine.clear``,
    # ``engine.storage_manager``, and ``engine.gpu_contexts`` directly.
    # The compositor refactor moved that state into the shared context
    # and individual modules; the forwarders below keep the HTTP layer
    # decoupled from module internals so adding/removing modules does
    # not ripple into HTTP handlers.
    # ------------------------------------------------------------------

    @property
    def storage_manager(self) -> StorageManager:
        """Forward to the shared context's storage manager.

        Used by the ``/quota`` and ``/status`` HTTP endpoints, which
        need quota-manager and usage-by-salt access on the shared
        storage layer.

        Returns:
            The :class:`StorageManager` owned by the shared engine context.
        """
        return self._context.storage_manager

    def clear(self) -> None:
        """Clear all stored KV cache data via the management module.

        Routes through the registered :class:`ManagementModule` so the
        HTTP ``/clear-cache`` endpoint keeps working without the HTTP
        layer importing module classes. Locking and the
        ``memcheck → clear(force=True) → memcheck`` sequence are owned
        by the module itself and are unchanged from the pre-refactor
        ``MPCacheEngine.clear`` behavior.

        Returns:
            None.

        Raises:
            RuntimeError: If no :class:`ManagementModule` is registered.
                Production loadouts in ``_build_modules`` always install
                one, so this is a defensive guard against custom
                compositions rather than a normal runtime path.
        """
        mgmt = self._find_module(ManagementModule)
        if mgmt is None:
            raise RuntimeError(
                "MPCacheEngine.clear() requires a registered ManagementModule"
            )
        mgmt.clear()

    @property
    def gpu_contexts(self) -> dict[int, GPUCacheContext]:
        """Snapshot of registered GPU contexts keyed by ``instance_id``.

        Mirrors the pre-refactor ``MPCacheEngine.gpu_contexts``
        contract: returns a fresh dict on every access, empty when no
        contexts are registered (or when the engine was assembled
        without a :class:`GPUTransferModule`). HTTP callers wanting to
        distinguish "no GPU support" from "no GPU registrations yet"
        should consult :attr:`supports_gpu_kvcache_check` first.

        Returns:
            A fresh ``dict[instance_id, GPUCacheContext]``. Mutating
            the returned dict has no effect on registration state.
        """
        gpu_module = self._find_module(GPUTransferModule)
        if gpu_module is None:
            return {}
        return gpu_module.gpu_contexts

    @property
    def supports_gpu_kvcache_check(self) -> bool:
        """Capability flag for the ``/kvcache/check`` HTTP endpoint.

        Decoupled from :attr:`gpu_contexts` so the data contract there
        can stay a pure mapping while HTTP callers still get a clean
        ``501 Not Implemented`` signal in non-GPU mode (e.g.
        ``transfer_mode != 'gpu'``) instead of a misleading ``404``
        meaning "instance_id not registered".

        Returns:
            ``True`` if a :class:`GPUTransferModule` is registered and
            can host GPU-backed KV checksum diagnostics; ``False``
            otherwise.
        """
        return self._find_module(GPUTransferModule) is not None

    def _find_module(self, module_cls: type[ModuleT]) -> ModuleT | None:
        """Return the first registered module of ``module_cls`` or ``None``.

        Args:
            module_cls: The module subclass to look up
                (e.g. :class:`ManagementModule`).

        Returns:
            The first matching module instance (typed as the requested
            subclass), or ``None`` if no instance of that class is
            registered.
        """
        for module in self._modules:
            if isinstance(module, module_cls):
                return module
        return None

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
        ValueError: If blend engine is requested with non-GPU transfer mode.
    """
    modules: list[EngineModule] = [
        LookupModule(ctx),
        ManagementModule(ctx),
    ]

    if mp_config.transfer_mode == "gpu":
        modules.append(GPUTransferModule(ctx))
    else:
        modules.append(NonGPUTransferModule(ctx))

    if mp_config.engine_type == "blend":
        if mp_config.transfer_mode != "gpu":
            raise ValueError(
                "Blend engine requires transfer_mode='gpu', "
                f"got '{mp_config.transfer_mode}'"
            )
        # First Party
        from lmcache.v1.multiprocess.modules.blend import BlendModule

        modules.append(BlendModule(ctx))

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
