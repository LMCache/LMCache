# SPDX-License-Identifier: Apache-2.0
"""Assembly of the engine modules composed into one MP cache server.

This module deliberately knows nothing about any concrete engine module. It
discovers them by scanning ``lmcache.v1.multiprocess.modules`` and asks each
one to construct itself from a :class:`ModuleBuildContext`, so adding a module
means adding a :class:`DiscoverableModule` subclass and nothing else.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
    discover_modules,
)
from lmcache.v1.multiprocess.ext_server_module import (
    TransportServiceRegistrar,
    build_server_module_router,
    load_server_module_components,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class ServerBuildComponents:
    """Modules and transport services composed for one MP server instance."""

    modules: list[EngineModule]
    grpc_service_registrars: tuple[TransportServiceRegistrar, ...] = ()
    zmq_service_registrars: tuple[TransportServiceRegistrar, ...] = ()


class ModuleCreator:
    """Builds the engine modules and transport registrars for one server.

    Reads ``--engine-type``, ``--supported-transfer-mode``, ``--enable``, and
    ``--server-module`` to decide which modules to construct. Built-in modules
    are discovered by scanning, not listed here.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which modules to load.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery and by the blend module for fleet matching.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        mp_config: MPServerConfig,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        self._ctx = ctx
        self._mp_config = mp_config
        self._coordinator_config = coordinator_config

    def create(self) -> ServerBuildComponents:
        """Assemble every module and transport registrar for one server.

        Construction order and close order differ on purpose. Each module
        declares a ``module_order`` used for *closing*; modules that depend on
        their siblings additionally rely on that rank for *construction*,
        because a lower rank is built first. ``ManagementModule`` is the one
        exception: it needs the out-of-tree plugin modules, so it declares
        ``deferred`` and is built after them -- while still closing early.

        Returns:
            Initialized modules in close() order, plus the transport-specific
            service registrars returned by out-of-tree module factories.

        Raises:
            ValueError: If a module rejects the configuration, or if
                ``--enable`` names an unknown experimental module.
            ImportError: If a discovered module file cannot be imported.
        """
        build_ctx = ModuleBuildContext(
            self._ctx, self._mp_config, self._coordinator_config
        )
        module_classes = discover_modules()

        for module_cls in module_classes:
            if not module_cls.deferred:
                _build_module(module_cls, build_ctx)

        # Out-of-tree modules see every built-in, and may themselves be
        # liveness targets, so they load before the deferred built-ins.
        plugin_components = load_server_module_components(
            self._mp_config.server_modules,
            server_context=self._ctx,
            mp_config=self._mp_config,
            coordinator_config=self._coordinator_config,
            built_modules=build_ctx.built,
        )
        plugin_modules = list(plugin_components.modules)
        for module in plugin_modules:
            build_ctx.add_liveness_target(module)

        for module_cls in module_classes:
            if module_cls.deferred:
                _build_module(module_cls, build_ctx)

        modules = [
            *build_ctx.built_in_close_order(),
            *plugin_modules,
        ]
        plugin_router = build_server_module_router(self._ctx, plugin_modules)
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


def _build_module(
    module_cls: type[DiscoverableModule],
    build_ctx: ModuleBuildContext,
) -> None:
    """Run one module factory and register whatever it returns.

    Args:
        module_cls: The discovered module class to construct.
        build_ctx: Build context handed to the factory.
    """
    module = module_cls.create(build_ctx)
    if module is not None:
        build_ctx.register(module)


def build_modules(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> list[EngineModule]:
    """Assemble only engine modules based on configuration.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which modules to load.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.

    Returns:
        List of initialized engine modules.
    """
    return ModuleCreator(ctx, mp_config, coordinator_config).create().modules


def build_server_components(
    ctx: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
) -> ServerBuildComponents:
    """Assemble the modules and transport registrars for one server.

    Args:
        ctx: The shared engine context.
        mp_config: Server configuration determining which modules to load.
        coordinator_config: Coordinator connection used by the P2P controller
            for peer discovery.

    Returns:
        Initialized modules and transport-specific service registrars.
    """
    return ModuleCreator(ctx, mp_config, coordinator_config).create()
