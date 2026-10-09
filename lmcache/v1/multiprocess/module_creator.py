# SPDX-License-Identifier: Apache-2.0
"""Assembly of the engine modules composed into one MP cache server."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TypeGuard

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import EngineModule, InstanceLivenessTarget
from lmcache.v1.multiprocess.ext_server_module import (
    TransportServiceRegistrar,
    build_server_module_router,
    load_server_module_components,
)
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental import EXPERIMENTAL_TRANSFER
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController

logger = init_logger(__name__)

_LIVENESS_TARGET_METHODS = (
    "touch_instance",
    "reap_stale_instances",
    "tracked_instance_count",
    "drop_instance_state",
)


def _is_liveness_target(module: EngineModule) -> TypeGuard[InstanceLivenessTarget]:
    """Return whether a module exposes the instance-liveness target contract."""
    return all(
        callable(getattr(module, name, None)) for name in _LIVENESS_TARGET_METHODS
    )


@dataclass(frozen=True)
class ServerBuildComponents:
    """Modules and transport services composed for one MP server instance."""

    modules: list[EngineModule]
    grpc_service_registrars: tuple[TransportServiceRegistrar, ...] = ()
    zmq_service_registrars: tuple[TransportServiceRegistrar, ...] = ()


class ModuleCreator:
    """Builds the engine modules and transport registrars for one server.

    Reads ``--engine-type``, ``--supported-transfer-mode``, ``--enable``, and
    ``--server-module`` to decide which built-in modules to construct, then
    layers out-of-tree plugin modules on top.  Instances are single-use:
    call :meth:`create` once per server.

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

        Returns:
            Initialized modules in close() order, plus the transport-specific
            service registrars returned by out-of-tree module factories.

        Raises:
            ValueError: If ``supported_transfer_mode`` is unknown, if the blend
                engine is requested with ``supported_transfer_mode`` set to
                ``engine_driven``, or if ``--enable`` names an unknown or
                unavailable experimental module.
        """
        lookup_module = LookupModule(self._ctx)
        p2p_controller = P2PController(
            self._ctx,
            self._mp_config.p2p_config,
            self._coordinator_config,
            self._mp_config.instance_id,
            self._mp_config.transport,
        )
        transfer_modules = self._create_transfer_modules()

        # Targets the reaper scans (and reap-notifies). The transfer modules own
        # per-instance liveness; BlendModule is added as a state mirror.
        liveness_targets: list[InstanceLivenessTarget] = [
            m
            for m in transfer_modules
            if isinstance(m, (LMCacheDrivenTransferModule, EngineDrivenTransferModule))
        ]

        blend_module = self._create_blend_module(transfer_modules)
        if blend_module is not None:
            # The blend module mirrors per-instance CB rope state, so the reaper
            # must notify it via drop_instance_state when an instance is reaped.
            liveness_targets.append(blend_module)

        experimental_modules, experimental_transfer = self._create_experimental_modules(
            transfer_modules
        )
        liveness_targets.extend(experimental_modules)

        plugin_components = load_server_module_components(
            self._mp_config.server_modules,
            server_context=self._ctx,
            mp_config=self._mp_config,
            coordinator_config=self._coordinator_config,
            built_modules=[
                lookup_module,
                p2p_controller,
                *transfer_modules,
                *experimental_modules,
                *([blend_module] if blend_module is not None else []),
            ],
        )
        plugin_modules = list(plugin_components.modules)
        for module in plugin_modules:
            if _is_liveness_target(module):
                liveness_targets.append(module)
        plugin_router = build_server_module_router(self._ctx, plugin_modules)

        management = ManagementModule(
            self._ctx,
            liveness_targets=liveness_targets,
            worker_reap_timeout_seconds=self._mp_config.worker_reap_timeout_seconds,
            worker_registration_grace_seconds=(
                self._mp_config.worker_registration_grace_seconds
            ),
            experimental_transfer=experimental_transfer,
        )

        # ManagementModule precedes the transfer/blend modules so close() stops
        # and joins the reaper before those modules clear their state and before
        # storage_manager.close() runs.
        return ServerBuildComponents(
            modules=[
                lookup_module,
                p2p_controller,
                management,
                *transfer_modules,
                *experimental_modules,
                *([blend_module] if blend_module is not None else []),
                *plugin_modules,
                *([plugin_router] if plugin_router is not None else []),
            ],
            grpc_service_registrars=tuple(plugin_components.grpc_service_registrars),
            zmq_service_registrars=tuple(plugin_components.zmq_service_registrars),
        )

    def _create_transfer_modules(self) -> list[EngineModule]:
        """Build the transfer modules selected by ``supported_transfer_mode``.

        Returns:
            One module for ``lmcache_driven`` or ``engine_driven``, both for
            ``auto``, in that order.

        Raises:
            ValueError: If ``supported_transfer_mode`` is not a known mode.
        """
        mode = self._mp_config.supported_transfer_mode
        transfer_modules: list[EngineModule] = []
        if mode == "lmcache_driven":
            transfer_modules.append(LMCacheDrivenTransferModule(self._ctx))
        elif mode == "engine_driven":
            transfer_modules.append(EngineDrivenTransferModule(self._ctx))
        elif mode == "auto":
            transfer_modules.append(LMCacheDrivenTransferModule(self._ctx))
            transfer_modules.append(EngineDrivenTransferModule(self._ctx))
        else:
            raise ValueError(f"Unsupported supported_transfer_mode '{mode}'")

        logger.info("Supported transfer mode: %s", mode)
        return transfer_modules

    def _create_blend_module(
        self,
        transfer_modules: list[EngineModule],
    ) -> BlendModule | None:
        """Build the blend module when ``engine_type`` requests it.

        Args:
            transfer_modules: Already-built transfer modules; blend wraps the
                LMCache-driven one.

        Returns:
            The blend module, or ``None`` when ``engine_type`` is not ``blend``.

        Raises:
            ValueError: If blend is requested with an engine-driven-only
                transfer mode, which has no module for blend to wrap.
        """
        if self._mp_config.engine_type != "blend":
            return None

        if self._mp_config.supported_transfer_mode == "engine_driven":
            raise ValueError(
                "blend engine requires supported_transfer_mode "
                f"'lmcache_driven' or 'auto', got "
                f"'{self._mp_config.supported_transfer_mode}'"
            )

        transfer_module = next(
            m for m in transfer_modules if isinstance(m, LMCacheDrivenTransferModule)
        )
        # Opt-in: enabled when a coordinator URL is configured (flag or
        # LMCACHE_COORDINATOR_URL, resolved at config parsing); otherwise
        # None and the blend module matches purely locally.
        #
        # Fleet matching also needs cache-event reporting on: the blend
        # index it queries is built from that stream.
        if (
            self._coordinator_config.url
            and not self._coordinator_config.event_reporting
        ):
            logger.warning(
                "Coordinator URL is set but cache-event reporting is off, so "
                "the coordinator has no cache state to match against: fleet "
                "CacheBlend matching is disabled and blend will match "
                "locally only. Pass --coordinator-event-reporting (or set "
                "LMCACHE_COORDINATOR_EVENT_REPORTING=true) to enable it."
            )
        coordinator = BlendCoordinatorClient.maybe_create(
            self._coordinator_config.url
            if self._coordinator_config.event_reporting
            else "",
            timeout=self._coordinator_config.blend_timeout,
            match_concurrency=self._coordinator_config.blend_match_concurrency,
        )
        return BlendModule(
            self._ctx,
            transfer_module,
            coordinator=coordinator,
            enable_segmented_prefix=self._mp_config.enable_segmented_prefix,
            enable_dedup_content=self._mp_config.enable_dedup_content,
        )

    def _create_experimental_modules(
        self,
        transfer_modules: list[EngineModule],
    ) -> tuple[list[QStoreModule], list[str]]:
        """Build the experimental transfer modules named by ``--enable``.

        Args:
            transfer_modules: Already-built transfer modules; the experimental
                modules wrap the LMCache-driven one.

        Returns:
            The built modules and the enabled feature names, deduplicated.

        Raises:
            ValueError: If ``--enable`` names an unknown feature, or names one
                that needs an LMCache-driven transfer module that is absent.
        """
        lmcache_driven_module = next(
            (m for m in transfer_modules if isinstance(m, LMCacheDrivenTransferModule)),
            None,
        )
        experimental_transfer: list[str] = []
        experimental_modules: list[QStoreModule] = []
        for enabled_module in set(self._mp_config.enable):
            if enabled_module not in EXPERIMENTAL_TRANSFER:
                raise ValueError(
                    f"Unknown --enable experimental module '{enabled_module}'."
                )
            if lmcache_driven_module is None:
                raise ValueError(
                    f"Experimental module '{enabled_module}' requires "
                    "supported_transfer_mode='lmcache_driven' or 'auto'."
                )
            experimental_module = QStoreModule(self._ctx)
            experimental_modules.append(experimental_module)
            experimental_transfer.append(enabled_module)
        return experimental_modules, experimental_transfer


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
