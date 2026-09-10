# SPDX-License-Identifier: Apache-2.0
"""Construct a gRPC request server from multiprocess business modules."""

# Standard
from typing import TypeVar, cast

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services import (
    BlendServiceImpl,
    ControllerServiceImpl,
    DebugServiceImpl,
    EngineDrivenServiceImpl,
    LMCacheDrivenServiceImpl,
    LookupServiceImpl,
    ObservabilityServiceImpl,
    P2PServiceImpl,
    QStoreServiceImpl,
)

_ModuleT = TypeVar("_ModuleT", bound=EngineModule)


class _ModuleResolver:
    """Resolve unique business modules by their public type."""

    def __init__(self, modules: list[EngineModule]) -> None:
        self._modules = tuple(modules)

    def require(self, module_type: type[_ModuleT]) -> _ModuleT:
        """Return the single module matching a required type.

        Args:
            module_type: Required business module type.

        Returns:
            The unique matching module.

        Raises:
            RuntimeError: If no module or multiple modules match.
        """
        matches = self._find(module_type)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {module_type.__name__}, found {len(matches)}"
            )
        return matches[0]

    def optional(self, module_type: type[_ModuleT]) -> _ModuleT | None:
        """Return the module matching an optional type.

        Args:
            module_type: Optional business module type.

        Returns:
            The matching module, or ``None`` when it is not configured.

        Raises:
            RuntimeError: If multiple modules match.
        """
        matches = self._find(module_type)
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most one {module_type.__name__}, found {len(matches)}"
            )
        return matches[0] if matches else None

    def _find(self, module_type: type[_ModuleT]) -> tuple[_ModuleT, ...]:
        return tuple(
            cast(_ModuleT, module)
            for module in self._modules
            if isinstance(module, module_type)
        )


def build_grpc_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> GrpcMultiprocessServer:
    """Build a gRPC server from concrete module-backed services.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.

    Returns:
        Configured, but not yet started, gRPC request server.

    Raises:
        RuntimeError: If a required module is missing or a module type is
            configured more than once.
    """
    resolver = _ModuleResolver(modules)
    lookup_module = resolver.require(LookupModule)
    management_module = resolver.require(ManagementModule)
    p2p_controller = resolver.require(P2PController)
    lmcache_driven_module = resolver.optional(LMCacheDrivenTransferModule)
    engine_driven_module = resolver.optional(EngineDrivenTransferModule)
    qstore_module = resolver.optional(QStoreModule)
    blend_module = None
    if mp_config.engine_type == "blend":
        # First Party
        from lmcache.v1.multiprocess.modules.blend import BlendModule

        blend_module = resolver.require(BlendModule)

    server = GrpcMultiprocessServer(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
    )
    service_implementations = (
        (
            "LMCacheDrivenService",
            LMCacheDrivenServiceImpl(
                lmcache_driven_module,
                blend_module if blend_module is not None else lmcache_driven_module,
            ),
        ),
        (
            "EngineDrivenService",
            EngineDrivenServiceImpl(engine_driven_module),
        ),
        ("LookupService", LookupServiceImpl(lookup_module)),
        ("QStoreService", QStoreServiceImpl(qstore_module)),
        ("ControllerService", ControllerServiceImpl(management_module)),
        ("DebugService", DebugServiceImpl(management_module)),
        (
            "ObservabilityService",
            ObservabilityServiceImpl(management_module),
        ),
        ("P2PService", P2PServiceImpl(p2p_controller)),
        ("BlendService", BlendServiceImpl(blend_module)),
    )
    for service_name, implementation in service_implementations:
        server.add_service(service_name, implementation)
    return server


__all__ = ["build_grpc_request_server"]
