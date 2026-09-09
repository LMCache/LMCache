# SPDX-License-Identifier: Apache-2.0
"""Construct a gRPC request server from multiprocess business modules."""

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
    """
    lookup_module = next(
        module for module in modules if isinstance(module, LookupModule)
    )
    management_module = next(
        module for module in modules if isinstance(module, ManagementModule)
    )
    p2p_controller = next(
        module for module in modules if isinstance(module, P2PController)
    )
    lmcache_driven_module = next(
        (
            module
            for module in modules
            if isinstance(module, LMCacheDrivenTransferModule)
        ),
        None,
    )
    engine_driven_module = next(
        (
            module
            for module in modules
            if isinstance(module, EngineDrivenTransferModule)
        ),
        None,
    )
    qstore_module = next(
        (module for module in modules if isinstance(module, QStoreModule)),
        None,
    )
    blend_module = None
    if mp_config.engine_type == "blend":
        # First Party
        from lmcache.v1.multiprocess.modules.blend import BlendModule

        blend_module = next(
            module for module in modules if isinstance(module, BlendModule)
        )

    server = GrpcMultiprocessServer(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
    )
    server.add_service(
        "LMCacheDrivenService",
        LMCacheDrivenServiceImpl(
            lmcache_driven_module,
            blend_module if blend_module is not None else lmcache_driven_module,
        ),
    )
    server.add_service(
        "EngineDrivenService",
        EngineDrivenServiceImpl(engine_driven_module),
    )
    server.add_service("LookupService", LookupServiceImpl(lookup_module))
    server.add_service("QStoreService", QStoreServiceImpl(qstore_module))
    server.add_service("ControllerService", ControllerServiceImpl(management_module))
    server.add_service("DebugService", DebugServiceImpl(management_module))
    server.add_service(
        "ObservabilityService", ObservabilityServiceImpl(management_module)
    )
    server.add_service("P2PService", P2PServiceImpl(p2p_controller))
    server.add_service("BlendService", BlendServiceImpl(blend_module))
    return server


__all__ = ["build_grpc_request_server"]
