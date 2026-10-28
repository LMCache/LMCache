# SPDX-License-Identifier: Apache-2.0
"""Selects and builds the gRPC request server for a multiprocess config."""

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.transport.grpc_impl.aio_server import (
    GrpcAioMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.native_server import (
    GrpcNativeMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_base import (
    GrpcMultiprocessServerBase,
)


def build_grpc_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> GrpcMultiprocessServerBase:
    """Build a gRPC request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration. ``grpc_server_mode``
            selects the sync, aio or native server.

    Returns:
        Configured, but not yet started, gRPC request server.
    """
    server_cls: type[GrpcMultiprocessServerBase]
    if mp_config.grpc_server_mode == "native":
        server_cls = GrpcNativeMultiprocessServer
    elif mp_config.grpc_server_mode == "aio":
        server_cls = GrpcAioMultiprocessServer
    else:
        server_cls = GrpcMultiprocessServer
    server = server_cls(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
        grpc_server_workers=mp_config.grpc_server_workers,
    )
    server.add_modules(modules)
    return server
