# SPDX-License-Identifier: Apache-2.0
"""Construct a gRPC request server from multiprocess business modules."""

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.transport.grpc_impl.server import (
    GrpcMultiprocessServer,
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
    server = GrpcMultiprocessServer(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
    )
    server.add_modules(modules)
    return server


__all__ = ["build_grpc_request_server"]
