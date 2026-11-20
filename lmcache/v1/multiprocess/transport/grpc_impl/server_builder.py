# SPDX-License-Identifier: Apache-2.0
"""Selects and builds the gRPC request server for a multiprocess config."""

# Standard
import importlib.util

# First Party
from lmcache.logging import init_logger
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
from lmcache.v1.transport.auth import load_request_authentication

logger = init_logger(__name__)


def resolve_grpc_server_mode(mode: str) -> str:
    """Resolve ``auto`` to ``native`` if pygrpc-native is installed, else ``aio``.

    Args:
        mode: The configured ``grpc_server_mode``.

    Returns:
        ``sync``, ``aio`` or ``native``.
    """
    if mode != "auto":
        return mode
    if importlib.util.find_spec("pygrpc_native") is not None:
        return "native"
    logger.info("pygrpc-native is not installed; using the aio gRPC server")
    return "aio"


def build_grpc_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
    *,
    rpc_client: type | None = None,
) -> GrpcMultiprocessServerBase:
    """Build a gRPC request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration. ``grpc_server_mode``
            selects the sync, aio or native server; ``auto`` resolves with
            ``resolve_grpc_server_mode``.
        rpc_client: RPC contract to expose; defaults to the worker RequestClient.

    Returns:
        Configured, but not yet started, gRPC request server.
    """
    server_cls: type[GrpcMultiprocessServerBase]
    mode = resolve_grpc_server_mode(mp_config.grpc_server_mode)
    if mode == "native":
        server_cls = GrpcNativeMultiprocessServer
    elif mode == "aio":
        server_cls = GrpcAioMultiprocessServer
    else:
        server_cls = GrpcMultiprocessServer
    server = server_cls(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
        grpc_server_workers=mp_config.grpc_server_workers,
        authentication=load_request_authentication(mp_config.api_key_files),
        rpc_client=rpc_client,
    )
    try:
        server.add_modules(modules)
    except BaseException:
        server.close(wait=True)
        raise
    return server
