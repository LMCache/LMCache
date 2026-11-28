# SPDX-License-Identifier: Apache-2.0
"""Selects and builds the gRPC request server for a multiprocess config."""

# Standard
import importlib.util

# First Party
from lmcache.v1.common.logging import init_logger
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


def resolve_grpc_server_mode(mode: str, *, tls_enabled: bool = False) -> str:
    """Resolve ``auto`` to a gRPC server that supports the configured transport.

    Args:
        mode: The configured ``grpc_server_mode``.
        tls_enabled: Whether the request listener uses TLS.

    Returns:
        The selected mode. With TLS, ``auto`` selects ``aio``. Otherwise it
        prefers ``native`` when pygrpc-native is installed.

    Raises:
        ValueError: If TLS is configured with explicit ``native`` mode.
    """
    if tls_enabled and mode == "native":
        raise ValueError(
            "TLS is not supported by the native gRPC transport; "
            "use grpc_server_mode='auto', 'sync', or 'aio'"
        )
    if mode != "auto":
        return mode
    if tls_enabled:
        return "aio"
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
            selects the backend, and ``grpc_tls`` supplies TLS credentials.
        rpc_client: RPC contract to expose; defaults to the worker RequestClient.

    Returns:
        Configured, but not yet started, gRPC request server.

    Raises:
        ValueError: If the selected backend cannot use TLS, a configured
            credential file is unreadable or empty, or auth config is invalid.
    """
    tls_config = mp_config.grpc_tls
    server_cls: type[GrpcMultiprocessServerBase]
    mode = resolve_grpc_server_mode(
        mp_config.grpc_server_mode, tls_enabled=tls_config is not None
    )
    if mode == "native":
        server_cls = GrpcNativeMultiprocessServer
    elif mode == "aio":
        server_cls = GrpcAioMultiprocessServer
    else:
        server_cls = GrpcMultiprocessServer
    server_args = (
        f"grpc://{mp_config.host}:{mp_config.port}",
        mp_config.max_cpu_workers,
        mp_config.max_gpu_workers,
        mp_config.grpc_server_workers,
        load_request_authentication(mp_config.api_key_files),
        rpc_client,
    )
    server = (
        server_cls(*server_args)
        if tls_config is None
        else server_cls(*server_args, tls_config=tls_config)
    )
    try:
        server.add_modules(modules)
    except BaseException:
        server.close(wait=True)
        raise
    return server
