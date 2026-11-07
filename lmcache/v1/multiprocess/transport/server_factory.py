# SPDX-License-Identifier: Apache-2.0
"""Request server construction behind a transport-neutral boundary."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.engine_module import EngineModule
    from lmcache.v1.multiprocess.transport.base import RequestServer


def create_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> RequestServer:
    """Create a configured request server for the selected transport.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration selecting ZMQ or gRPC.

    Returns:
        Configured, but not yet started, request server.
    """
    if mp_config.transport == "grpc":
        # First Party
        from lmcache.v1.multiprocess.transport.auth import (
            load_request_authentication,
        )

        if load_request_authentication(mp_config.api_key_files).enabled:
            logger.warning(
                "API keys protect only the HTTP API with gRPC transport; "
                "gRPC requests remain unauthenticated. Use --transport zmq "
                "to authenticate requests."
            )

        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl.server_builder import (
            build_grpc_request_server,
        )

        return build_grpc_request_server(modules, mp_config)

    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.server import (
        build_zmq_request_server,
    )

    return build_zmq_request_server(modules, mp_config)
