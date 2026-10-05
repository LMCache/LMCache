# SPDX-License-Identifier: Apache-2.0
"""Request server construction behind a transport-neutral boundary."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.engine_module import EngineModule
    from lmcache.v1.multiprocess.transport.base import RequestServer
    from lmcache.v1.multiprocess.transport.grpc_impl.server import (
        GrpcMultiprocessServer,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer


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
    # First Party
    from lmcache.v1.multiprocess.modules.management import ManagementModule

    server: GrpcMultiprocessServer | MessageQueueServer
    if mp_config.transport == "grpc":
        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl.server import (
            build_grpc_request_server,
        )

        server = build_grpc_request_server(modules, mp_config)
    else:
        # First Party
        from lmcache.v1.multiprocess.transport.zmq_impl.server import (
            build_zmq_request_server,
        )

        server = build_zmq_request_server(modules, mp_config)
    for module in modules:
        if isinstance(module, ManagementModule):
            module.add_liveness_target(server)
    return server
