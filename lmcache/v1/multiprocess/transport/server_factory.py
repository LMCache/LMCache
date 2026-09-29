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
    from lmcache.v1.multiprocess.server_module import TransportServiceRegistrar
    from lmcache.v1.multiprocess.transport.base import RequestServer


def create_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
    *,
    grpc_service_registrars: tuple[TransportServiceRegistrar, ...] = (),
    zmq_service_registrars: tuple[TransportServiceRegistrar, ...] = (),
) -> RequestServer:
    """Create a configured request server for the selected transport.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration selecting ZMQ or gRPC.
        grpc_service_registrars: Out-of-tree gRPC service registrars.
        zmq_service_registrars: Out-of-tree ZMQ service registrars.

    Returns:
        Configured, but not yet started, request server.
    """
    if mp_config.transport == "grpc":
        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl.server import (
            build_grpc_request_server,
        )

        return build_grpc_request_server(
            modules,
            mp_config,
            service_registrars=grpc_service_registrars,
        )

    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.server import (
        build_zmq_request_server,
    )

    return build_zmq_request_server(
        modules,
        mp_config,
        service_registrars=zmq_service_registrars,
    )
