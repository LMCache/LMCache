# SPDX-License-Identifier: Apache-2.0
"""Request server construction behind a transport-neutral boundary."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Callable

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.engine_module import EngineModule
    from lmcache.v1.multiprocess.ext_server_module import TransportServiceRegistrar
    from lmcache.v1.multiprocess.transport.base import RequestServer

logger = init_logger(__name__)


def create_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
    *,
    grpc_service_registrars: tuple[TransportServiceRegistrar, ...] = (),
    zmq_service_registrars: tuple[TransportServiceRegistrar, ...] = (),
    on_peer_disconnected: Callable[[bytes], None] | None = None,
) -> RequestServer:
    """Create a configured request server for the selected transport.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration selecting ZMQ or gRPC.
        grpc_service_registrars: Out-of-tree gRPC service registrars.
        zmq_service_registrars: Out-of-tree ZMQ service registrars.
        on_peer_disconnected: Optional callback receiving the connection id
            (see ``current_request_peer``) of each client connection that
            closes. Only the ZMQ transport reports connection loss.

    Returns:
        Configured, but not yet started, request server.
    """
    if mp_config.transport == "grpc":
        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl.server import (
            build_grpc_request_server,
        )

        if on_peer_disconnected is not None:
            logger.info(
                "The grpc transport does not report closed client "
                "connections; dead workers are reclaimed by the heartbeat "
                "timeout only"
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
        on_peer_disconnected=on_peer_disconnected,
    )
