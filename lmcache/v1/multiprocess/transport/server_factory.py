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
    from lmcache.v1.multiprocess.mq import MessageQueueServer


def create_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> MessageQueueServer:
    """Create the ZMQ request server used by the multiprocess runtime.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.

    Returns:
        Configured, but not yet started, ZMQ message queue server.
    """
    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.server import (
        build_zmq_request_server,
    )

    return build_zmq_request_server(modules, mp_config)


__all__ = ["create_request_server"]
