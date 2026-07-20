# SPDX-License-Identifier: Apache-2.0
"""Transport layer for LMCache mp-mode message queue.

Importing this package registers built-in transport plug-ins (currently
ZMQ over ``ipc://`` and ``tcp://``).  Third-party plug-ins can register
themselves via :func:`~.registry.register_client` /
:func:`~.registry.register_server` decorators.
"""

# First Party
# Import built-in transports for their side-effect: registering
# themselves with the URL-scheme registry.
from lmcache.v1.multiprocess.transport import zmq_transport  # noqa: F401
from lmcache.v1.multiprocess.transport.base import (
    ClientContext,
    ClientTransport,
    PollHandle,
    ServerTransport,
)
from lmcache.v1.multiprocess.transport.registry import (
    available_client_schemes,
    available_server_schemes,
    create_client_transport,
    create_server_transport,
    register_client,
    register_server,
)

__all__ = [
    "ClientContext",
    "ClientTransport",
    "PollHandle",
    "ServerTransport",
    "available_client_schemes",
    "available_server_schemes",
    "create_client_transport",
    "create_server_transport",
    "register_client",
    "register_server",
]
