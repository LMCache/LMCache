# SPDX-License-Identifier: Apache-2.0
"""ZMQ handler table for the layer-wise operations."""

# First Party
from lmcache.v1.multiprocess.request_handler_layerwise import (
    iter_layerwise_request_handlers,
)
from lmcache.v1.multiprocess.transport.zmq_impl.server import HandlerSpec


def get_layerwise_zmq_handler_specs(module: object) -> list[HandlerSpec]:
    """Build the ZMQ handler table for a module's layer-wise handlers.

    The counterpart of
    :func:`~lmcache.v1.multiprocess.transport.zmq_impl.server.get_zmq_handler_specs`
    for handlers declared with ``@layerwise_request_handler``.

    Args:
        module: Business module that may serve layer-wise requests.

    Returns:
        Ordered ZMQ handler specifications; empty if the module has none.
    """
    return [
        HandlerSpec(
            operation=registered.operation,
            handler=registered.handler,
            handler_type=registered.options.handler_type,
            requires_client_affinity=registered.options.requires_client_affinity,
        )
        for registered in iter_layerwise_request_handlers(module)
    ]
