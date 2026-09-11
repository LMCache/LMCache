# SPDX-License-Identifier: Apache-2.0
"""ZMQ request handlers and server construction for multiprocess requests."""

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.request_handler import iter_request_handlers


class ThreadPoolType(Enum):
    """Select the ZMQ worker pool for a request handler."""

    SYNC = auto()
    AFFINITY = auto()
    NORMAL = auto()


@dataclass(frozen=True)
class HandlerSpec:
    """Describe one ZMQ request handler and its worker pool.

    Args:
        request_type: ZMQ request type served by the handler.
        handler: Callable that processes the decoded request payloads.
        handler_type: Whether to execute inline or on a worker.
        requires_client_affinity: Whether to use the client-affinity pool.
    """

    request_type: RequestType
    handler: Callable[..., Any]
    handler_type: HandlerType
    requires_client_affinity: bool

    @property
    def pool(self) -> ThreadPoolType:
        """Return the ZMQ worker pool implied by common handler metadata."""
        if self.handler_type is HandlerType.SYNC:
            return ThreadPoolType.SYNC
        if self.requires_client_affinity:
            return ThreadPoolType.AFFINITY
        return ThreadPoolType.NORMAL


def add_handler_helper(
    server: MessageQueueServer,
    request_type: RequestType,
    handler_function: Callable[..., Any],
    handler_type: HandlerType,
) -> None:
    """Register one annotated request handler with a ZMQ server.

    Args:
        server: ZMQ message queue server.
        request_type: Request type to register.
        handler_function: Callable that handles the decoded payloads.
        handler_type: Execution type from the common handler annotation.

    Returns:
        None.
    """
    server.add_handler(request_type, handler_type, handler_function)


def get_zmq_handler_specs(module: object) -> list[HandlerSpec]:
    """Build the ZMQ handler table for one transport-neutral engine module.

    Args:
        module: Business module whose public methods should serve ZMQ requests.

    Returns:
        Ordered ZMQ handler specifications for the module.

    """
    return [
        HandlerSpec(
            request_type=registered.options.request_type,
            handler=registered.handler,
            handler_type=registered.options.handler_type,
            requires_client_affinity=registered.options.requires_client_affinity,
        )
        for registered in iter_request_handlers(module)
    ]


def build_zmq_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> MessageQueueServer:
    """Build a ZMQ request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.

    Returns:
        Configured, but not yet started, ZMQ message queue server.
    """
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=zmq.Context.instance(),
    )
    all_specs = [spec for module in modules for spec in get_zmq_handler_specs(module)]
    for spec in all_specs:
        add_handler_helper(
            server,
            spec.request_type,
            spec.handler,
            spec.handler_type,
        )

    affinity_types = [
        spec.request_type for spec in all_specs if spec.pool is ThreadPoolType.AFFINITY
    ]
    normal_types = [
        spec.request_type for spec in all_specs if spec.pool is ThreadPoolType.NORMAL
    ]
    if affinity_types:
        server.add_affinity_thread_pool(
            affinity_types, max_workers=mp_config.max_gpu_workers
        )
    if normal_types:
        server.add_normal_thread_pool(
            normal_types, max_workers=mp_config.max_cpu_workers
        )
    return server


__all__ = [
    "HandlerSpec",
    "ThreadPoolType",
    "add_handler_helper",
    "build_zmq_request_server",
    "get_zmq_handler_specs",
]
