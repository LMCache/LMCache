# SPDX-License-Identifier: Apache-2.0
"""ZMQ request handlers and server construction for multiprocess requests."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.request_handler import HandlerType, iter_request_handlers
from lmcache.v1.multiprocess.rpc import RpcOperation
from lmcache.v1.multiprocess.server_module import (
    TransportServiceRegistrar,
    register_zmq_services,
)
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer


class ThreadPoolType(Enum):
    """Select the ZMQ worker pool for a request handler."""

    SYNC = auto()
    AFFINITY = auto()
    NORMAL = auto()


@dataclass(frozen=True)
class HandlerSpec:
    """Describe one ZMQ request handler and its worker pool.

    Args:
        operation: RPC operation served by the handler.
        handler: Callable that processes the decoded request payloads.
        handler_type: Whether to execute inline or on a worker.
        requires_client_affinity: Whether to use the client-affinity pool.
    """

    operation: RpcOperation
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
    operation: RpcOperation,
    handler_function: Callable[..., Any],
    handler_type: HandlerType = HandlerType.SYNC,
) -> None:
    """Register one RPC request handler with a ZMQ server.

    Args:
        server: ZMQ message queue server.
        operation: Stable snake-case RPC name.
        handler_function: Callable that handles the decoded payloads.
        handler_type: Execution type from the common handler annotation.

    Returns:
        None.
    """
    server.add_handler(
        operation,
        handler_type,
        handler_function,
    )


def get_zmq_handler_specs(module: object) -> list[HandlerSpec]:
    """Build the ZMQ handler table for one transport-neutral engine module.

    Args:
        module: Business module whose public methods should serve ZMQ requests.

    Returns:
        Ordered ZMQ handler specifications for the module.

    """
    return [
        HandlerSpec(
            operation=registered.operation,
            handler=registered.handler,
            handler_type=registered.options.handler_type,
            requires_client_affinity=registered.options.requires_client_affinity,
        )
        for registered in iter_request_handlers(module)
    ]


def build_zmq_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
    *,
    service_registrars: Sequence[TransportServiceRegistrar] = (),
) -> MessageQueueServer:
    """Build a ZMQ request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.
        service_registrars: Out-of-tree ZMQ service registrars returned by
            server-module factories.

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
            spec.operation,
            spec.handler,
            spec.handler_type,
        )

    affinity_types = [
        spec.operation for spec in all_specs if spec.pool is ThreadPoolType.AFFINITY
    ]
    normal_types = [
        spec.operation for spec in all_specs if spec.pool is ThreadPoolType.NORMAL
    ]
    if affinity_types:
        server.add_affinity_thread_pool(
            affinity_types, max_workers=mp_config.max_gpu_workers
        )
    if normal_types:
        server.add_normal_thread_pool(
            normal_types, max_workers=mp_config.max_cpu_workers
        )
    register_zmq_services(modules, server, service_registrars)
    return server
