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
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.experimental.qstore import QStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)


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
        pool: ZMQ worker pool used to execute the handler.
    """

    request_type: RequestType
    handler: Callable[..., Any]
    pool: ThreadPoolType


def add_handler_helper(
    server: MessageQueueServer,
    request_type: RequestType,
    handler_function: Callable[..., Any],
) -> None:
    """Register one legacy request handler with a ZMQ server.

    Args:
        server: ZMQ message queue server.
        request_type: Legacy request type to register.
        handler_function: Callable that handles the decoded payloads.

    Returns:
        None.
    """
    server.add_handler(
        request_type,
        get_payload_classes(request_type),
        get_handler_type(request_type),
        handler_function,
    )


def get_zmq_handler_specs(module: EngineModule) -> list[HandlerSpec]:
    """Build the ZMQ handler table for one transport-neutral engine module.

    Args:
        module: Business module whose public methods should serve ZMQ requests.

    Returns:
        Ordered ZMQ handler specifications for the module.

    Raises:
        TypeError: If the module has no ZMQ adapter.
    """
    if isinstance(module, LookupModule):
        return [
            HandlerSpec(RequestType.LOOKUP, module.lookup, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_STATUS,
                module.query_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.WAIT_PREFETCH_STATUS,
                module.wait_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                module.query_prefetch_lookup_hits,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.FREE_LOOKUP_LOCKS,
                module.free_lookup_locks,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.END_SESSION,
                module.end_session,
                ThreadPoolType.NORMAL,
            ),
        ]
    if isinstance(module, P2PController):
        return [
            HandlerSpec(
                RequestType.P2P_LOOKUP_AND_LOCK,
                module.p2p_lookup_and_lock,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.P2P_QUERY_LOOKUP_RESULTS,
                module.p2p_query_lookup_results,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.P2P_UNLOCK_OBJECTS,
                module.p2p_unlock_objects,
                ThreadPoolType.NORMAL,
            ),
        ]
    if isinstance(module, ManagementModule):
        return [
            HandlerSpec(RequestType.CLEAR, module.clear, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.GET_CHUNK_SIZE,
                module.get_chunk_size,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.GET_EXPERIMENTAL,
                module.get_experimental,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.PING, module.ping, ThreadPoolType.NORMAL),
            HandlerSpec(RequestType.NOOP, module.debug, ThreadPoolType.SYNC),
            HandlerSpec(
                RequestType.REPORT_BLOCK_ALLOCATION,
                module.report_block_allocations,
                ThreadPoolType.NORMAL,
            ),
        ]
    if isinstance(module, LMCacheDrivenTransferModule):
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE,
                module.register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
                module.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.STORE, module.store, ThreadPoolType.AFFINITY),
            HandlerSpec(
                RequestType.RETRIEVE,
                module.retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]
    if isinstance(module, EngineDrivenTransferModule):
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                module.register_kv_cache_engine_driven_context,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                module.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.PREPARE_STORE,
                module.prepare_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_STORE,
                module.commit_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.PREPARE_RETRIEVE,
                module.prepare_retrieve,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_RETRIEVE,
                module.commit_retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]
    if isinstance(module, QStoreModule):
        return [
            HandlerSpec(
                RequestType.REGISTER_Q_CACHE,
                module.register_q_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_Q_CACHE,
                module.unregister_q_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.STORE_Q, module.store_q, ThreadPoolType.AFFINITY),
        ]

    # Blend is optional and expensive to import, so resolve its adapter only
    # after all always-loaded module types have been ruled out.
    # First Party
    from lmcache.v1.multiprocess.modules.blend import BlendModule

    if isinstance(module, BlendModule):
        return [
            # STORE intentionally shadows LMCacheDrivenTransferModule.store;
            # module order keeps the Blend handler last.
            HandlerSpec(RequestType.STORE, module.store, ThreadPoolType.AFFINITY),
            HandlerSpec(
                RequestType.CB_REGISTER_ROPE,
                module.cb_register_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNREGISTER_ROPE,
                module.cb_unregister_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNIFIED_LOOKUP,
                module.cb_unified_lookup,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.CB_RETRIEVE_PRE_COMPUTED,
                module.cb_retrieve_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.CB_PROTOCOL_HANDSHAKE,
                module.cb_protocol_handshake,
                ThreadPoolType.SYNC,
            ),
        ]
    raise TypeError(f"No ZMQ handler adapter for {type(module).__name__}")


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
        add_handler_helper(server, spec.request_type, spec.handler)

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
