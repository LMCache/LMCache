# SPDX-License-Identifier: Apache-2.0
"""gRPC request server construction for multiprocess requests."""

# Standard
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    iter_request_handlers,
)
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.grpc_impl.client import parse_grpc_target
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    ServiceBinding,
    get_service_bindings,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
)

logger = init_logger(__name__)

_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)
_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


@dataclass
class _GrpcRequestHandler:
    request_type: RequestType
    handler: Callable[..., Any] | None
    handler_type: HandlerType
    requires_client_affinity: bool
    request_decoder: RequestDecoder
    response_encoder: ResponseEncoder


class _GeneratedServicer:
    def __init__(
        self,
        binding: ServiceBinding,
        handlers: dict[str, _GrpcRequestHandler],
        normal_pool: ThreadPoolExecutor,
        affinity_pool: AffinityThreadPool,
        affinity_submit_lock: threading.Lock,
        sync_handler_lock: threading.Lock,
    ) -> None:
        self._binding = binding
        self._handlers = handlers
        self._normal_pool = normal_pool
        self._affinity_pool = affinity_pool
        self._affinity_submit_lock = affinity_submit_lock
        self._sync_handler_lock = sync_handler_lock

    def __getattr__(self, method_name: str) -> Callable[[Any, Any], Any]:
        full_name = f"{self._binding.descriptor.full_name}.{method_name}"
        handler = self._handlers.get(full_name)
        if handler is None:
            raise AttributeError(method_name)

        def invoke(request: Any, context: grpc.ServicerContext) -> Any:
            return self._dispatch(handler, request, context)

        return invoke

    def _dispatch(
        self,
        registered: _GrpcRequestHandler,
        request: Any,
        context: grpc.ServicerContext,
    ) -> Any:
        try:
            if registered.handler is None:
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    f"{registered.request_type.name} is not enabled on this server",
                )
                raise RuntimeError("gRPC context abort unexpectedly returned")
            payloads = registered.request_decoder(request)
            if registered.handler_type is HandlerType.SYNC:
                with self._sync_handler_lock:
                    result = registered.handler(*payloads)
            elif registered.handler_type is HandlerType.BLOCKING and (
                registered.requires_client_affinity
            ):
                affinity_key = self._affinity_key(context)
                with self._affinity_submit_lock:
                    future = self._affinity_pool.submit(
                        registered.handler,
                        *payloads,
                        affinity_key=affinity_key,
                    )
                result = future.result()
            elif registered.handler_type is HandlerType.BLOCKING:
                result = self._normal_pool.submit(
                    registered.handler, *payloads
                ).result()
            else:
                raise NotImplementedError(
                    f"{registered.handler_type.name} handlers are not supported"
                )
            return registered.response_encoder(result)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            raise RuntimeError("gRPC context abort unexpectedly returned") from exc

    @staticmethod
    def _affinity_key(context: grpc.ServicerContext) -> int:
        for key, value in context.invocation_metadata():
            if key == _CLIENT_ID_METADATA_KEY:
                return hash(value)
        return hash(context.peer())


class GrpcMultiprocessServer(RequestServer):
    """Register transport-neutral modules against generated gRPC services."""

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_server_workers: int,
    ) -> None:
        self._bind_url = bind_url
        self._handlers: dict[str, _GrpcRequestHandler] = {}
        self._normal_pool = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="grpc-normal",
        )
        self._affinity_pool = AffinityThreadPool(
            max_workers=max_gpu_workers,
            thread_name_prefix="grpc-affinity",
        )
        self._affinity_submit_lock = threading.Lock()
        # HandlerType.SYNC is a transport-neutral single-main-loop contract.
        self._sync_handler_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=grpc_server_workers,
            thread_name_prefix="grpc-server",
        )
        self._server = grpc.server(self._executor, options=_GRPC_OPTIONS)
        self._bound_port = self._server.add_insecure_port(parse_grpc_target(bind_url))
        if self._bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC multiprocess server: {bind_url}")
        self._closed = threading.Event()

    @property
    def bound_port(self) -> int:
        """Return the TCP port selected by gRPC, including for port zero."""
        return self._bound_port

    def add_modules(self, modules: Sequence[object]) -> None:
        """Register decorated module methods as generated gRPC services.

        Args:
            modules: Ordered business modules. A later module overrides an
                earlier handler for the same request type.

        Raises:
            TypeError: If a module handler does not match its protocol types.
            ValueError: If a module exposes invalid handler metadata.
        """
        handlers_by_request: dict[RequestType, BoundRequestHandler] = {}
        for module in modules:
            for registered in iter_request_handlers(module):
                handlers_by_request[registered.options.request_type] = registered

        for binding in get_service_bindings().values():
            self._add_generated_service(binding, handlers_by_request)

    def _add_generated_service(
        self,
        binding: ServiceBinding,
        handlers_by_request: dict[RequestType, BoundRequestHandler],
    ) -> None:
        service_name = binding.descriptor.name

        service_handlers: dict[str, _GrpcRequestHandler] = {}
        codec_registry = get_method_codec_registry()
        for method in binding.descriptor.methods:
            method_codec = codec_registry.by_full_name[method.full_name]
            bound_handler = handlers_by_request.get(method_codec.request_type)
            if bound_handler is not None:
                method_codec.validate_handler(bound_handler.handler)
            full_name = method.full_name
            registered = _GrpcRequestHandler(
                request_type=method_codec.request_type,
                handler=(bound_handler.handler if bound_handler is not None else None),
                handler_type=(
                    bound_handler.options.handler_type
                    if bound_handler is not None
                    else HandlerType.SYNC
                ),
                requires_client_affinity=(
                    bound_handler.options.requires_client_affinity
                    if bound_handler is not None
                    else False
                ),
                request_decoder=method_codec.request_decoder,
                response_encoder=method_codec.response_encoder,
            )
            self._handlers[full_name] = registered
            service_handlers[full_name] = registered

        servicer = _GeneratedServicer(
            binding,
            service_handlers,
            self._normal_pool,
            self._affinity_pool,
            self._affinity_submit_lock,
            self._sync_handler_lock,
        )
        add_servicer = getattr(
            binding.grpc_module,
            f"add_{service_name}Servicer_to_server",
        )
        add_servicer(servicer, self._server)

    def start(self) -> None:
        """Start accepting gRPC requests."""
        self._server.start()
        logger.info("LMCache gRPC cache server is running on %s", self._bind_url)

    def close(self) -> None:
        """Stop the gRPC server and its request executors."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._server.stop(grace=None)
        self._normal_pool.shutdown(wait=False)
        self._affinity_pool.shutdown(wait=False)
        self._executor.shutdown(wait=False)


def build_grpc_request_server(
    modules: list[EngineModule],
    mp_config: MPServerConfig,
) -> GrpcMultiprocessServer:
    """Build a gRPC request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.

    Returns:
        Configured, but not yet started, gRPC request server.
    """
    server = GrpcMultiprocessServer(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
        grpc_server_workers=mp_config.grpc_server_workers,
    )
    server.add_modules(modules)
    return server
