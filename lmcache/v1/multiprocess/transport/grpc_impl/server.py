# SPDX-License-Identifier: Apache-2.0
"""Multiprocess server driven by generated gRPC service descriptors."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.transport.grpc_impl.client import parse_grpc_target
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    ServiceBinding,
    get_service_bindings,
    message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
    compile_request_decoder,
    compile_response_encoder,
)
from lmcache.v1.multiprocess.transport.grpc_impl.services.base import (
    GrpcHandlerType,
    get_grpc_method_options,
)

logger = init_logger(__name__)

_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)
_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


@dataclass
class _GrpcRequestHandler:
    handler: Callable[..., Any]
    handler_type: GrpcHandlerType
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
    ) -> None:
        self._binding = binding
        self._handlers = handlers
        self._normal_pool = normal_pool
        self._affinity_pool = affinity_pool
        self._affinity_submit_lock = affinity_submit_lock

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
        payloads = registered.request_decoder(request)
        try:
            if registered.handler_type is GrpcHandlerType.SYNC:
                result = registered.handler(*payloads)
            elif registered.requires_client_affinity:
                affinity_key = self._affinity_key(context)
                with self._affinity_submit_lock:
                    future = self._affinity_pool.submit(
                        registered.handler,
                        *payloads,
                        affinity_key=affinity_key,
                    )
                result = future.result()
            else:
                result = self._normal_pool.submit(
                    registered.handler, *payloads
                ).result()
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


class GrpcMultiprocessServer:
    """Register concrete implementations against generated gRPC services."""

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_workers: int = 32,
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
        self._executor = ThreadPoolExecutor(
            max_workers=grpc_workers,
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

    def add_service(self, service_name: str, implementation: object) -> None:
        """Register methods from a concrete protobuf service implementation.

        Args:
            service_name: Name declared by the generated protobuf service.
            implementation: Object with one same-named method per proto RPC.

        Raises:
            ValueError: If the generated service does not exist.
            TypeError: If an RPC implementation is missing.
        """
        binding = get_service_bindings().get(service_name)
        if binding is None:
            raise ValueError(f"Unknown generated gRPC service: {service_name}")

        service_handlers: dict[str, _GrpcRequestHandler] = {}
        for method in binding.descriptor.methods:
            handler = getattr(implementation, method.name, None)
            if not callable(handler):
                raise TypeError(
                    f"{implementation.__class__.__name__} must implement "
                    f"{service_name}.{method.name}"
                )
            request_decoder, _ = compile_request_decoder(
                message_class(method.input_type), handler
            )
            response_encoder, _ = compile_response_encoder(
                message_class(method.output_type), handler
            )
            handler_type, requires_affinity = get_grpc_method_options(handler)
            full_name = method.full_name
            registered = _GrpcRequestHandler(
                handler=handler,
                handler_type=handler_type,
                requires_client_affinity=requires_affinity,
                request_decoder=request_decoder,
                response_encoder=response_encoder,
            )
            self._handlers[full_name] = registered
            service_handlers[full_name] = registered

        servicer = _GeneratedServicer(
            binding,
            service_handlers,
            self._normal_pool,
            self._affinity_pool,
            self._affinity_submit_lock,
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
