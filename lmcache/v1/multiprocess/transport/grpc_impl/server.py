# SPDX-License-Identifier: Apache-2.0
"""gRPC request server construction for multiprocess requests."""

# Standard
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.ext_server_module import register_grpc_services
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    HandlerType,
)
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.grpc_impl.aio_server import (
    GrpcAioMultiprocessServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.client import parse_grpc_target
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    ServiceBinding,
    get_service_bindings,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_handlers import (
    GRPC_SERVER_OPTIONS,
    GrpcRequestHandler,
    bind_module_handlers,
    build_service_handlers,
    client_affinity_key,
)

logger = init_logger(__name__)


class _GeneratedServicer:
    def __init__(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
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

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        full_name = f"{self._binding.descriptor.full_name}.{method_name}"
        handler = self._handlers.get(full_name)
        if handler is None:
            raise AttributeError(method_name)

        if self._binding.descriptor.methods_by_name[method_name].server_streaming:

            def subscribe(
                request: Any, context: grpc.ServicerContext, send: Callable
            ) -> None:
                try:
                    stream = self._dispatch(handler, request, context)
                except ValueError as exc:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                    return
                if not context.add_callback(stream.close):
                    stream.close()
                    return

                def forward() -> None:
                    try:
                        for response in stream:
                            send(handler.response_encoder(response))
                    except Exception as exc:
                        if context.is_active():
                            logger.exception(
                                "Error in streaming handler %s", handler.operation
                            )
                            context.set_code(grpc.StatusCode.INTERNAL)
                            context.set_details(str(exc))
                    finally:
                        stream.close()
                        send(None)

                # The same nonblocking hook used by grpc.health.v1.Health.Watch:
                # an idle stream must not occupy the unary request executor.
                threading.Thread(
                    target=forward, daemon=True, name="grpc-event-stream"
                ).start()

            subscribe.experimental_non_blocking = True  # type: ignore[attr-defined]
            return subscribe

        def invoke(request: Any, context: grpc.ServicerContext) -> Any:
            return self._dispatch(handler, request, context)

        return invoke

    def _dispatch(
        self,
        registered: GrpcRequestHandler,
        request: Any,
        context: grpc.ServicerContext,
    ) -> Any:
        try:
            if registered.handler is None:
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    f"{registered.operation} is not enabled on this server",
                )
                raise RuntimeError("gRPC context abort unexpectedly returned")
            payloads = registered.request_decoder(request)
            if registered.handler_type is HandlerType.SYNC:
                with self._sync_handler_lock:
                    result = registered.handler(*payloads)
            elif registered.handler_type is HandlerType.BLOCKING and (
                registered.requires_client_affinity
            ):
                affinity_key = client_affinity_key(context)
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
            if isinstance(result, MessagingStream):
                return result
            return registered.response_encoder(result)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            raise RuntimeError("gRPC context abort unexpectedly returned") from exc


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
        self._handlers: dict[str, GrpcRequestHandler] = {}
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
        self._server = grpc.server(self._executor, options=GRPC_SERVER_OPTIONS)
        self._bound_port = self._server.add_insecure_port(parse_grpc_target(bind_url))
        if self._bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC multiprocess server: {bind_url}")
        self._closed = threading.Event()

    @property
    def bound_port(self) -> int:
        """Return the TCP port selected by gRPC, including for port zero."""
        return self._bound_port

    def add_modules(
        self,
        modules: Sequence[object],
    ) -> None:
        """Register decorated module methods as generated gRPC services.

        Args:
            modules: Ordered business modules. A later module overrides an
                earlier handler for the same request type.

        Raises:
            TypeError: If a module handler does not match its protocol types.
            ValueError: If a module exposes invalid handler metadata.
        """
        handlers_by_operation = bind_module_handlers(modules)
        for binding in get_service_bindings().values():
            self._add_generated_service(binding, handlers_by_operation)
        register_grpc_services(modules, self._server)

    def _add_generated_service(
        self,
        binding: ServiceBinding,
        handlers_by_operation: dict[str, BoundRequestHandler],
    ) -> None:
        service_name = binding.descriptor.name

        service_handlers = build_service_handlers(binding, handlers_by_operation)
        self._handlers.update(service_handlers)

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
) -> GrpcMultiprocessServer | GrpcAioMultiprocessServer:
    """Build a gRPC request server for the supplied business modules.

    Args:
        modules: Ordered business modules composing the cache server.
        mp_config: Multiprocess server configuration.

    Returns:
        Configured, but not yet started, gRPC request server.
    """
    server_cls: type[GrpcMultiprocessServer] | type[GrpcAioMultiprocessServer]
    if mp_config.grpc_server_mode == "aio":
        server_cls = GrpcAioMultiprocessServer
    else:
        server_cls = GrpcMultiprocessServer
    server = server_cls(
        bind_url=f"grpc://{mp_config.host}:{mp_config.port}",
        max_gpu_workers=mp_config.max_gpu_workers,
        max_cpu_workers=mp_config.max_cpu_workers,
        grpc_server_workers=mp_config.grpc_server_workers,
    )
    server.add_modules(modules)
    return server
