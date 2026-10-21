# SPDX-License-Identifier: Apache-2.0
"""gRPC request server construction for multiprocess requests."""

# Standard
from typing import Any, Callable
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import ServiceBinding
from lmcache.v1.multiprocess.transport.grpc_impl.server_base import (
    GrpcMultiprocessServerBase,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_handlers import (
    GRPC_SERVER_OPTIONS,
    GrpcRequestHandler,
    client_affinity_key,
)

logger = init_logger(__name__)


class _GeneratedServicer:
    def __init__(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
        normal_slots: threading.BoundedSemaphore,
        affinity_pool: AffinityThreadPool,
        affinity_submit_lock: threading.Lock,
        serialized_handler_lock: threading.Lock,
    ) -> None:
        self._binding = binding
        self._handlers = handlers
        self._normal_slots = normal_slots
        self._affinity_pool = affinity_pool
        self._affinity_submit_lock = affinity_submit_lock
        self._serialized_handler_lock = serialized_handler_lock

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
            if registered.handler_type is HandlerType.SERIALIZED:
                with self._serialized_handler_lock:
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
                # gRPC already runs this call on a worker. Bound handler
                # concurrency without a second executor submission and wait.
                with self._normal_slots:
                    result = registered.handler(*payloads)
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


class GrpcMultiprocessServer(GrpcMultiprocessServerBase):
    """Serve the gRPC request services on grpc's thread-pool server.

    Each call runs on one of ``grpc_server_workers`` threads. Normal handlers
    run on that thread, at most ``max_cpu_workers`` at a time, and affinity
    handlers are handed to the worker chosen for their client.
    """

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_server_workers: int,
    ) -> None:
        super().__init__(
            bind_url, max_cpu_workers, max_gpu_workers, grpc_server_workers
        )
        self._normal_slots = threading.BoundedSemaphore(max_cpu_workers)
        self._affinity_submit_lock = threading.Lock()
        # HandlerType.SERIALIZED is a transport-neutral single-main-loop contract.
        self._serialized_handler_lock = threading.Lock()
        self._server = grpc.server(self._executor, options=GRPC_SERVER_OPTIONS)
        self._bind(self._server)

    def start(self) -> None:
        """Start accepting gRPC requests."""
        self._server.start()
        logger.info("LMCache gRPC cache server is running on %s", self._bind_url)

    def _grpc_server(self) -> grpc.Server:
        return self._server

    def _make_servicer(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
    ) -> object:
        return _GeneratedServicer(
            binding,
            handlers,
            self._normal_slots,
            self._affinity_pool,
            self._affinity_submit_lock,
            self._serialized_handler_lock,
        )

    def _stop_server(self) -> None:
        self._server.stop(grace=None)
