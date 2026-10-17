# SPDX-License-Identifier: Apache-2.0
"""gRPC request server on grpc.aio for multiprocess requests."""

# Standard
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import asyncio
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.ext_server_module import register_grpc_services
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    HandlerType,
)
from lmcache.v1.multiprocess.transport.base import RequestServer
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

_STREAM_END = object()


class _AioServicer:
    def __init__(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
        server: "GrpcAioMultiprocessServer",
    ) -> None:
        self._binding = binding
        self._handlers = handlers
        self._server = server

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        full_name = f"{self._binding.descriptor.full_name}.{method_name}"
        handler = self._handlers.get(full_name)
        if handler is None:
            raise AttributeError(method_name)
        server = self._server

        if self._binding.descriptor.methods_by_name[method_name].server_streaming:

            async def subscribe(
                request: Any, context: grpc.aio.ServicerContext
            ) -> AsyncIterator[Any]:
                try:
                    stream = await server.dispatch(handler, request, context)
                except ValueError as exc:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                async for response in server.forward(stream, handler, context):
                    yield response

            return subscribe

        async def invoke(request: Any, context: grpc.aio.ServicerContext) -> Any:
            return await server.dispatch(handler, request, context)

        return invoke


class GrpcAioMultiprocessServer(RequestServer):
    """Serve the same handlers as ``GrpcMultiprocessServer`` on grpc.aio.

    Calls are received on one event loop thread and handed to the handler
    pools, so no gRPC thread sits waiting while a handler runs.
    Handler types keep their meaning: ``SYNC`` handlers run one at a time,
    normal ``BLOCKING`` handlers run on up to ``max_cpu_workers`` threads, and
    affinity handlers stay on the worker chosen for their client.
    """

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_server_workers: int,
    ) -> None:
        self._bind_url = bind_url
        self._normal_pool = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="grpc-normal",
        )
        self._affinity_pool = AffinityThreadPool(
            max_workers=max_gpu_workers,
            thread_name_prefix="grpc-affinity",
        )
        self._sync_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="grpc-sync",
        )
        # Out-of-tree services may register plain functions, which grpc.aio
        # runs on this pool.
        self._executor = ThreadPoolExecutor(
            max_workers=grpc_server_workers,
            thread_name_prefix="grpc-server",
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="grpc-aio-loop",
        )
        self._loop_thread.start()
        try:
            self._server: grpc.aio.Server = self._run(self._create_server())
            self._bound_port = self._server.add_insecure_port(
                parse_grpc_target(bind_url)
            )
            if self._bound_port == 0:
                raise RuntimeError(
                    f"Failed to bind gRPC multiprocess server: {bind_url}"
                )
        except BaseException:
            # Nothing has been submitted yet, so waiting for the workers is quick.
            self._release_threads(wait=True)
            raise
        self._closed = threading.Event()

    @property
    def bound_port(self) -> int:
        """Return the TCP port selected by gRPC, including for port zero."""
        return self._bound_port

    async def dispatch(
        self,
        registered: GrpcRequestHandler,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        """Run one handler on the pool its handler type calls for.

        Args:
            registered: The handler bound to the called method.
            request: The decoded protobuf request.
            context: The call's servicer context.

        Returns:
            The encoded response, or the handler's ``MessagingStream`` for a
            streaming method.
        """
        handler = registered.handler
        if handler is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{registered.operation} is not enabled on this server",
            )
            raise RuntimeError("gRPC context abort unexpectedly returned")
        loop = asyncio.get_running_loop()
        try:
            payloads = registered.request_decoder(request)
            if registered.handler_type is HandlerType.SYNC:
                result = await loop.run_in_executor(self._sync_pool, handler, *payloads)
            elif registered.handler_type is not HandlerType.BLOCKING:
                raise NotImplementedError(
                    f"{registered.handler_type.name} handlers are not supported"
                )
            elif registered.requires_client_affinity:
                # Only the loop thread submits, so this needs no submit lock.
                future = self._affinity_pool.submit(
                    handler,
                    *payloads,
                    affinity_key=client_affinity_key(context),
                )
                result = await asyncio.wrap_future(future)
            else:
                result = await loop.run_in_executor(
                    self._normal_pool, handler, *payloads
                )
        except NotImplementedError as exc:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            raise RuntimeError("gRPC context abort unexpectedly returned") from exc
        if isinstance(result, MessagingStream):
            return result
        return registered.response_encoder(result)

    async def forward(
        self,
        stream: MessagingStream,
        registered: GrpcRequestHandler,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        """Yield encoded responses from a handler's stream until it ends.

        An idle stream must not hold a worker, so each one is read on its own
        thread, as in the sync server.

        Args:
            stream: The stream returned by the handler.
            registered: The handler bound to the called method.
            context: The call's servicer context.

        Yields:
            Encoded responses, in the order the stream produces them.
        """
        loop = asyncio.get_running_loop()
        items: asyncio.Queue[Any] = asyncio.Queue()

        def read() -> None:
            try:
                for response in stream:
                    loop.call_soon_threadsafe(items.put_nowait, response)
            except Exception as exc:
                loop.call_soon_threadsafe(items.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(items.put_nowait, _STREAM_END)

        threading.Thread(target=read, daemon=True, name="grpc-event-stream").start()
        try:
            while (item := await items.get()) is not _STREAM_END:
                if isinstance(item, Exception):
                    logger.error(
                        "Error in streaming handler %s",
                        registered.operation,
                        exc_info=item,
                    )
                    await context.abort(grpc.StatusCode.INTERNAL, str(item))
                yield registered.response_encoder(item)
        finally:
            stream.close()

    def add_modules(self, modules: Sequence[object]) -> None:
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

    def start(self) -> None:
        """Start accepting gRPC requests."""
        self._run(self._server.start())
        logger.info("LMCache gRPC cache server (aio) is running on %s", self._bind_url)

    def close(self) -> None:
        """Stop the gRPC server, its event loop and its request executors."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._run(self._server.stop(grace=None))
        self._release_threads(wait=False)

    def _release_threads(self, wait: bool) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()
        self._normal_pool.shutdown(wait=wait)
        self._sync_pool.shutdown(wait=wait)
        self._affinity_pool.shutdown(wait=wait)
        self._executor.shutdown(wait=wait)

    async def _create_server(self) -> grpc.aio.Server:
        return grpc.aio.server(
            migration_thread_pool=self._executor,
            options=GRPC_SERVER_OPTIONS,
        )

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _add_generated_service(
        self,
        binding: ServiceBinding,
        handlers_by_operation: dict[str, BoundRequestHandler],
    ) -> None:
        service_handlers = build_service_handlers(binding, handlers_by_operation)
        add_servicer = getattr(
            binding.grpc_module,
            f"add_{binding.descriptor.name}Servicer_to_server",
        )
        add_servicer(_AioServicer(binding, service_handlers, self), self._server)
