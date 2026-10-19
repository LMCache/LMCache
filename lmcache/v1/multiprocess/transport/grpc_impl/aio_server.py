# SPDX-License-Identifier: Apache-2.0
"""gRPC request server on grpc.aio for multiprocess requests."""

# Standard
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import asyncio
import threading

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
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

_STREAM_END = object()
_STREAM_READ_AHEAD = 1


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


class GrpcAioMultiprocessServer(GrpcMultiprocessServerBase):
    """Serve the gRPC request services on grpc.aio.

    Calls are received on one event loop thread and handed to the handler
    pools, so no gRPC thread sits waiting while a handler runs.
    Handler types keep their meaning: ``SERIALIZED`` handlers run one at a time,
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
        super().__init__(
            bind_url, max_cpu_workers, max_gpu_workers, grpc_server_workers
        )
        # The loop must not block, so normal handlers need their own threads
        # here instead of the sync server's semaphore.
        self._normal_pool = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="grpc-normal",
        )
        self._serialized_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="grpc-serialized",
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
            self._bind(self._server)
        except BaseException:
            # Nothing has been submitted yet, so waiting for the workers is quick.
            self._release_threads(wait=True)
            raise

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
            if registered.handler_type is HandlerType.SERIALIZED:
                result = await loop.run_in_executor(
                    self._serialized_pool, handler, *payloads
                )
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
        # Don't read further ahead than the client drains.
        credits = threading.Semaphore(_STREAM_READ_AHEAD)
        done = threading.Event()

        def put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(items.put_nowait, item)
            except RuntimeError:
                pass  # loop closed on shutdown

        def read() -> None:
            try:
                while credits.acquire() and not done.is_set():
                    try:
                        response = next(stream)
                    except StopIteration:
                        break
                    put(response)
            except Exception as exc:
                put(exc)
            finally:
                put(_STREAM_END)

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
                credits.release()
        finally:
            done.set()
            credits.release()
            stream.close()

    def start(self) -> None:
        """Start accepting gRPC requests."""
        self._run(self._server.start())
        logger.info("LMCache gRPC cache server (aio) is running on %s", self._bind_url)

    async def _create_server(self) -> grpc.aio.Server:
        return grpc.aio.server(
            migration_thread_pool=self._executor,
            options=GRPC_SERVER_OPTIONS,
        )

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _grpc_server(self) -> grpc.aio.Server:
        return self._server

    def _make_servicer(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
    ) -> object:
        return _AioServicer(binding, handlers, self)

    def _stop_server(self) -> None:
        self._run(self._server.stop(grace=None))

    def _release_threads(self, wait: bool) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()
        self._serialized_pool.shutdown(wait=wait)
        self._normal_pool.shutdown(wait=wait)
        super()._release_threads(wait)
