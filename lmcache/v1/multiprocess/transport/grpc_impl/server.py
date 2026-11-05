# SPDX-License-Identifier: Apache-2.0
"""gRPC request server construction for multiprocess requests."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable
import threading
import time

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
    AFFINITY_TURN_TIMEOUT_S,
    GRPC_SERVER_OPTIONS,
    AffinityTurns,
    GrpcRequestHandler,
    client_turn,
)

logger = init_logger(__name__)

# Affinity calls wait for their turn on a thread, and the call they wait for needs
# one too, so they get their own pool. More in flight than this can still run out
# of order. Much larger pools cost throughput.
_AFFINITY_CALL_THREADS = 128


class _GeneratedServicer:
    def __init__(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
        normal_slots: threading.BoundedSemaphore,
        affinity_pool: AffinityThreadPool,
        affinity_turn: threading.Condition,
        affinity_turns: AffinityTurns,
        closed: threading.Event,
        affinity_call_pool: ThreadPoolExecutor,
        serialized_handler_lock: threading.Lock,
    ) -> None:
        self._binding = binding
        self._handlers = handlers
        self._normal_slots = normal_slots
        self._affinity_pool = affinity_pool
        self._affinity_turn = affinity_turn
        self._affinity_turns = affinity_turns
        self._closed = closed
        self._affinity_call_pool = affinity_call_pool
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

        if handler.requires_client_affinity:
            invoke.experimental_thread_pool = (  # type: ignore[attr-defined]
                self._affinity_call_pool
            )
        return invoke

    def _dispatch(
        self,
        registered: GrpcRequestHandler,
        request: Any,
        context: grpc.ServicerContext,
    ) -> Any:
        try:
            if registered.handler is None:
                self._pass_turn(context)
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    f"{registered.operation} is not enabled on this server",
                )
                raise RuntimeError("gRPC context abort unexpectedly returned")
            try:
                payloads = registered.request_decoder(request)
            except Exception:
                self._pass_turn(context)
                raise
            if registered.handler_type is HandlerType.SERIALIZED:
                with self._serialized_handler_lock:
                    result = registered.handler(*payloads)
            elif registered.handler_type is HandlerType.BLOCKING and (
                registered.requires_client_affinity
            ):
                result = self._submit_in_turn(
                    registered.handler, payloads, context
                ).result()
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

    def _submit_in_turn(
        self,
        handler: Callable[..., Any],
        payloads: tuple[Any, ...],
        context: grpc.ServicerContext,
    ) -> Future[Any]:
        """Submit an affinity handler once the client's earlier calls have been."""
        client_key, sequence = client_turn(context)
        with self._affinity_turn:
            if sequence is not None:
                self._wait_for_turn(client_key, sequence)
            if self._closed.is_set():
                # The pool has stopped, so a submit would never finish.
                context.abort(grpc.StatusCode.UNAVAILABLE, "Server is closing")
            future = self._affinity_pool.submit(
                handler, *payloads, affinity_key=client_key
            )
            if sequence is not None:
                self._affinity_turns.finish(client_key, sequence)
                self._affinity_turn.notify_all()
        return future

    def _pass_turn(self, context: grpc.ServicerContext) -> None:
        """Keep a call that won't reach the affinity pool from holding up others."""
        client_key, sequence = client_turn(context)
        if sequence is None:
            return
        with self._affinity_turn:
            self._wait_for_turn(client_key, sequence)
            self._affinity_turns.finish(client_key, sequence)
            self._affinity_turn.notify_all()

    def _wait_for_turn(self, client_key: int, sequence: int) -> None:
        """Wait for the client's earlier calls. Needs ``_affinity_turn`` held."""
        turns = self._affinity_turns
        if turns.is_turn(client_key, sequence):
            return
        deadline = time.monotonic() + AFFINITY_TURN_TIMEOUT_S
        turns.add_waiter(client_key, sequence)
        try:
            while not self._closed.is_set() and not turns.is_turn(client_key, sequence):
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._affinity_turn.wait(remaining)
                elif turns.skip_to(client_key, sequence):
                    logger.warning(
                        "Call %d skipped a call that didn't arrive", sequence
                    )
                else:
                    self._affinity_turn.wait(AFFINITY_TURN_TIMEOUT_S)
        finally:
            turns.remove_waiter(client_key, sequence)


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
        self._affinity_turn = threading.Condition(threading.Lock())
        self._affinity_turns = AffinityTurns()
        self._affinity_call_pool = ThreadPoolExecutor(
            max_workers=_AFFINITY_CALL_THREADS,
            thread_name_prefix="grpc-affinity-call",
        )
        # HandlerType.SERIALIZED is a transport-neutral single-main-loop contract.
        self._serialized_handler_lock = threading.Lock()
        try:
            self._server = grpc.server(self._executor, options=GRPC_SERVER_OPTIONS)
            self._bind(self._server)
        except BaseException:
            self._release_threads(wait=True)
            raise

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
            self._affinity_turn,
            self._affinity_turns,
            self._closed,
            self._affinity_call_pool,
            self._serialized_handler_lock,
        )

    def _stop_server(self) -> None:
        with self._affinity_turn:
            self._affinity_turn.notify_all()
        self._server.stop(grace=None)

    def _release_threads(self, wait: bool) -> None:
        self._affinity_call_pool.shutdown(wait=wait)
        super()._release_threads(wait)
