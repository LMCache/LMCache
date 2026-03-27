# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
import inspect
import os
import queue
import threading
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
    get_response_class,
)

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int


# Helper functions
def encode_request_uid(uid: RequestUID) -> bytes:
    return msgspec.msgpack.encode(uid)


def decode_request_uid(b_uid: bytes) -> RequestUID:
    return msgspec.msgpack.decode(b_uid, type=RequestUID)


def unwrap_request_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> list[Any]:
    if len(b_payloads) != len(payload_clss):
        raise ValueError("Payload count does not match expected count")

    decoded_payloads = [
        msgspec_decode(payload, cls=cls)
        for payload, cls in zip(b_payloads, payload_clss, strict=False)
    ]
    return decoded_payloads


def prepare_internal_push_pull_sockets(
    ctx: zmq.Context,
) -> tuple[zmq.Socket, zmq.Socket]:
    """Create 2 inproc socket pair for the zmq-poller compatible task
    queue

    Returns:
        tuple[zmq.Socket, zmq.Socket]: The (push_socket, pull_socket)
    """
    inproc_url = "inproc://mq_internal_push_pull/" + str(uuid.uuid4())
    push_socket = ctx.socket(zmq.PUSH)
    pull_socket = ctx.socket(zmq.PULL)
    pull_socket.bind(inproc_url)
    push_socket.connect(inproc_url)
    return push_socket, pull_socket


_SPECIAL_ENCODER_DECODERS = {
    CudaIPCWrapper: (
        get_customized_encoder(CudaIPCWrapper),
        get_customized_decoder(CudaIPCWrapper),
    ),
    list[CudaIPCWrapper]: (
        get_customized_encoder(list[CudaIPCWrapper]),
        get_customized_decoder(list[CudaIPCWrapper]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    return msgspec.msgpack.decode(b_obj, type=cls)


# Main classes
class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any]
        request_type: RequestType
        request_payloads: list[Any]

    def __init__(self, server_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(server_url)

        # Input queue
        self.task_notifier, self.task_waiter = prepare_internal_push_pull_sockets(
            self.ctx
        )
        self.input_queue: queue.Queue = queue.Queue()

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.task_waiter, zmq.POLLIN)

        # main thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-client-thread"
        )
        self.worker_thread.start()

        # Pending job's futures
        self.request_counter = 0
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}

    def _process_outbound_task(self):
        try:
            while wrapped_request := self.input_queue.get_nowait():
                # wrapped_request = self.input_queue.get_nowait()

                # Update the pending futures
                request_uid = wrapped_request.request_uid
                self.pending_futures[request_uid] = wrapped_request.future

                # Send the request
                b_request_uid = msgspec_encode(request_uid, cls=RequestUID)
                b_request_type = msgspec_encode(
                    wrapped_request.request_type, cls=RequestType
                )
                payload_classes = get_payload_classes(wrapped_request.request_type)
                if len(payload_classes) != len(wrapped_request.request_payloads):
                    raise ValueError("Payload count does not match expected count")

                b_payloads = [
                    msgspec_encode(payload, cls=cls)
                    for payload, cls in zip(
                        wrapped_request.request_payloads,
                        payload_classes,
                        strict=False,
                    )
                ]
                self.socket.send_multipart([b_request_uid, b_request_type] + b_payloads)
        except queue.Empty:
            pass

    def _main_loop(self):
        # NOTE: make sure we only edit the pending_futures dict in this thread
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(self.task_waiter, None)

            if outbound_state and outbound_state & zmq.POLLIN:
                # Drain the notifier
                while True:
                    try:
                        self.task_waiter.recv(zmq.DONTWAIT)
                    except zmq.Again:
                        break

                # Process the output tasks
                self._process_outbound_task()

            if inbound_state and inbound_state & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                assert len(msg) >= 2, (
                    "Expected at least 2 message part "
                    "[request_uid, request_type, *response]"
                )
                b_request_uid, b_request_type, *b_response = msg
                request_uid = msgspec_decode(b_request_uid, cls=RequestUID)
                request_type = msgspec_decode(b_request_type, cls=RequestType)
                response_cls = get_response_class(request_type)

                if request_uid in self.pending_futures:
                    future = self.pending_futures.pop(request_uid)
                    if b_response:
                        response = msgspec_decode(b_response[0], cls=response_cls)
                        future.set_result(response)
                    else:
                        future.set_result(None)

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
        response_cls: Optional[T] = None,
    ) -> MessagingFuture[T]:
        """Submit a request to the server.

        Args:
            request_type (RequestType): The type of the request.
            request_payloads (list[Any]): The payloads of the request.
            response_cls (Optional[T]): The expected response class.
                This should be get from `get_response_class(request_type)`.

        Returns:
            MessagingFuture[T]: A future that will hold the response.
        """
        future: MessagingFuture[T] = MessagingFuture()
        request_uid = self.request_counter
        self.request_counter += 1
        self.input_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=request_uid,
                future=future,
                request_type=request_type,
                request_payloads=request_payloads,
            )
        )
        self.task_notifier.send(b"1")
        return future

    def close(self) -> None:
        self.is_finished.set()
        self.worker_thread.join()
        self.socket.close()


ResponseType = TypeVar("ResponseType", covariant=True)
StateType = TypeVar("StateType", covariant=True)


class RequestHandlerBase(Generic[ResponseType]):
    def __call__(self, payloads: list[bytes]):
        raise NotImplementedError

    def get_response_class(self) -> ResponseType:
        raise NotImplementedError

    def get_handler_type(self) -> HandlerType:
        raise NotImplementedError


class SyncRequestHandler(RequestHandlerBase[ResponseType]):
    """
    The handler for those "fast" functions that can be executed in the main loop
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.payload_clss = payload_clss
        self.response_cls = response_cls
        self.handler = handler

    def __call__(self, payloads: list[bytes]) -> ResponseType:
        return self.handler(*unwrap_request_payloads(payloads, self.payload_clss))

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.SYNC


class BlockingRequestHandler(RequestHandlerBase[ResponseType]):
    """
    Returns the future of the response.

    The ``executor`` field is initially ``None`` and must be assigned via
    :meth:`MessageQueueServer.add_normal_thread_pool` or
    :meth:`MessageQueueServer.add_affinity_thread_pool` before the server
    is started.
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.executor: ThreadPoolExecutor | AffinityThreadPool | None = None
        self.payload_clss = payload_clss
        self.handler = handler
        self.response_cls = response_cls

    def __call__(
        self, payloads: list[bytes], affinity_key: int = 0
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "BlockingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler, *decoded_payloads, affinity_key=affinity_key
            )
        return self.executor.submit(self.handler, *decoded_payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.BLOCKING


class NonBlockingRequestHandler(Generic[ResponseType, StateType]):
    """
    The handler for the "fire and probe" functions that launch async tasks
    and have special mechanism to probe the task status.

    It requires 2 callables as the input:
    - the first one is to launch the async task. This function should return
        a 'state handle' that can be used to probe the task status later.
    - the second one is to probe the task status and get the return value
        with the 'state handle' returned by the first function.
    """

    # TODO: implement this in the future versions if needed
    pass


class MessageQueueServer:
    def __init__(self, bind_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(bind_url)
        # Use eventfd instead of zmq PUSH/PULL sockets because blocking
        # handler callbacks run on ThreadPoolExecutor threads, and zmq
        # sockets are not thread-safe. eventfd_write() is atomic.
        self._output_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.output_queue: queue.Queue = queue.Queue()

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self._output_efd, zmq.POLLIN)

        # Main loop thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-server-thread"
        )

        # Registered handlers: request_type -> (payload_cls, handler)
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}

        # Thread pools assigned via add_normal_thread_pool / add_affinity_thread_pool
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []

    def _call_sync_handler(
        self,
        handler_entry: SyncRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        """
        Call the sync handler and send the response back to the client.

        Args:
            handler_entry (SyncRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
        """
        response = handler_entry(payloads)
        response_cls = handler_entry.get_response_class()
        b_response = msgspec_encode(response, cls=response_cls)
        if response is not None:
            self.socket.send_multipart(prefix_frames + [b_response])
        else:
            self.socket.send_multipart(prefix_frames)

    def _call_blocking_handler(
        self,
        handler_entry: BlockingRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        """
        Call the blocking handler in a separate thread and send the response
        back to the client.

        Args:
            handler_entry (BlockingRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
                prefix_frames[0] is the zmq identity used as affinity key.
        """
        affinity_key = hash(prefix_frames[0])
        future = handler_entry(payloads, affinity_key=affinity_key)

        def _notify_response(fut: Future):
            try:
                response = fut.result()
                response_cls = handler_entry.get_response_class()
                b_response = msgspec_encode(response, cls=response_cls)
                frames_to_send = (
                    prefix_frames + [b_response]
                    if response is not None
                    else prefix_frames
                )

                self.output_queue.put(frames_to_send)
                os.eventfd_write(self._output_efd, 1)

            except Exception:
                logger.exception("Error in blocking handler")

        future.add_done_callback(_notify_response)

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        match handler_entry.get_handler_type():
            case HandlerType.SYNC:
                assert isinstance(handler_entry, SyncRequestHandler)
                self._call_sync_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.BLOCKING:
                assert isinstance(handler_entry, BlockingRequestHandler)
                self._call_blocking_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError("Unknown handler type")

    def _main_loop(self):
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(self._output_efd, None)

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                assert len(msg) >= 3, (
                    "Expected at least 3 message parts "
                    "[identity, request_uid, request_type, *payloads]"
                )

                identity, b_request_uid, b_request_type, *payloads = msg
                request_type = msgspec_decode(b_request_type, cls=RequestType)

                if handler_entry := self.handlers.get(request_type):
                    try:
                        self._call_handler(
                            handler_entry=handler_entry,
                            payloads=payloads,
                            prefix_frames=[identity, b_request_uid, b_request_type],
                        )
                    except Exception:
                        logger.exception("Error handling request %s", request_type)
                else:
                    logger.error(
                        "No handler registered for request type %s", request_type
                    )
                    logger.error("Available handlers: %s", list(self.handlers.keys()))

            # Send the responses
            if outbound_state and outbound_state & zmq.POLLIN:
                # Consume the eventfd counter (resets atomically)
                os.eventfd_read(self._output_efd)

                # Process the output tasks
                try:
                    while frames_to_send := self.output_queue.get_nowait():
                        self.socket.send_multipart(frames_to_send)
                except queue.Empty:
                    pass

    def _inspect_handler_signature(self, request_type: RequestType, handler) -> bool:
        """Inspect the handler signature to ensure it matches the expected
        payload classes.

        Args:
            handler (callable): The handler function.

        Returns:
            bool: True if the signature matches, False otherwise.
        """

        def same_type(a, b) -> bool:
            if a is None:
                a = type(None)
            if b is None:
                b = type(None)
            return a == b

        sig = inspect.signature(handler)
        hints = get_type_hints(handler)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        payload_clss = get_payload_classes(request_type)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d arguments, but got %d",
                request_type,
                len(payload_clss),
                len(params),
            )
            return False

        for i, (param, expected_cls) in enumerate(
            zip(params, payload_clss, strict=False)
        ):
            ann = hints.get(param.name, param.annotation)
            if not same_type(ann, expected_cls):
                logger.error(
                    "Handler for %s argument %d expects type %s, but got %s",
                    request_type,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(request_type)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return type %s, but got %s",
                request_type,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler,
    ) -> None:
        """Register a handler for a specific request type.

        Args:
            request_type (RequestType): The type of the request to handle.
            payload_clss (list[Any]): The expected payload classes for the request.
                This should be get from `get_payload_classes(request_type)`.
            handler (callable): The handler function that takes the payloads
                as arguments.
        """
        if not self._inspect_handler_signature(request_type, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {request_type}"
            )

        match handler_type:
            case HandlerType.SYNC:
                self.add_sync_handler(request_type, payload_clss, handler)
            case HandlerType.BLOCKING:
                self.add_blocking_handler(request_type, payload_clss, handler)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = SyncRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_blocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = BlockingRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        request_types: list[RequestType],
        method_name: str,
    ) -> None:
        """Validate that all request types are registered BlockingRequestHandlers."""
        for request_type in request_types:
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(
                    f"No handler registered for request type: {request_type}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {request_type} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler. "
                    f"Only blocking handlers can use thread pools."
                )

    def add_normal_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign a ThreadPoolExecutor to specific request types.

        Use this for non-GPU blocking handlers (e.g. LOOKUP, END_SESSION).

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().  Each request_type must
        already be registered as a BlockingRequestHandler; otherwise a
        ValueError or TypeError is raised.

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_normal_thread_pool")
        if not request_types:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign an AffinityThreadPool to specific request types.

        Use this for GPU-bound blocking handlers (e.g. STORE, RETRIEVE).
        Requests from the same zmq client identity are always dispatched
        to the same worker thread, eliminating the need for per-instance
        GPU transfer locks.

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_affinity_thread_pool")
        if not request_types:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def start(self):
        # Validate all blocking handlers have an executor assigned
        for rt, handler in self.handlers.items():
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    f"assigned. Call add_normal_thread_pool or "
                    f"add_affinity_thread_pool before start()."
                )
        self.worker_thread.start()

    def close(self) -> None:
        self.is_finished.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        self.socket.close()
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
        os.close(self._output_efd)
