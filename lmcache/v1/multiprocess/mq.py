# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar
import queue
import threading
import uuid

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
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
class MessagingFuture(Generic[T]):
    def __init__(self):
        self.is_done_ = threading.Event()
        self.result_ = None

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise TimeoutError("Future result not available within timeout")
        return self.result_

    def set_result(self, result: T) -> None:
        self.result_ = result
        self.is_done_.set()


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
        self.task_notifier, self.task_waiter = self._prepare_task_sockets()
        self.input_queue: queue.Queue = queue.Queue()

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.task_waiter, zmq.POLLIN)

        # main thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.worker_thread.start()

        # Pending job's futures
        self.request_counter = 0
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}

    def _prepare_task_sockets(self) -> tuple[zmq.Socket, zmq.Socket]:
        """Create 2 inproc socket pair for the zmq-poller compatible task
        queue

        Returns:
            tuple[zmq.Socket, zmq.Socket]: The (push_socket, pull_socket)
        """
        inproc_url = "inproc://mq_client_task_queue/" + str(uuid.uuid4())
        push_socket = self.ctx.socket(zmq.PUSH)
        pull_socket = self.ctx.socket(zmq.PULL)
        pull_socket.bind(inproc_url)
        push_socket.connect(inproc_url)
        return push_socket, pull_socket

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

                # TODO: we need a typing system for responses
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
    """

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.executor = executor
        self.payload_clss = payload_clss
        self.handler = handler
        self.response_cls = response_cls

    def __call__(self, payloads: list[bytes]) -> Future[ResponseType]:
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        future = self.executor.submit(self.handler, *decoded_payloads)
        return future

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
    def __init__(self, bind_url: str, context: zmq.Context, max_workers: int = 4):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(bind_url)

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Main loop thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(target=self._main_loop, daemon=True)

        # Thread pool for blocking handlers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Registered handlers: request_type -> (payload_cls, handler)
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}

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
        """
        future = handler_entry(payloads)

        def _send_response(fut: Future):
            try:
                response = fut.result()
                response_cls = handler_entry.get_response_class()
                b_response = msgspec_encode(response, cls=response_cls)
                if response is not None:
                    self.socket.send_multipart(prefix_frames + [b_response])
                else:
                    self.socket.send_multipart(prefix_frames)
            except Exception as e:
                logger.error("Error in blocking handler: %s", e)

        future.add_done_callback(_send_response)

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
            if socks.get(self.socket) == zmq.POLLIN:
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
                    except Exception as e:
                        logger.error("Error handling request %s: %s", request_type, e)
                else:
                    logger.error(
                        "No handler registered for request type %s", request_type
                    )
                    logger.error("Available handlers: %s", list(self.handlers.keys()))

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
        match handler_type:
            case HandlerType.SYNC:
                self.add_sync_handler(request_type, payload_clss, handler)
            case HandlerType.BLOCKING:
                self.add_blocking_handler(request_type, payload_clss, handler)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError(f"Unknown handler type: {handler_type}")
        # self.handlers[request_type] = self.HandlerEntry(payload_clss, handler)

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
            self.thread_pool, payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        raise NotImplementedError

    def start(self):
        self.worker_thread.start()

    def close(self) -> None:
        self.is_finished.set()
        self.worker_thread.join()
        self.socket.close()
