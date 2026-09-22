# SPDX-License-Identifier: Apache-2.0
"""Low-level ZMQ message queue transport for multiprocess requests."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, get_type_hints
import enum
import inspect
import itertools
import queue
import threading

# Third Party
from zmq.utils.monitor import recv_monitor_message
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
    MessagingStream,
)
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.rpc import RpcOperation, RpcSpec, get_rpc_spec
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    decode_operation,
    encode_operation,
)
from lmcache.v1.platform import EventNotifier, create_event_notifier

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
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    MemoryLayoutDesc: (
        get_customized_encoder(MemoryLayoutDesc),
        get_customized_decoder(MemoryLayoutDesc),
    ),
    dict[int, MemoryLayoutDesc]: (
        get_customized_encoder(dict[int, MemoryLayoutDesc]),
        get_customized_decoder(dict[int, MemoryLayoutDesc]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    # Defensive guard: coerce obj to the declared cls so that
    # e.g. a bool passed as int (or vice-versa) is encoded in the
    # wire format that msgspec_decode expects for that cls.
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    # Defensive guard: msgspec strict-validates wire format
    # (bool ≠ int in msgpack), but runtime type may not match
    # declared cls. Decode untyped, then coerce.
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


# Shared polling loop for MessageQueueClient instances


class _OpKind(enum.Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"


@dataclass
class _PollOp:
    kind: _OpKind
    client: "MessageQueueClient"
    done: threading.Event


class ClientPollingLoop:
    """Singleton polling loop shared by all MessageQueueClient instances.

    Instead of each client running its own daemon thread and zmq.Poller,
    a single loop polls all clients' DEALER sockets and dispatches
    inbound/outbound work.

    Use ``get_instance()`` / ``release_instance()`` for lifecycle
    management — the loop starts lazily on first client and stops
    automatically when the last client releases.
    """

    _instance: "ClientPollingLoop | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._is_finished = threading.Event()
        self._notifier: EventNotifier = create_event_notifier()
        self._ops_queue: queue.Queue[_PollOp] = queue.Queue()
        self._poller = zmq.Poller()
        self._poller.register(self._notifier.fileno(), zmq.POLLIN)
        self._socket_to_client: dict[zmq.Socket, "MessageQueueClient"] = {}
        self._monitor_to_client: dict[zmq.Socket, "MessageQueueClient"] = {}
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-client-shared-loop"
        )
        self._thread.start()

    @classmethod
    def get_instance(cls) -> "ClientPollingLoop":
        """Get or create the singleton, incrementing the ref count.

        Returns:
            ClientPollingLoop: The shared polling loop instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ClientPollingLoop()
            cls._instance._ref_count += 1
            return cls._instance

    @classmethod
    def release_instance(cls) -> None:
        """Decrement the ref count; tear down the loop when it reaches 0."""
        with cls._instance_lock:
            inst = cls._instance
            if inst is None:
                return
            inst._ref_count -= 1
            if inst._ref_count > 0:
                return
            inst._is_finished.set()
            inst._notifier.notify()
            cls._instance = None

        inst._thread.join()
        inst._notifier.close()
        logger.debug("ClientPollingLoop shut down")

    def register(self, client: "MessageQueueClient") -> None:
        """Register a client's DEALER socket with the shared poller.

        Blocks until the loop thread has completed the registration.

        Args:
            client: The MessageQueueClient to register.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.REGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def unregister(self, client: "MessageQueueClient") -> None:
        """Unregister a client's DEALER socket from the shared poller.

        Blocks until the loop thread has completed the unregistration.

        Args:
            client: The MessageQueueClient to unregister.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.UNREGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def notify(self) -> None:
        """Wake the polling loop to process outbound tasks."""
        self._notifier.notify()

    def _process_ops(self) -> None:
        """Drain the ops queue and apply register/unregister to the poller."""
        try:
            while True:
                op = self._ops_queue.get_nowait()
                if op.kind is _OpKind.REGISTER:
                    self._poller.register(op.client.socket, zmq.POLLIN)
                    self._socket_to_client[op.client.socket] = op.client
                    self._poller.register(op.client.monitor, zmq.POLLIN)
                    self._monitor_to_client[op.client.monitor] = op.client
                    logger.debug("Registered client socket %s", op.client.socket)
                elif op.kind is _OpKind.UNREGISTER:
                    op.client.cancel_streams()
                    op.client.process_outbound_task()
                    self._poller.unregister(op.client.socket)
                    self._socket_to_client.pop(op.client.socket, None)
                    self._poller.unregister(op.client.monitor)
                    self._monitor_to_client.pop(op.client.monitor, None)
                    logger.debug("Unregistered client socket %s", op.client.socket)
                op.done.set()
        except queue.Empty:
            pass

    def _main_loop(self) -> None:
        """Unified poll loop for all registered clients."""
        notifier_fd = self._notifier.fileno()

        while not self._is_finished.is_set():
            self._process_ops()

            socks = dict(self._poller.poll(1000))

            # Outbound: shared notifier woke us — drain it, then flush
            # all clients' output queues.
            if socks.get(notifier_fd) and socks[notifier_fd] & zmq.POLLIN:
                self._notifier.consume()
                for client in self._socket_to_client.values():
                    client.process_outbound_task()

            # Inbound: dispatch each ready DEALER socket to its client.
            for sock, event in socks.items():
                if sock is notifier_fd:
                    continue
                if event & zmq.POLLIN:
                    if sock in self._monitor_to_client:
                        client = self._monitor_to_client[sock]
                        if (
                            recv_monitor_message(sock)["event"]
                            == zmq.EVENT_DISCONNECTED
                        ):
                            client.cancel_streams()
                        continue
                    owner = self._socket_to_client.get(sock)
                    if owner is not None:
                        try:
                            owner.process_inbound()
                        except Exception:
                            # This loop is shared by every client in the
                            # process, so one undecodable or unexpected frame
                            # must not terminate it: that would strand all
                            # other clients' pending and future requests.
                            logger.exception(
                                "process_inbound failed; dropping this frame "
                                "and continuing the shared polling loop"
                            )

        # Drain remaining ops so any waiting threads unblock.
        self._process_ops()


# Main classes
class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any] | MessagingStream[Any]
        rpc_spec: RpcSpec
        request_payloads: list[Any]
        responses: queue.Queue | None = None

    def __init__(self, server_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.DEALER)
        self.monitor = self.socket.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        self.socket.connect(server_url)

        # Input queue
        self.input_queue: queue.Queue = queue.Queue()

        # Pending job's futures
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, tuple[MessagingFuture[Any], RpcSpec]] = {}
        self.pending_streams: dict[
            int, tuple[MessagingStream[Any], RpcSpec, queue.Queue]
        ] = {}

        # Register with the shared polling loop
        self._polling_loop = ClientPollingLoop.get_instance()
        self._polling_loop.register(self)

    def process_outbound_task(self):
        try:
            while wrapped_request := self.input_queue.get_nowait():
                if isinstance(wrapped_request, list):
                    self.socket.send_multipart(wrapped_request)
                    if wrapped_request[-1] == b"cancel":
                        self.pending_streams.pop(
                            decode_request_uid(wrapped_request[0]), None
                        )
                    continue
                request_uid = wrapped_request.request_uid
                try:
                    b_request_uid = msgspec_encode(request_uid, cls=RequestUID)
                    rpc_spec = wrapped_request.rpc_spec
                    b_operation = encode_operation(rpc_spec.operation)
                    payload_classes = rpc_spec.payload_types
                    if len(payload_classes) != len(wrapped_request.request_payloads):
                        expected_classes = [cls.__name__ for cls in payload_classes]
                        actual_classes = [
                            type(payload).__name__
                            for payload in wrapped_request.request_payloads
                        ]
                        raise ValueError(
                            f"Payload count mismatch for request "
                            f"{rpc_spec.operation}: "
                            f"expected {len(payload_classes)} payloads "
                            f"{expected_classes}, "
                            f"got {len(wrapped_request.request_payloads)} payloads "
                            f"{actual_classes}. "
                            f"This is likely caused by a version mismatch between "
                            f"the lmcache client and lmcache server."
                        )

                    b_payloads = [
                        msgspec_encode(payload, cls=cls)
                        for payload, cls in zip(
                            wrapped_request.request_payloads,
                            payload_classes,
                            strict=False,
                        )
                    ]
                    if isinstance(wrapped_request.future, MessagingStream):
                        self.pending_streams[request_uid] = (
                            wrapped_request.future,
                            rpc_spec,
                            wrapped_request.responses,
                        )
                    else:
                        self.pending_futures[request_uid] = (
                            wrapped_request.future,
                            rpc_spec,
                        )
                    self.socket.send_multipart(
                        [b_request_uid, b_operation] + b_payloads
                    )
                except Exception as exc:
                    self.pending_futures.pop(request_uid, None)
                    if isinstance(wrapped_request.future, MessagingStream):
                        self.pending_streams.pop(request_uid, None)
                        wrapped_request.responses.put_nowait(exc)
                    else:
                        wrapped_request.future.set_exception(exc)
        except queue.Empty:
            pass

    def process_inbound(self) -> None:
        """Process one inbound response from the server.

        Called by the shared ClientPollingLoop when the DEALER socket
        is readable.  Only touches ``pending_futures``, which is
        exclusively accessed from the loop thread.
        """
        msg = self.socket.recv_multipart()
        if len(msg) < 2:
            logger.error(
                "Malformed response: expected at least 2 message parts "
                "[request_uid, operation, *response], got %d",
                len(msg),
            )
            return
        b_request_uid, b_operation, *b_response = msg
        request_uid = msgspec_decode(b_request_uid, cls=RequestUID)

        if request_uid in self.pending_streams:
            stream, rpc_spec, responses = self.pending_streams[request_uid]
            try:
                if (
                    decode_operation(b_operation) != rpc_spec.operation
                    or not b_response
                ):
                    raise ConnectionError("Event subscription ended")
                response = msgspec_decode(b_response[0], cls=rpc_spec.response_type)
                responses.put_nowait(response)
            except Exception:
                stream.close()
            return

        if request_uid in self.pending_futures:
            future, rpc_spec = self.pending_futures.pop(request_uid)
            operation = decode_operation(b_operation)
            if operation != rpc_spec.operation:
                future.set_exception(
                    ValueError(
                        f"Response operation {operation!r} does not match "
                        f"request {rpc_spec.operation!r}"
                    )
                )
                return
            if b_response:
                response = msgspec_decode(b_response[0], cls=rpc_spec.response_type)
                future.set_result(response)
            else:
                future.set_result(None)

    def submit_request(
        self,
        operation: RpcOperation,
        request_payloads: list[Any],
    ) -> MessagingFuture[T] | MessagingStream[T]:
        """Submit a request to the server.

        Args:
            operation: Stable snake-case RPC name.
            request_payloads (list[Any]): The payloads of the request.

        Returns:
            MessagingFuture[T]: A future that will hold the response.
        """
        rpc_spec = get_rpc_spec(operation)
        request_uid = next(self._request_counter)
        responses: queue.Queue | None = None
        future: MessagingFuture[T] | MessagingStream[T]
        if rpc_spec.streaming:
            responses = queue.Queue(maxsize=1)

            def control(action: bytes) -> None:
                self.input_queue.put([encode_request_uid(request_uid), b"", action])
                self._polling_loop.notify()

            def read(closed: threading.Event) -> T:
                response = responses.get()
                if closed.is_set():
                    raise StopIteration
                if isinstance(response, BaseException):
                    raise response
                control(b"ack")
                return response

            def cancel() -> None:
                try:
                    responses.put_nowait(None)
                except queue.Full:
                    pass
                control(b"cancel")

            future = MessagingStream(read, cancel)
        else:
            future = MessagingFuture()
        self.input_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=request_uid,
                future=future,
                rpc_spec=rpc_spec,
                request_payloads=request_payloads,
                responses=responses,
            )
        )
        self._polling_loop.notify()
        return future

    def cancel_streams(self) -> None:
        """Wake subscription readers on disconnect; called by the socket loop."""
        for stream, _, _ in list(self.pending_streams.values()):
            stream.close()
        self.pending_streams.clear()

    def close(self) -> None:
        self._polling_loop.unregister(self)
        ClientPollingLoop.release_instance()
        self.monitor.close(linger=0)
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


class MessageQueueServer(RequestServer):
    def __init__(self, bind_url: str, context: zmq.Context) -> None:
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(bind_url)
        # Use a cross-platform Notifier instead of zmq PUSH/PULL sockets
        # because blocking handler callbacks run on ThreadPoolExecutor
        # threads, and zmq sockets are not thread-safe. Notifier.notify()
        # is atomic (eventfd on Linux, self-pipe elsewhere).
        self._output_efd = create_event_notifier()
        self.output_queue: queue.Queue = queue.Queue()

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self._output_efd.fileno(), zmq.POLLIN)

        # Main loop thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-server-thread"
        )

        # Registered handlers: operation -> decoded handler wrapper
        self.handlers: dict[RpcOperation, RequestHandlerBase[Any]] = {}

        # Thread pools assigned via add_normal_thread_pool / add_affinity_thread_pool
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self.streams: dict[
            tuple[bytes, bytes],
            tuple[MessagingStream, threading.Event, threading.Thread],
        ] = {}

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
        if isinstance(response, MessagingStream):
            self._start_stream(
                response, handler_entry.get_response_class(), prefix_frames
            )
            return
        response_cls = handler_entry.get_response_class()
        b_response = msgspec_encode(response, cls=response_cls)
        if response is not None:
            self.socket.send_multipart(prefix_frames + [b_response])
        else:
            self.socket.send_multipart(prefix_frames)

    def _start_stream(
        self, stream: MessagingStream, response_cls: Any, prefix: list[bytes]
    ) -> None:
        """Forward one batch per credit; idle readers wait on event notification."""
        credit = threading.Event()
        credit.set()
        stream.add_close_callback(credit.set)

        def forward() -> None:
            try:
                while not self.is_finished.is_set():
                    credit.wait()
                    credit.clear()
                    response = next(stream)
                    self.output_queue.put(
                        prefix + [msgspec_encode(response, response_cls)]
                    )
                    self._output_efd.notify()
            except StopIteration:
                pass
            except Exception:
                logger.exception("Error in ZMQ event stream")
            finally:
                stream.close()
                if not self.is_finished.is_set():
                    self.output_queue.put(prefix)
                    self._output_efd.notify()

        thread = threading.Thread(target=forward, daemon=True, name="zmq-event-stream")
        self.streams[(prefix[0], prefix[1])] = (stream, credit, thread)
        thread.start()

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
                self._output_efd.notify()

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
        output_fd = self._output_efd.fileno()
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(output_fd, None)

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                assert len(msg) >= 3, (
                    "Expected at least 3 message parts "
                    "[identity, request_uid, operation, *payloads]"
                )

                identity, b_request_uid, b_operation, *payloads = msg
                if b_operation == b"":
                    state = self.streams.get((identity, b_request_uid))
                    if state is not None:
                        if payloads == [b"ack"]:
                            state[1].set()
                        elif payloads == [b"cancel"]:
                            state[0].close()
                    continue
                try:
                    operation = decode_operation(b_operation)
                except (TypeError, ValueError):
                    logger.exception("Invalid ZMQ operation identifier")
                    continue

                if handler_entry := self.handlers.get(operation):
                    try:
                        self._call_handler(
                            handler_entry=handler_entry,
                            payloads=payloads,
                            prefix_frames=[identity, b_request_uid, b_operation],
                        )
                    except Exception:
                        logger.exception("Error handling operation %s", operation)
                else:
                    logger.error("No handler registered for operation %s", operation)
                    logger.error("Available handlers: %s", list(self.handlers.keys()))

            # Send the responses
            if outbound_state and outbound_state & zmq.POLLIN:
                # Consume the notifier counter (resets atomically)
                self._output_efd.consume()

                # Process the output tasks
                try:
                    while frames_to_send := self.output_queue.get_nowait():
                        key = (frames_to_send[0], frames_to_send[1])
                        try:
                            self.socket.send_multipart(frames_to_send)
                        except zmq.ZMQError:
                            if key in self.streams:
                                self.streams[key][0].close()
                            else:
                                logger.warning("Response destination disconnected")
                        if len(frames_to_send) == 3:
                            self.streams.pop(key, None)
                except queue.Empty:
                    pass

    def _inspect_handler_signature(
        self, rpc_spec: RpcSpec, handler: Callable[..., Any]
    ) -> bool:
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

        payload_clss = rpc_spec.payload_types
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d arguments, but got %d",
                rpc_spec.operation,
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
                    rpc_spec.operation,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = rpc_spec.handler_response_type
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return type %s, but got %s",
                rpc_spec.operation,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        operation: RpcOperation,
        handler_type: HandlerType,
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for a specific RPC operation.

        Args:
            operation: Stable snake-case RPC name.
            handler_type: Whether to execute inline or on a worker.
            handler: Function that accepts the declared RPC payloads.

        Raises:
            ValueError: If the handler signature does not match the RPC contract.
            NotImplementedError: If a non-blocking handler is requested.
        """
        rpc_spec = get_rpc_spec(operation)
        if not self._inspect_handler_signature(rpc_spec, handler):
            raise ValueError(
                f"Handler signature does not match for operation: {operation}"
            )

        match handler_type:
            case HandlerType.SYNC:
                self.add_sync_handler(rpc_spec, handler)
            case HandlerType.BLOCKING:
                self.add_blocking_handler(rpc_spec, handler)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(self, rpc_spec: RpcSpec, handler: Callable[..., Any]) -> None:
        self.handlers[rpc_spec.operation] = SyncRequestHandler(
            list(rpc_spec.payload_types), rpc_spec.response_type, handler
        )

    def add_blocking_handler(
        self, rpc_spec: RpcSpec, handler: Callable[..., Any]
    ) -> None:
        self.handlers[rpc_spec.operation] = BlockingRequestHandler(
            list(rpc_spec.payload_types), rpc_spec.response_type, handler
        )

    def add_nonblocking_handler(
        self, rpc_spec: RpcSpec, handler: Callable[..., Any]
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        operations: list[RpcOperation],
        method_name: str,
    ) -> None:
        """Validate that all operations are registered blocking handlers."""
        for operation in operations:
            handler = self.handlers.get(operation)
            if handler is None:
                raise ValueError(
                    f"No handler registered for operation: {operation}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {operation} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler. "
                    f"Only blocking handlers can use thread pools."
                )

    def add_normal_thread_pool(
        self,
        operations: list[RpcOperation],
        max_workers: int,
    ) -> None:
        """Assign a ThreadPoolExecutor to specific request types.

        Use this for non-GPU blocking handlers (e.g. LOOKUP, END_SESSION).

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start(). Each operation must
        already be registered as a BlockingRequestHandler; otherwise a
        ValueError or TypeError is raised.

        Args:
            operations: The RPC operations that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(operations, "add_normal_thread_pool")
        if not operations:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for operation in operations:
            handler = self.handlers[operation]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for request types: %s",
            max_workers,
            operations,
        )

    def add_affinity_thread_pool(
        self,
        operations: list[RpcOperation],
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
            operations: The RPC operations that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(operations, "add_affinity_thread_pool")
        if not operations:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for operation in operations:
            handler = self.handlers[operation]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for request types: %s",
            max_workers,
            operations,
        )

    def start(self) -> None:
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
        for stream, _, _ in list(self.streams.values()):
            stream.close()
        for _, _, thread in list(self.streams.values()):
            thread.join()
        self.streams.clear()
        self.socket.close()
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
        self._output_efd.close()
