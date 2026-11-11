# SPDX-License-Identifier: Apache-2.0
"""Low-level ZMQ message queue transport for multiprocess requests."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Generic, TypeVar, get_type_hints
import inspect
import queue
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.security.api_key import APIKeyVerifier
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    QRingInfo,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.transport.futures import MessagingStream
from lmcache.v1.transport.rpc import RpcOperation, RpcSpec, get_rpc_spec
from lmcache.v1.transport.zmq.wire import (
    AUTH_ERROR_FRAME,
    ZMQ_ERROR_OPERATION,
    ZmqErrorResponse,
    decode_authentication_frame,
    decode_operation,
    encode_error_response,
    encode_operation,
    is_authentication_frame,
)

logger = init_logger(__name__)


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
    list[QRingInfo]: (
        get_customized_encoder(list[QRingInfo]),
        get_customized_decoder(list[QRingInfo]),
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
        return HandlerType.SERIALIZED


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
    MAX_REQUESTS_PER_POLL = 64

    def __init__(
        self,
        bind_url: str,
        context: zmq.Context,
        authentication: APIKeyVerifier | None = None,
    ) -> None:
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(bind_url)
        self.authentication = authentication or APIKeyVerifier()
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

            except Exception as exc:
                operation = decode_operation(prefix_frames[2])
                logger.exception("Error in blocking handler %s", operation)
                error_code = (
                    "invalid_argument"
                    if isinstance(exc, ValueError)
                    else "handler_error"
                )
                self.output_queue.put(
                    prefix_frames[:2]
                    + [
                        encode_operation(ZMQ_ERROR_OPERATION),
                        encode_error_response(
                            ZmqErrorResponse(
                                code=error_code,
                                operation=operation,
                                message=str(exc),
                            )
                        ),
                    ]
                )
                self._output_efd.notify()

        future.add_done_callback(_notify_response)

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        match handler_entry.get_handler_type():
            case HandlerType.SERIALIZED:
                assert isinstance(handler_entry, SyncRequestHandler)
                self._call_sync_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.BLOCKING:
                assert isinstance(handler_entry, BlockingRequestHandler)
                self._call_blocking_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError("Unknown handler type")

    def _handle_request(self, msg: list[bytes]) -> None:
        if len(msg) < 3:
            logger.error(
                "Malformed request: expected at least 3 message parts "
                "[identity, request_uid, operation, *payloads], got %d",
                len(msg),
            )
            return

        identity, *request_frames = msg
        authenticated_frames = self._authenticate_request(identity, request_frames)
        if authenticated_frames is None:
            return
        if len(authenticated_frames) < 2:
            logger.error("Malformed ZMQ request after authentication")
            return
        b_request_uid, b_operation, *payloads = authenticated_frames
        if b_operation == b"":
            state = self.streams.get((identity, b_request_uid))
            if state is not None:
                if payloads == [b"ack"]:
                    state[1].set()
                elif payloads == [b"cancel"]:
                    state[0].close()
            return
        try:
            operation = decode_operation(b_operation)
        except (TypeError, ValueError):
            logger.exception("Invalid ZMQ operation identifier")
            return

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
            message = (
                f"Operation {operation!r} is not enabled on this server. "
                "Check whether the required server module is enabled; transfer "
                "operations may require a matching --supported-transfer-mode."
            )
            logger.error("%s", message)
            logger.error("Available handlers: %s", list(self.handlers.keys()))
            self.socket.send_multipart(
                [
                    identity,
                    b_request_uid,
                    encode_operation(ZMQ_ERROR_OPERATION),
                    encode_error_response(
                        ZmqErrorResponse(
                            code="operation_not_enabled",
                            operation=operation,
                            message=message,
                        )
                    ),
                ]
            )

    def _authenticate_request(
        self, identity: bytes, request_frames: list[bytes]
    ) -> list[bytes] | None:
        """Authenticate and remove the optional wire envelope before dispatch."""
        token: str | None = None
        authenticated_frames = request_frames
        if request_frames and is_authentication_frame(request_frames[0]):
            authenticated_frames = request_frames[1:]
            try:
                token = decode_authentication_frame(request_frames[0])
            except (ValueError, msgspec.DecodeError):
                logger.warning("Rejected malformed ZMQ authentication envelope")

        if not self.authentication.enabled:
            return authenticated_frames
        if self.authentication.verify_token(token):
            return authenticated_frames

        logger.warning("Rejected unauthenticated ZMQ request")
        if authenticated_frames:
            self.socket.send_multipart(
                [identity, authenticated_frames[0], AUTH_ERROR_FRAME]
            )
        return None

    def _send_responses(self) -> None:
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

    def _main_loop(self):
        output_fd = self._output_efd.fileno()
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(output_fd, None)

            if outbound_state and outbound_state & zmq.POLLIN:
                # Consume the notifier counter (resets atomically)
                self._output_efd.consume()

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                for _ in range(self.MAX_REQUESTS_PER_POLL):
                    try:
                        msg = self.socket.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._handle_request(msg)
                    self._send_responses()

            # Send the responses
            self._send_responses()

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
            case HandlerType.SERIALIZED:
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
