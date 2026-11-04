# SPDX-License-Identifier: Apache-2.0
"""Low-level ZMQ message queue transport for multiprocess requests."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
import enum
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
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
    MessagingStream,
)
from lmcache.v1.multiprocess.rpc import RpcOperation, RpcSpec, get_rpc_spec
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    ZMQ_ERROR_OPERATION,
    decode_error_response,
    decode_operation,
    encode_operation,
)
from lmcache.v1.platform import EventNotifier, create_event_notifier

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import (
        BlockingRequestHandler as BlockingRequestHandler,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import (
        MessageQueueServer as MessageQueueServer,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import (
        NonBlockingRequestHandler as NonBlockingRequestHandler,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import (
        RequestHandlerBase as RequestHandlerBase,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import (
        SyncRequestHandler as SyncRequestHandler,
    )

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int

# Moved to server_mq, which only the lmcache server distribution ships.
_SERVER_CLASSES = frozenset(
    {
        "BlockingRequestHandler",
        "MessageQueueServer",
        "NonBlockingRequestHandler",
        "RequestHandlerBase",
        "SyncRequestHandler",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve the ZMQ server classes that lived here before the split.

    They are imported only when accessed, so importing this module never
    loads the server.

    Raises:
        AttributeError: If ``name`` is not one of those classes.
        ModuleNotFoundError: If ``lmcache-server`` is not installed; the
            message names the missing module and the install command.
    """
    if name in _SERVER_CLASSES:
        try:
            # `from <package> import server_mq` would turn a missing module into
            # a bare ImportError that does not name it.
            # First Party
            import lmcache.v1.multiprocess.transport.zmq_impl.server_mq as server_mq
        except ModuleNotFoundError as e:
            if not (e.name or "").startswith("lmcache."):
                raise
            raise ModuleNotFoundError(
                f"{e.msg}; {name} ships with the full LMCache runtime: "
                "pip install 'lmcache[server]'",
                name=e.name,
            ) from e
        return getattr(server_mq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        # Sockets also polled for POLLOUT because their client has requests
        # waiting for room to send.
        self._waiting_to_send: set[zmq.Socket] = set()
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
                    self._waiting_to_send.discard(op.client.socket)
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
                    self._send_outbound(client)

            # Inbound: dispatch each ready DEALER socket to its client.
            for sock, event in socks.items():
                if sock is notifier_fd:
                    continue
                if event & zmq.POLLOUT and sock in self._socket_to_client:
                    self._send_outbound(self._socket_to_client[sock])
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

    def _send_outbound(self, client: "MessageQueueClient") -> None:
        """Send what the client has queued, and poll for room if some must wait."""
        waiting = client.process_outbound_task()
        if waiting == (client.socket in self._waiting_to_send):
            return
        if waiting:
            self._waiting_to_send.add(client.socket)
            self._poller.register(client.socket, zmq.POLLIN | zmq.POLLOUT)
        else:
            self._waiting_to_send.discard(client.socket)
            self._poller.register(client.socket, zmq.POLLIN)


CONNECT_TIMEOUT_MS = 5000
RECONNECT_IVL_MAX_MS = 1000


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
        self.socket.setsockopt(zmq.CONNECT_TIMEOUT, CONNECT_TIMEOUT_MS)
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, RECONNECT_IVL_MAX_MS)
        self.monitor = self.socket.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        self.socket.connect(server_url)

        # Input queue
        self.input_queue: queue.Queue = queue.Queue()
        # Encoded messages the socket had no room for, in send order.
        self._unsent: deque[list[bytes]] = deque()

        # Pending job's futures
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, tuple[MessagingFuture[Any], RpcSpec]] = {}
        self.pending_streams: dict[
            int, tuple[MessagingStream[Any], RpcSpec, queue.Queue]
        ] = {}

        # Register with the shared polling loop
        self._polling_loop = ClientPollingLoop.get_instance()
        self._polling_loop.register(self)

    def process_outbound_task(self) -> bool:
        """Send queued requests without blocking the shared polling loop.

        A request the socket has no room for (no connected peer, or the send
        high-water mark is reached) is kept with everything queued after it,
        and sent first on the next call.

        Returns:
            True if requests are still waiting for room to send.
        """
        try:
            while self._unsent:
                self.socket.send_multipart(self._unsent[0], flags=zmq.NOBLOCK)
                self._unsent.popleft()
        except zmq.Again:
            return True
        try:
            while wrapped_request := self.input_queue.get_nowait():
                if isinstance(wrapped_request, list):
                    sent = self._send(wrapped_request)
                    if wrapped_request[-1] == b"cancel":
                        self.pending_streams.pop(
                            decode_request_uid(wrapped_request[0]), None
                        )
                    if not sent:
                        return True
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
                    if not self._send([b_request_uid, b_operation] + b_payloads):
                        return True
                except Exception as exc:
                    self.pending_futures.pop(request_uid, None)
                    if isinstance(wrapped_request.future, MessagingStream):
                        self.pending_streams.pop(request_uid, None)
                        wrapped_request.responses.put_nowait(exc)
                    else:
                        wrapped_request.future.set_exception(exc)
        except queue.Empty:
            pass
        return False

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
            if operation == ZMQ_ERROR_OPERATION:
                if len(b_response) != 1:
                    future.set_exception(
                        RuntimeError("Malformed error response from LMCache server")
                    )
                    return
                error = decode_error_response(b_response[0])
                future.set_exception(
                    RuntimeError(
                        f"LMCache server rejected RPC {error.operation!r} "
                        f"({error.code}): {error.message}"
                    )
                )
                return
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

    def _send(self, frames: list[bytes]) -> bool:
        """Send frames now, or keep them for the next call if there is no room.

        A blocking send here would stall every client sharing the polling
        loop. libzmq blocks it once the high-water mark is reached while the
        server is down, and for good after a non-ZMQ peer fails the
        handshake, since it does not reconnect after that.
        """
        try:
            self.socket.send_multipart(frames, flags=zmq.NOBLOCK)
        except zmq.Again:
            self._unsent.append(frames)
            return False
        return True
