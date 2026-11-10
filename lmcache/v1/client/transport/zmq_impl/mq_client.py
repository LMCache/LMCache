# SPDX-License-Identifier: Apache-2.0
"""ZMQ DEALER client that sends requests to a multiprocess server."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar
import itertools
import queue
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.client.transport.errors import MPRequestError
from lmcache.v1.client.transport.zmq_impl.polling_loop import ClientPollingLoop
from lmcache.v1.multiprocess.rpc import RpcOperation, RpcSpec, get_rpc_spec
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    msgspec_decode,
    msgspec_encode,
)
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    ZMQ_ERROR_OPERATION,
    decode_error_response,
    decode_operation,
    encode_authentication_frame,
    encode_operation,
    is_authentication_error_frame,
)
from lmcache.v1.transport.futures import MessagingFuture, MessagingStream

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int


# Helper functions
def encode_request_uid(uid: RequestUID) -> bytes:
    return msgspec.msgpack.encode(uid)


def decode_request_uid(b_uid: bytes) -> RequestUID:
    return msgspec.msgpack.decode(b_uid, type=RequestUID)


CONNECT_TIMEOUT_MS = 5000
RECONNECT_IVL_MAX_MS = 1000


class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any] | MessagingStream[Any]
        rpc_spec: RpcSpec
        request_payloads: list[Any]
        responses: queue.Queue | None = None

    def __init__(
        self, server_url: str, context: zmq.Context, api_key: str | None = None
    ) -> None:
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.CONNECT_TIMEOUT, CONNECT_TIMEOUT_MS)
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, RECONNECT_IVL_MAX_MS)
        self.monitor = self.socket.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        self.socket.connect(server_url)
        self._authentication_frame = (
            encode_authentication_frame(api_key) if api_key is not None else None
        )

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
                    sent = self._send(self._with_authentication(wrapped_request))
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
                    if not self._send(
                        self._with_authentication(
                            [b_request_uid, b_operation] + b_payloads
                        )
                    ):
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

    def _with_authentication(self, frames: list[bytes]) -> list[bytes]:
        """Prepend the configured versioned authentication frame."""
        if self._authentication_frame is None:
            return frames
        return [self._authentication_frame, *frames]

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

        if is_authentication_error_frame(b_operation):
            auth_error = PermissionError("ZMQ request authentication failed")
            if request_uid in self.pending_streams:
                _, _, responses = self.pending_streams.pop(request_uid)
                responses.put_nowait(auth_error)
            elif request_uid in self.pending_futures:
                future, _ = self.pending_futures.pop(request_uid)
                future.set_exception(auth_error)
            return

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
                    MPRequestError(error.code, f"{error.operation}: {error.message}")
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
