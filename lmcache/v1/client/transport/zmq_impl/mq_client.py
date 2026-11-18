# SPDX-License-Identifier: Apache-2.0
"""ZMQ DEALER client that sends requests to a multiprocess server."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, TypeVar
from urllib.parse import urlsplit
import itertools
import queue
import socket
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.client.transport.errors import MPRequestError
from lmcache.v1.client.transport.zmq_impl.polling_loop import ClientPollingLoop
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    msgspec_decode,
    msgspec_encode,
)
from lmcache.v1.transport.futures import MessagingFuture, MessagingStream
from lmcache.v1.transport.rpc import RpcOperation, RpcSpec, get_rpc_spec
from lmcache.v1.transport.zmq.wire import (
    ZMQ_ERROR_OPERATION,
    decode_error_response,
    decode_operation,
    encode_authentication_frame,
    encode_operation,
    is_authentication_error_frame,
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


CONNECT_TIMEOUT_MS = 5000
RECONNECT_IVL_MAX_MS = 1000

# HTTP/2 frame header: length (3 bytes), type, flags, stream id (4 bytes).
_HTTP2_FRAME_HEADER_LEN = 9
_HTTP2_SETTINGS = 0x04
_PEER_PROBE_TIMEOUT_S = 1.0


def _peer_speaks_http2(host: str, port: int) -> bool:
    """Return whether the server at ``host:port`` greets like an HTTP/2 server.

    Args:
        host: Server host.
        port: Server TCP port.

    Returns:
        True if the first bytes the server sends are an HTTP/2 SETTINGS frame
        header. False if it sends anything else, nothing in time, or refuses
        the connection (for example while it is restarting).
    """
    header = b""
    try:
        with socket.create_connection((host, port), _PEER_PROBE_TIMEOUT_S) as sock:
            sock.settimeout(_PEER_PROBE_TIMEOUT_S)
            while len(header) < _HTTP2_FRAME_HEADER_LEN:
                chunk = sock.recv(_HTTP2_FRAME_HEADER_LEN - len(header))
                if not chunk:
                    break
                header += chunk
    except OSError:
        return False
    return (
        len(header) == _HTTP2_FRAME_HEADER_LEN
        and header[3] == _HTTP2_SETTINGS
        and header[5:] == bytes(4)
    )


def put_stream_error(responses: queue.Queue, error: BaseException) -> None:
    """Hand error to a stream reader, replacing a response it has not read.

    The stream is failing, so dropping its last unread response is better than
    dropping the error: without the error the reader would wait forever.

    Args:
        responses: The stream's single-slot response queue.
        error: Raised by the reader's next read.
    """
    try:
        responses.put_nowait(error)
        return
    except queue.Full:
        pass
    try:
        responses.get_nowait()
    except queue.Empty:
        pass
    try:
        responses.put_nowait(error)
    except queue.Full:
        pass


def _tcp_host_port(server_url: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` for a ``tcp://`` URL, or None for other URLs."""
    parts = urlsplit(server_url)
    try:
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "tcp" or parts.hostname is None or port is None:
        return None
    return parts.hostname, port


class PeerCheck:
    """Track the checks a ZMQ disconnect starts to see if the server is gRPC.

    Only the socket loop uses it. A disconnect holds the streams open at that
    moment and starts a check; a disconnect while a check runs asks for one
    more, since the server may have changed in between. A gRPC result is
    final. Otherwise the held streams are released once a check ends with no
    disconnect after it started.
    """

    def __init__(self) -> None:
        self.speaks_grpc = False
        self._running = False
        self._again = False
        self._held: set[int] = set()

    def disconnected(self, open_streams: Iterable[int]) -> bool:
        """Record a disconnect.

        Args:
            open_streams: Request ids of the streams open at the disconnect.

        Returns:
            Whether to start a check. At most one check runs at a time.
        """
        if self.speaks_grpc:
            return False
        self._held.update(open_streams)
        if self._running:
            self._again = True
            return False
        self._running = True
        return True

    def checked(self, speaks_grpc: bool) -> tuple[bool, set[int]]:
        """Record the result of the running check.

        Args:
            speaks_grpc: Whether the server answered like a gRPC server.

        Returns:
            Whether to start another check, and the streams to release.
        """
        if speaks_grpc:
            self.speaks_grpc = True
            self._running = False
            self._again = False
            self._held = set()
            return False, set()
        if self._again:
            self._again = False
            return True, set()
        self._running = False
        held, self._held = self._held, set()
        return False, held


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
        self._server_url = server_url
        self._peer = _tcp_host_port(server_url)
        self._wrong_transport: str | None = None
        self._peer_check = PeerCheck()
        self._probe_results: queue.SimpleQueue[bool] = queue.SimpleQueue()
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
        self._apply_probe_results()
        if self._wrong_transport is not None:
            self._fail_all(self._wrong_transport)
            return False
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

    def cancel_streams(self, request_uids: set[int] | None = None) -> None:
        """Wake subscription readers; called by the socket loop.

        Args:
            request_uids: Streams to close. None closes every pending stream.
        """
        uids = list(self.pending_streams) if request_uids is None else request_uids
        for uid in uids:
            pending = self.pending_streams.pop(uid, None)
            if pending is not None:
                pending[0].close()

    def handle_disconnect(self) -> None:
        """Close pending streams, after checking whether the server speaks gRPC.

        libzmq drops a peer that fails the ZMTP handshake and never reconnects,
        so requests to a gRPC server would wait until they time out, and this
        disconnect is the only chance to notice. The check runs in the
        background and only reports its result; the socket loop applies it.
        Streams stay open until then, so a gRPC server fails them with the
        error instead of ending them. Called by the socket loop.
        """
        self._apply_probe_results()
        if self._peer is None:
            self.cancel_streams()
            return
        if self._peer_check.disconnected(self.pending_streams):
            self._start_probe()

    def close(self) -> None:
        self._polling_loop.unregister(self)
        ClientPollingLoop.release_instance()
        self.monitor.close(linger=0)
        self.socket.close()

    def _apply_probe_results(self) -> None:
        """Act on finished checks; runs on the socket loop."""
        while True:
            try:
                speaks_grpc = self._probe_results.get_nowait()
            except queue.Empty:
                return
            check_again, release = self._peer_check.checked(speaks_grpc)
            if self._peer_check.speaks_grpc and self._wrong_transport is None:
                assert self._peer is not None
                host, port = self._peer
                self._wrong_transport = (
                    f"The server at {self._server_url} speaks gRPC, not ZMQ. Use "
                    f"grpc://{host}:{port}, or start the server with "
                    "--transport zmq."
                )
                logger.error("%s", self._wrong_transport)
            self.cancel_streams(release)
            if check_again:
                self._start_probe()

    def _start_probe(self) -> None:
        assert self._peer is not None
        threading.Thread(
            target=self._probe_peer,
            args=self._peer,
            daemon=True,
            name="zmq-peer-probe",
        ).start()

    def _probe_peer(self, host: str, port: int) -> None:
        self._probe_results.put(_peer_speaks_http2(host, port))
        try:
            self._polling_loop.notify()
        except OSError:
            pass  # The client closed while probing.

    def _fail_all(self, message: str) -> None:
        """Fail every pending and queued request; runs on the socket loop."""
        self._unsent.clear()
        for future, _ in self.pending_futures.values():
            future.set_exception(ConnectionError(message))
        self.pending_futures.clear()
        for _, _, responses in self.pending_streams.values():
            self._fail_stream(responses, message)
        self.pending_streams.clear()
        try:
            while True:
                wrapped_request = self.input_queue.get_nowait()
                if isinstance(wrapped_request, list):
                    continue
                if isinstance(wrapped_request.future, MessagingStream):
                    self._fail_stream(wrapped_request.responses, message)
                else:
                    wrapped_request.future.set_exception(ConnectionError(message))
        except queue.Empty:
            pass

    @staticmethod
    def _fail_stream(responses: queue.Queue | None, message: str) -> None:
        if responses is not None:
            put_stream_error(responses, ConnectionError(message))

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
