# SPDX-License-Identifier: Apache-2.0
"""gRPC implementation of :class:`ClientTransport` / :class:`ServerTransport`.

The wire protocol is a single bidirectional stream per client (see
``proto/lmcache_mq.proto``).  All framing / msgspec encoding stays in
``mq.py`` — this module only shovels opaque ``bytes`` frames.

Design notes:

* Both sides use grpc's *synchronous* API so the transport plays well
  with the existing ``zmq.Poller`` based main loops in ``mq.py``.
  Each side exposes a self-pipe fd via :meth:`readable_handle` so the
  Poller can wait on it exactly like it waits on a ZMQ socket.
* One background thread pumps the bidi stream on the client (grpc
  requires that); the server uses grpc's built-in thread pool.
* The server side keeps an in-memory map from :class:`ClientContext`
  to a per-stream outbox queue so ``send_response`` can route replies
  to the right stream without knowing anything about grpc internals.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlsplit
import itertools
import queue
import threading
import uuid

# Third Party
import grpc
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.transport.base import (
    ClientContext,
    ClientTransport,
    ServerTransport,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2,
    lmcache_mq_pb2_grpc,
)
from lmcache.v1.multiprocess.transport.registry import (
    register_client,
    register_server,
)

logger = init_logger(__name__)

_SENTINEL = object()


class _ZmqSelfPipe:
    """Wake a ``zmq.Poller`` from any thread using an inproc PAIR socket.

    ``os.pipe`` fds do not fire ``POLLIN`` reliably inside pyzmq's
    poller on all platforms (notably macOS).  A zmq PAIR socket over
    ``inproc://`` avoids that entirely and is what pyzmq recommends for
    cross-thread wakeups.
    """

    def __init__(self, ctx: zmq.Context) -> None:
        endpoint = "inproc://mq-selfpipe-" + uuid.uuid4().hex
        self._r = ctx.socket(zmq.PAIR)
        self._r.bind(endpoint)
        self._w = ctx.socket(zmq.PAIR)
        self._w.connect(endpoint)
        self._w_lock = threading.Lock()

    def readable_handle(self) -> zmq.Socket:
        return self._r

    def notify(self) -> None:
        with self._w_lock:
            try:
                self._w.send(b"x", flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

    def drain(self) -> None:
        while True:
            try:
                self._r.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return

    def close(self) -> None:
        for s in (self._r, self._w):
            try:
                s.close(linger=0)
            except zmq.ZMQError:
                pass


def _parse_url(url: str) -> tuple[str, dict[str, str]]:
    """Turn a ``grpc://`` / ``grpc+unix://`` URL into (grpc_target, options).

    Supported query parameters:

    * ``compression=gzip|deflate|none`` -- enables per-call payload
      compression.  Gzip is a good default for KV metadata payloads
      that carry ascii identifiers / json-ish blobs; leave off for
      already-compressed / tiny frames where the cpu cost outweighs
      the byte savings.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme == "grpc":
        target = parts.netloc
    elif scheme == "grpc+unix":
        target = "unix:" + parts.path
    else:
        raise ValueError("unsupported grpc URL scheme: " + scheme)

    opts: dict[str, str] = {}
    if parts.query:
        for chunk in parts.query.split("&"):
            if not chunk:
                continue
            key, _, value = chunk.partition("=")
            opts[key] = value
    return target, opts


_COMPRESSION_MAP = {
    "gzip": grpc.Compression.Gzip,
    "deflate": grpc.Compression.Deflate,
    "none": grpc.Compression.NoCompression,
    "": grpc.Compression.NoCompression,
}


def _resolve_compression(opts: dict[str, str]) -> grpc.Compression:
    raw = opts.get("compression", "").lower()
    if raw not in _COMPRESSION_MAP:
        raise ValueError(
            "unknown compression '"
            + raw
            + "'; expected one of "
            + ", ".join(sorted(k for k in _COMPRESSION_MAP if k))
        )
    return _COMPRESSION_MAP[raw]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GrpcClientTransport(ClientTransport):
    """One bidi stream per client, pumped by a background thread."""

    def __init__(self, context: Optional[zmq.Context] = None) -> None:
        self._ctx = context if context is not None else zmq.Context.instance()
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[lmcache_mq_pb2_grpc.MessageQueueStub] = None
        self._outbox: queue.Queue = queue.Queue()
        self._inbox: queue.Queue = queue.Queue()
        self._pipe = _ZmqSelfPipe(self._ctx)
        self._pump: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self._compression: grpc.Compression = grpc.Compression.NoCompression

    # -- request iterator fed by ``send_frames`` ---------------------------

    def _request_iter(self):
        while not self._closed.is_set():
            item = self._outbox.get()
            if item is _SENTINEL:
                return
            yield lmcache_mq_pb2.MqRequest(frames=item)

    def _pump_loop(self) -> None:
        assert self._stub is not None
        try:
            call = self._stub.Exchange(
                self._request_iter(), compression=self._compression
            )
            for response in call:
                self._inbox.put(list(response.frames))
                self._pipe.notify()
        except grpc.RpcError as exc:
            if not self._closed.is_set():
                logger.warning("grpc client stream ended: %s", exc)
        except Exception:
            logger.exception("grpc client pump crashed")

    # -- ClientTransport API ----------------------------------------------

    def connect(self, url: str) -> None:
        assert self._channel is None, "GrpcClientTransport already connected"
        target, opts = _parse_url(url)
        self._compression = _resolve_compression(opts)
        self._channel = grpc.insecure_channel(target)
        self._stub = lmcache_mq_pb2_grpc.MessageQueueStub(self._channel)
        self._pump = threading.Thread(
            target=self._pump_loop,
            daemon=True,
            name="mq-grpc-client-pump",
        )
        self._pump.start()

    def send_frames(self, frames: list[bytes]) -> None:
        self._outbox.put(list(frames))

    def recv_frames(self) -> list[bytes] | None:
        try:
            frames = self._inbox.get_nowait()
        except queue.Empty:
            frames = None
        if self._inbox.empty():
            self._pipe.drain()
        return frames

    def readable_handle(self) -> zmq.Socket:
        return self._pipe.readable_handle()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Unblock the request iterator.
        self._outbox.put(_SENTINEL)
        if self._pump is not None:
            self._pump.join(timeout=2)
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._pipe.close()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class _GrpcServerServicer(lmcache_mq_pb2_grpc.MessageQueueServicer):
    """Adapts every bidi RPC into per-stream inbox/outbox queues."""

    def __init__(self, transport: "GrpcServerTransport") -> None:
        self._transport = transport

    def Exchange(self, request_iterator, context):
        stream_id = next(self._transport.stream_id_counter)
        client_ctx = ClientContext(key=stream_id.to_bytes(8, "little"))
        outbox: queue.Queue = queue.Queue()
        self._transport.register_stream(client_ctx, outbox)

        def _reader():
            try:
                for req in request_iterator:
                    self._transport.enqueue_request(client_ctx, list(req.frames))
            except grpc.RpcError:
                pass
            finally:
                outbox.put(_SENTINEL)

        reader = threading.Thread(
            target=_reader,
            daemon=True,
            name="mq-grpc-server-reader",
        )
        reader.start()
        try:
            while True:
                item = outbox.get()
                if item is _SENTINEL:
                    return
                yield lmcache_mq_pb2.MqResponse(frames=item)
        finally:
            self._transport.unregister_stream(client_ctx)


class GrpcServerTransport(ServerTransport):
    """gRPC server side of the mp message queue."""

    def __init__(
        self,
        context: Optional[zmq.Context] = None,
        max_workers: int = 32,
    ) -> None:
        self._ctx = context if context is not None else zmq.Context.instance()
        self._server: Optional[grpc.Server] = None
        self._servicer = _GrpcServerServicer(self)
        self._inbox: queue.Queue = queue.Queue()
        self._pipe = _ZmqSelfPipe(self._ctx)
        # Per-connection outboxes; key is ClientContext because it is
        # hashable and unique per stream.
        self._outboxes: dict[ClientContext, queue.Queue] = {}
        self._outboxes_lock = threading.Lock()
        self.stream_id_counter = itertools.count(1)
        self._max_workers = max_workers

    # -- helpers called from servicer threads ------------------------------

    def register_stream(self, client_ctx: ClientContext, outbox: queue.Queue) -> None:
        with self._outboxes_lock:
            self._outboxes[client_ctx] = outbox

    def unregister_stream(self, client_ctx: ClientContext) -> None:
        with self._outboxes_lock:
            self._outboxes.pop(client_ctx, None)

    def enqueue_request(self, client_ctx: ClientContext, frames: list[bytes]) -> None:
        self._inbox.put((client_ctx, frames))
        self._pipe.notify()

    # -- ServerTransport API ----------------------------------------------

    def bind(self, url: str) -> None:
        assert self._server is None, "GrpcServerTransport already bound"
        target, opts = _parse_url(url)
        compression = _resolve_compression(opts)
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=self._max_workers),
            compression=compression,
        )
        lmcache_mq_pb2_grpc.add_MessageQueueServicer_to_server(
            self._servicer, self._server
        )
        self._server.add_insecure_port(target)
        self._server.start()

    def recv_request(self) -> tuple[ClientContext, list[bytes]] | None:
        try:
            item = self._inbox.get_nowait()
        except queue.Empty:
            item = None
        if self._inbox.empty():
            self._pipe.drain()
        return item

    def send_response(self, client_ctx: ClientContext, frames: list[bytes]) -> None:
        with self._outboxes_lock:
            outbox = self._outboxes.get(client_ctx)
        if outbox is None:
            logger.debug(
                "grpc server: dropping response for gone stream %s",
                client_ctx,
            )
            return
        outbox.put(list(frames))

    def readable_handle(self) -> zmq.Socket:
        return self._pipe.readable_handle()

    def close(self) -> None:
        if self._server is not None:
            self._server.stop(grace=0.5)
            self._server = None
        with self._outboxes_lock:
            for outbox in self._outboxes.values():
                outbox.put(_SENTINEL)
            self._outboxes.clear()
        self._pipe.close()


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


@register_client("grpc")
@register_client("grpc+unix")
def _make_grpc_client(
    context: Optional[zmq.Context] = None,
    **_kwargs,
) -> GrpcClientTransport:
    return GrpcClientTransport(context=context)


@register_server("grpc")
@register_server("grpc+unix")
def _make_grpc_server(
    context: Optional[zmq.Context] = None,
    **_kwargs,
) -> GrpcServerTransport:
    return GrpcServerTransport(context=context)
