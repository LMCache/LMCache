# SPDX-License-Identifier: Apache-2.0
"""Method-oriented client built directly from generated gRPC descriptors."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable
from urllib.parse import urlparse
import queue
import threading
import time
import uuid

# Third Party
import grpc

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    GrpcMethodCodec,
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import ResponseDecoder
from lmcache.v1.multiprocess.transport.grpc_impl.stream import (
    STREAM_METHOD,
    get_stream_method_ids,
    identity_bytes,
    pack_batch,
    pack_request_frame,
    unpack_batch,
    unpack_response_frame,
)

_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)
_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"
_STREAM_BATCH_DELAY_SECONDS = 0.000005
_STREAM_MAX_BATCH = 64


def parse_grpc_target(server_url: str) -> str:
    """Convert an LMCache multiprocess endpoint into a gRPC target.

    Args:
        server_url: gRPC endpoint.

    Returns:
        A target accepted by ``grpc.insecure_channel``.

    Raises:
        ValueError: If the URL scheme or target is invalid.
    """
    if "://" not in server_url:
        if not server_url:
            raise ValueError("gRPC server target must not be empty")
        return server_url

    parsed = urlparse(server_url)
    if parsed.scheme == "grpc":
        if not parsed.netloc:
            raise ValueError(f"Missing host in gRPC URL: {server_url!r}")
        return parsed.netloc
    if parsed.scheme == "grpc+unix":
        path = f"/{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        if not path:
            raise ValueError(f"Missing socket path in gRPC URL: {server_url!r}")
        return f"unix://{path}"
    raise ValueError(f"Unsupported gRPC URL scheme: {parsed.scheme!r}")


@dataclass(frozen=True)
class _ClientRpc:
    codec: GrpcMethodCodec
    method_id: int


@dataclass(frozen=True)
class _PendingStreamResponse:
    future: MessagingFuture[Any]
    response_message_class: type[Any]
    response_decoder: ResponseDecoder


ClientRpcCallable = Callable[..., MessagingFuture[Any]]


class _GrpcStreamTransport:
    """Shared persistent gRPC stream for one target."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._channel = grpc.insecure_channel(target, options=_GRPC_OPTIONS)
        self._metadata = ((_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)
        self._closed = threading.Event()
        self._stream_requests: queue.Queue[bytes | None] = queue.Queue()
        self._stream_pending: dict[int, _PendingStreamResponse] = {}
        self._stream_lock = threading.Lock()
        self._stream_request_ids = count(1)
        self._ref_count = 0
        stream_callable = self._channel.stream_stream(
            STREAM_METHOD,
            request_serializer=identity_bytes,
            response_deserializer=identity_bytes,
        )
        self._stream_call = stream_callable(
            self._iter_stream_batches(),
            metadata=self._metadata,
            wait_for_ready=True,
        )
        self._stream_reader = threading.Thread(
            target=self._read_stream_responses,
            name="lmcache-grpc-stream-reader",
            daemon=True,
        )
        self._stream_reader.start()

    def add_ref(self) -> None:
        self._ref_count += 1

    @property
    def closed(self) -> bool:
        """Return whether this shared stream transport is closed."""
        return self._closed.is_set()

    def release_ref(self) -> bool:
        self._ref_count -= 1
        return self._ref_count == 0

    def submit(
        self,
        client_key: int,
        rpc: _ClientRpc,
        request: Any,
    ) -> MessagingFuture[Any]:
        future: MessagingFuture[Any] = MessagingFuture()
        with self._stream_lock:
            if self._closed.is_set():
                future.set_exception(RuntimeError("gRPC client is closed"))
                return future
            request_id = next(self._stream_request_ids)
            self._stream_pending[request_id] = _PendingStreamResponse(
                future=future,
                response_message_class=rpc.codec.response_message_class,
                response_decoder=rpc.codec.response_decoder,
            )
        self._stream_requests.put(
            pack_request_frame(
                request_id,
                client_key,
                rpc.method_id,
                request.SerializeToString(),
            )
        )
        return future

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._stream_requests.put(None)
        self._stream_call.cancel()
        self._channel.close()
        self._stream_reader.join(timeout=1)
        self._fail_pending_streams(RuntimeError("gRPC client stream closed"))

    def _iter_stream_batches(self) -> Any:
        while True:
            frame = self._stream_requests.get()
            if frame is None:
                return
            frames = [frame]
            deadline = time.perf_counter() + _STREAM_BATCH_DELAY_SECONDS
            while len(frames) < _STREAM_MAX_BATCH:
                timeout = deadline - time.perf_counter()
                try:
                    if timeout > 0:
                        frame = self._stream_requests.get(timeout=timeout)
                    else:
                        frame = self._stream_requests.get_nowait()
                except queue.Empty:
                    break
                if frame is None:
                    self._stream_requests.put(None)
                    break
                frames.append(frame)
            yield pack_batch(frames)

    def _read_stream_responses(self) -> None:
        try:
            for batch in self._stream_call:
                for frame in unpack_batch(batch):
                    self._handle_stream_response(frame)
        except BaseException as exc:
            if not self._closed.is_set():
                self._closed.set()
                self._fail_pending_streams(exc)

    def _handle_stream_response(self, frame: bytes) -> None:
        request_id, ok, payload = unpack_response_frame(frame)
        with self._stream_lock:
            pending = self._stream_pending.pop(request_id, None)
        if pending is None:
            return
        if not ok:
            pending.future.set_exception(RuntimeError(payload.decode(errors="replace")))
            return
        try:
            response = pending.response_message_class.FromString(payload)
            pending.future.set_result(pending.response_decoder(response))
        except BaseException as exc:
            pending.future.set_exception(exc)

    def _fail_pending_streams(self, exc: BaseException) -> None:
        with self._stream_lock:
            pending_responses = tuple(self._stream_pending.values())
            self._stream_pending.clear()
        for pending in pending_responses:
            pending.future.set_exception(exc)


_STREAM_TRANSPORTS: dict[str, _GrpcStreamTransport] = {}
_STREAM_TRANSPORTS_LOCK = threading.Lock()


def _acquire_stream_transport(target: str) -> _GrpcStreamTransport:
    with _STREAM_TRANSPORTS_LOCK:
        transport = _STREAM_TRANSPORTS.get(target)
        if transport is None or transport.closed:
            transport = _GrpcStreamTransport(target)
            _STREAM_TRANSPORTS[target] = transport
        transport.add_ref()
        return transport


def _release_stream_transport(target: str, transport: _GrpcStreamTransport) -> None:
    with _STREAM_TRANSPORTS_LOCK:
        if not transport.release_ref():
            return
        if _STREAM_TRANSPORTS.get(target) is transport:
            _STREAM_TRANSPORTS.pop(target, None)
    transport.close()


class GrpcMultiprocessClient(RequestClient):
    """Expose every generated unary RPC as a snake-case client method."""

    def __init__(self, server_url: str) -> None:
        self._target = parse_grpc_target(server_url)
        self._transport = _acquire_stream_transport(self._target)
        self._client_key = int.from_bytes(uuid.uuid4().bytes[:8], "big")
        self._rpc_methods: dict[str, _ClientRpc] = {}
        codec_registry = get_method_codec_registry()
        stream_method_ids = get_stream_method_ids()
        for _binding, method in iter_methods():
            name = client_method_name(method.name)
            if name in self._rpc_methods:
                raise RuntimeError(f"Duplicate gRPC client method: {name}")
            self._rpc_methods[name] = _ClientRpc(
                codec=codec_registry.by_full_name[method.full_name],
                method_id=stream_method_ids[method.full_name],
            )
        self._closed = False

    def __getattr__(self, name: str) -> ClientRpcCallable:
        """Resolve a generated RPC as a method-oriented client call."""
        rpc = self._rpc_methods.get(name)
        if rpc is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {name!r}"
            )

        def invoke(*args: Any, **kwargs: Any) -> MessagingFuture[Any]:
            return self._call(rpc, args, kwargs)

        return invoke

    def __dir__(self) -> list[str]:
        """Include descriptor-derived RPC methods in introspection output."""
        return sorted(set(super().__dir__()) | set(self._rpc_methods))

    def cb_register_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbRegisterRope``."""
        return self.cb_register_rope(*args, **kwargs)

    def cb_unregister_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbUnregisterRope``."""
        return self.cb_unregister_rope(*args, **kwargs)

    def cb_retrieve_pre_computed_v3(
        self, *args: Any, **kwargs: Any
    ) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbRetrievePreComputed``."""
        return self.cb_retrieve_pre_computed(*args, **kwargs)

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._closed:
            return
        self._closed = True
        _release_stream_transport(self._target, self._transport)

    def _call(
        self,
        rpc: _ClientRpc,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MessagingFuture[Any]:
        request = rpc.codec.request_encoder(args, kwargs)
        if self._closed:
            future: MessagingFuture[Any] = MessagingFuture()
            future.set_exception(RuntimeError("gRPC client is closed"))
            return future
        return self._transport.submit(self._client_key, rpc, request)


def _make_client_rpc_method(name: str) -> ClientRpcCallable:
    def rpc_method(
        self: GrpcMultiprocessClient,
        *args: Any,
        **kwargs: Any,
    ) -> MessagingFuture[Any]:
        return self._call(self._rpc_methods[name], args, kwargs)

    rpc_method.__name__ = name
    rpc_method.__qualname__ = f"GrpcMultiprocessClient.{name}"
    return rpc_method


def _install_client_rpc_methods() -> None:
    """Install descriptor-derived RPC methods on the concrete client class."""
    for _, method in iter_methods():
        name = client_method_name(method.name)
        if name in GrpcMultiprocessClient.__dict__:
            raise RuntimeError(f"gRPC client method conflicts with {name!r}")
        setattr(GrpcMultiprocessClient, name, _make_client_rpc_method(name))


_install_client_rpc_methods()
