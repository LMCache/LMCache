# SPDX-License-Identifier: Apache-2.0
"""Method-oriented client built directly from generated gRPC descriptors."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse
import functools
import socket
import threading
import time
import uuid

# Third Party
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture, MessagingStream
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    GrpcMethodCodec,
    get_method_codec_registry,
)

CONNECT_TIMEOUT_MS = 5000
INITIAL_RECONNECT_BACKOFF_MS = 100
MAX_RECONNECT_BACKOFF_MS = 1000

# Same bounds as the ZMQ client (#4671); gRPC defaults are 20 s and 120 s.
# "min_reconnect_backoff_ms" is gRPC's per-attempt connect timeout.
_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.min_reconnect_backoff_ms", CONNECT_TIMEOUT_MS),
    ("grpc.initial_reconnect_backoff_ms", INITIAL_RECONNECT_BACKOFF_MS),
    ("grpc.max_reconnect_backoff_ms", MAX_RECONNECT_BACKOFF_MS),
)
_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"

# A ZMQ socket greets every new connection with this 10-byte ZMTP signature.
_ZMTP_SIGNATURE_LEN = 10
_PEER_PROBE_TIMEOUT_S = 1.0
_PEER_PROBE_INTERVAL_S = 5.0

logger = init_logger(__name__)


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


def _peer_speaks_zmtp(host: str, port: int) -> bool:
    """Return whether the server at ``host:port`` greets like a ZMQ socket.

    Args:
        host: Server host.
        port: Server TCP port.

    Returns:
        True if the first bytes the server sends are a ZMTP signature. False if
        it sends anything else, nothing in time, or refuses the connection
        (for example while it is still starting).
    """
    greeting = b""
    try:
        with socket.create_connection((host, port), _PEER_PROBE_TIMEOUT_S) as sock:
            sock.settimeout(_PEER_PROBE_TIMEOUT_S)
            while len(greeting) < _ZMTP_SIGNATURE_LEN:
                chunk = sock.recv(_ZMTP_SIGNATURE_LEN - len(greeting))
                if not chunk:
                    break
                greeting += chunk
    except OSError:
        return False
    return (
        len(greeting) == _ZMTP_SIGNATURE_LEN
        and greeting[0] == 0xFF
        and greeting[-1] == 0x7F
    )


def _tcp_host_port(target: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` for a TCP gRPC target, or None for other targets."""
    if target.startswith("unix:"):
        return None
    host, sep, port = target.rpartition(":")
    if not sep or not port.isdigit():
        return None
    return host.strip("[]"), int(port)


@dataclass(frozen=True)
class _ClientRpc:
    stub_method: Any
    codec: GrpcMethodCodec


ClientRpcCallable = Callable[..., MessagingFuture[Any] | MessagingStream[Any]]


class GrpcMultiprocessClient(RequestClient):
    """Expose generated unary and server-streaming RPCs as client methods."""

    def __init__(self, server_url: str) -> None:
        target = parse_grpc_target(server_url)
        self._server_url = server_url
        self._channel = grpc.insecure_channel(target, options=_GRPC_OPTIONS)
        # Calls wait for the server to come up (wait_for_ready), so a ZMQ
        # server on the port would otherwise look like one that never starts.
        self._wrong_transport: str | None = None
        self._pending: set[grpc.Future] = set()
        self._pending_lock = threading.Lock()
        self._next_probe_at = 0.0
        self._connectivity_callback: Callable[[grpc.ChannelConnectivity], None] | None
        self._connectivity_callback = None
        peer = _tcp_host_port(target)
        if peer is not None:
            self._connectivity_callback = functools.partial(
                self._on_connectivity, *peer
            )
            self._channel.subscribe(self._connectivity_callback, try_to_connect=False)
        stubs: dict[str, Any] = {}
        self._rpc_methods: dict[str, _ClientRpc] = {}
        codec_registry = get_method_codec_registry()
        for binding, method in iter_methods():
            service_name = binding.descriptor.name
            stub = stubs.get(service_name)
            if stub is None:
                stub_class = getattr(binding.grpc_module, f"{service_name}Stub")
                stub = stub_class(self._channel)
                stubs[service_name] = stub
            name = client_method_name(method.name)
            if name in self._rpc_methods:
                raise RuntimeError(f"Duplicate gRPC client method: {name}")
            self._rpc_methods[name] = _ClientRpc(
                stub_method=getattr(stub, method.name),
                codec=codec_registry.by_full_name[method.full_name],
            )
        self._metadata = ((_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)

    def __getattr__(self, name: str) -> ClientRpcCallable:
        """Resolve a generated RPC as a method-oriented client call."""
        rpc = self._rpc_methods.get(name)
        if rpc is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {name!r}"
            )

        def invoke(
            *args: Any, **kwargs: Any
        ) -> MessagingFuture[Any] | MessagingStream[Any]:
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
        if self._connectivity_callback is not None:
            self._channel.unsubscribe(self._connectivity_callback)
        self._channel.close()

    def _on_connectivity(
        self, host: str, port: int, state: grpc.ChannelConnectivity
    ) -> None:
        if state is not grpc.ChannelConnectivity.TRANSIENT_FAILURE:
            return
        now = time.monotonic()
        if self._wrong_transport is not None or now < self._next_probe_at:
            return
        self._next_probe_at = now + _PEER_PROBE_INTERVAL_S
        # Runs on gRPC's polling thread, which must not block on the probe.
        threading.Thread(
            target=self._probe_peer,
            args=(host, port),
            daemon=True,
            name="grpc-peer-probe",
        ).start()

    def _probe_peer(self, host: str, port: int) -> None:
        if not _peer_speaks_zmtp(host, port):
            return
        self._wrong_transport = (
            f"The server at {self._server_url} speaks ZMQ, not gRPC. Use "
            f"tcp://{host}:{port}, or start the server with --transport grpc."
        )
        logger.error("%s", self._wrong_transport)
        with self._pending_lock:
            pending = list(self._pending)
        for call in pending:
            call.cancel()

    def _call(
        self,
        rpc: _ClientRpc,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MessagingFuture[Any] | MessagingStream[Any]:
        request = rpc.codec.request_encoder(args, kwargs)
        if isinstance(rpc.stub_method, grpc.UnaryStreamMultiCallable):
            call = rpc.stub_method(
                request, metadata=self._metadata, wait_for_ready=True
            )
            self._track(call)

            def read(closed: threading.Event) -> Any:
                try:
                    return rpc.codec.response_decoder(next(call))
                except grpc.RpcError as exc:
                    if self._wrong_transport is not None:
                        raise ConnectionError(self._wrong_transport) from exc
                    if exc.code() in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.CANCELLED,
                    ):
                        raise ConnectionError("gRPC event stream disconnected") from exc
                    raise

            return MessagingStream(read, call.cancel)
        future: MessagingFuture[Any] = MessagingFuture()
        if self._wrong_transport is not None:
            future.set_exception(ConnectionError(self._wrong_transport))
            return future
        call = rpc.stub_method.future(
            request,
            metadata=self._metadata,
            wait_for_ready=True,
        )

        def on_done(grpc_future: grpc.Future[Any]) -> None:
            if grpc_future.cancelled() and self._wrong_transport is not None:
                future.set_exception(ConnectionError(self._wrong_transport))
                return
            try:
                result = rpc.codec.response_decoder(grpc_future.result())
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        call.add_done_callback(on_done)
        self._track(call)
        return future

    def _track(self, call: grpc.Future) -> None:
        """Keep ``call`` cancellable until it ends, so a wrong-transport
        verdict can fail it. Checked after adding, so a call cannot slip
        between the verdict and its cancel sweep."""
        with self._pending_lock:
            self._pending.add(call)
        call.add_done_callback(self._untrack)
        if self._wrong_transport is not None:
            call.cancel()

    def _untrack(self, call: grpc.Future) -> None:
        with self._pending_lock:
            self._pending.discard(call)


def _make_client_rpc_method(name: str) -> ClientRpcCallable:
    def rpc_method(
        self: GrpcMultiprocessClient,
        *args: Any,
        **kwargs: Any,
    ) -> MessagingFuture[Any] | MessagingStream[Any]:
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
