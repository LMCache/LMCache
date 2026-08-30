# SPDX-License-Identifier: Apache-2.0
"""LMCache mp-mode gRPC transport.

Each RPC method maps to a distinct typed unary gRPC method on one of the
services defined in ``proto/lmcache_mq.proto``. Concurrent control-plane calls
go directly over typed unary RPCs. The old msgspec envelope (uid + request type
+ payload frames) is gone; gRPC service/method routing and protobuf
request/response messages now define the wire protocol.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeVar
from urllib.parse import parse_qs, urlparse
import threading
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.protocol import (
    RPC_METHODS,
    HandlerType,
    RpcMethod,
    coerce_rpc_method,
    get_grpc_method_options,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
    compile_request_decoder,
    compile_response_encoder,
    decode_response_to_python,
    encode_request_from_call,
    get_request_message_class,
    get_response_message_class,
    get_service_names,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    msgspec_decode as msgspec_decode,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    msgspec_encode as msgspec_encode,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    request_type_to_method_name as request_type_to_method_name,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    validate_protocol_descriptor,
)

# Message classes come out of the protobuf descriptor pool at runtime
# and are invisible to static analysis; rebind through Any so mypy
# does not chase every attribute lookup.
lmcache_mq_pb2: Any = _pb2_typed
grpc: Any = None
lmcache_mq_pb2_grpc: Any = None
_grpc_runtime_lock = threading.Lock()

logger = init_logger(__name__)

T = TypeVar("T")
ClientRpcCallable = Callable[..., MessagingFuture[Any]]

# gRPC channel/server options. LMCache multiprocess is a loopback
# (localhost TCP or unix socket) IPC boundary carrying KV cache
# payloads that routinely exceed the 4 MiB default; disable both
# caps so registers/stores never trip on message size.
_GRPC_UNLIMITED_MSG_OPTS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
]
_GRPC_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


def _ensure_grpc_runtime() -> None:
    """Load gRPC and generated service bindings on first transport use."""
    global grpc, lmcache_mq_pb2_grpc

    if grpc is not None:
        return
    with _grpc_runtime_lock:
        if grpc is not None:
            return

        # Third Party
        import grpc as grpc_module

        # First Party
        from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
            lmcache_mq_pb2_grpc as pb2_grpc_module,
        )

        lmcache_mq_pb2_grpc = pb2_grpc_module
        grpc = grpc_module


_RPC_METHOD_NAMES = {
    rpc_method: request_type_to_method_name(rpc_method) for rpc_method in RPC_METHODS
}
_CLIENT_RPC_METHOD_NAMES = {
    rpc_method: rpc_method.client_method_name for rpc_method in RPC_METHODS
}
_CLIENT_RPC_METHODS_BY_NAME = {
    method_name: rpc_method
    for rpc_method, method_name in _CLIENT_RPC_METHOD_NAMES.items()
}
_RPC_METHODS_BY_SERVICE: dict[str, tuple[RpcMethod, ...]] = {
    service_name: tuple(
        rpc_method
        for rpc_method in RPC_METHODS
        if rpc_method.service_name == service_name
    )
    for service_name in {rpc_method.service_name for rpc_method in RPC_METHODS}
}
_RPC_METHODS_BY_METHOD_NAME: dict[str, RpcMethod] = {
    str(rpc_method): rpc_method for rpc_method in RPC_METHODS
}

validate_protocol_descriptor()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def _parse_grpc_url(url: str) -> str:
    """Return a gRPC target string for the mp-mode transport URL.

    Backward compatibility matters here: older LMCache configs still pass
    ``tcp://host:port`` or ``ipc:///path`` from the historical ZMQ transport.
    The wire is gRPC now, but those legacy schemes still map cleanly to gRPC's
    TCP and unix-domain-socket targets, so accept them as aliases instead of
    breaking existing deploys and tests.
    """
    if "://" not in url:
        target, _, _query = url.partition("?")
        return target
    parsed = urlparse(url)
    if parsed.scheme in ("grpc", "tcp"):
        if not parsed.netloc:
            raise ValueError(f"missing host in url {url!r}")
        if parsed.scheme == "tcp":
            logger.warning(
                "treating legacy mp transport url %r as grpc://%s", url, parsed.netloc
            )
        return parsed.netloc

    if parsed.scheme in ("grpc+unix", "unix", "ipc"):
        unix_path = parsed.path
        if parsed.netloc:
            unix_path = f"/{parsed.netloc}{parsed.path}"
        if not unix_path:
            raise ValueError(f"missing unix socket path in url {url!r}")
        if parsed.scheme in ("unix", "ipc"):
            logger.warning(
                "treating legacy mp transport url %r as grpc+unix://%s",
                url,
                unix_path,
            )
        return f"unix://{unix_path}"

    raise ValueError(
        f"unsupported transport scheme {parsed.scheme!r} for url {url!r}; "
        "supported schemes are grpc://, grpc+unix://, and the legacy aliases "
        "tcp:// / ipc:// / unix://"
    )


def _parse_grpc_compression(url: str) -> Any:
    """Resolve the optional gRPC compression mode from a transport URL."""
    _ensure_grpc_runtime()
    values = parse_qs(urlparse(url).query, keep_blank_values=True).get(
        "compression", ["none"]
    )
    if len(values) != 1:
        raise ValueError("compression must be specified at most once")

    name = values[0].lower()
    modes = {
        "": grpc.Compression.NoCompression,
        "none": grpc.Compression.NoCompression,
        "gzip": grpc.Compression.Gzip,
        "deflate": grpc.Compression.Deflate,
    }
    if name not in modes:
        raise ValueError(
            f"unknown gRPC compression {name!r}; expected gzip, deflate, or none"
        )
    return modes[name]


def _grpc_affinity_key(context: "grpc.ServicerContext") -> int:
    """Return the stable client identity carried in gRPC metadata."""
    for key, value in context.invocation_metadata():
        if key == _GRPC_CLIENT_ID_METADATA_KEY:
            return hash(value)
    return hash(context.peer())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class _PendingUnaryRequest:
    rpc_method: RpcMethod
    method_name: str
    stub_method: Any
    proto_request: Any
    future: MessagingFuture[Any]


class MultiprocessGrpcClient:
    """Typed gRPC client for the LMCache mp cache server.

    Instances are cheap; a shared ``grpc.Channel`` is created per client
    and callers can share one client across many threads (gRPC channels
    are thread-safe).

    Args:
        server_url: Either ``grpc://host:port``, ``grpc+unix:///path``, a bare
            ``host:port``, or a legacy alias such as ``tcp://host:port`` /
            ``ipc:///path``.
        context: Legacy positional slot kept for backwards compatibility
            with the historical zmq-based constructor; ignored.
    """

    def __init__(
        self,
        server_url: str,
        context: Optional[Any] = None,
        transport: Optional[Any] = None,
    ):
        del context, transport  # legacy positional slots, no longer used
        _ensure_grpc_runtime()
        target = _parse_grpc_url(server_url)
        compression = _parse_grpc_compression(server_url)
        self._server_url = server_url
        self._channel = grpc.insecure_channel(
            target,
            options=_GRPC_UNLIMITED_MSG_OPTS,
            compression=compression,
        )
        self._stubs = {
            service_name: getattr(lmcache_mq_pb2_grpc, f"{service_name}Stub")(
                self._channel
            )
            for service_name in get_service_names()
        }
        self._call_metadata = ((_GRPC_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)
        self._rpc_methods = {
            rpc_method: (
                _RPC_METHOD_NAMES[rpc_method],
                getattr(
                    self._stubs[rpc_method.service_name],
                    _RPC_METHOD_NAMES[rpc_method],
                ),
                get_request_message_class(rpc_method),
            )
            for rpc_method in RPC_METHODS
        }

    def __getattr__(self, name: str) -> ClientRpcCallable:
        """Resolve dynamically generated snake_case RPC methods for type checkers.

        Args:
            name: Attribute name being looked up.

        Returns:
            Callable RPC method returning a :class:`MessagingFuture`.

        Raises:
            AttributeError: If ``name`` is not a known RPC method.
        """
        rpc_method = _CLIENT_RPC_METHODS_BY_NAME.get(name)
        if rpc_method is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {name!r}"
            )

        def _rpc_call(
            *request_payloads: Any, **request_fields: Any
        ) -> MessagingFuture[Any]:
            return self._call_rpc(rpc_method, *request_payloads, **request_fields)

        return _rpc_call

    def __dir__(self) -> list[str]:
        """Include generated RPC method names in introspection output."""
        return sorted(set(super().__dir__()) | set(_CLIENT_RPC_METHODS_BY_NAME))

    def _call_rpc(
        self,
        rpc_method: RpcMethod,
        *request_payloads: Any,
        **request_fields: Any,
    ) -> MessagingFuture[T]:
        """Submit one typed RPC by its protocol method.

        Args:
            rpc_method: Descriptor-derived RPC method token.
            request_payloads: Positional Python payloads for the method.

        Returns:
            A ``MessagingFuture`` completed by the gRPC callback.
        """
        method_name, stub_method, request_message_class = self._rpc_methods[rpc_method]
        future: MessagingFuture[T] = MessagingFuture()

        proto_request = encode_request_from_call(
            request_message_class,
            request_payloads,
            request_fields,
        )
        pending = _PendingUnaryRequest(
            rpc_method=rpc_method,
            method_name=method_name,
            stub_method=stub_method,
            proto_request=proto_request,
            future=future,
        )
        self._submit_unary(pending)
        return future

    def close(self) -> None:
        self._channel.close()

    def _submit_unary(self, pending: _PendingUnaryRequest) -> None:
        def _on_done_typed(call: "grpc.Future[Any]") -> None:
            self._on_unary_done(call, pending)

        # Requests submitted while the daemon starts remain pending until it is
        # reachable, matching the old DEALER socket behavior.
        call = pending.stub_method.future(
            pending.proto_request,
            metadata=self._call_metadata,
            wait_for_ready=True,
        )
        call.add_done_callback(_on_done_typed)

    def _on_unary_done(
        self,
        call: "grpc.Future[Any]",
        pending: _PendingUnaryRequest,
    ) -> None:
        try:
            proto_response = call.result()
        except grpc.RpcError as exc:
            if exc.code() is grpc.StatusCode.UNAVAILABLE:
                logger.warning(
                    "gRPC call %s lost its server and remains pending: %s",
                    pending.method_name,
                    exc,
                )
                return
            logger.error("gRPC call %s failed: %s", pending.method_name, exc)
            pending.future.set_exception(exc)
            return
        except Exception as exc:  # defensive
            logger.exception("gRPC call %s failed", pending.method_name)
            pending.future.set_exception(exc)
            return

        try:
            decoded = decode_response_to_python(proto_response)
        except Exception as exc:
            logger.exception(
                "failed to decode protobuf response for %s", pending.method_name
            )
            pending.future.set_exception(exc)
            return
        pending.future.set_result(decoded)


def _make_client_rpc_method(rpc_method: RpcMethod) -> ClientRpcCallable:
    method_name = _CLIENT_RPC_METHOD_NAMES[rpc_method]

    def _rpc_method(
        self: MultiprocessGrpcClient,
        *request_payloads: Any,
        **request_fields: Any,
    ) -> MessagingFuture[Any]:
        return self._call_rpc(rpc_method, *request_payloads, **request_fields)

    _rpc_method.__name__ = method_name
    _rpc_method.__qualname__ = f"MultiprocessGrpcClient.{method_name}"
    _rpc_method.__doc__ = (
        f"Submit the {rpc_method.name} RPC and return a ``MessagingFuture``."
    )
    return _rpc_method


def _install_client_rpc_methods() -> None:
    """Install one snake_case method per RPC on ``MultiprocessGrpcClient``."""
    for rpc_method, method_name in _CLIENT_RPC_METHOD_NAMES.items():
        if hasattr(MultiprocessGrpcClient, method_name):
            raise RuntimeError(
                f"RPC client method {method_name!r} conflicts with an existing "
                "MultiprocessGrpcClient attribute"
            )
        setattr(
            MultiprocessGrpcClient, method_name, _make_client_rpc_method(rpc_method)
        )


_install_client_rpc_methods()


# ---------------------------------------------------------------------------
# Server: direct gRPC method dispatch
# ---------------------------------------------------------------------------


@dataclass
class GrpcRequestHandler:
    """Registered Python implementation for one protobuf service method.

    Args:
        method_name: Protobuf service method name.
        handler: Python callable implementing the method.
        handler_type: Scheduler mode for the registered handler.
        requires_client_affinity: Whether blocking calls need client affinity.
        request_decoder: Converts protobuf requests to handler args.
        response_encoder: Converts handler results to protobuf responses.
        executor: Dedicated executor for blocking handlers, if assigned.
    """

    method_name: str
    handler: Callable[..., Any]
    handler_type: HandlerType
    requires_client_affinity: bool
    request_decoder: RequestDecoder
    response_encoder: ResponseEncoder
    payload_types: tuple[Any, ...]
    response_type: Any
    executor: ThreadPoolExecutor | AffinityThreadPool | None = None

    def run(self, request: Any, affinity_key: int) -> Any:
        """Execute the registered Python handler and return its result.

        Args:
            request: Protobuf request from gRPC.
            affinity_key: Stable client key used by affinity executors.

        Returns:
            The protobuf response encoded from the handler's Python result.

        Raises:
            RuntimeError: If a blocking handler has no executor.
            NotImplementedError: If the handler type is not implemented.
        """
        payloads = self.request_decoder(request)
        if self.handler_type is HandlerType.SYNC:
            return self.response_encoder(self.handler(*payloads))
        if self.handler_type is HandlerType.BLOCKING:
            if self.executor is None:
                raise RuntimeError(
                    f"Blocking handler for {self.method_name} has no "
                    "thread pool assigned."
                )
            if isinstance(self.executor, AffinityThreadPool):
                future = self.executor.submit(
                    self.handler, *payloads, affinity_key=affinity_key
                )
            else:
                future = self.executor.submit(self.handler, *payloads)
            return self.response_encoder(future.result())
        raise NotImplementedError(f"handler_type {self.handler_type} not supported")


class _GrpcServicer:
    """Concrete servicer implementing generated LMCache gRPC methods."""

    def __init__(self, handlers: dict[RpcMethod, GrpcRequestHandler]) -> None:
        self._handlers = handlers

    def _dispatch(
        self,
        method_name: str,
        request: Any,
        context: "grpc.ServicerContext",
    ) -> Any:
        """Decode a protobuf request, call its handler, and encode the response."""
        rpc_method = _RPC_METHODS_BY_METHOD_NAME[method_name]
        handler = self._handlers.get(rpc_method)
        if handler is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"No handler registered for {method_name}",
            )
            raise RuntimeError("unreachable")

        affinity_key = (
            _grpc_affinity_key(context)
            if isinstance(handler.executor, AffinityThreadPool)
            else 0
        )
        try:
            return handler.run(request, affinity_key)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
            raise RuntimeError("unreachable") from exc

    def __getattr__(self, method_name: str) -> Callable[[Any, Any], Any]:
        if method_name not in _RPC_METHODS_BY_METHOD_NAME:
            raise AttributeError(method_name)

        def rpc_method(request: Any, context: "grpc.ServicerContext") -> Any:
            return self._dispatch(method_name, request, context)

        return rpc_method


# ---------------------------------------------------------------------------
# Server: typed gRPC server
# ---------------------------------------------------------------------------


@dataclass
class _ServerConfig:
    bind_url: str
    max_concurrency: int = 32


class MultiprocessGrpcServer:
    """gRPC server that registers concrete Python gRPC service implementations.

    The normal path is ``add_service()``: callers provide the generated
    protobuf service name and an implementation object with same-named
    CamelCase RPC methods. Direct ``add_handler`` remains available for focused
    transport tests and external compatibility.

    Args:
        bind_url: Either ``grpc://host:port`` or a bare ``host:port``.
        context: Legacy positional slot (used to be zmq.Context); ignored.
        transport: Legacy positional slot; ignored.
        grpc_max_workers: Size of the base grpc thread pool.  Sync
            handlers run here directly; blocking handlers hand off to
            their dedicated thread pool so this executor stays free
            for dispatch and shouldn't need many threads.
    """

    def __init__(
        self,
        bind_url: str,
        context: Optional[Any] = None,
        transport: Optional[Any] = None,
        grpc_max_workers: int = 32,
    ):
        del context, transport  # legacy positional slots, no longer used
        self._bind_url = bind_url
        self._grpc_max_workers = grpc_max_workers
        self.handlers: dict[RpcMethod, GrpcRequestHandler] = {}
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self._server: grpc.Server | None = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Direct handler registration for transport tests.
    # ------------------------------------------------------------------

    def add_handler(
        self,
        request_type: RpcMethod | str,
        *args: Any,
    ) -> None:
        """Register one protobuf method handler.

        Args:
            request_type: Descriptor-derived protobuf method token.
            *args: Either ``(handler,)`` or the legacy test-only
                ``(_payload_classes, handler_type, handler)`` shape.

        Raises:
            TypeError: If arguments are malformed.
            NotImplementedError: If non-blocking handlers are requested.
            ValueError: If the handler type is unknown.
        """
        rpc_method = coerce_rpc_method(request_type)
        if len(args) == 1 and callable(args[0]):
            handler = args[0]
            handler_type, requires_client_affinity = get_grpc_method_options(handler)
        elif len(args) == 3:
            _payload_clss, handler_type, handler = args
            _decorated_type, requires_client_affinity = get_grpc_method_options(handler)
        else:
            raise TypeError(
                "add_handler expects either (rpc_method, handler) or "
                "(rpc_method, payload_clss, handler_type, handler)"
            )

        if not callable(handler):
            raise TypeError("handler must be callable")

        if handler_type is HandlerType.SYNC:
            self.add_sync_handler(rpc_method, handler)
        elif handler_type is HandlerType.BLOCKING:
            self.add_blocking_handler(
                rpc_method,
                handler,
                requires_client_affinity=requires_client_affinity,
            )
        elif handler_type is HandlerType.NON_BLOCKING:
            raise NotImplementedError("Non-blocking handler is not supported yet")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def _register_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
        *,
        handler_type: HandlerType,
        requires_client_affinity: bool,
    ) -> None:
        rpc_method = coerce_rpc_method(request_type)
        request_cls = get_request_message_class(rpc_method)
        response_cls = get_response_message_class(rpc_method)
        request_decoder, payload_types = compile_request_decoder(request_cls, handler)
        response_encoder, response_type = compile_response_encoder(
            response_cls, handler
        )
        self.handlers[rpc_method] = GrpcRequestHandler(
            method_name=str(rpc_method),
            handler=handler,
            handler_type=handler_type,
            requires_client_affinity=requires_client_affinity,
            request_decoder=request_decoder,
            response_encoder=response_encoder,
            payload_types=payload_types,
            response_type=response_type,
        )

    def add_sync_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
    ) -> None:
        self._register_handler(
            request_type,
            handler=handler,
            handler_type=HandlerType.SYNC,
            requires_client_affinity=False,
        )

    def add_blocking_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
        *,
        requires_client_affinity: bool = False,
    ) -> None:
        self._register_handler(
            request_type,
            handler=handler,
            handler_type=HandlerType.BLOCKING,
            requires_client_affinity=requires_client_affinity,
        )

    def add_nonblocking_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
    ) -> None:
        del request_type, handler
        raise NotImplementedError

    def add_service(self, service_name: str, implementation: object) -> None:
        """Register a generated gRPC service implementation.

        Args:
            service_name: Generated protobuf service name, for example
                ``"EngineService"``.
            implementation: Python object implementing every protobuf method
                in the service with the same CamelCase method name.

        Raises:
            ValueError: If ``service_name`` is not in the protobuf descriptor.
            TypeError: If the implementation is missing a required method.
        """
        if service_name not in _RPC_METHODS_BY_SERVICE:
            raise ValueError(f"Unknown gRPC service {service_name!r}")
        for rpc_method in _RPC_METHODS_BY_SERVICE[service_name]:
            handler = getattr(implementation, str(rpc_method), None)
            if not callable(handler):
                raise TypeError(
                    f"{implementation.__class__.__name__} must implement "
                    f"{service_name}.{rpc_method}"
                )
            self.add_handler(rpc_method, handler)

    def assign_thread_pools(
        self,
        *,
        max_cpu_workers: int,
        max_gpu_workers: int,
    ) -> None:
        """Assign thread pools for all currently registered blocking handlers.

        Args:
            max_cpu_workers: Worker count for normal blocking handlers.
            max_gpu_workers: Worker count for affinity-routed blocking handlers.
        """
        mounted_types = list(self.handlers)
        affinity_types = [
            rpc_method
            for rpc_method in mounted_types
            if self.handlers[rpc_method].handler_type is HandlerType.BLOCKING
            and self.handlers[rpc_method].requires_client_affinity
        ]
        normal_types = [
            rpc_method
            for rpc_method in mounted_types
            if self.handlers[rpc_method].handler_type is HandlerType.BLOCKING
            and not self.handlers[rpc_method].requires_client_affinity
        ]
        if affinity_types:
            self.add_affinity_thread_pool(
                affinity_types,
                max_workers=max_gpu_workers,
            )
        if normal_types:
            self.add_normal_thread_pool(
                normal_types,
                max_workers=max_cpu_workers,
            )

    def _validate_blocking_handlers(
        self,
        request_types: Sequence[RpcMethod | str],
        method_name: str,
    ) -> None:
        for request_type in request_types:
            rpc_method = coerce_rpc_method(request_type)
            handler = self.handlers.get(rpc_method)
            if handler is None:
                raise ValueError(
                    f"No handler registered for RPC method: {rpc_method}. "
                    f"Register handlers before calling {method_name}."
                )
            if handler.handler_type is not HandlerType.BLOCKING:
                raise TypeError(
                    f"Handler for {rpc_method} is "
                    f"{handler.handler_type.name}, not BLOCKING."
                )

    def add_normal_thread_pool(
        self,
        request_types: Sequence[RpcMethod | str],
        max_workers: int,
    ) -> None:
        self._validate_blocking_handlers(request_types, "add_normal_thread_pool")
        if not request_types:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            rpc_method = coerce_rpc_method(request_type)
            handler = self.handlers[rpc_method]
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for %s",
            max_workers,
            [coerce_rpc_method(rt).name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: Sequence[RpcMethod | str],
        max_workers: int,
    ) -> None:
        self._validate_blocking_handlers(request_types, "add_affinity_thread_pool")
        if not request_types:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            rpc_method = coerce_rpc_method(request_type)
            handler = self.handlers[rpc_method]
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for %s",
            max_workers,
            [coerce_rpc_method(rt).name for rt in request_types],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        _ensure_grpc_runtime()
        for rt, handler in self.handlers.items():
            if (
                handler.handler_type is HandlerType.BLOCKING
                and handler.executor is None
            ):
                raise RuntimeError(
                    f"Blocking handler for {rt} has no thread pool assigned. "
                    "Call add_normal_thread_pool or add_affinity_thread_pool "
                    "before start()."
                )

        target = _parse_grpc_url(self._bind_url)
        compression = _parse_grpc_compression(self._bind_url)
        server = grpc.server(
            ThreadPoolExecutor(
                max_workers=self._grpc_max_workers,
                thread_name_prefix="mq-grpc-server",
            ),
            options=_GRPC_UNLIMITED_MSG_OPTS,
            compression=compression,
        )
        servicer = _GrpcServicer(self.handlers)
        for service_name in sorted(get_service_names()):
            add_servicer = getattr(
                lmcache_mq_pb2_grpc,
                f"add_{service_name}Servicer_to_server",
            )
            add_servicer(servicer, server)
        server.add_insecure_port(target)
        server.start()
        self._server = server
        logger.info("MultiprocessGrpcServer listening on %s", self._bind_url)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._server is not None:
            self._server.stop(grace=None)
            self._server = None
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
