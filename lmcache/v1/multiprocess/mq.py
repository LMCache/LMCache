# SPDX-License-Identifier: Apache-2.0
"""LMCache mp-mode gRPC transport.

Each RPC method maps to a distinct typed unary gRPC method on one of the
services defined in ``proto/lmcache_mq.proto``. Concurrent control-plane calls
go directly over typed unary RPCs. The old msgspec envelope (uid + request type
+ payload frames) is gone; gRPC service/method routing and protobuf
request/response messages now define the wire protocol.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, get_type_hints
from urllib.parse import parse_qs, urlparse
import inspect
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
    get_handler_type,
    get_payload_classes,
    get_response_class,
    requires_client_affinity,
)
from lmcache.v1.multiprocess.service import MPService
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    TYPED_RPCS as _TYPED_RPCS,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    TypedRpcSpec,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    msgspec_decode as msgspec_decode,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    msgspec_encode as msgspec_encode,
)
from lmcache.v1.multiprocess.transport.grpc_impl.typed_rpc import (
    request_type_to_method_name as request_type_to_method_name,
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
_RPC_METHODS_BY_SERVICE: dict[str, tuple[RpcMethod, ...]] = {
    service_name: tuple(
        rpc_method
        for rpc_method in RPC_METHODS
        if rpc_method.service_name == service_name
    )
    for service_name in {rpc_method.service_name for rpc_method in RPC_METHODS}
}


def _service_handler_name(service: MPService, rpc_method: RpcMethod) -> str | None:
    """Resolve the Python handler name for a service/rpc pair."""
    service_type = type(service)
    skipped: frozenset[str] = getattr(service_type, "GRPC_SKIP_METHODS", frozenset())
    method_name = str(rpc_method)
    if method_name in skipped or rpc_method.name in skipped:
        return None
    aliases: dict[str, str] = getattr(service_type, "GRPC_METHOD_ALIASES", {})
    return aliases.get(method_name, rpc_method.name.lower())


def _has_declared_handler(service: MPService, handler_name: str) -> bool:
    """Return True only for methods declared by the concrete service class."""
    try:
        inspect.getattr_static(service, handler_name)
    except AttributeError:
        return False
    return callable(getattr(service, handler_name))


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
    spec: TypedRpcSpec
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
            for service_name in {
                typed_spec.service_name for typed_spec in _TYPED_RPCS.values()
            }
        }
        self._call_metadata = ((_GRPC_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)
        self._rpc_methods = {
            rpc_method: (
                _RPC_METHOD_NAMES[rpc_method],
                getattr(
                    self._stubs[typed_spec.service_name],
                    _RPC_METHOD_NAMES[rpc_method],
                ),
                typed_spec,
            )
            for rpc_method, typed_spec in _TYPED_RPCS.items()
        }

    def submit_request(
        self,
        request_type: RpcMethod | str,
        request_payloads: list[Any],
        response_cls: Optional[T] = None,
    ) -> MessagingFuture[T]:
        """Submit a request and return a future for its response.

        Args:
            request_type: Which RPC to invoke.
            request_payloads: Positional payloads matching
                ``get_payload_classes(request_type)``.
            response_cls: Kept for signature compatibility; ignored
                (the response class is resolved from ``request_type``).

        Returns:
            A ``MessagingFuture`` completed by the gRPC callback.
        """
        del response_cls
        rpc_method = coerce_rpc_method(request_type)
        method_name, stub_method, typed_spec = self._rpc_methods[rpc_method]
        future: MessagingFuture[T] = MessagingFuture()

        proto_request = typed_spec.python_to_request(*request_payloads)
        pending = _PendingUnaryRequest(
            rpc_method=rpc_method,
            method_name=method_name,
            stub_method=stub_method,
            spec=typed_spec,
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
            decoded = pending.spec.response_to_python(proto_response)
        except Exception as exc:
            logger.exception(
                "failed to decode typed response for %s", pending.method_name
            )
            pending.future.set_exception(exc)
            return
        pending.future.set_result(decoded)


# ---------------------------------------------------------------------------
# Server: RequestHandlerBase + concrete handler types (unchanged interface)
# ---------------------------------------------------------------------------


ResponseType = TypeVar("ResponseType", covariant=True)
StateType = TypeVar("StateType", covariant=True)


class RequestHandlerBase(Generic[ResponseType]):
    def __call__(self, payloads: tuple[Any, ...]):
        raise NotImplementedError

    def get_response_class(self) -> ResponseType:
        raise NotImplementedError

    def get_handler_type(self) -> HandlerType:
        raise NotImplementedError


class SyncRequestHandler(RequestHandlerBase[ResponseType]):
    """Handler that runs in the calling grpc worker thread."""

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.payload_clss = payload_clss
        self.response_cls = response_cls
        self.handler = handler

    def __call__(self, payloads: tuple[Any, ...]) -> ResponseType:
        return self.handler(*payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.SYNC


class BlockingRequestHandler(RequestHandlerBase[ResponseType]):
    """Handler dispatched to a dedicated thread pool (normal or affinity)."""

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
        self, payloads: tuple[Any, ...], affinity_key: Any = 0
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "BlockingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler, *payloads, affinity_key=affinity_key
            )
        return self.executor.submit(self.handler, *payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.BLOCKING


class NonBlockingRequestHandler(Generic[ResponseType, StateType]):
    """Reserved for future async handlers; not currently instantiated."""

    pass


# ---------------------------------------------------------------------------
# Server: gRPC servicer bridging RpcMethod -> RequestHandlerBase
# ---------------------------------------------------------------------------


class _RequestHandlerServicer:
    """Bridge every rpc method to its registered ``RequestHandlerBase``."""

    def __init__(
        self,
        handlers: dict[RpcMethod, RequestHandlerBase[Any]],
    ):
        self._handlers = handlers

    @staticmethod
    def _submit_handler(
        handler: RequestHandlerBase[Any],
        payloads: tuple[Any, ...],
        affinity_key: int,
    ) -> Any:
        handler_type = handler.get_handler_type()
        if handler_type is HandlerType.SYNC:
            assert isinstance(handler, SyncRequestHandler)
            return handler(payloads)
        if handler_type is HandlerType.BLOCKING:
            assert isinstance(handler, BlockingRequestHandler)
            return handler(payloads, affinity_key=affinity_key)
        raise NotImplementedError(f"handler_type {handler_type} not supported")

    def _run_handler(
        self,
        handler: RequestHandlerBase[Any],
        payloads: tuple[Any, ...],
        context: "grpc.ServicerContext",
    ) -> Any:
        """Route typed Python payloads into the registered request handler."""
        affinity_key = 0
        if isinstance(handler, BlockingRequestHandler) and isinstance(
            handler.executor, AffinityThreadPool
        ):
            # Stable client metadata keeps old DEALER affinity semantics.
            affinity_key = _grpc_affinity_key(context)
        execution = self._submit_handler(handler, payloads, affinity_key)
        return execution.result() if isinstance(execution, Future) else execution

    def _dispatch_typed(
        self,
        request: Any,
        context: "grpc.ServicerContext",
        request_type: RpcMethod,
        spec: TypedRpcSpec,
    ) -> Any:
        """Decode a typed request, run its handler, and encode the response."""
        handler = self._handlers.get(request_type)
        if handler is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"No handler registered for {request_type}",
            )
            raise RuntimeError("unreachable")

        py_args = spec.request_to_python(request)
        result = self._run_handler(handler, py_args, context)
        return spec.python_to_response(result)


def _install_servicer_methods() -> None:
    """Attach one typed dispatch method per RpcMethod to the servicer."""
    for rt in RPC_METHODS:
        method_name = _RPC_METHOD_NAMES[rt]
        typed_spec = _TYPED_RPCS[rt]
        method: Callable[..., Any]
        _resolved_spec: TypedRpcSpec = typed_spec

        def _typed_method(  # noqa: E501 (captured ``rt`` / ``spec`` via default arg)
            self: _RequestHandlerServicer,
            request: Any,
            context: "grpc.ServicerContext",
            _rt: RpcMethod = rt,
            _spec: TypedRpcSpec = _resolved_spec,
        ) -> Any:
            return self._dispatch_typed(request, context, _rt, _spec)

        method = _typed_method
        method.__name__ = method_name
        method.__qualname__ = f"_RequestHandlerServicer.{method_name}"
        setattr(_RequestHandlerServicer, method_name, method)


_install_servicer_methods()


# ---------------------------------------------------------------------------
# Server: typed gRPC server
# ---------------------------------------------------------------------------


@dataclass
class _ServerConfig:
    bind_url: str
    max_concurrency: int = 32


class MultiprocessGrpcServer:
    """gRPC server that mounts descriptor-derived service methods.

    The normal path is ``mount_services()``: services declare generated gRPC
    service names, and the transport discovers the Python handler method for
    each protobuf method from that descriptor. Direct ``add_handler`` remains
    available for focused transport tests and external compatibility.

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
        self.handlers: dict[RpcMethod, RequestHandlerBase[Any]] = {}
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self._server: grpc.Server | None = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Direct handler registration for transport tests and compatibility.
    # ------------------------------------------------------------------

    def _inspect_handler_signature(
        self, request_type: RpcMethod | str, handler: Callable[..., Any]
    ) -> bool:
        """Verify a handler's parameter / return annotations match the
        registered ``ProtocolDefinition``.

        Returns:
            True if the signature matches or the annotations are omitted
            in a way that keeps us backwards compatible; False otherwise.
        """

        def same_type(a: Any, b: Any) -> bool:
            if a is None:
                a = type(None)
            if b is None:
                b = type(None)
            return a == b

        rpc_method = coerce_rpc_method(request_type)
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

        payload_clss = get_payload_classes(rpc_method)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d args, but got %d",
                rpc_method,
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
                    "Handler for %s arg %d expects %s, got %s",
                    rpc_method,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(rpc_method)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return %s, got %s",
                rpc_method,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RpcMethod | str,
        *args: Any,
    ) -> None:
        rpc_method = coerce_rpc_method(request_type)
        if len(args) == 1 and callable(args[0]):
            handler = args[0]
            payload_clss = get_payload_classes(rpc_method)
            handler_type = get_handler_type(rpc_method)
        elif len(args) == 3:
            payload_clss, handler_type, handler = args
        else:
            raise TypeError(
                "add_handler expects either (rpc_method, handler) or "
                "(rpc_method, payload_clss, handler_type, handler)"
            )

        if not callable(handler):
            raise TypeError("handler must be callable")
        if not self._inspect_handler_signature(rpc_method, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {rpc_method}"
            )

        if handler_type is HandlerType.SYNC:
            self.add_sync_handler(rpc_method, handler)
        elif handler_type is HandlerType.BLOCKING:
            self.add_blocking_handler(rpc_method, handler)
        elif handler_type is HandlerType.NON_BLOCKING:
            raise NotImplementedError("Non-blocking handler is not supported yet")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
    ) -> None:
        rpc_method = coerce_rpc_method(request_type)
        response_cls = get_response_class(rpc_method)
        self.handlers[rpc_method] = SyncRequestHandler(
            get_payload_classes(rpc_method), response_cls, handler
        )

    def add_blocking_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
    ) -> None:
        rpc_method = coerce_rpc_method(request_type)
        response_cls = get_response_class(rpc_method)
        self.handlers[rpc_method] = BlockingRequestHandler(
            get_payload_classes(rpc_method), response_cls, handler
        )

    def add_nonblocking_handler(
        self,
        request_type: RpcMethod | str,
        handler: Callable[..., Any],
    ) -> None:
        del request_type, handler
        raise NotImplementedError

    def mount_services(
        self,
        services: Sequence[MPService],
        *,
        max_cpu_workers: int,
        max_gpu_workers: int,
    ) -> None:
        """Mount RPC handlers implemented by service objects.

        Services declare generated gRPC service names through
        ``GRPC_SERVICE_NAMES``. For each method in those descriptors, the
        transport binds a same-named Python method using the lower-case
        request name (``P2P_LOOKUP_AND_LOCK`` -> ``p2p_lookup_and_lock``),
        with optional class-level aliases for Python method names that
        intentionally differ.

        Args:
            services: Service objects implementing generated gRPC operations.
            max_cpu_workers: Worker count for normal blocking handlers.
            max_gpu_workers: Worker count for affinity-routed blocking handlers.
        """
        mounted_types: list[RpcMethod] = []
        mounted_seen: set[RpcMethod] = set()
        for service in services:
            for service_name in getattr(type(service), "GRPC_SERVICE_NAMES", ()):
                if service_name not in _RPC_METHODS_BY_SERVICE:
                    raise ValueError(f"Unknown gRPC service {service_name!r}")
                for rpc_method in _RPC_METHODS_BY_SERVICE[service_name]:
                    handler_name = _service_handler_name(service, rpc_method)
                    if handler_name is None or not _has_declared_handler(
                        service, handler_name
                    ):
                        continue
                    self.add_handler(rpc_method, getattr(service, handler_name))
                    if rpc_method not in mounted_seen:
                        mounted_types.append(rpc_method)
                        mounted_seen.add(rpc_method)

        affinity_types = [
            rpc_method
            for rpc_method in mounted_types
            if get_handler_type(rpc_method) is HandlerType.BLOCKING
            and requires_client_affinity(rpc_method)
        ]
        normal_types = [
            rpc_method
            for rpc_method in mounted_types
            if get_handler_type(rpc_method) is HandlerType.BLOCKING
            and not requires_client_affinity(rpc_method)
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
                    f"No handler registered for request type: {rpc_method}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {rpc_method} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler."
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
            assert isinstance(handler, BlockingRequestHandler)
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
            assert isinstance(handler, BlockingRequestHandler)
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
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    "assigned. Call add_normal_thread_pool or "
                    "add_affinity_thread_pool before start()."
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
        servicer = _RequestHandlerServicer(self.handlers)
        for service_name in sorted(
            {typed_spec.service_name for typed_spec in _TYPED_RPCS.values()}
        ):
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


# Compatibility aliases for existing public imports. New code should use the
# gRPC names above; these aliases intentionally carry no separate behavior.
MessageQueueClient = MultiprocessGrpcClient
MessageQueueServer = MultiprocessGrpcServer
