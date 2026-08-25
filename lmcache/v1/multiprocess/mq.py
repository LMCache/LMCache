# SPDX-License-Identifier: Apache-2.0
"""LMCache mp-mode message queue, backed by gRPC.

Each ``RequestType`` maps to a distinct typed unary RPC method on the
``MessageQueue`` service defined in ``proto/lmcache_mq.proto``. Concurrent
control-plane calls may share a typed micro-batch. The old msgspec envelope
(uid + request type + payload frames) is gone; gRPC method routing and protobuf
request/response messages now define the wire protocol.
"""

# Standard
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
from urllib.parse import parse_qs, urlparse
import inspect
import threading
import time
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
    get_response_class,
)
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
# Amortize one gRPC exchange across concurrent control-plane callers. Single
# callers bypass Batch entirely, so this window does not delay unary traffic.
_GRPC_BATCH_COALESCE_SECONDS = 150 / 1_000_000
_GRPC_BATCH_MAX_ITEMS = 64
_GRPC_BATCH_MAX_ITEM_BYTES = 64 * 1024
_GRPC_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"
# Data-plane and staged transfer operations retain dedicated unary calls because
# they may carry large payloads or depend on per-call affinity and ordering.
_GRPC_BATCH_UNSAFE_REQUEST_TYPES = {
    RequestType.STORE,
    RequestType.STORE_Q,
    RequestType.RETRIEVE,
    RequestType.PREPARE_STORE,
    RequestType.COMMIT_STORE,
    RequestType.PREPARE_RETRIEVE,
    RequestType.COMMIT_RETRIEVE,
    RequestType.CB_STORE_PRE_COMPUTED,
    RequestType.CB_STORE_FINAL,
    RequestType.CB_RETRIEVE_PRE_COMPUTED,
    RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
    RequestType.CB_RETRIEVE_PRE_COMPUTED_V3,
}


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
    request_type: request_type_to_method_name(request_type)
    for request_type in RequestType
}
_RPC_REQUEST_TYPES = {request_type.value: request_type for request_type in RequestType}


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
class _PendingBatchRequest:
    request_type: RequestType
    method_name: str
    stub_method: Any
    spec: TypedRpcSpec
    proto_request: Any
    future: MessagingFuture[Any]


class MessageQueueClient:
    """gRPC-backed client for the LMCache mp cache server.

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
        self._stub = lmcache_mq_pb2_grpc.MessageQueueStub(self._channel)
        self._call_metadata = ((_GRPC_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)
        self._rpc_methods = {
            request_type: (
                _RPC_METHOD_NAMES[request_type],
                getattr(self._stub, _RPC_METHOD_NAMES[request_type]),
                typed_spec,
            )
            for request_type, typed_spec in _TYPED_RPCS.items()
        }
        self._batch_condition = threading.Condition()
        self._batch_queue: deque[_PendingBatchRequest] = deque()
        self._batch_thread: threading.Thread | None = None
        self._inflight_requests = 0
        self._closing = False

    def submit_request(
        self,
        request_type: RequestType,
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
        method_name, stub_method, typed_spec = self._rpc_methods[request_type]
        future: MessagingFuture[T] = MessagingFuture()

        proto_request = typed_spec.python_to_request(*request_payloads)
        pending = _PendingBatchRequest(
            request_type=request_type,
            method_name=method_name,
            stub_method=stub_method,
            spec=typed_spec,
            proto_request=proto_request,
            future=future,
        )

        use_batch = False
        with self._batch_condition:
            if self._closing:
                raise RuntimeError("MessageQueueClient is closed")
            use_batch = (
                self._inflight_requests > 0
                and request_type not in _GRPC_BATCH_UNSAFE_REQUEST_TYPES
                and proto_request.ByteSize() <= _GRPC_BATCH_MAX_ITEM_BYTES
            )
            self._inflight_requests += 1
            if use_batch:
                self._batch_queue.append(pending)
                self._ensure_batch_thread()
                self._batch_condition.notify()

        if not use_batch:
            self._submit_unary(pending)
        return future

    def close(self) -> None:
        with self._batch_condition:
            if self._closing:
                return
            self._closing = True
            self._batch_condition.notify()
            batch_thread = self._batch_thread
        if batch_thread is not None:
            batch_thread.join()
        self._channel.close()

    def _submit_unary(self, pending: _PendingBatchRequest) -> None:
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
        pending: _PendingBatchRequest,
    ) -> None:
        try:
            proto_response = call.result()
        except grpc.RpcError as exc:
            self._finish_requests(1)
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
            self._finish_requests(1)
            logger.exception("gRPC call %s failed", pending.method_name)
            pending.future.set_exception(exc)
            return

        try:
            decoded = pending.spec.response_to_python(proto_response)
        except Exception as exc:
            self._finish_requests(1)
            logger.exception(
                "failed to decode typed response for %s", pending.method_name
            )
            pending.future.set_exception(exc)
            return
        self._finish_requests(1)
        pending.future.set_result(decoded)

    def _ensure_batch_thread(self) -> None:
        if self._batch_thread is not None:
            return
        self._batch_thread = threading.Thread(
            target=self._batch_loop,
            daemon=True,
            name="mq-grpc-batch",
        )
        self._batch_thread.start()

    def _batch_loop(self) -> None:
        while True:
            with self._batch_condition:
                self._batch_condition.wait_for(
                    lambda: self._batch_queue or self._closing
                )
                if self._closing and not self._batch_queue:
                    return

            time.sleep(_GRPC_BATCH_COALESCE_SECONDS)
            with self._batch_condition:
                count = min(len(self._batch_queue), _GRPC_BATCH_MAX_ITEMS)
                batch = [self._batch_queue.popleft() for _ in range(count)]

            request = lmcache_mq_pb2.BatchRequest()
            for pending in batch:
                item = request.items.add()
                item.method_id = pending.request_type.value
                item.payload = pending.proto_request.SerializeToString()

            def _on_done_batch(
                call: "grpc.Future[Any]",
                _batch: list[_PendingBatchRequest] = batch,
            ) -> None:
                self._on_batch_done(call, _batch)

            call = self._stub.Batch.future(
                request,
                metadata=self._call_metadata,
                wait_for_ready=True,
            )
            call.add_done_callback(_on_done_batch)

    def _on_batch_done(
        self,
        call: "grpc.Future[Any]",
        batch: list[_PendingBatchRequest],
    ) -> None:
        try:
            response = call.result()
        except grpc.RpcError as exc:
            if exc.code() is grpc.StatusCode.UNIMPLEMENTED:
                for pending in batch:
                    self._submit_unary(pending)
                return
            self._finish_requests(len(batch))
            if exc.code() is grpc.StatusCode.UNAVAILABLE:
                logger.warning(
                    "gRPC batch lost its server and %d requests remain pending: %s",
                    len(batch),
                    exc,
                )
                return
            logger.error("gRPC batch failed: %s", exc)
            for pending in batch:
                pending.future.set_exception(exc)
            return
        except Exception as exc:  # defensive
            self._finish_requests(len(batch))
            logger.exception("gRPC batch failed")
            for pending in batch:
                pending.future.set_exception(exc)
            return

        if len(response.items) != len(batch):
            response_count_error = RuntimeError(
                f"gRPC batch returned {len(response.items)} responses "
                f"for {len(batch)} requests"
            )
            self._finish_requests(len(batch))
            for pending in batch:
                pending.future.set_exception(response_count_error)
            return

        outcomes: list[tuple[_PendingBatchRequest, Any, BaseException | None]] = []
        for pending, item in zip(batch, response.items, strict=True):
            result_field = item.WhichOneof("result")
            if result_field == "error":
                item_error = RuntimeError(
                    f"batched gRPC call {pending.method_name} failed "
                    f"with {item.error.status}: {item.error.details}"
                )
                outcomes.append((pending, None, item_error))
                continue
            if result_field != "payload":
                response_type_error = RuntimeError(
                    f"batched gRPC call {pending.method_name} returned "
                    f"an invalid result {result_field!r}"
                )
                outcomes.append((pending, None, response_type_error))
                continue
            try:
                proto_response = pending.spec.response_message()
                proto_response.ParseFromString(item.payload)
                decoded = pending.spec.response_to_python(proto_response)
            except Exception as exc:
                outcomes.append((pending, None, exc))
            else:
                outcomes.append((pending, decoded, None))

        self._finish_requests(len(batch))
        for pending, decoded, outcome_error in outcomes:
            if outcome_error is not None:
                pending.future.set_exception(outcome_error)
            else:
                pending.future.set_result(decoded)

    def _finish_requests(self, count: int) -> None:
        with self._batch_condition:
            self._inflight_requests -= count
            assert self._inflight_requests >= 0


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
# Server: gRPC servicer bridging RequestType -> RequestHandlerBase
# ---------------------------------------------------------------------------


class _RequestHandlerServicer:
    """Bridge every rpc method to the ``RequestHandlerBase`` registered
    under the matching ``RequestType``.

    Each generated method just calls :meth:`_dispatch` with the right
    ``RequestType``; keeping one implementation avoids 36 near-identical
    thunks in this file.  gRPC's method routing already runs before we
    get here, so ``_dispatch`` is the whole request path.
    """

    def __init__(
        self,
        handlers: dict[RequestType, RequestHandlerBase[Any]],
    ):
        self._handlers = handlers

    def Batch(self, request: Any, context: "grpc.ServicerContext") -> Any:
        """Execute a typed micro-batch while preserving response order."""
        if len(request.items) > _GRPC_BATCH_MAX_ITEMS:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"batch has {len(request.items)} items; "
                f"maximum is {_GRPC_BATCH_MAX_ITEMS}",
            )
        response = lmcache_mq_pb2.BatchResponse()
        executions: list[tuple[Any, str, TypedRpcSpec, Any]] = []
        affinity_key = _grpc_affinity_key(context)

        for request_item in request.items:
            response_item = response.items.add()
            method_id = request_item.method_id
            if not method_id:
                self._set_batch_error(
                    response_item,
                    "INVALID_ARGUMENT",
                    "batch request item has no method id",
                )
                continue
            request_type = _RPC_REQUEST_TYPES.get(method_id)
            if request_type is None:
                self._set_batch_error(
                    response_item,
                    "INVALID_ARGUMENT",
                    f"unknown batch method id {method_id}",
                )
                continue
            method_name = _RPC_METHOD_NAMES[request_type]

            handler = self._handlers.get(request_type)
            if handler is None:
                self._set_batch_error(
                    response_item,
                    "UNIMPLEMENTED",
                    f"No handler registered for {request_type}",
                )
                continue

            spec = _TYPED_RPCS[request_type]
            try:
                proto_request = spec.request_message()
                proto_request.ParseFromString(request_item.payload)
                py_args = spec.request_to_python(proto_request)
                execution = self._submit_handler(handler, py_args, affinity_key)
            except Exception as exc:
                logger.exception("failed to start batched %s", request_type)
                self._set_batch_error(
                    response_item,
                    "UNKNOWN",
                    str(exc),
                )
                continue
            executions.append((response_item, method_name, spec, execution))

        for response_item, method_name, spec, execution in executions:
            try:
                result = (
                    execution.result() if isinstance(execution, Future) else execution
                )
                proto_response = spec.python_to_response(result)
                response_item.payload = proto_response.SerializeToString()
            except Exception as exc:
                logger.exception("failed to complete batched %s", method_name)
                self._set_batch_error(
                    response_item,
                    "UNKNOWN",
                    str(exc),
                )

        return response

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
        request_type: RequestType,
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

    @staticmethod
    def _set_batch_error(item: Any, status: str, details: str) -> None:
        item.error.status = status
        item.error.details = details


def _install_servicer_methods() -> None:
    """Attach one typed dispatch method per ``RequestType`` to the servicer."""
    for rt in RequestType:
        method_name = _RPC_METHOD_NAMES[rt]
        typed_spec = _TYPED_RPCS[rt]
        method: Callable[..., Any]
        _resolved_spec: TypedRpcSpec = typed_spec

        def _typed_method(  # noqa: E501 (captured ``rt`` / ``spec`` via default arg)
            self: _RequestHandlerServicer,
            request: Any,
            context: "grpc.ServicerContext",
            _rt: RequestType = rt,
            _spec: TypedRpcSpec = _resolved_spec,
        ) -> Any:
            return self._dispatch_typed(request, context, _rt, _spec)

        method = _typed_method
        method.__name__ = method_name
        method.__qualname__ = f"_RequestHandlerServicer.{method_name}"
        setattr(_RequestHandlerServicer, method_name, method)


_install_servicer_methods()


# ---------------------------------------------------------------------------
# Server: public MessageQueueServer API preserved
# ---------------------------------------------------------------------------


@dataclass
class _ServerConfig:
    bind_url: str
    max_concurrency: int = 32


class MessageQueueServer:
    """gRPC server that wraps ``RequestHandlerBase`` instances.

    Public API mirrors the historical zmq-backed one so no module needs
    to change: ``add_handler`` / ``add_normal_thread_pool`` /
    ``add_affinity_thread_pool`` / ``start`` / ``close`` all keep their
    old semantics.

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
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self._server: grpc.Server | None = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Handler registration (identical semantics to the old zmq server)
    # ------------------------------------------------------------------

    def _inspect_handler_signature(
        self, request_type: RequestType, handler: Callable[..., Any]
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

        payload_clss = get_payload_classes(request_type)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d args, but got %d",
                request_type,
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
                    request_type,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(request_type)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return %s, got %s",
                request_type,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler: Callable[..., Any],
    ) -> None:
        if not self._inspect_handler_signature(request_type, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {request_type}"
            )

        if handler_type is HandlerType.SYNC:
            self.add_sync_handler(request_type, payload_clss, handler)
        elif handler_type is HandlerType.BLOCKING:
            self.add_blocking_handler(request_type, payload_clss, handler)
        elif handler_type is HandlerType.NON_BLOCKING:
            raise NotImplementedError("Non-blocking handler is not supported yet")
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = SyncRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_blocking_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = BlockingRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler: Callable[..., Any],
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        request_types: list[RequestType],
        method_name: str,
    ) -> None:
        for request_type in request_types:
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(
                    f"No handler registered for request type: {request_type}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {request_type} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler."
                )

    def add_normal_thread_pool(
        self,
        request_types: list[RequestType],
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
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: list[RequestType],
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
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for %s",
            max_workers,
            [rt.name for rt in request_types],
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
        lmcache_mq_pb2_grpc.add_MessageQueueServicer_to_server(servicer, server)
        server.add_insecure_port(target)
        server.start()
        self._server = server
        logger.info("MessageQueueServer listening on %s (gRPC)", self._bind_url)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._server is not None:
            self._server.stop(grace=None)
            self._server = None
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
