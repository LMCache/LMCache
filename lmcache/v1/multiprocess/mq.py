# SPDX-License-Identifier: Apache-2.0
"""LMCache mp-mode message queue, backed by gRPC.

Each ``RequestType`` maps to a distinct unary rpc method on the
``MessageQueue`` service defined in ``proto/lmcache_mq.proto`` -- the
old msgspec envelope (uid + request_type frame + payloads) is gone and
gRPC's method routing takes over.  The request/response payload bytes
themselves still carry msgspec-encoded values today, so the surrounding
handler / client business code keeps the same signatures; a follow-up
PR can promote individual rpc methods to typed proto messages without
touching this file.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
from urllib.parse import urlparse
import inspect
import threading

# Third Party
import grpc
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
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
    lmcache_mq_pb2,
    lmcache_mq_pb2_grpc,
)

logger = init_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Typed rpc registry (proto messages as first-class citizens).
#
# Each entry says "for this RequestType, don't touch the msgspec envelope
# -- serialize / deserialize through these two typed proto messages
# directly".  Migrating an rpc off the legacy BytesRequest / BytesResponse
# envelope is a matter of:
#
#   1. Add a real message pair to ``lmcache_mq.proto`` (see PingRequest /
#      PingResponse) and change the rpc to use them.
#   2. Add one entry to this dict wiring the RequestType to those
#      messages and the two small Python <-> proto adapters.
#
# The adapters intentionally stay next to the registry (rather than in
# the business handler) so the whole "typed-vs-legacy" decision surface
# lives in one file that grep's cheap to audit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedRpcSpec:
    """Metadata describing a typed rpc.

    Attributes:
        request_message: The generated proto message class for the
            request; instances of it hit the wire directly.
        response_message: Same for the response.
        request_to_python: Servicer-side unpack.  Turns the incoming
            proto request into the positional Python arguments the
            handler function expects.  The tuple's shape must match
            ``protocol.get_payload_classes(request_type)``.
        python_to_request: Client-side pack.  Inverse of
            ``request_to_python`` -- takes the same positional Python
            arguments and builds the proto request that hits the wire.
        python_to_response: Servicer-side pack.  Turns the handler's
            Python return value into the proto response.
        response_to_python: Client-side unpack.  Inverse of
            ``python_to_response`` -- turns the proto response back
            into the Python value the caller expects (or ``None``).
    """

    request_message: Any
    response_message: Any
    request_to_python: Callable[[Any], tuple[Any, ...]]
    python_to_request: Callable[..., Any]
    python_to_response: Callable[[Any], Any]
    response_to_python: Callable[[Any], Any]


def _ping_request_to_python(req: "lmcache_mq_pb2.PingRequest") -> tuple[Any, ...]:
    # ``-1`` on the wire is the sentinel for "untracked prober" (the
    # legacy msgspec path used ``None`` for the same case).
    instance_id: Optional[int] = None if req.instance_id == -1 else req.instance_id
    return (instance_id,)


def _ping_python_to_request(instance_id: Optional[int]) -> "lmcache_mq_pb2.PingRequest":
    wire_id = -1 if instance_id is None else instance_id
    return lmcache_mq_pb2.PingRequest(instance_id=wire_id)


def _ping_python_to_response(result: Any) -> "lmcache_mq_pb2.PingResponse":
    return lmcache_mq_pb2.PingResponse(ok=bool(result))


def _ping_response_to_python(resp: "lmcache_mq_pb2.PingResponse") -> bool:
    return bool(resp.ok)


# ---------------------------------------------------------------------------
# msgspec encode / decode helpers (payload bytes wrapped inside proto)
# ---------------------------------------------------------------------------

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
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


def unwrap_request_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> list[Any]:
    if len(b_payloads) != len(payload_clss):
        raise ValueError("Payload count does not match expected count")

    return [
        msgspec_decode(payload, cls=cls)
        for payload, cls in zip(b_payloads, payload_clss, strict=False)
    ]


# The one source of truth for "which RequestType has been promoted to a
# typed proto message pair".  Entries here take priority over the
# msgspec-envelope path in both ``submit_request`` and the servicer.
_TYPED_RPCS: dict[RequestType, TypedRpcSpec] = {
    RequestType.PING: TypedRpcSpec(
        request_message=lmcache_mq_pb2.PingRequest,
        response_message=lmcache_mq_pb2.PingResponse,
        request_to_python=_ping_request_to_python,
        python_to_request=_ping_python_to_request,
        python_to_response=_ping_python_to_response,
        response_to_python=_ping_response_to_python,
    ),
}


# ---------------------------------------------------------------------------
# RequestType <-> gRPC method name
# ---------------------------------------------------------------------------


def request_type_to_method_name(request_type: RequestType) -> str:
    """Return the CamelCase gRPC method name for a ``RequestType``.

    ``STORE`` -> ``Store``; ``CB_LOOKUP_PRE_COMPUTED_V2`` ->
    ``CbLookupPreComputedV2``; ``P2P_LOOKUP_AND_LOCK`` ->
    ``P2PLookupAndLock``.  These names are baked into ``lmcache_mq.proto``
    so any drift shows up immediately at handshake time.
    """
    parts = request_type.name.split("_")
    out: list[str] = []
    for part in parts:
        if part == "P2P":
            out.append("P2P")
        else:
            out.append(part[:1].upper() + part[1:].lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def _parse_grpc_url(url: str) -> str:
    """Return a ``host:port`` target that ``grpc.insecure_channel`` accepts.

    Accepts ``grpc://host:port`` or a bare ``host:port``.  Any other
    transport scheme (``tcp://`` / ``ipc://`` / etc.) is rejected up front
    now that gRPC is the only supported transport.
    """
    if "://" not in url:
        return url
    parsed = urlparse(url)
    if parsed.scheme != "grpc":
        raise ValueError(
            f"unsupported transport scheme {parsed.scheme!r} for url {url!r}; "
            f"only grpc:// (or a bare host:port) is supported"
        )
    if not parsed.netloc:
        raise ValueError(f"missing host in url {url!r}")
    return parsed.netloc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MessageQueueClient:
    """gRPC-backed client for the LMCache mp cache server.

    Instances are cheap; a shared ``grpc.Channel`` is created per client
    and callers can share one client across many threads (gRPC channels
    are thread-safe).

    Args:
        server_url: Either ``grpc://host:port`` or a bare ``host:port``.
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
        target = _parse_grpc_url(server_url)
        self._server_url = server_url
        self._channel = grpc.insecure_channel(target)
        self._stub = lmcache_mq_pb2_grpc.MessageQueueStub(self._channel)

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
        method_name = request_type_to_method_name(request_type)
        stub_method = getattr(self._stub, method_name)
        future: MessagingFuture[T] = MessagingFuture()

        typed_spec = _TYPED_RPCS.get(request_type)
        if typed_spec is not None:
            # Typed path: build the proto request from positional Python
            # args directly; no msgspec envelope on the wire.
            proto_request = typed_spec.python_to_request(*request_payloads)

            def _on_done_typed(call: "grpc.Future[Any]") -> None:
                try:
                    proto_response = call.result()
                except grpc.RpcError as exc:
                    logger.error("gRPC call %s failed: %s", method_name, exc)
                    future.set_result(None)  # type: ignore[arg-type]
                    return
                except Exception:  # defensive
                    logger.exception("gRPC call %s failed", method_name)
                    future.set_result(None)  # type: ignore[arg-type]
                    return
                try:
                    decoded = typed_spec.response_to_python(proto_response)
                except Exception:
                    logger.exception(
                        "failed to decode typed response for %s", method_name
                    )
                    future.set_result(None)  # type: ignore[arg-type]
                    return
                future.set_result(decoded)

            call = stub_method.future(proto_request)
            call.add_done_callback(_on_done_typed)
            return future

        # Legacy msgspec-envelope path (to be retired once every rpc
        # has an entry in ``_TYPED_RPCS``).
        payload_classes = get_payload_classes(request_type)
        if len(payload_classes) != len(request_payloads):
            expected = [cls.__name__ for cls in payload_classes]
            actual = [type(p).__name__ for p in request_payloads]
            raise ValueError(
                "Payload count mismatch for request "
                f"{request_type}: expected {len(payload_classes)} {expected}, "
                f"got {len(request_payloads)} {actual}. Likely a version "
                "skew between the lmcache client and server."
            )
        b_payloads = [
            msgspec_encode(payload, cls=cls)
            for payload, cls in zip(request_payloads, payload_classes, strict=False)
        ]
        # A rpc's request is opaque bytes on the wire so we flatten
        # any multi-payload request with msgpack (matches the daemon
        # side's ``unwrap_request_payloads`` reconstruction).
        wire = msgspec.msgpack.encode(b_payloads)
        proto_request = lmcache_mq_pb2.BytesRequest(payload=wire)
        response_type = get_response_class(request_type)

        def _on_done(call: "grpc.Future[lmcache_mq_pb2.BytesResponse]") -> None:
            try:
                proto_response = call.result()
            except grpc.RpcError as exc:
                logger.error("gRPC call %s failed: %s", method_name, exc)
                future.set_result(None)  # type: ignore[arg-type]
                return
            except Exception:  # defensive
                logger.exception("gRPC call %s failed", method_name)
                future.set_result(None)  # type: ignore[arg-type]
                return
            if response_type is None or not proto_response.result:
                future.set_result(None)  # type: ignore[arg-type]
                return
            try:
                decoded = msgspec_decode(proto_response.result, cls=response_type)
            except Exception:  # decoding failed
                logger.exception("failed to decode response for %s", method_name)
                future.set_result(None)  # type: ignore[arg-type]
                return
            future.set_result(decoded)

        call = stub_method.future(proto_request)
        call.add_done_callback(_on_done)
        return future

    def close(self) -> None:
        self._channel.close()


# ---------------------------------------------------------------------------
# Server: RequestHandlerBase + concrete handler types (unchanged interface)
# ---------------------------------------------------------------------------


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

    def __call__(self, payloads: list[bytes]) -> ResponseType:
        return self.handler(*unwrap_request_payloads(payloads, self.payload_clss))

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
        self, payloads: list[bytes], affinity_key: Any = 0
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
    """Reserved for future async handlers; not currently instantiated."""

    pass


# ---------------------------------------------------------------------------
# Server: gRPC servicer bridging RequestType -> RequestHandlerBase
# ---------------------------------------------------------------------------


class _RequestHandlerServicer(lmcache_mq_pb2_grpc.MessageQueueServicer):
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

    def _run_handler(
        self,
        request_type: RequestType,
        payloads: list[bytes],
        peer: str,
    ) -> Any:
        """Route a legacy-envelope payload list into the registered
        ``RequestHandlerBase`` and return the raw Python result.

        Split out of the msgspec path so the typed path can share the
        same executor / affinity dispatch without duplicating it.
        """
        handler = self._handlers.get(request_type)
        if handler is None:
            raise RuntimeError(f"No handler registered for {request_type}")

        handler_type = handler.get_handler_type()
        if handler_type is HandlerType.SYNC:
            assert isinstance(handler, SyncRequestHandler)
            return handler(payloads)
        if handler_type is HandlerType.BLOCKING:
            assert isinstance(handler, BlockingRequestHandler)
            # Peer id keeps the same affinity semantics as the old zmq
            # DEALER-ROUTER identity: one thread per client, forever.
            fut = handler(payloads, affinity_key=hash(peer))
            return fut.result()
        raise NotImplementedError(f"handler_type {handler_type} not supported")

    def _dispatch(
        self,
        request: lmcache_mq_pb2.BytesRequest,
        context: "grpc.ServicerContext",
        request_type: RequestType,
    ) -> lmcache_mq_pb2.BytesResponse:
        handler = self._handlers.get(request_type)
        if handler is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"No handler registered for {request_type}",
            )
            # ``context.abort`` raises, so this is unreachable; kept as
            # an explicit fall-through for mypy's control-flow analysis.
            raise RuntimeError("unreachable")
        payloads: list[bytes] = msgspec.msgpack.decode(request.payload, type=list)
        response_cls = handler.get_response_class()
        result = self._run_handler(request_type, payloads, context.peer())

        if result is None:
            return lmcache_mq_pb2.BytesResponse(result=b"")
        b_result = msgspec_encode(result, cls=response_cls)
        return lmcache_mq_pb2.BytesResponse(result=b_result)

    def _dispatch_typed(
        self,
        request: Any,
        context: "grpc.ServicerContext",
        request_type: RequestType,
        spec: TypedRpcSpec,
    ) -> Any:
        """Typed-rpc entry point.  Shares the executor / affinity logic
        with ``_dispatch`` via ``_run_handler``; the only difference is
        the wire format on either end.

        The registered ``RequestHandlerBase`` still speaks the msgspec
        payload-list ABI internally (business handlers haven't changed),
        so we re-encode the unpacked positional args back to msgspec
        bytes here.  That's a temporary crutch -- once every rpc is
        typed, ``RequestHandlerBase`` itself will lose the ``list[bytes]``
        parameter and take positional Python args directly.
        """
        handler = self._handlers.get(request_type)
        if handler is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"No handler registered for {request_type}",
            )
            raise RuntimeError("unreachable")

        py_args = spec.request_to_python(request)
        payload_classes = get_payload_classes(request_type)
        if len(py_args) != len(payload_classes):
            context.abort(
                grpc.StatusCode.INTERNAL,
                (
                    f"typed rpc {request_type} produced {len(py_args)} args, "
                    f"but protocol expects {len(payload_classes)}"
                ),
            )
            raise RuntimeError("unreachable")
        b_payloads = [
            msgspec_encode(arg, cls=cls)
            for arg, cls in zip(py_args, payload_classes, strict=False)
        ]
        result = self._run_handler(request_type, b_payloads, context.peer())
        return spec.python_to_response(result)


def _install_servicer_methods() -> None:
    """Attach one dispatch method per ``RequestType`` to the servicer.

    Typed rpcs get a ``_dispatch_typed`` thunk (proto message in / proto
    message out); legacy rpcs get the msgspec-envelope ``_dispatch``.
    """
    for rt in RequestType:
        method_name = request_type_to_method_name(rt)
        typed_spec = _TYPED_RPCS.get(rt)
        method: Callable[..., Any]

        if typed_spec is not None:
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
        else:

            def _method(  # noqa: E501 (captured ``rt`` via default arg)
                self: _RequestHandlerServicer,
                request: lmcache_mq_pb2.BytesRequest,
                context: "grpc.ServicerContext",
                _rt: RequestType = rt,
            ) -> lmcache_mq_pb2.BytesResponse:
                return self._dispatch(request, context, _rt)

            method = _method

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
        for rt, handler in self.handlers.items():
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    "assigned. Call add_normal_thread_pool or "
                    "add_affinity_thread_pool before start()."
                )

        target = _parse_grpc_url(self._bind_url)
        server = grpc.server(
            ThreadPoolExecutor(
                max_workers=self._grpc_max_workers,
                thread_name_prefix="mq-grpc-server",
            )
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
