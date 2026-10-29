# SPDX-License-Identifier: Apache-2.0
"""gRPC request server on the native (Rust) front for multiprocess requests."""

# Standard
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, cast
import os
import shutil
import tempfile
import threading

# Third Party
from google.protobuf.message import DecodeError, Message
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.otel_init import (
    grpc_metrics_enabled,
    register_metrics_source,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.grpc_impl.client import parse_grpc_target
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    ServiceBinding,
    get_service_bindings,
    message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.native_metrics import (
    native_metrics_source,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_base import (
    GrpcMultiprocessServerBase,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_handlers import (
    CLIENT_ID_METADATA_KEY,
    GRPC_SERVER_OPTIONS,
    GrpcRequestHandler,
)

logger = init_logger(__name__)

_MAX_EVENTS_PER_RECV = 64
_UNIMPLEMENTED = grpc.StatusCode.UNIMPLEMENTED.value[0]
_INVALID_ARGUMENT = grpc.StatusCode.INVALID_ARGUMENT.value[0]
_INTERNAL = grpc.StatusCode.INTERNAL.value[0]
_UNKNOWN = grpc.StatusCode.UNKNOWN.value[0]


class _FrontServedServicer:
    """Stands in for a generated service on the out-of-tree grpcio server.

    The front serves every generated method itself and never forwards their
    paths there, but grpcio needs a servicer to add the service.
    """

    def __getattr__(self, name: str) -> Callable[..., None]:
        def unreachable(request: Any, context: grpc.ServicerContext) -> None:
            context.abort(grpc.StatusCode.INTERNAL, f"{name} is served by the front")

        return unreachable


def _rejection(registered: GrpcRequestHandler) -> str | None:
    """Return why a method is answered with UNIMPLEMENTED, or None to serve it."""
    if registered.handler is None:
        return f"{registered.operation} is not enabled on this server"
    if registered.handler_type not in (HandlerType.SERIALIZED, HandlerType.BLOCKING):
        return f"{registered.handler_type.name} handlers are not supported"
    return None


@dataclass(frozen=True)
class _NativeMethod:
    """A generated method as the native server dispatches it."""

    registered: GrpcRequestHandler
    request_class: type[Message]
    server_streaming: bool
    # Why the method is answered with UNIMPLEMENTED; None when it is served.
    rejection: str | None
    submit: Callable[..., Future[Any]]


def _wants_metrics(front_class: type) -> bool:
    """Whether to collect gRPC server metrics on the native front."""
    if not grpc_metrics_enabled():
        return False
    # metrics() arrived in pygrpc-native 0.3.0.
    if not hasattr(front_class, "metrics"):
        logger.warning(
            "gRPC server metrics in native mode need pygrpc-native 0.3.0 or "
            "later; serving without them. Upgrade pygrpc-native or pass "
            "--disable-grpc-metrics to silence this."
        )
        return False
    return True


class GrpcNativeMultiprocessServer(GrpcMultiprocessServerBase):
    """Serve the gRPC request services on the native front.

    HTTP/2 and gRPC framing run on Rust threads without the GIL. One
    dispatcher thread receives calls in batches and hands each one to the
    pool its handler type calls for, where the request is decoded, the
    handler runs, and the response is encoded and sent. Handler types keep
    their meaning: ``SERIALIZED`` handlers run one at a time, normal
    ``BLOCKING`` handlers run on up to ``max_cpu_workers`` threads, and
    affinity handlers stay on the worker chosen for their client. Each
    server-streaming call is read on its own thread, as in the other servers.

    Out-of-tree services register on a real ``grpc.Server``, which runs on
    ``grpc_server_workers`` threads. When a module has a registrar, that
    server listens on a private Unix socket and the front forwards every
    method it does not know there; in its handlers ``context.peer()`` is that
    socket, not the client.

    Raises:
        RuntimeError: If the ``pygrpc-native`` package is not installed,
            or an address cannot be bound.
        ValueError: If ``max_cpu_workers`` is not positive.
    """

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_server_workers: int,
    ) -> None:
        try:
            # Third Party
            from pygrpc_native import EVENT_CANCEL, NativeGrpcFront
        except ImportError as exc:
            raise RuntimeError(
                "grpc_server_mode='native' needs the pygrpc-native package. "
                "Install it with `pip install pygrpc-native`, and retry."
            ) from exc
        super().__init__(
            bind_url, max_cpu_workers, max_gpu_workers, grpc_server_workers
        )
        self._event_cancel = EVENT_CANCEL
        self._methods: list[_NativeMethod | None] = []
        self._method_ids: dict[str, int] = {}
        paths: list[tuple[str, bool]] = []
        for binding in get_service_bindings().values():
            service = binding.descriptor.full_name
            for method in binding.descriptor.methods:
                self._method_ids[method.full_name] = len(paths)
                paths.append((f"/{service}/{method.name}", method.server_streaming))
                self._methods.append(None)

        metrics = _wants_metrics(NativeGrpcFront)
        try:
            self._front = NativeGrpcFront(
                parse_grpc_target(bind_url),
                paths,
                client_key_metadata=CLIENT_ID_METADATA_KEY,
                **({"metrics": True} if metrics else {}),
            )
        except OSError as exc:
            super()._release_threads(wait=True)
            raise RuntimeError(
                f"Failed to bind gRPC multiprocess server: {bind_url}"
            ) from exc
        self._unregister_metrics: Callable[[], None] = (
            register_metrics_source(native_metrics_source(self._front))
            if metrics
            else lambda: None
        )
        self._bound_port = self._front.bound_port
        self._ext_server = grpc.server(self._executor, options=GRPC_SERVER_OPTIONS)
        self._has_ext_services = False
        self._ext_socket_dir: str | None = None

        self._normal_pool = ThreadPoolExecutor(
            max_workers=max_cpu_workers,
            thread_name_prefix="grpc-normal",
        )
        self._serialized_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="grpc-serialized",
        )
        self._streams: dict[int, MessagingStream[Any]] = {}
        self._streams_lock = threading.Lock()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="grpc-native-dispatch",
        )

    def start(self) -> None:
        """Start accepting gRPC requests.

        Call ``add_modules`` first: out-of-tree services are only served if
        their module was added before ``start``.

        Raises:
            RuntimeError: If the server was already started or closed, or the
                private socket for out-of-tree services cannot be bound.
        """
        if self._closed.is_set():
            raise RuntimeError("gRPC server is closed")
        if self._dispatcher.is_alive():
            raise RuntimeError("gRPC server already started")
        if self._has_ext_services:
            self._start_ext_services()
        self._dispatcher.start()
        self._front.start()
        logger.info(
            "LMCache gRPC cache server (native) is running on %s", self._bind_url
        )

    def add_modules(self, modules: Sequence[object]) -> None:
        """Register decorated module methods as generated gRPC services.

        Args:
            modules: Ordered business modules. A later module overrides an
                earlier handler for the same request type.

        Raises:
            TypeError: If a module handler does not match its protocol types.
            ValueError: If a module exposes invalid handler metadata.
        """
        super().add_modules(modules)
        self._has_ext_services |= any(
            getattr(module, "register_grpc_services", None) is not None
            for module in modules
        )

    def _grpc_server(self) -> grpc.Server:
        return self._ext_server

    def _start_ext_services(self) -> None:
        self._ext_socket_dir = tempfile.mkdtemp(prefix="lmcache-grpc-ext-")
        socket_path = os.path.join(self._ext_socket_dir, "ext.sock")
        try:
            self._ext_server.add_insecure_port(f"unix:{socket_path}")
        except RuntimeError as exc:
            shutil.rmtree(self._ext_socket_dir, ignore_errors=True)
            self._ext_socket_dir = None
            raise RuntimeError(
                f"Failed to bind out-of-tree gRPC services: {socket_path}"
            ) from exc
        self._front.set_fallback(socket_path)
        self._ext_server.start()

    def _make_servicer(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
    ) -> object:
        for method in binding.descriptor.methods:
            registered = handlers[method.full_name]
            self._methods[self._method_ids[method.full_name]] = _NativeMethod(
                registered=registered,
                request_class=message_class(method.input_type),
                server_streaming=method.server_streaming,
                rejection=_rejection(registered),
                submit=self._submitter(registered),
            )
        return _FrontServedServicer()

    def _stop_server(self) -> None:
        self._unregister_metrics()
        self._front.close()
        if self._dispatcher.is_alive():
            self._dispatcher.join()
        with self._streams_lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            try:
                stream.close()
            except Exception:
                logger.exception("Error closing a gRPC stream on shutdown")
        self._ext_server.stop(grace=None)

    def _release_threads(self, wait: bool) -> None:
        self._serialized_pool.shutdown(wait=wait)
        self._normal_pool.shutdown(wait=wait)
        if self._ext_socket_dir is not None:
            shutil.rmtree(self._ext_socket_dir, ignore_errors=True)
        super()._release_threads(wait)

    def _submitter(self, registered: GrpcRequestHandler) -> Callable[..., Future[Any]]:
        """Return the pool submit a served method's calls go to."""
        if registered.handler_type is HandlerType.SERIALIZED:
            return self._serialized_pool.submit
        if registered.requires_client_affinity:
            return self._submit_affinity
        return self._normal_pool.submit

    def _submit_affinity(
        self,
        fn: Callable[..., None],
        method: _NativeMethod,
        call_id: int,
        payload: bytes,
        client_key: bytes | str,
    ) -> Future[Any]:
        return self._affinity_pool.submit(
            fn, method, call_id, payload, client_key, affinity_key=hash(client_key)
        )

    def _dispatch_loop(self) -> None:
        front = self._front
        methods = self._methods
        run = self._run
        event_cancel = self._event_cancel
        while (events := front.recv_batch(_MAX_EVENTS_PER_RECV)) is not None:
            for kind, call_id, method_id, payload, client_key in events:
                # One bad event (say a stream's close callback raising) must
                # not stop the loop, or every later call would hang.
                try:
                    if kind == event_cancel:
                        self._close_stream(call_id)
                        continue
                    method = methods[method_id]
                    if method is None or method.rejection is not None:
                        self._reject(method, call_id)
                        continue
                    method.submit(run, method, call_id, payload, client_key)
                except Exception:
                    logger.exception("Error dispatching gRPC call %d", call_id)
                    if kind != event_cancel:
                        front.fail(call_id, _INTERNAL, "Error dispatching call")

    def _reject(self, method: _NativeMethod | None, call_id: int) -> None:
        if method is None:
            # Started before add_modules(): no service is registered yet.
            self._front.fail(call_id, _UNIMPLEMENTED, "Method not found!")
            return
        self._front.fail(call_id, _UNIMPLEMENTED, cast(str, method.rejection))

    def _run(
        self,
        method: _NativeMethod,
        call_id: int,
        payload: bytes,
        client_key: bytes | str,
    ) -> None:
        """Decode, run and answer one call on a handler thread.

        Status codes follow the sync server: an undecodable request is
        INTERNAL, NotImplementedError is UNIMPLEMENTED, a ValueError from a
        streaming handler is INVALID_ARGUMENT, and any other exception is
        UNKNOWN. A streaming method's stream is then sent by ``_forward``.
        """
        registered = method.registered
        front = self._front
        try:
            request = method.request_class.FromString(payload)
        except DecodeError:
            logger.exception("Exception deserializing %s request", registered.operation)
            front.fail(call_id, _INTERNAL, "Exception deserializing request!")
            return
        try:
            # Only methods with a handler get a submitter.
            handler = cast(Callable[..., Any], registered.handler)
            result = handler(*registered.request_decoder(request))
            if method.server_streaming:
                if not isinstance(result, MessagingStream):
                    raise TypeError(
                        f"{registered.operation} must return a MessagingStream"
                    )
                self._start_stream(registered, call_id, result)
                return
            if isinstance(result, MessagingStream):
                result.close()
                raise TypeError(f"{registered.operation} returned a stream")
            response = registered.response_encoder(result).SerializeToString()
        except NotImplementedError as exc:
            front.fail(call_id, _UNIMPLEMENTED, str(exc))
            return
        except ValueError as exc:
            if method.server_streaming:
                front.fail(call_id, _INVALID_ARGUMENT, str(exc))
            else:
                self._fail_application(registered, call_id, exc)
            return
        except Exception as exc:
            self._fail_application(registered, call_id, exc)
            return
        front.respond(call_id, response)

    def _fail_application(
        self, registered: GrpcRequestHandler, call_id: int, exc: Exception
    ) -> None:
        logger.exception("Error in handler %s", registered.operation, exc_info=exc)
        self._front.fail(call_id, _UNKNOWN, f"Exception calling application: {exc}")

    def _start_stream(
        self,
        registered: GrpcRequestHandler,
        call_id: int,
        stream: MessagingStream[Any],
    ) -> None:
        with self._streams_lock:
            self._streams[call_id] = stream
        # The client may have left before the handler returned; its cancel
        # event then found no stream to close.
        if not self._front.stream_open(call_id):
            self._close_stream(call_id)
            return
        threading.Thread(
            target=self._forward,
            args=(registered, call_id, stream),
            daemon=True,
            name="grpc-event-stream",
        ).start()

    def _forward(
        self,
        registered: GrpcRequestHandler,
        call_id: int,
        stream: MessagingStream[Any],
    ) -> None:
        """Send a handler's stream until it ends or the client leaves.

        Each send waits while the client is not reading, so a stalled
        subscriber stops reads from the handler's stream.
        """
        front = self._front
        try:
            for response in stream:
                data = registered.response_encoder(response).SerializeToString()
                if not front.stream_send(call_id, data):
                    break
            else:
                front.stream_end(call_id)
        except Exception as exc:
            if front.stream_open(call_id):
                logger.exception("Error in streaming handler %s", registered.operation)
                front.fail(call_id, _INTERNAL, str(exc))
        finally:
            self._close_stream(call_id)
            stream.close()

    def _close_stream(self, call_id: int) -> None:
        with self._streams_lock:
            stream = self._streams.pop(call_id, None)
        if stream is not None:
            stream.close()
