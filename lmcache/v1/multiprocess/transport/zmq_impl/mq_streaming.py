# SPDX-License-Identifier: Apache-2.0
"""Multi-frame ("streaming") request support for the message queue.

A streaming handler answers one request with several responses under the same
request uid.  Nothing here interprets those frames: the server forwards each
one as the handler emits it, and the client hands each to the waiting future,
keeping the request registered until that future reports itself done.  What
counts as a partial answer is therefore decided entirely by the response type
rather than by the transport.

This lives outside ``mq.py`` so the capability stays opt-in.  A deployment
with layer-wise disabled constructs the plain :class:`MessageQueueServer` and
runs its dispatch unmodified; only a server built as
:class:`StreamingMessageQueueServer` can serve the layer-wise operations.
Those are looked up in the layer-wise RPC specs rather than the shared ones,
which the shared ``add_handler``/``submit_request`` would reject.
"""

# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.rpc import RpcOperation, RpcSpec
from lmcache.v1.multiprocess.transport.base_layerwise import (
    STREAMING_OPERATIONS,
    get_layerwise_rpc_spec,
    get_layerwise_rpc_specs,
)
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    BlockingRequestHandler,
    MessageQueueClient,
    MessageQueueServer,
    RequestHandlerBase,
    ResponseType,
    msgspec_encode,
    unwrap_request_payloads,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.futures_layerwise import LayerwiseRawFuture

logger = init_logger(__name__)


class StreamingRequestHandler(BlockingRequestHandler[ResponseType]):
    """
    Runs like :class:`BlockingRequestHandler`, but the handler is additionally
    handed a ``response_channel`` it may call any number of times to answer
    with more than one frame under the same request uid.

    Nothing in the transport interprets those frames: the client hands each
    one to the waiting future and keeps the request registered until that
    future reports itself done, so what counts as a partial answer is decided
    entirely by the response type.

    ``response_channel`` must be keyword-only on the handler itself, because
    :meth:`MessageQueueServer._inspect_handler_signature` matches the
    positional parameters against the declared payload classes.

    Its handler type stays ``HandlerType.BLOCKING``, so the thread-pool
    wiring treats it like any blocking handler; the server tells it apart
    by class when dispatching.
    """

    def __call__(
        self,
        payloads: list[bytes],
        affinity_key: int = 0,
        response_channel: Optional[Callable[[Any], None]] = None,
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "StreamingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler,
                *decoded_payloads,
                affinity_key=affinity_key,
                response_channel=response_channel,
            )
        return self.executor.submit(
            self.handler, *decoded_payloads, response_channel=response_channel
        )


class StreamingMessageQueueServer(MessageQueueServer):
    """A message queue server that can also serve the layer-wise operations.

    Every other operation is delegated straight to
    :class:`MessageQueueServer`, so registering and dispatching the existing
    operations goes through exactly the same code as before.
    """

    def _call_streaming_handler(
        self,
        handler_entry: StreamingRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        """
        Call the streaming handler in a separate thread and forward every
        response it produces to the client under the same request uid.

        Args:
            handler_entry (StreamingRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
                prefix_frames[0] is the zmq identity used as affinity key.
        """
        affinity_key = hash(prefix_frames[0])

        def _send_response(response: Any) -> None:
            frames_to_send = list(prefix_frames)
            if response is not None:
                response_cls = handler_entry.get_response_class()
                frames_to_send.append(msgspec_encode(response, cls=response_cls))
            self.output_queue.put(frames_to_send)
            self._output_efd.notify()

        future = handler_entry(
            payloads, affinity_key=affinity_key, response_channel=_send_response
        )

        def _notify_response(fut: Future):
            try:
                _send_response(fut.result())
            except Exception:
                logger.exception("Error in streaming handler")

        future.add_done_callback(_notify_response)

    def add_streaming_handler(
        self, rpc_spec: RpcSpec, handler: Callable[..., Any]
    ) -> None:
        self.handlers[rpc_spec.operation] = StreamingRequestHandler(
            list(rpc_spec.payload_types), rpc_spec.response_type, handler
        )

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        # Checked first: a streaming handler also reports BLOCKING.
        if isinstance(handler_entry, StreamingRequestHandler):
            self._call_streaming_handler(handler_entry, payloads, prefix_frames)
            return None
        return super()._call_handler(handler_entry, payloads, prefix_frames)

    def add_handler(
        self,
        operation: RpcOperation,
        handler_type: HandlerType,
        handler: Callable[..., Any],
    ) -> None:
        if operation not in get_layerwise_rpc_specs():
            super().add_handler(operation, handler_type, handler)
            return
        # Mirrors the base class, with the spec taken from the layer-wise
        # contract; whether the reply streams is a property of the operation.
        rpc_spec = get_layerwise_rpc_spec(operation)
        if not self._inspect_handler_signature(rpc_spec, handler):
            raise ValueError(
                f"Handler signature does not match for operation: {operation}"
            )
        streaming = operation in STREAMING_OPERATIONS
        if streaming and handler_type is HandlerType.BLOCKING:
            self.add_streaming_handler(rpc_spec, handler)
        elif not streaming and handler_type is HandlerType.SYNC:
            self.add_sync_handler(rpc_spec, handler)
        elif not streaming and handler_type is HandlerType.BLOCKING:
            self.add_blocking_handler(rpc_spec, handler)
        else:
            raise ValueError(
                f"Unsupported handler type {handler_type} for layer-wise "
                f"operation {operation!r}"
            )


class StreamingMessageQueueClient(MessageQueueClient):
    """Client counterpart of :class:`StreamingMessageQueueServer`.

    Adds the submit paths for the layer-wise operations, including one whose
    reply arrives as several frames. The base client is left untouched, so a
    deployment that never issues one keeps its dispatch and its request table
    unchanged.
    """

    def submit_layerwise_request(
        self,
        operation: RpcOperation,
        request_payloads: list[Any],
    ) -> MessagingFuture[Any]:
        """Submit a single-reply layer-wise request.

        Same as
        :meth:`~lmcache.v1.multiprocess.transport.zmq_impl.mq.MessageQueueClient.submit_request`,
        with the spec taken from the layer-wise contract, which the base
        method does not know.

        Args:
            operation: A layer-wise operation name.
            request_payloads: The payloads of the request.

        Returns:
            A future that will hold the response.
        """
        rpc_spec = get_layerwise_rpc_spec(operation)
        future: MessagingFuture[Any] = MessagingFuture()
        request_uid = next(self._request_counter)
        self.input_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=request_uid,
                future=future,
                rpc_spec=rpc_spec,
                request_payloads=request_payloads,
            )
        )
        self._polling_loop.notify()
        return future

    def submit_streaming_request(
        self,
        operation: RpcOperation,
        request_payloads: list[Any],
        future: "LayerwiseRawFuture",
    ) -> "LayerwiseRawFuture":
        """Submit a request whose reply arrives as several frames.

        The streaming counterpart of
        :meth:`~lmcache.v1.multiprocess.transport.zmq_impl.mq.MessageQueueClient.submit_request`.
        The only difference is that the future is supplied by the caller and
        bound to the pending-request table first, so it can re-arm itself
        between frames. Binding must happen before the request reaches the
        polling loop, because a reply can land the moment it does.

        The request-building below deliberately mirrors ``submit_request``;
        the two are pinned together by
        ``test_streaming_submit_matches_the_base_client`` so this copy cannot
        silently fall behind.

        Args:
            operation: A streaming layer-wise operation name.
            request_payloads: The payloads of the request.
            future: The future to complete, created by the caller so that
                per-layer state is attached before any frame can arrive.

        Returns:
            The same future, now registered with the polling loop.
        """
        rpc_spec = get_layerwise_rpc_spec(operation)
        request_uid = next(self._request_counter)
        future.bind_registry(self.pending_futures, request_uid, rpc_spec)
        self.input_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=request_uid,
                future=future,
                rpc_spec=rpc_spec,
                request_payloads=request_payloads,
            )
        )
        self._polling_loop.notify()
        return future
