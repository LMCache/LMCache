# SPDX-License-Identifier: Apache-2.0
"""Multi-frame ("streaming") request support for the message queue.

A streaming handler answers one request with several responses under the same
request uid.  Nothing here interprets those frames: the server forwards each
one as the handler emits it, and the client hands each to the waiting future,
keeping the request registered until that future reports itself done.  What
counts as a partial answer is therefore decided entirely by the response type
rather than by the transport.

This lives outside ``mq.py`` so the capability stays opt-in.  A deployment
that never registers a ``HandlerType.STREAMING`` handler constructs the plain
:class:`MessageQueueServer` and runs its dispatch unmodified; only a server
built as :class:`StreamingMessageQueueServer` can route one.
"""

# Standard
from concurrent.futures import Future
from typing import Any, Callable, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.mq import (
    BlockingRequestHandler,
    MessageQueueServer,
    RequestHandlerBase,
    ResponseType,
    msgspec_encode,
    unwrap_request_payloads,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_response_class,
)

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

    def get_handler_type(self) -> HandlerType:
        return HandlerType.STREAMING


class StreamingMessageQueueServer(MessageQueueServer):
    """A message queue server that can also route ``HandlerType.STREAMING``.

    Every other handler type is delegated straight to
    :class:`MessageQueueServer`, so registering and dispatching the existing
    request types goes through exactly the same code as before.
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
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = StreamingRequestHandler(
            payload_clss, response_cls, handler
        )

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        if handler_entry.get_handler_type() is HandlerType.STREAMING:
            assert isinstance(handler_entry, StreamingRequestHandler)
            self._call_streaming_handler(handler_entry, payloads, prefix_frames)
            return None
        return super()._call_handler(handler_entry, payloads, prefix_frames)

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler,
    ) -> None:
        if handler_type is HandlerType.STREAMING:
            # Mirrors the validation the base class performs before it
            # dispatches on handler_type.
            if not self._inspect_handler_signature(request_type, handler):
                raise ValueError(
                    f"Handler signature does not match for request type: {request_type}"
                )
            self.add_streaming_handler(request_type, payload_clss, handler)
            return
        super().add_handler(request_type, payload_clss, handler_type, handler)
