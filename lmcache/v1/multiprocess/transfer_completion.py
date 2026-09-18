# SPDX-License-Identifier: Apache-2.0
"""Stream-ordered release of what a transfer handler borrows.

A STORE / RETRIEVE handler enqueues device work on a transfer stream and
returns while that work is still running. Two things it holds are only safe
to let go once the stream has caught up:

* **The worker's imported event.** The wait the handler queues on it refers
  to the event until the stream has consumed the wait. Dropping the import
  at handler return would let the event be destroyed under a queued wait.
* **The reply.** The worker treats the reply as "the transfer is complete",
  so it may only be sent once the stream has run the copy.

Both are released by a host callback queued on the same stream directly
behind the operation that must finish first. Release therefore follows from
an observed fact -- the stream reaching the callback -- rather than from any
assumption about how long the device work takes or when the peer looks.
"""

# Standard
from concurrent.futures import Future
from typing import Callable, NamedTuple, Protocol
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFunc,
    submit_callback_to_stream,
)
from lmcache.v1.platform.base.event_ipc import EventIPCBackend

logger = init_logger(__name__)

RELEASE_IMPORTED_EVENT_KIND = "release_imported_event"
RESOLVE_DEFERRED_REPLY_KIND = "resolve_deferred_reply"

# Wire shape of a transfer reply: (device event handle, succeeded). The server
# never sends a handle; the reply itself is sent once the transfer is complete.
TransferReply = tuple[bytes, bool]

# ``DeviceHostFuncDispatcher.register``: (kind, handler, payload_type).
HostFuncRegistrar = Callable[[str, DeviceHostFunc, type], None]


class TransferStream(Protocol):
    """The stream a transfer is enqueued on, as both a torch and a CuPy stream.

    Every cache context satisfies this. A handler that runs on a stream of
    its own builds a :class:`TransferStreams`.
    """

    @property
    def device(self) -> object: ...

    @property
    def stream(self) -> object: ...

    @property
    def cupy_stream(self) -> object: ...


class TransferStreams(NamedTuple):
    """A :class:`TransferStream` for a handler that owns its own stream."""

    device: object
    stream: object
    cupy_stream: object


class TransferCompletion:
    """Hold imported events and deferred replies until the stream releases them.

    One instance per server process, shared by every module that enqueues
    transfers. Thread-safe: handlers call :meth:`wait_for_producer` and
    :meth:`reply_when_done` on request threads; the device host-function
    dispatcher calls :meth:`release_imported_event` and :meth:`resolve_reply`
    on its drain thread.
    """

    def __init__(self) -> None:
        # Imports keyed by the worker's handle. Two handlers may import the
        # same handle (STORE and STORE_Q of one forward), so hold a list and
        # release one entry per consumed wait.
        self._imported_events: dict[bytes, list[object]] = {}
        self._deferred_replies: dict[
            int, tuple[Future[TransferReply], TransferReply]
        ] = {}
        self._next_reply_id = 0
        self._lock = threading.Lock()

    def register_host_funcs(self, register: HostFuncRegistrar) -> None:
        """Register the two release callbacks on the dispatcher.

        Args:
            register: The dispatcher's ``register(kind, handler, payload_type)``.
        """
        register(RELEASE_IMPORTED_EVENT_KIND, self.release_imported_event, bytes)
        register(RESOLVE_DEFERRED_REPLY_KIND, self.resolve_reply, int)

    def wait_for_producer(
        self, backend: EventIPCBackend, handle: bytes, target: TransferStream
    ) -> None:
        """Make ``target`` wait on the worker's event and hold the import.

        The import is held until a host callback queued on ``target`` right
        behind the wait fires, which proves the wait has been consumed.

        Args:
            backend: Event backend of the context that owns ``target``.
            handle: Serialized event handle from the worker.
            target: Stream the transfer will be enqueued on.

        Raises:
            Exception: Whatever ``import_event`` or ``wait_event`` raises. No
                import is held when this method raises.
        """
        event = backend.import_event(handle, target.device)
        with self._lock:
            self._imported_events.setdefault(handle, []).append(event)
        try:
            backend.wait_event(event, target.stream)
        except Exception:
            self.release_imported_event(handle)
            raise
        submit_callback_to_stream(
            target.cupy_stream, RELEASE_IMPORTED_EVENT_KIND, handle
        )

    def release_imported_event(self, handle: bytes) -> None:
        """Drop one held import of ``handle``; its stream wait has been consumed.

        Args:
            handle: The worker's handle the import was made from.
        """
        with self._lock:
            events = self._imported_events.get(handle)
            if not events:
                logger.warning("No held import for event handle %r", handle)
                return
            events.pop()
            if not events:
                del self._imported_events[handle]

    def reply_when_done(
        self, target: TransferStream, succeeded: bool
    ) -> Future[TransferReply]:
        """Return the transfer reply as a future that resolves once ``target``
        has run everything queued on it before this call.

        A handler that calls this after enqueuing its copy hands the peer a
        reply whose arrival means "the copy is complete".

        Args:
            target: Stream whose queued work the reply must follow.
            succeeded: The reply's success flag.

        Returns:
            A future the request transport sends once it resolves.
        """
        future: Future[TransferReply] = Future()
        with self._lock:
            reply_id = self._next_reply_id
            self._next_reply_id += 1
            self._deferred_replies[reply_id] = (future, (b"", succeeded))
        submit_callback_to_stream(
            target.cupy_stream, RESOLVE_DEFERRED_REPLY_KIND, reply_id
        )
        return future

    def resolve_reply(self, reply_id: int) -> None:
        """Resolve the deferred reply ``reply_id``; its stream work has run.

        Args:
            reply_id: Id allocated by :meth:`reply_when_done`.
        """
        with self._lock:
            pending = self._deferred_replies.pop(reply_id, None)
        if pending is None:
            logger.warning("No deferred reply with id %d", reply_id)
            return
        future, reply = pending
        future.set_result(reply)

    def held_import_count(self) -> int:
        """Return how many imported events are currently held."""
        with self._lock:
            return sum(len(events) for events in self._imported_events.values())

    def pending_reply_count(self) -> int:
        """Return how many replies are waiting for their stream callback."""
        with self._lock:
            return len(self._deferred_replies)
