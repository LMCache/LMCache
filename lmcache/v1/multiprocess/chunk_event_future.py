# SPDX-License-Identifier: Apache-2.0
"""Host-polled chunk completion with an explicit remote event lease."""

# Standard
from collections.abc import Callable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.platform.base.event_ipc import EventIPCBackend

ChunkStoreResponse = tuple[bytes, list[tuple[bytes, int, int]], bool, str]
logger = init_logger(__name__)


class ChunkEventDeviceMessagingFuture(MessagingFuture[bool]):
    """Report source-safe ranges and release exported events after completion.

    ``query``, ``wait`` and ``result`` depend on the terminal device event.
    ``take_completed_ranges`` is nonblocking and returns each range once.
    The future owns all imported events; callers must not extract their handles
    or enqueue stream waits using them. This host-polled interface lets it
    acknowledge the remote lease as soon as terminal completion is observed.

    Args:
        raw_future: Response containing terminal handle, chunk handles/ranges,
            success, and the server's lease ID. An empty ID means no device work.
        device: Device on which to import the events.
        event_backend: Backend selected when the worker registered.
        release: Nonblocking submission of the server's lease-release RPC.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[ChunkStoreResponse],
        device: object,
        event_backend: EventIPCBackend,
        release: Callable[[str], MessagingFuture[None]],
    ) -> None:
        super().__init__()
        self._raw_future = raw_future
        self._device = device
        self._backend = event_backend
        self._release = release
        self._lock = threading.RLock()
        self._prepared = False
        self._terminal: object | None = None
        self._pending: list[tuple[object, tuple[int, int]]] = []
        self._ready: list[tuple[int, int]] = []
        self._lease_id = ""
        self._release_future: MessagingFuture[None] | None = None

    def query(self) -> bool:
        """Poll terminal completion without waiting for the server or device."""
        with self._lock:
            if self.is_done_.is_set():
                return True
            if not self._prepare():
                return False
            if self._terminal is not None and not self._backend.query_event(
                self._terminal
            ):
                return False
            self._complete()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for terminal completion; timeout bounds the raw RPC wait only.

        Args:
            timeout: Maximum seconds for the server response, or None.

        Returns:
            False if the RPC wait timed out, otherwise True after device completion.

        Raises:
            BaseException: The error carried by a failed raw RPC.
        """
        if not self._raw_future.wait(timeout):
            return False
        with self._lock:
            if self.is_done_.is_set():
                return True
            self._prepare()
            if self._terminal is not None:
                self._backend.synchronize_event(self._terminal, self._device)
            self._complete()
            return True

    def take_completed_ranges(self) -> tuple[tuple[int, int], ...]:
        """Drain source-safe token ranges, returning each range at most once.

        Returns:
            Newly completed absolute, end-exclusive token ranges in server order.
            Returns an empty tuple while the RPC response is unavailable.
        """
        with self._lock:
            if not self.query():
                if not self._prepared:
                    return ()
                pending = []
                for event, token_range in self._pending:
                    if self._backend.query_event(event):
                        self._ready.append(token_range)
                    else:
                        pending.append((event, token_range))
                self._pending = pending
            ready = tuple(self._ready)
            self._ready.clear()
            return ready

    def wait_for_release(self, timeout: float | None = None) -> None:
        """Drain this store and its release acknowledgement before client close.

        A failed raw RPC has no imported events or known lease to release.
        Leave that error on result() and let worker unregister reclaim any
        server-side lease whose response was lost.

        Args:
            timeout: RPC timeout in seconds, or None.

        Raises:
            TimeoutError: A response did not arrive within the timeout.
            BaseException: Event import or the release RPC failed.
        """
        try:
            self.result(timeout)
        except Exception:
            if self.exception_ is not None:
                return
            raise
        with self._lock:
            self._submit_release()
            if self._release_future is not None:
                self._acknowledge_release(timeout)

    def release_complete(self) -> bool:
        """Poll completion and the release acknowledgement without blocking.

        Returns:
            True when device work and lease release have both completed.
            Also True when the raw RPC failed without returning any handles;
            result() still raises the original error in that case.

        Raises:
            BaseException: Event import or the release RPC failed.
        """
        with self._lock:
            try:
                if not self.query():
                    return False
            except Exception:
                if self.exception_ is not None:
                    return True
                raise
            self._submit_release()
            if self._release_future is None:
                return True
            if not self._release_future.query():
                return False
            self._acknowledge_release(0)
            return True

    def _prepare(self) -> bool:
        if self._prepared:
            return True
        if not self._raw_future.query():
            return False
        try:
            terminal, chunks, result, lease_id = self._raw_future.result(0)
        except Exception as exc:
            # Cache transport failure separately from event import failure:
            # only the former has no known lease and must rely on unregister.
            self.exception_ = exc
            raise
        self._lease_id = lease_id
        self._terminal = (
            self._backend.import_event(terminal, self._device) if terminal else None
        )
        self._pending = [
            (self._backend.import_event(handle, self._device), (start, end))
            for handle, start, end in chunks
        ]
        self.result_ = result
        self._prepared = True
        return True

    def _complete(self) -> None:
        # Materialize remaining ranges before dropping imported handles. Later
        # polls and drains must not use handles whose exporter has been released.
        self._ready.extend(token_range for _, token_range in self._pending)
        self._pending.clear()
        self._terminal = None
        self.is_done_.set()
        # A cleanup transport failure must not change the data-path result.
        # Retain the lease ID so the context can retry its idempotent release.
        try:
            self._submit_release()
        except Exception:
            logger.exception("Failed to submit chunk event release; retained for retry")

    def _submit_release(self) -> None:
        if self._lease_id and self._release_future is None:
            self._release_future = self._release(self._lease_id)

    def _acknowledge_release(self, timeout: float | None) -> None:
        assert self._release_future is not None
        try:
            self._release_future.result(timeout)
        except Exception:
            # Only resubmit failed RPCs. A pending RPC may still acknowledge
            # after its caller times out, and must stay tracked until then.
            if self._release_future.query():
                self._release_future = None
            raise
        self._release_future = None
        self._lease_id = ""
