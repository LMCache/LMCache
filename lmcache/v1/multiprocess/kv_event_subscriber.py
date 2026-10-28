# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral reception of the MP CPU KV-event stream."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)

_BATCH_SIZE = 1024
_RECONNECT_DELAY = 1.0


@dataclass(frozen=True)
class KVEventStreamClosed:
    """Invalidate the consumer's CPU placements after a stream failure.

    ``error=None`` signals a transient disconnect followed by replay from zero.
    A non-None error is terminal; the consumer owns the failure policy.
    """

    error: str | None = None


class MPKVEventSubscriber:
    """Borrow a request client and own one event stream and receiver thread.

    The engine owns registration, heartbeat, subscriber election, and event
    conversion. Batches and stream failures retain their arrival order. The
    consumer must drain updates on notification; no second consumer may drain
    concurrently. Disabled batches and terminal failures stop reception.

    Args:
        req_client: Borrowed client, kept alive until after ``close`` returns.
        instance_id: Elected engine attachment's existing instance identity.
        model_name: Model namespace used to filter the server's event log.
        on_update: Receiver-thread callback invoked outside the subscriber lock.
            It must drain updates before returning and must not call ``close``.
            Stream-closure updates must be handled without raising.
            This preserves asynchronous adapter metrics and failure handling.
    """

    def __init__(
        self,
        req_client: RequestClient,
        instance_id: int,
        model_name: str,
        on_update: Callable[["MPKVEventSubscriber"], None],
    ) -> None:
        self._client = req_client
        self._instance_id = instance_id
        self._model_name = model_name
        self._on_update = on_update
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream: MessagingStream[KVEventBatch] | None = None
        self._pending: list[KVEventBatch | KVEventStreamClosed] = []
        self._thread = threading.Thread(
            target=self._receive, name="lmcache-kv-events", daemon=True
        )

    def start(self) -> None:
        """Start reception once; repeated calls do not restart a stopped stream.

        Raises:
            RuntimeError: If the subscriber has been closed.
        """
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber is closed")
            if self._thread.ident is None:
                self._thread.start()

    def drain(self) -> list[KVEventBatch | KVEventStreamClosed]:
        """Remove and return pending batches and stream failures in source order.

        Call from ``on_update``. The consumer interprets batch incarnation/loss
        fields and applies its event-conversion and failure policies.

        Returns:
            Updates received since the previous drain, or an empty list.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            return pending

    def close(self) -> None:
        """Cancel and join reception without closing the borrowed request client.

        Idempotent, including before ``start``. Call from the owning engine
        thread without holding locks needed by ``on_update``. An in-flight
        subscription call must return before its stream can be cancelled.
        """
        with self._lock:
            self._stop.set()
            stream = self._stream
            started = self._thread.ident is not None
        if stream is not None:
            stream.close()
        if started:
            self._thread.join()

    def _publish(self, update: KVEventBatch | KVEventStreamClosed) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._pending.append(update)
        self._on_update(self)

    def _receive(self) -> None:
        while not self._stop.is_set():
            stream = None
            try:
                stream = self._client.subscribe_kv_events(
                    self._instance_id, self._model_name, 0, _BATCH_SIZE
                )
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._stream = stream
                for batch in stream:
                    if self._stop.is_set():
                        return
                    self._publish(batch)
                    if not batch.enabled:
                        return
                raise ConnectionError("KV event subscription ended")
            except ConnectionError as exc:
                if not self._stop.is_set():
                    logger.warning("Reconnecting KV event subscription: %s", exc)
                    self._publish(KVEventStreamClosed())
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("KV event subscription failed")
                    self._publish(KVEventStreamClosed(error=str(exc)))
                return
            finally:
                if stream is not None:
                    stream.close()
                with self._lock:
                    self._stream = None
            if self._stop.wait(_RECONNECT_DELAY):
                return
