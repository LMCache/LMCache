# SPDX-License-Identifier: Apache-2.0
"""MP-server-side L2 event client.

Implements :class:`L2AdapterListener` to receive store/lookup/delete
notifications from the L2 adapter, converts ``ObjectKey`` to
``CacheKey``, buffers events, and flushes them to the coordinator in
batches on a timer.

Thread-safe: listener callbacks can fire from any thread while
``run`` drains the buffer on the event loop.
"""

# Standard
import asyncio
import threading

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.mp_coordinator.schemas import (
    CacheKey,
    EventType,
    ReportUsageRequest,
    ReportUsageResponse,
    UsageEvent,
)

logger = init_logger(__name__)

_DEFAULT_FLUSH_INTERVAL = 1.0


def _object_key_to_cache_key(obj: ObjectKey) -> CacheKey:
    """Convert an ``ObjectKey`` to a ``CacheKey``.

    Args:
        obj: The object key to convert.

    Returns:
        The equivalent cache key.
    """
    return CacheKey(
        chunk_hash_hex=obj.chunk_hash.hex(),
        model_name=obj.model_name,
        kv_rank=obj.kv_rank,
        cache_salt=obj.cache_salt,
    )


class L2EventListener(L2AdapterListener):
    """L2 adapter listener that batches events and flushes to the coordinator.

    Register as a listener on the L2 adapter via
    ``adapter.register_listener(client)``. The ``run`` coroutine should
    be started as a background task and cancelled on shutdown.

    Args:
        client: The HTTP client to send with.
        coordinator_url: Coordinator base URL (e.g. ``http://host:9300``).
        instance_id: Identifier of this MP server (included in every batch).
        flush_interval: Seconds between flush attempts.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        coordinator_url: str,
        instance_id: str,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self._client = client
        self._base_url = coordinator_url.rstrip("/")
        self._instance_id = instance_id
        self._flush_interval = flush_interval
        self._seq = 0
        self._lock = threading.Lock()
        self._buffer: list[UsageEvent] = []

    # -- L2AdapterListener implementation ------------------------------------

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        """Buffer store events for each key. Thread-safe."""
        for obj, size in zip(keys, sizes, strict=True):
            event = UsageEvent(
                type=EventType.STORE,
                key=_object_key_to_cache_key(obj),
                bytes=size,
            )
            with self._lock:
                self._buffer.append(event)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        """Buffer lookup events for each key. Thread-safe."""
        for obj in keys:
            event = UsageEvent(
                type=EventType.LOOKUP,
                key=_object_key_to_cache_key(obj),
                bytes=0,
            )
            with self._lock:
                self._buffer.append(event)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        """No-op — the coordinator handles deletion separately."""

    # -- Flush loop ----------------------------------------------------------

    async def run(self) -> None:
        """Drain the buffer on a timer until cancelled.

        Resilient: flush failures are logged and the batch is dropped
        to prevent unbounded growth when the coordinator is down.
        """
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered events to the coordinator."""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
            self._seq += 1
            seq = self._seq

        body = ReportUsageRequest(
            instance_id=self._instance_id,
            seq=seq,
            events=batch,
        )
        try:
            resp = await self._client.post(
                f"{self._base_url}/l2/events",
                json=body.model_dump(),
            )
            resp.raise_for_status()
            result = ReportUsageResponse.model_validate(resp.json())
            logger.debug("Flushed %d L2 events to coordinator", result.recorded)
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "Failed to flush %d L2 events to coordinator: %s",
                len(batch),
                e,
            )
