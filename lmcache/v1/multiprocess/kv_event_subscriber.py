# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral reception of the MP native-binding event stream."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEvent, CacheRemoveEvent, CacheStoreEvent
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KV_EVENT_SCHEMA_VERSION,
    EventBinding,
    KVEventBatch,
    KVEventRecord,
    NativeBlockHash,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.custom_types import KVEventProfile
    from lmcache.v1.multiprocess.futures import MessagingStream
    from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVEventReset:
    """Withdraw the engine's CPU ledger before applying subsequent batches.

    A non-None ``error`` is terminal: the engine must report the failure and
    must not switch to a different source. GPU advertisements are unaffected.
    """

    reason: str
    error: str | None = None


class KVEventLedger:
    """Track chunk contributions to native CPU placements on the engine thread.

    One ledger belongs to one subscribed model/profile. A chunk is identified by
    its ordered native hashes; sibling chunks can share constituent blocks.
    Only the first contribution publishes a block and the last removal withdraws
    it. Replayed transitions for an already-known chunk are idempotent.
    """

    def __init__(self) -> None:
        self._chunks: set[tuple[NativeBlockHash, ...]] = set()
        self._block_refcounts: dict[NativeBlockHash, int] = {}

    def apply(self, record: KVEventRecord) -> list[CacheEvent]:
        """Apply a validated chunk transition on the engine's publication thread.

        Args:
            record: Complete native chunk descriptor validated by the subscriber.

        Returns:
            CPU placement changes, or an empty list for a duplicate or an unknown
            removal. Stored runs contain only newly advertised blocks, with their
            corresponding original tokens and preceding native parent hash.
        """
        hashes = record.block_hashes
        if record.kind == KV_EVENT_KIND_REMOVED:
            if hashes not in self._chunks:
                return []
            self._chunks.remove(hashes)
            removed = []
            for block_hash in hashes:
                count = self._block_refcounts[block_hash] - 1
                if count:
                    self._block_refcounts[block_hash] = count
                else:
                    del self._block_refcounts[block_hash]
                    removed.append(block_hash)
            return (
                [CacheRemoveEvent(removed, "CPU", group_idx=record.group_idx)]
                if removed
                else []
            )
        if hashes in self._chunks:
            return []
        self._chunks.add(hashes)
        ranges: list[tuple[int, int]] = []
        for index, block_hash in enumerate(hashes):
            count = self._block_refcounts.get(block_hash, 0)
            self._block_refcounts[block_hash] = count + 1
            if count:
                continue
            if ranges and ranges[-1][1] == index:
                ranges[-1] = (ranges[-1][0], index + 1)
            else:
                ranges.append((index, index + 1))
        return [
            CacheStoreEvent(
                block_hashes=list(hashes[start:end]),
                parent_block_hash=hashes[start - 1]
                if start
                else record.parent_block_hash,
                token_ids=list(
                    record.token_ids[
                        start * record.block_size : end * record.block_size
                    ]
                ),
                block_size=record.block_size,
                lora_id=None,
                medium="CPU",
                lora_name=None,
                group_idx=record.group_idx,
                kv_cache_spec_kind=record.kv_cache_spec_kind,
            )
            for start, end in ranges
        ]

    def reset(self) -> list[CacheEvent]:
        """Forget chunk ownership and withdraw each advertised CPU block once.

        Returns:
            One CPU removal event, or an empty list when no blocks were advertised.
            Subsequent replay starts with an empty ledger.
        """
        events: list[CacheEvent] = (
            [CacheRemoveEvent(list(self._block_refcounts), "CPU", group_idx=0)]
            if self._block_refcounts
            else []
        )
        self._chunks.clear()
        self._block_refcounts.clear()
        return events


class MPKVEventSubscriber:
    """Borrow a request client and own only one stream and receiver thread.

    Registration, heartbeat, election, and native publication belong to the
    engine. Call ``start`` after registration, then ``drain`` on the engine's
    publication thread. Every reconnect withdraws CPU advertisements and
    replays from cursor zero. Warm entries outside retained history may remain
    undiscovered; this class never reconstructs them from request history.

    Args:
        req_client: Borrowed, already connected MP client.
        instance_id: Elected subscriber's registered worker identity.
        model_name: Registered cache namespace.
        world_size: Number of KV ranks owned by the private MP.
        profile: Immutable native identity profile negotiated with MP.
        max_pending_events: Maximum buffered records, batches, and resets.
        on_buffer_change: Optional gauge update, called under the buffer lock.

    Raises:
        ValueError: If the buffer bound cannot hold a reset and a batch.
    """

    def __init__(
        self,
        req_client: "RequestClient",
        instance_id: int,
        model_name: str,
        *,
        world_size: int,
        profile: "KVEventProfile",
        max_pending_events: int = 65536,
        on_buffer_change: Callable[[int], None] | None = None,
    ) -> None:
        if max_pending_events < 2:
            raise ValueError("max_pending_events must be at least two")
        self._client = req_client
        self._instance_id = instance_id
        self._model_name = model_name
        self._world_size = world_size
        self._profile = profile
        self._capacity = max_pending_events
        self._on_buffer_change = on_buffer_change
        self._batch_size = min(1024, max_pending_events - 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._terminal_error: str | None = None
        self._thread = threading.Thread(
            target=self._receive, name="lmcache-kv-events", daemon=True
        )
        self._stream: MessagingStream[KVEventBatch] | None = None
        self._pending: list[KVEventBatch | KVEventReset] = []
        self._pending_size = 0
        self._incarnation: int | None = None
        self._cursor = 0

    def start(self, *, timeout: float | None = None) -> None:
        """Start reception once; repeated calls before close are harmless.

        Args:
            timeout: When provided, wait for an enabled, compatible first batch
                before engine readiness. None starts reception asynchronously.

        Raises:
            RuntimeError: If closed, or negotiation fails during a bounded start.
            LMCacheTimeoutError: If negotiation does not complete before timeout.
        """
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber is closed")
            if self._thread.ident is None:
                self._thread.start()
        if timeout is not None:
            if not self._ready.wait(timeout):
                self.close()
                raise LMCacheTimeoutError("MP KV event profile negotiation timed out")
            if self._terminal_error is not None:
                raise RuntimeError(self._terminal_error)
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber closed during negotiation")

    def drain(self) -> list[KVEventBatch | KVEventReset]:
        """Return pending batches and invalidations in source order.

        A reset always precedes data recovered after loss. The caller owns its
        native CPU ledger and must process resets even when no data follows.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            self._pending_size = 0
            if self._on_buffer_change is not None:
                self._on_buffer_change(0)
            return pending

    def close(self) -> None:
        """Idempotently cancel and join reception, keeping the client alive."""
        with self._lock:
            self._stop.set()
            self._ready.set()
            stream = self._stream
            started = self._thread.ident is not None
        if stream is not None:
            stream.close()
        if started:
            self._thread.join()

    def _enqueue(self, item: KVEventBatch | KVEventReset) -> None:
        """Keep an invalidation path even when the data buffer overflows."""
        size = max(1, len(item.events)) if isinstance(item, KVEventBatch) else 1
        with self._lock:
            if self._pending_size + size > self._capacity:
                self._pending = [KVEventReset("queue_overflow")]
                self._pending_size = 1
            self._pending.append(item)
            self._pending_size += size
            if self._on_buffer_change is not None:
                self._on_buffer_change(self._pending_size)

    def _accept(self, batch: KVEventBatch) -> None:
        """Validate source continuity; numeric gaps are valid filtered streams."""
        if not batch.enabled:
            raise ValueError("MP native KV event stream is disabled")
        if batch.schema_version != KV_EVENT_SCHEMA_VERSION:
            raise ValueError("MP native KV event stream schema mismatch")
        if len(batch.events) > self._batch_size:
            raise ValueError("MP event batch exceeds the requested limit")
        if batch.next_cursor < 0:
            raise ValueError("MP event cursor must be nonnegative")
        # An incarnation can reuse sequence numbers from a previous server.
        if self._incarnation != batch.incarnation:
            if self._incarnation is not None:
                self._enqueue(KVEventReset("server_restart"))
            self._incarnation = batch.incarnation
            self._cursor = 0
        if batch.lost:
            self._enqueue(KVEventReset("events_lost"))
            self._cursor = 0
        if batch.next_cursor < self._cursor:
            raise ValueError("MP event cursor regressed without a loss marker")
        previous = 0
        for record in batch.events:
            if record.seq <= previous or record.seq > batch.next_cursor:
                raise ValueError("MP event records are not ordered within the cursor")
            if (
                record.profile_id != self._profile.profile_id
                or record.block_size != self._profile.block_size
                or record.model_name != self._model_name
                or record.world_size != self._world_size
                or record.cache_salt
                or record.medium != KV_EVENT_MEDIUM_CPU
                or record.kind not in (KV_EVENT_KIND_STORED, KV_EVENT_KIND_REMOVED)
                or record.group_idx != 0
                or record.kv_cache_spec_kind != "full_attention"
            ):
                raise ValueError(
                    "MP event does not match the subscribed native profile"
                )
            if record.kind == KV_EVENT_KIND_STORED:
                EventBinding(
                    record.profile_id,
                    record.block_hashes,
                    record.parent_block_hash,
                    record.token_ids,
                    record.block_size,
                    record.group_idx,
                    record.kv_cache_spec_kind,
                )
            elif record.token_ids or not record.block_hashes:
                raise ValueError(
                    "MP removal must identify blocks without token payloads"
                )
            previous = record.seq
        records = [record for record in batch.events if record.seq > self._cursor]
        self._cursor = batch.next_cursor
        if records:
            self._enqueue(replace(batch, events=records))

    def _receive(self) -> None:
        """Restore the same stream with bounded retry, or expose terminal failure."""
        delay = 0.1
        while not self._stop.is_set():
            stream = None
            try:
                stream = self._client.subscribe_kv_events(
                    self._instance_id,
                    self._model_name,
                    0,
                    self._batch_size,
                    self._world_size,
                    self._profile,
                )
                with self._lock:
                    self._stream = stream
                if self._stop.is_set():
                    return
                self._cursor = 0
                for batch in stream:
                    if self._stop.is_set():
                        return
                    self._accept(batch)
                    self._ready.set()
                    delay = 0.1
                raise ConnectionError("MP KV event subscription ended")
            except Exception as exc:
                if self._stop.is_set():
                    return
                if not isinstance(exc, (ConnectionError, TimeoutError)):
                    logger.exception("MP KV event stream failed")
                    self._enqueue(KVEventReset("protocol_error", str(exc)))
                    self._terminal_error = str(exc)
                    self._ready.set()
                    return
                logger.warning("Reconnecting MP KV event stream: %s", exc)
                self._enqueue(KVEventReset("stream_closed"))
            finally:
                if stream is not None:
                    stream.close()
                with self._lock:
                    self._stream = None
            if self._stop.wait(delay):
                return
            delay = min(delay * 2, 5.0)
