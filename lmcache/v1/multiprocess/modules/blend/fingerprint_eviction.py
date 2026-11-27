# SPDX-License-Identifier: Apache-2.0
"""Blend proactive fingerprint eviction: drop a chunk's fingerprint when
storage reports that one of its keys has no copy left."""

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType

if TYPE_CHECKING:
    # Standard
    import threading

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.blend.matcher import BlendTokenRangeMatcher


class FingerprintEvictionTracker:
    """Decide from storage key events which chunks lost a key for good.

    A blended chunk is usable only when every one of its keys (one per kv
    rank and object group) can be loaded, so a chunk is gone as soon as one
    key has no copy in L1 or in L2. The tracker keeps the L1 copies it was
    told about and the keys L2 deleted while L1 still held them:

    - no L2 adapter: a key is gone when its last L1 copy is evicted;
    - one L2 adapter: a key is gone when L2 deleted it and L1 does not hold
      it, in either order. An L1 eviction alone says nothing, because L2
      may still serve the key;
    - several L2 adapters: an L2 delete is ignored, because the event does
      not say which adapter deleted the key and another may still hold it.

    Whatever the tracker cannot decide is left to the lookup's stale
    strikes. Not thread-safe: every method runs on the event bus's drain
    thread.
    """

    def __init__(self) -> None:
        # key -> tags of the L1 managers holding a readable copy
        self._l1_copies: dict[ObjectKey, set[str]] = {}
        # keys deleted from L2 while an L1 copy remained
        self._l2_gone: set[ObjectKey] = set()

    def on_l1_committed(self, keys: list[ObjectKey], l1_tag: str) -> None:
        """Record readable L1 copies.

        Args:
            keys: The keys whose write finished.
            l1_tag: The L1 manager that holds them.
        """
        for key in keys:
            self._l1_copies.setdefault(key, set()).add(l1_tag)

    def on_l1_evicted(
        self, keys: list[ObjectKey], l1_tag: str, num_l2_adapters: int
    ) -> list[bytes]:
        """Forget evicted L1 copies and report the chunks that lost a key.

        Args:
            keys: The keys evicted from L1.
            l1_tag: The L1 manager that evicted them.
            num_l2_adapters: How many L2 adapters are configured right now.

        Returns:
            The chunk hashes whose key has no copy left, without duplicates.
        """
        gone: dict[bytes, None] = {}
        for key in keys:
            owners = self._l1_copies.get(key)
            if owners is not None:
                owners.discard(l1_tag)
                if owners:
                    continue
                del self._l1_copies[key]
            deleted_from_l2 = key in self._l2_gone
            self._l2_gone.discard(key)
            if num_l2_adapters == 0 or deleted_from_l2:
                gone[key.chunk_hash] = None
        return list(gone)

    def on_l2_stored(self, keys: list[ObjectKey]) -> None:
        """Record that L2 holds ``keys`` again.

        Args:
            keys: The keys stored to L2.
        """
        self._l2_gone.difference_update(keys)

    def on_l2_deleted(self, keys: list[ObjectKey], num_l2_adapters: int) -> list[bytes]:
        """Record keys deleted from L2 and report the chunks that lost a key.

        Args:
            keys: The keys deleted from L2.
            num_l2_adapters: How many L2 adapters are configured right now.

        Returns:
            The chunk hashes whose key has no copy left, without duplicates.
            Empty unless exactly one L2 adapter is configured.
        """
        if num_l2_adapters != 1:
            return []
        gone: dict[bytes, None] = {}
        for key in keys:
            if key in self._l1_copies:
                self._l2_gone.add(key)
            else:
                gone[key.chunk_hash] = None
        return list(gone)

    def clear(self) -> None:
        """Forget everything, after lost events made the record unreliable.

        Until it is rebuilt, a key whose L1 commit was forgotten can be
        reported gone while L1 still holds it. That costs one reuse of the
        chunk, whose next store registers the fingerprint again.
        """
        self._l1_copies.clear()
        self._l2_gone.clear()


class FingerprintEvictionMixin:
    """Storage key-event handlers of ``BlendModule``; state lives on the
    composed instance."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__; declared so the mixin type-checks.
        _ctx: "MPCacheServerContext"
        _event_bus: Any
        _token_range_matcher: "BlendTokenRangeMatcher"
        _pending_fp_hashes: set[bytes]
        _pending_fp_lock: "threading.Lock"
        _stale_strike: dict[bytes, int]
        _fingerprint_eviction: FingerprintEvictionTracker

    def _subscribe_fingerprint_eviction(self) -> None:
        """Subscribe to the storage key events on the event bus. Nothing
        arrives while the bus is disabled; the lookup's stale strikes then
        remain the only eviction."""
        bus = self._event_bus
        bus.subscribe(EventType.L1_WRITE_FINISHED, self._on_l1_keys_committed)
        bus.subscribe(
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, self._on_l1_keys_committed
        )
        bus.subscribe(EventType.L1_KEYS_EVICTED, self._on_l1_keys_evicted)
        bus.subscribe(EventType.L2_KEYS_STORED, self._on_l2_keys_stored)
        bus.subscribe(EventType.L2_KEYS_DELETED, self._on_l2_keys_deleted)
        bus.subscribe_drops(self._fingerprint_eviction.clear)

    def _on_l1_keys_committed(self, event: Event) -> None:
        self._fingerprint_eviction.on_l1_committed(
            event.metadata["keys"], event.metadata["l1_tag"]
        )

    def _on_l1_keys_evicted(self, event: Event) -> None:
        self._evict_fingerprints(
            self._fingerprint_eviction.on_l1_evicted(
                event.metadata["keys"],
                event.metadata["l1_tag"],
                len(self._ctx.storage_manager.l2_adapters()),
            )
        )

    def _on_l2_keys_stored(self, event: Event) -> None:
        self._fingerprint_eviction.on_l2_stored(event.metadata["keys"])

    def _on_l2_keys_deleted(self, event: Event) -> None:
        self._evict_fingerprints(
            self._fingerprint_eviction.on_l2_deleted(
                event.metadata["keys"],
                len(self._ctx.storage_manager.l2_adapters()),
            )
        )

    def _evict_fingerprints(self, chunk_hashes: list[bytes]) -> None:
        """Remove ``chunk_hashes`` from the matcher, sparing in-flight ones.

        A hash whose fingerprint registration is still queued belongs to a
        store that has just written the chunk; the event may describe the
        copy that store replaced.

        Args:
            chunk_hashes: The chunks storage can no longer serve whole.
        """
        if not chunk_hashes:
            return
        with self._pending_fp_lock:
            settled = [h for h in chunk_hashes if h not in self._pending_fp_hashes]
            for chunk_hash in settled:
                self._stale_strike.pop(chunk_hash, None)
        num_removed = self._token_range_matcher.remove_chunks(settled)
        if num_removed:
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_CHUNKS_EVICTED,
                    metadata={"num_chunks": num_removed},
                )
            )
