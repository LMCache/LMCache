# SPDX-License-Identifier: Apache-2.0
"""Blend STORE hook: async chunk-fingerprint registration + drainers."""

# Standard
from queue import Empty as QueueEmpty
from typing import TYPE_CHECKING, Any
import time

if TYPE_CHECKING:
    # Standard
    from queue import Queue
    import threading

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.blend.matcher import (
        BlendTokenRangeMatcher,
    )
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)

#: Dispatcher kind for stream-ordered fingerprint enqueues.
CB_FINGERPRINTS_KIND = "cb_fingerprints"

#: Dispatcher kind for the stream-ordered "store landed" mark: the last
#: callback a store submits, so it runs after the store's L1 commit and
#: fingerprint enqueue. Payload: the store's landing sequence number.
CB_STORE_LANDED_KIND = "cb_store_landed"

#: One fingerprint-registration job:
#: (tokens_in_range, chunk_hashes, start_chunk_idx, position_offset,
#: request_id, store_seq). ``store_seq`` is the landing sequence number of
#: the store that produced it. Also the msgspec decode type for the
#: dispatcher payload.
FpJob = tuple[list[int], list[bytes], int, int, str, int]


class StoreMixin:
    """STORE fingerprint handlers of ``BlendModule``; state lives on the
    composed instance."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__; declared so the mixin type-checks.
        _ctx: "MPCacheServerContext"
        _event_bus: Any
        _transfer_module: "LMCacheDrivenTransferModule"
        _token_range_matcher: "BlendTokenRangeMatcher"
        _fingerprint_queue: "Queue[FpJob]"
        _fingerprint_stop: "threading.Event"
        _pending_fp_hashes: set[bytes]
        _pending_fp_lock: "threading.Lock"
        _prompt_store: Any
        _store_landing_lock: "threading.Lock"
        _store_seq: int
        _unlanded_stores: dict[int, int]
        _cb_landing_tickets: dict[str, tuple[int, float]]
        _STORE_VISIBILITY_TIMEOUT_S: float

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Paged store, then register the stored chunks as match fingerprints.

        Delegates the KV write to ``LMCacheDrivenTransfer.store``, then
        (worker 0 only) enqueues chunk hashes for async fingerprint
        registration ordered after the L1 commit. Fingerprint failures are
        logged, never raised — they do not affect store correctness.

        The store holds a landing sequence number from before the KV write
        until its L1 commit has run and its fingerprints are indexed; a blend
        lookup that arrives meanwhile waits for it (``_stores_landed_before``)
        instead of missing content this store already acknowledged.

        Returns:
            The underlying ``LMCacheDrivenTransfer.store`` result
            (event handle, success).
        """
        seq = self._open_store_landing()
        handle = b""
        try:
            handle, store_ok = self._store_and_enqueue_fingerprints(
                key, instance_id, gpu_block_ids, event_ipc_handle, seq
            )
            return handle, store_ok
        finally:
            self._close_store_landing(seq, instance_id if handle else None)

    def _store_and_enqueue_fingerprints(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        seq: int,
    ) -> tuple[bytes, bool]:
        """The KV write plus the fingerprint enqueue of :meth:`store`; each
        enqueued job holds one part of the store's landing ``seq``."""
        handle, store_ok, stored_mask = self._transfer_module.store_with_chunk_mask(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )
        result = (handle, store_ok)

        # The matcher is engine-shared; only worker 0 registers.
        if key.worker_id not in (0, None):
            return result

        # Stream-ordered enqueue via the device host-func dispatcher: the
        # completion fires at this stream position (after the L1-commit
        # callback, else lookups drop the whole group as stale) without
        # acquiring the GIL on the driver thread.
        chunk_hashes: list[bytes] = []
        tokens_in_range: list[int] = []
        try:
            session = self._ctx.session_manager.get_or_create(key.request_id)
            # Request-end cleanup may have replaced the session; re-set tokens
            # (idempotent if it survived, corrective if not).
            session.set_tokens(list(key.token_ids))
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h)
                for h in session.get_hashes(key.start, key.end)
            ]
            if not chunk_hashes:
                return result
            # Record prompts whose KV was stored, for the reorder planner.
            if getattr(self, "_prompt_store", None) is not None and any(stored_mask):
                self._record_prompt(key, session)
            tokens_in_range = list(key.token_ids)[key.start : key.end]
            # Chunk 0 is owned by the prefix lookup leg; skip its fingerprint.
            start_chunk_idx = 0 if key.start != 0 else 1

            # Register only committed chunks: a fingerprint for a skipped
            # chunk would advertise content that was never persisted.
            def _stored(i: int) -> bool:
                return stored_mask[i] if i < len(stored_mask) else False

            jobs: list[FpJob] = []
            chunk_size = self._ctx.chunk_size
            run_start: int | None = None
            for i in range(len(chunk_hashes) + 1):
                if i < len(chunk_hashes) and _stored(i):
                    if run_start is None:
                        run_start = i
                    continue
                if run_start is None:
                    continue
                run_sci = max(start_chunk_idx - run_start, 0)
                if run_sci < i - run_start:
                    jobs.append(
                        (
                            tokens_in_range[run_start * chunk_size : i * chunk_size],
                            chunk_hashes[run_start:i],
                            run_sci,
                            key.start + run_start * chunk_size,
                            key.request_id,
                            seq,
                        )
                    )
                run_start = None
            if not jobs:
                return result

            with self._pending_fp_lock:
                for _, hashes, sci, _, _, _ in jobs:
                    self._pending_fp_hashes.update(hashes[sci:])
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            gpu_ctx = entry.cache_context if entry is not None else None
            for job in jobs:
                # Held before the job can run; released once it is indexed.
                self._hold_store_landing(seq)
                try:
                    if gpu_ctx is not None and gpu_ctx.cupy_stream is not None:
                        submit_callback_to_stream(
                            gpu_ctx.cupy_stream, CB_FINGERPRINTS_KIND, job
                        )
                    else:
                        self._fingerprint_queue.put_nowait(job)
                except Exception:
                    self._release_store_landing(seq)
                    raise
        except Exception:
            logger.exception(
                "CB fingerprint enqueue failed for request %s "
                "(does not affect store correctness)",
                key.request_id,
            )

        return result

    def _record_prompt(self, key: IPCCacheServerKey, session: Any) -> None:
        """Remember a stored prompt for the reorder planner. The chain starts
        at chunk 0 because a request's first store starts at its prefix hit.
        Never raises: reorder planning is best-effort."""
        try:
            chain = [
                TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0, key.end)
            ]
            self._prompt_store.record(
                (key.model_name, key.world_size, key.cache_salt),
                key.request_id,
                key.token_ids,
                chain,
            )
        except Exception:
            logger.exception("CB reorder: recording %s failed", key.request_id)

    def _drain_fingerprints_sync(self) -> None:
        """Sync-drain pending fingerprint registrations (the async drainer
        races at low max_tokens). Must not clear ``_pending_fp_hashes`` —
        only the async drainer owns that."""
        while True:
            try:
                job = self._fingerprint_queue.get_nowait()
            except QueueEmpty:
                break
            (
                tokens_in_range,
                chunk_hashes,
                start_chunk_idx,
                position_offset,
                rid,
                store_seq,
            ) = job
            try:
                n_new = self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
                self._emit_fingerprints_registered(rid, n_new)
            except Exception:
                logger.exception("CB fingerprint registration failed (sync drain)")
            finally:
                self._release_store_landing(store_seq)

    def _emit_fingerprints_registered(self, rid: str, num_chunks: int) -> None:
        """Publish CB_FINGERPRINTS_REGISTERED for one drained registration job.

        ``num_chunks`` counts only chunks the matcher newly indexed (skipped
        and deduplicated chunks excluded); nothing is published when it is
        zero — a re-store of known content is not a registration.
        """
        if num_chunks <= 0:
            return
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                session_id=rid,
                metadata={
                    "num_chunks": num_chunks,
                    # Only full chunks are hashed, so every indexed chunk
                    # covers exactly chunk_size tokens.
                    "num_tokens": num_chunks * self._token_range_matcher.chunk_size,
                },
            )
        )

    def _drain_fingerprint_queue(self) -> None:
        """Best-effort background drainer for _fingerprint_queue."""
        while not self._fingerprint_stop.is_set():
            try:
                job = self._fingerprint_queue.get(timeout=0.1)
            except QueueEmpty:
                continue
            (
                tokens_in_range,
                chunk_hashes,
                start_chunk_idx,
                position_offset,
                rid,
                store_seq,
            ) = job
            try:
                n_new = self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
                self._emit_fingerprints_registered(rid, n_new)
            except Exception:
                logger.exception("CB fingerprint registration failed (async)")
            finally:
                with self._pending_fp_lock:
                    self._pending_fp_hashes.difference_update(
                        chunk_hashes[start_chunk_idx:]
                    )
                self._release_store_landing(store_seq)

    # ------------------------------------------------------------------
    # Store landing: a lookup sees every store acknowledged before it
    # ------------------------------------------------------------------

    def _open_store_landing(self) -> int:
        """Take a landing sequence number for a store that is starting.

        It holds one part, released by :meth:`_close_store_landing`'s
        stream-ordered mark (or at once when no device work was submitted);
        each fingerprint job adds a part, released once the job is indexed.
        """
        with self._store_landing_lock:
            self._store_seq += 1
            seq = self._store_seq
            self._unlanded_stores[seq] = 1
        return seq

    def _hold_store_landing(self, seq: int) -> None:
        """Add one outstanding part to the landing of store ``seq``."""
        with self._store_landing_lock:
            self._unlanded_stores[seq] += 1

    def _release_store_landing(self, seq: int) -> None:
        """Release one part of store ``seq``; it lands at zero parts.
        Also the dispatcher handler for ``CB_STORE_LANDED_KIND``."""
        with self._store_landing_lock:
            parts = self._unlanded_stores.get(seq)
            if parts is None:
                return
            if parts <= 1:
                del self._unlanded_stores[seq]
            else:
                self._unlanded_stores[seq] = parts - 1

    def _close_store_landing(self, seq: int, instance_id: int | None) -> None:
        """Release the store's own landing part after its device work.

        With device work (``instance_id`` set) the release is a callback
        submitted last on the GPU context's stream, so it runs after the
        store's L1 commit (``finish_write_by_owner``) and its fingerprint
        enqueue. Otherwise, or if the callback cannot be submitted, the part
        is released now. Never raises.
        """
        stream = None
        if instance_id is not None:
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            if entry is not None:
                stream = entry.cache_context.cupy_stream
        if stream is not None:
            try:
                submit_callback_to_stream(stream, CB_STORE_LANDED_KIND, seq)
                return
            except Exception:
                logger.exception("CB store landing mark failed for store %d", seq)
        self._release_store_landing(seq)

    def _stores_landed_before(self, rid: str) -> bool:
        """Whether every store acknowledged before lookup ``rid`` arrived
        has landed: L1-committed on every rank and fingerprint-indexed.

        The first call fixes the lookup's ticket (the newest store sequence
        number) so later stores never delay it. Queued fingerprint jobs are
        indexed here rather than waiting for the background drainer. Past
        ``_STORE_VISIBILITY_TIMEOUT_S`` the lookup proceeds without them.

        Returns:
            ``True`` when the lookup may run, ``False`` to defer (the client
            re-polls).
        """
        self._drain_fingerprints_sync()
        now = time.monotonic()
        with self._store_landing_lock:
            ticket = self._cb_landing_tickets.pop(rid, None)
            if ticket is None:
                ticket = (self._store_seq, now + self._STORE_VISIBILITY_TIMEOUT_S)
            seq, deadline = ticket
            waiting = sum(1 for s in self._unlanded_stores if s <= seq)
            if waiting == 0:
                return True
            if now < deadline:
                # Drop tickets of lookups that stopped polling (aborted).
                for stale in [
                    r for r, (_, d) in self._cb_landing_tickets.items() if d < now
                ]:
                    del self._cb_landing_tickets[stale]
                self._cb_landing_tickets[rid] = ticket
                return False
        logger.warning(
            "CB lookup %s: %d earlier store(s) not indexed after %.1f s; "
            "looking up without them",
            rid,
            waiting,
            self._STORE_VISIBILITY_TIMEOUT_S,
        )
        return True
