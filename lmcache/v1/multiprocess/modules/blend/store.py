# SPDX-License-Identifier: Apache-2.0
"""Blend STORE hook: async chunk-fingerprint registration + drainers."""

# Standard
from queue import Empty as QueueEmpty
from typing import TYPE_CHECKING, Any

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
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)

#: Dispatcher kind for stream-ordered fingerprint enqueues.
CB_FINGERPRINTS_KIND = "cb_fingerprints"

#: One fingerprint-registration job:
#: (tokens_in_range, chunk_hashes, start_chunk_idx, position_offset,
#: request_id). Also the msgspec decode type for the dispatcher payload.
FpJob = tuple[list[int], list[bytes], int, int, str]


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

        Returns:
            The underlying ``LMCacheDrivenTransfer.store`` result
            (event handle, success).
        """
        result = self._transfer_module.store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

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
            tokens_in_range = list(key.token_ids)[key.start : key.end]
            # Chunk 0 is owned by the prefix lookup leg; skip its fingerprint.
            start_chunk_idx = 0 if key.start != 0 else 1
            job: FpJob = (
                tokens_in_range,
                chunk_hashes,
                start_chunk_idx,
                key.start,
                key.request_id,
            )
            with self._pending_fp_lock:
                self._pending_fp_hashes.update(chunk_hashes[start_chunk_idx:])
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            gpu_ctx = entry.cache_context if entry is not None else None
            if gpu_ctx is not None and gpu_ctx.cupy_stream is not None:
                submit_callback_to_stream(
                    gpu_ctx.cupy_stream, CB_FINGERPRINTS_KIND, job
                )
            else:
                self._fingerprint_queue.put_nowait(job)
        except Exception:
            logger.exception(
                "CB fingerprint enqueue failed for request %s "
                "(does not affect store correctness)",
                key.request_id,
            )

        return result

    def _drain_fingerprints_sync(self) -> None:
        """Sync-drain pending fingerprint registrations (the async drainer
        races at low max_tokens). Must not clear ``_pending_fp_hashes`` —
        only the async drainer owns that."""
        while True:
            try:
                job = self._fingerprint_queue.get_nowait()
            except QueueEmpty:
                break
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset, rid = job
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
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset, rid = job
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
