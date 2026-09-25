# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass
from functools import partial
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import (
    SKIP_L2_REQUEST_CONFIG_KEY,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


def resolve_prefetched_obj_keys(
    ctx: MPCacheServerContext,
    key: IPCCacheServerKey,
    hit_chunks: int,
    locked_gids: tuple,
    group_windows: tuple[int, ...] | None = None,
) -> list[ObjectKey]:
    """Resolve the subset of a request range that lookup actually locked.

    ``key.worker_id=None`` resolves every KV rank for scheduler-owned cleanup.
    A worker-specific key resolves only that worker's shard (or one MLA reader
    share), which is required for per-instance RETRIEVE failure cleanup.
    """
    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(
        list(key.token_ids), start=key.start, end=key.end
    )
    if not chunk_hashes:
        return []

    start_chunk = key.start // ctx.chunk_size
    end_chunk = start_chunk + len(chunk_hashes)
    if group_windows is None:
        group_windows = tuple(
            ctx.layout_desc_registry.find_attn_desc(
                key.model_name, key.world_size
            ).num_chunks_in_sw
        )

    obj_keys: list[ObjectKey] = []
    for group_idx, window in enumerate(group_windows):
        if locked_gids and group_idx not in locked_gids:
            continue
        if hit_chunks < 0:
            if window >= 0:
                continue
            lo, hi = start_chunk, end_chunk
        else:
            # Locked range per ``unfold``: the whole hit prefix for full
            # attention, its trailing ``window`` chunks otherwise.
            lo = 0 if window < 0 else max(0, hit_chunks - window)
            lo = max(lo, start_chunk)
            hi = min(hit_chunks, end_chunk)
        if lo >= hi:
            continue
        group_hashes = chunk_hashes[lo - start_chunk : hi - start_chunk]
        obj_keys.extend(ipc_key_to_object_keys(key, group_hashes, [group_idx])[0])
    return obj_keys


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    # Sliding window of each submitted key row, in row order; empty when no
    # prefetch was submitted.
    row_windows: tuple[int, ...]
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 on the early-exit paths (see
    # ``early_exit_reason``).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    num_object_groups: int = 1
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""
    # Names the ``lookup()`` branch that returned before submitting a prefetch
    # task; empty on the normal path.
    early_exit_reason: str = ""


class LookupModule:
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS,
    QUERY_PREFETCH_LOOKUP_HITS, FREE_LOOKUP_LOCKS, and END_SESSION
    request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        return {
            "active_prefetch_jobs": self._active_prefetch_count(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""
        pass

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    @request_handler(HandlerType.BLOCKING)
    def lookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status. ``lmcache.skip_l2=True`` in the request
        configuration limits the lookup to objects already visible in L1.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        model_name, world_size = key.model_name, key.world_size
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._ctx.layout_desc_registry.find(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    row_windows=(),
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    early_exit_reason="no_gpu_context",
                )
            )
            return

        num_kv_readers = key.require_num_kv_readers()

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), end=key.end
        )
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    row_windows=(),
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    early_exit_reason="empty_chunk_hashes",
                )
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._ctx.event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # Submit one key row per (object group, kv rank), each with its own
        # attention window.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.begin_lookup(key, tuple(attn_desc.num_chunks_in_sw))

        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(
            model_name, world_size
        )
        if not group_layout_descs:
            logger.error(
                "No group layout descs found for model %s with world size %d "
                "during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    row_windows=(),
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                    early_exit_reason="no_group_layout_descs",
                )
            )
            return

        spec = PrefetchTaskSpec(
            key_groups=ipc_key_to_grouped_object_keys(
                key,
                chunk_hashes,
                list(range(attn_desc.num_object_groups)),
                group_layout_descs,
                attn_desc,
            ),
            num_kv_readers=num_kv_readers,
        )
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec,
            external_request_id=key.request_id,
            skip_l2=(key.request_configs or {}).get(SKIP_L2_REQUEST_CONFIG_KEY) is True,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                row_windows=tuple(row.sliding_window_size for row in spec.key_groups),
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                model_name=model_name,
                cache_salt=key.cache_salt,
            )
        )

    @request_handler(HandlerType.BLOCKING)
    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        # The storage manager reports the prefix hit in chunks once the
        # prefetch has finished, and None before that.
        return self._ctx.storage_manager.query_prefetch_lookup_hits(job.handle)

    @request_handler(HandlerType.BLOCKING)
    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        result = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if result is None:
            return None
        found_rows = result.hit_cells
        hit_counts = (result.l1_hit_count, result.l2_hit_count)

        if job.row_windows:
            found_count, _retain = fold_unfold_grouped(found_rows, job.row_windows)
        else:
            # Nothing was submitted (early exit), so nothing can be hit.
            found_count = 0

        # Record the model-wide hit length on the session so a later
        # free_lookup_locks can reconstruct which keys the prefetch
        # read-locked (see ``unfold``: full-attention groups lock the whole
        # hit prefix, sliding-window groups only its in-window suffix).
        session = self._ctx.session_manager.get_or_create(job.request_id)
        session.record_prefetch_result(
            found_count,
            tuple(range(job.attn_desc.num_object_groups)),
        )

        # The hit counts are cells (one per row and chunk); one chunk spans
        # one cell per row, so divide by the row count to get chunks and
        # attribute the remainder of the hit to L2.
        l1_cells, _l2_cells = hit_counts
        num_rows = max(len(job.row_windows), 1)
        l1_chunks = min(l1_cells // num_rows, found_count)
        l2_chunks = found_count - l1_chunks
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    "hit_tokens": found_count * self._ctx.chunk_size,
                    "l1_hit_tokens": l1_chunks * self._ctx.chunk_size,
                    "l2_hit_tokens": l2_chunks * self._ctx.chunk_size,
                    "early_exit_reason": job.early_exit_reason,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)

        return found_count

    @request_handler(HandlerType.BLOCKING)
    def wait_prefetch_status(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Block until a prefetch job completes, then return its chunk count.

        Like query_prefetch_status, but waits for the daemon to publish the
        result instead of returning None while the prefetch is still in
        progress, so the caller does not have to busy-poll. The job entry is
        removed once a non-None result is returned (exactly-once semantics).

        Args:
            request_id: The external request ID passed in the lookup key.
            timeout: Maximum number of seconds to wait for the prefetch.

        Returns:
            Chunk count (int) when done, None if the wait timed out, 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if not self._ctx.storage_manager.wait_prefetch_status(job.handle, timeout):
            return None
        return self.query_prefetch_status(request_id)

    @request_handler(HandlerType.BLOCKING)
    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Only the keys the prefetch actually read-locked are released.

        Releases the same per-object count the lookup reserved
        (``key.num_kv_readers``).

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        if key.start >= key.end:
            return

        hit_chunks = self._ctx.session_manager.get_or_create(
            key.request_id
        ).prefetch_hit_chunks
        if hit_chunks < 0:
            logger.warning(
                "free_lookup_locks for request %s before its prefetch result "
                "was consumed; releasing full-attention groups only",
                key.request_id,
            )

        # Release exactly the groups the prefetch locked (std lookup: all;
        # CB prefix leg: its prefix set) -- releasing an unlocked group
        # would drop another request's lock on the shared object key.
        locked_gids = self._ctx.session_manager.get_or_create(
            key.request_id
        ).prefetch_locked_gids
        obj_keys = resolve_prefetched_obj_keys(self._ctx, key, hit_chunks, locked_gids)

        if not obj_keys:
            return

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys, read_locks=key.require_num_kv_readers()
        )

    @request_handler(HandlerType.BLOCKING)
    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self._ctx.session_manager.remove(request_id)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch",
                request_id,
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = self._all_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _all_object_keys(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> list[ObjectKey]:
        """Resolve every object key of a request across all object groups.

        The object-group count is read from the layout registry for
        ``key``'s ``(model_name, world_size)``. The order of the returned keys
        is unspecified.

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.

        Returns:
            The object keys of all groups and kv ranks for ``chunk_hashes``.
        """
        num_groups = self._ctx.layout_desc_registry.find_attn_desc(
            key.model_name, key.world_size
        ).num_object_groups
        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))
        return [obj_key for group_keys in per_group for obj_key in group_keys]

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
