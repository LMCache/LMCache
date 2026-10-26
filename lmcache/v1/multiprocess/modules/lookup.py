# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from concurrent.futures import Future
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, replace
from functools import partial
from typing import Sequence, cast
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    MemoryTags,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.kv_load_policy import (
    DefaultKVLoadPolicy,
    KVLoadContext,
    KVLoadPolicy,
    create_kv_load_policy,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.token_codec import num_packed_tokens, unpack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.tp_reshard import (
    HeteroTPFallback,
    HeteroTPPlan,
    SourceStage,
)

logger = init_logger(__name__)

# Stateless, so one instance can serve every LookupModule.
_DEFAULT_LOAD_POLICY = DefaultKVLoadPolicy({})


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
    chunk_hashes = ctx.token_hasher.compute_packed_chunk_hashes(
        key.token_bytes, start=key.start, end=key.end
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
class _HeteroTPState:
    """Cross-TP fallback of one lookup.

    See ``docs/design/v1/multiprocess/hetero_tp.md``. The lookup first runs
    at its own world size (the primary stage). If that finds no chunk, it
    runs each source stage ``plan`` gives it; a hit there is converted into
    objects keyed and shaped for the lookup's own world size, which a final
    L1-only prefetch then locks like any other hit.

    Attributes:
        key: The lookup's own key.
        suffix_hashes: Chunk hashes of the looked-up range.
        group_layout_descs: Per-object-group layouts at the lookup's own
            world size.
        plan: Source stages to try, and the conversion of a source hit.
        in_source: Whether the running stage is a source stage.
        source_result: The source stage's result, kept while its hit is
            converted.
        reshard: The background conversion of a source-stage hit; it
            resolves to whether it submitted the final stage.
    """

    key: IPCCacheServerKey
    suffix_hashes: list[bytes]
    group_layout_descs: dict[int, MemoryLayoutDesc]
    plan: HeteroTPPlan
    in_source: bool = False
    source_result: PrefetchResult | None = None
    reshard: Future[bool] | None = None


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    # Sliding window of each submitted key row, in row order; empty when no
    # prefetch was submitted.
    row_windows: tuple[int, ...]
    request_id: str
    # Chunk-aligned tokens LMCache was asked to look up, past the engine-covered
    # prefix (hit-rate denominator); 0 on early exits.
    requested_tokens: int
    num_object_groups: int = 1
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    # First chunk of the looked-up range (``key.start``); chunks below it are
    # engine-covered and skipped, so the status offsets hits back to absolute.
    start_chunk: int = 0
    # Engine-held chunks below ``key.start``, reported as ``apc_covered_tokens``;
    # equals ``start_chunk`` except on a declined lookup, which offsets no hits.
    engine_covered_chunks: int = 0
    # Request-level view on every path: the whole prompt and the engine's own
    # prefix-cache hit from the ``engine_computed_tokens`` hint (0 without it).
    prompt_tokens: int = 0
    engine_hit_tokens: int = 0
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""
    # Names the ``lookup()`` branch that returned before submitting a prefetch
    # task; empty on the normal path.
    early_exit_reason: str = ""
    # Set while a cross-TP fallback may still replace ``handle``. A new
    # stage swaps ``handle``, ``row_windows`` and ``hetero`` together under
    # the module's ``_prefetch_job_lock``. Otherwise ``hetero`` is cleared
    # only by the thread that took the current stage's result, which the
    # storage manager hands out once; a failed reshard's result is claimed by
    # clearing ``hetero`` under the lock.
    hetero: _HeteroTPState | None = None


class LookupModule(DiscoverableModule):
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS, FREE_LOOKUP_LOCKS, and
    END_SESSION request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
        load_policy: Decides per lookup whether to serve the request or let
            the engine recompute it. Defaults to ``DefaultKVLoadPolicy``.
    """

    module_name = "lookup"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the lookup module; it applies to every configuration."""
        return cls(
            build_ctx.engine_context,
            create_kv_load_policy(
                build_ctx.mp_config.kv_load_policy,
                build_ctx.mp_config.runtime_plugin_config.extra_config,
            ),
            hetero_tp_source_world_sizes=(
                build_ctx.mp_config.hetero_tp_source_world_sizes
            ),
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        load_policy: KVLoadPolicy = _DEFAULT_LOAD_POLICY,
        hetero_tp_source_world_sizes: Sequence[int] = (),
    ) -> None:
        self._ctx = ctx
        self._load_policy = load_policy
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        self._hetero_tp = HeteroTPFallback(
            ctx.layout_desc_registry, ctx.storage_manager, hetero_tp_source_world_sizes
        )
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
        """Stop the reshard threads after every submitted reshard finishes.

        Queued reshards still run: cancelling one would leave its source
        read locks held until their TTL.
        """
        self._hetero_tp.close()

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
        via query_prefetch_status. When the KV load policy declines the
        request, no prefetch is submitted and the job reports zero hits.

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
            self._register_empty_prefetch_job(key, early_exit_reason="no_gpu_context")
            return

        num_kv_readers = key.require_num_kv_readers()

        chunk_hashes = self._ctx.token_hasher.compute_packed_chunk_hashes(
            key.token_bytes, end=key.end
        )
        if not chunk_hashes:
            self._register_empty_prefetch_job(
                key, early_exit_reason="empty_chunk_hashes"
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
                        "seq_len": key.num_tokens,
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                        "token_ids": unpack_token_ids(key.token_bytes),
                    },
                )
            )

        # Submit one key row per (object group, kv rank), each with its own
        # attention window.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(key.token_bytes)

        start_chunk = min(max(0, key.start // self._ctx.chunk_size), len(chunk_hashes))
        # The engine-covered prefix is never looked up, so it is excluded from
        # the hit-rate denominator and reported separately.
        looked_up_tokens = requested_tokens - start_chunk * self._ctx.chunk_size
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
            self._register_empty_prefetch_job(
                key, early_exit_reason="no_group_layout_descs"
            )
            return

        if not self._load_policy.should_load(
            KVLoadContext(
                request_id=key.request_id,
                model_name=model_name,
                chunk_size=self._ctx.chunk_size,
                num_lookup_tokens=requested_tokens,
                request_configs=key.request_configs,
            )
        ):
            # Report a miss without prefetching: nothing is fetched from L2
            # and no read lock is taken, so the engine has nothing to release.
            # requested_tokens stays non-zero so hit-rate metrics count the
            # declined tokens as unserved.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                engine_covered_chunks=start_chunk,
                early_exit_reason="load_policy_recompute",
            )
            return

        # Keep the covered prefix's recency without locking or prefetching
        # it. After the load policy, so a declined lookup leaves no trace.
        if start_chunk:
            self._ctx.storage_manager.touch_cached_keys(
                [
                    obj_key
                    for group_keys in ipc_key_to_object_keys(
                        key,
                        chunk_hashes[:start_chunk],
                        list(range(attn_desc.num_object_groups)),
                    )
                    for obj_key in group_keys
                ]
            )

        # Prefetch only [start, end); status offsets hits back to absolute.
        suffix_hashes = chunk_hashes[start_chunk:]
        if not suffix_hashes:
            # Everything is engine-covered: nothing to lock or prefetch. The
            # empty job still reports the covered prefix as the hit.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
            )
            return

        handle, row_windows = self._submit_prefetch(
            key, suffix_hashes, group_layout_descs, attn_desc, num_kv_readers
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                row_windows=row_windows,
                request_id=key.request_id,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
                prompt_tokens=num_packed_tokens(key.token_bytes),
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=model_name,
                cache_salt=key.cache_salt,
                hetero=self._hetero_tp_state(
                    key, suffix_hashes, attn_desc, group_layout_descs
                ),
            )
        )

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

        With a cross-TP fallback configured, a finished stage can make this
        call submit the next stage and return None. A source-stage hit is
        resharded into L1 on a background thread, which then submits the
        final stage; until then this call returns None.

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

        return self._consume_prefetch_result(job)

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

        With a cross-TP fallback configured, the wait follows each stage the
        fallback submits, and the background reshard; ``timeout`` covers all
        of them.

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

        deadline = time.monotonic() + timeout
        remaining = timeout
        while True:
            with self._prefetch_job_lock:
                handle = job.handle
                reshard = self._pending_reshard(job)
            if reshard is not None:
                # The finished source stage's handle never completes again;
                # the reshard is what submits the next prefetch.
                if not wait_futures([reshard], remaining).done:
                    return None
            elif not self._ctx.storage_manager.wait_prefetch_status(handle, remaining):
                return None
            result = self.query_prefetch_status(request_id)
            # A cross-TP fallback started a reshard or swapped in a new
            # prefetch: wait for that too.
            with self._prefetch_job_lock:
                moved = job.handle is not handle or (
                    self._pending_reshard(job) is not None
                )
            if result is not None or not moved:
                return result
            remaining = max(0.0, deadline - time.monotonic())

    @request_handler(HandlerType.BLOCKING)
    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
        extra_ranges: tuple[tuple[int, int], ...] = (),
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Only the keys the prefetch actually read-locked are released: the
        key's range is floored to the current lookup's ``start``, and each
        extra range resolves against its own boundaries.

        Releases the same per-object count the lookup reserved
        (``key.num_kv_readers``).

        The release is derived from the lookup result recorded on the
        session (hit length, lock model, L1 owners). If that result has not
        been consumed yet it is handed over here first. While the prefetch
        is still running nothing is released: its locks still belong to the
        prefetch controller and expire with the read-lock TTL. A running
        cross-TP reshard is the exception: this call blocks until it finishes
        (no timeout), so the final stage's locks are released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Legacy wire field; ignored (kept for payload arity).
            extra_ranges: Lock ranges left over from this request's earlier,
                abandoned lookups, as ``(start, end)`` token pairs. Each end
                must be that lookup's hit boundary, so the range resolves to
                exactly the keys it locked (sliding-window groups translate
                from the hit).
        """
        if key.start >= key.end and not extra_ranges:
            return

        session = self._ctx.session_manager.get_or_create(key.request_id)
        if session.prefetch_hit_chunks < 0:
            # The lookup result has not reached the session yet (no
            # QUERY/WAIT_PREFETCH_STATUS consumed it). Hand it over now so
            # the release below sees the real hit length, lock model and L1
            # owners. While the prefetch is still running its locks belong
            # to the prefetch controller, which settles them itself in
            # _finish_request; releasing them here would double-decrement,
            # so leave them to the controller and the read-lock TTL.
            with self._prefetch_job_lock:
                job = self._prefetch_jobs.get(key.request_id)
            if (
                job is not None
                and self._consume_prefetch_result(job, advance_hetero=False) is None
            ):
                logger.warning(
                    "free_lookup_locks for request %s while its prefetch is "
                    "still running; leaving its locks to the prefetch controller",
                    key.request_id,
                )
                return

        hit_chunks = session.prefetch_hit_chunks
        if hit_chunks < 0:
            # No lookup result exists (never looked up, or the session was
            # recreated after END_SESSION). L1 locks are anonymous refcounts:
            # guessing a range could strip a concurrent reader's lock.
            logger.warning(
                "free_lookup_locks for request %s without a lookup result; "
                "nothing to release",
                key.request_id,
            )
            return
        # Release exactly the groups the prefetch locked (std lookup: all;
        # CB prefix leg: its prefix set) -- releasing an unlocked group
        # would drop another request's lock on the shared object key.
        lookup_key = session.lookup_ipc_key
        lock_floor = lookup_key.start if lookup_key is not None else 0
        obj_keys: list[ObjectKey] = []
        if key.start < key.end:
            # The lookup never locked below its own ``start``.
            obj_keys.extend(
                resolve_prefetched_obj_keys(
                    self._ctx,
                    replace(key, start=max(key.start, lock_floor)),
                    hit_chunks,
                    session.prefetch_locked_gids,
                )
            )
        for range_start, range_end in extra_ranges:
            if range_start >= range_end:
                continue
            # An abandoned lookup's range ends at its hit boundary, so the
            # range is its own lock model: resolve with that hit, not the
            # current lookup's.
            obj_keys.extend(
                resolve_prefetched_obj_keys(
                    self._ctx,
                    replace(key, start=range_start, end=range_end),
                    range_end // self._ctx.chunk_size,
                    session.prefetch_locked_gids,
                )
            )
        if not obj_keys:
            return

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys,
            read_locks=key.require_num_kv_readers(),
            l1_owners=self._ctx.get_read_owners(key.request_id),
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
        self._ctx.storage_manager.touch_cached_keys(obj_keys)

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

    def _register_empty_prefetch_job(
        self,
        key: IPCCacheServerKey,
        requested_tokens: int = 0,
        num_object_groups: int = 1,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        start_chunk: int = 0,
        engine_covered_chunks: int = 0,
        early_exit_reason: str = "",
    ) -> None:
        """Register a job that resolves immediately without a prefetch.

        Args:
            key: The lookup's cache key.
            requested_tokens: Hit-rate denominator tokens; 0 on early exits.
            num_object_groups: Object groups of the request's layout.
            attn_desc: The request's attention window descriptor.
            start_chunk: First chunk of the looked-up range; reported as the
                hit when everything below it is engine-covered.
            engine_covered_chunks: Engine-covered chunks below ``key.start``,
                reported as ``apc_covered_tokens``.
            early_exit_reason: The ``lookup()`` branch that returned before
                submitting a prefetch; empty when the lookup simply had
                nothing to fetch.
        """
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
                requested_tokens=requested_tokens,
                num_object_groups=num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=engine_covered_chunks,
                prompt_tokens=num_packed_tokens(key.token_bytes),
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=key.model_name,
                cache_salt=key.cache_salt,
                early_exit_reason=early_exit_reason,
            )
        )

    def _consume_prefetch_result(
        self, job: _PrefetchJob, advance_hetero: bool = True
    ) -> int | None:
        """Hand a finished prefetch result over to the request's session.

        Takes the result from the storage manager (each result is returned
        once), records the model-wide hit length, lock model and retained L1
        owners on the session, emits ``MP_LOOKUP_PREFETCH_END`` and drops the
        job. This is the only path that moves lookup state into the session,
        so ``query_prefetch_status`` and ``free_lookup_locks`` cannot disagree
        about what a lookup locked.

        Args:
            job: The lookup's job.
            advance_hetero: Whether a cross-TP fallback may submit its next
                stage. ``free_lookup_locks`` passes False because the request
                is being released, so the current result is final; it also
                blocks until a running reshard finishes, so the release sees
                the final stage's locks.

        Returns:
            The hit chunk count, or None while the lookup is still running:
            its prefetch has not finished, a reshard is running, or a
            cross-TP stage just submitted the next one (the job is kept so a
            later call can resolve it).
        """
        with self._prefetch_job_lock:
            reshard = self._pending_reshard(job)
        if reshard is not None and advance_hetero and not reshard.done():
            return None
        row_windows: tuple[int, ...]
        if reshard is not None and not reshard.result():
            # The final stage was never submitted: answer with the source
            # stage's result and lock nothing at this world size. Nothing
            # else hands this result out once, so the consumer that clears
            # ``hetero`` owns it.
            with self._prefetch_job_lock:
                state = job.hetero
                if state is None or state.reshard is not reshard:
                    return None
                job.hetero = None
            result, row_windows = cast(PrefetchResult, state.source_result), ()
        else:
            with self._prefetch_job_lock:
                handle, row_windows = job.handle, job.row_windows
            queried = self._ctx.storage_manager.query_prefetch_status(handle)
            if queried is None:
                return None
            result = queried
        if job.hetero is not None:
            try:
                final_windows = self._advance_hetero_tp(
                    job, job.hetero, result, row_windows, advance_hetero
                )
            except Exception:
                # RPC boundary: a failed fallback must still answer the lookup,
                # or the job never resolves and its session is never released.
                logger.exception(
                    "Cross-TP fallback failed for request %s; reporting a miss",
                    job.request_id,
                )
                job.hetero = None
                final_windows = ()
            if final_windows is None:
                return None
            row_windows = final_windows
        if row_windows:
            found_count, _retain = fold_unfold_grouped(result.hit_cells, row_windows)
            l1_found_count, _l1_retain = fold_unfold_grouped(
                result.l1_hit_cells, row_windows
            )
        else:
            # Nothing was submitted (early exit), or a cross-TP source stage
            # ended without a reshard, so nothing is locked at this world size.
            found_count = 0
            l1_found_count = 0

        # The fold covers only the looked-up sub-range: the tier split is
        # attributed on it, so the covered prefix never counts as an L1 hit.
        l1_chunks = min(l1_found_count, found_count)
        l2_chunks = found_count - l1_chunks
        # The engine resolves the hit in absolute chunks; shift it back.
        found_count += job.start_chunk

        # Record the model-wide hit length on the session so a later
        # free_lookup_locks can reconstruct which keys the prefetch
        # read-locked (see ``unfold``: full-attention groups lock the whole
        # hit prefix, sliding-window groups only its in-window suffix).
        session = self._ctx.session_manager.get_or_create(job.request_id)
        session.record_prefetch_result(
            found_count,
            tuple(range(job.attn_desc.num_object_groups)),
            result.l1_owners,
        )

        chunk_size = self._ctx.chunk_size
        covered_tokens = job.engine_covered_chunks * chunk_size
        hit_tokens = (l1_chunks + l2_chunks) * chunk_size
        # Prefix served by either cache: both ranges start at 0 and overlap in
        # the sub-chunk gap below the lookup start, so take the longer, not the sum.
        request_hit_tokens = max(job.engine_hit_tokens, covered_tokens + hit_tokens)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    # Hit-rate counters cover only the looked-up range; the
                    # engine-covered prefix gets its own field.
                    "apc_covered_tokens": covered_tokens,
                    "hit_tokens": hit_tokens,
                    "l1_hit_tokens": l1_chunks * chunk_size,
                    "l2_hit_tokens": l2_chunks * chunk_size,
                    # Request-level view: the whole prompt as denominator and
                    # what the engine plus LMCache served as numerator.
                    "prompt_tokens": job.prompt_tokens,
                    "engine_hit_tokens": job.engine_hit_tokens,
                    "request_hit_tokens": request_hit_tokens,
                    "l1_hit_keys": result.l1_hit_count,
                    "l2_hit_keys": result.l2_hit_count,
                    "early_exit_reason": job.early_exit_reason,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(job.request_id, None)

        return found_count

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _hetero_tp_state(
        self,
        key: IPCCacheServerKey,
        suffix_hashes: list[bytes],
        attn_desc: AttnWindowDesc,
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> _HeteroTPState | None:
        """Build the cross-TP fallback for a lookup, or ``None`` if it has none.

        Args:
            key: The lookup's key.
            suffix_hashes: Chunk hashes of the looked-up range.
            attn_desc: Attention windows at the lookup's world size.
            group_layout_descs: Per-object-group layouts at the lookup's
                world size.

        Returns:
            The fallback state, or ``None`` if the lookup does not qualify.
        """
        plan = self._hetero_tp.plan(key, suffix_hashes, attn_desc, group_layout_descs)
        if plan is None:
            return None
        return _HeteroTPState(key, suffix_hashes, group_layout_descs, plan)

    def _advance_hetero_tp(
        self,
        job: _PrefetchJob,
        state: _HeteroTPState,
        result: PrefetchResult,
        row_windows: tuple[int, ...],
        advance: bool,
    ) -> tuple[int, ...] | None:
        """Move a lookup to its next cross-TP stage after a stage finished.

        Primary stage: a hit, or nothing left to try, makes ``result`` final.
        Otherwise the next candidate's source stage is submitted. Source
        stage: a hit is handed to a background reshard, which submits the
        final L1-only prefetch; a miss moves on to the next candidate. With
        ``advance`` False nothing is submitted: a source stage's locks are
        released and the lookup reports a miss.

        Args:
            job: The lookup's job.
            state: ``job.hetero``.
            result: The finished stage's result, already taken from the
                storage manager.
            row_windows: The windows ``result`` was submitted with.
            advance: Whether a new stage may be submitted.

        Returns:
            ``None`` if a new stage replaced ``job.handle`` or a reshard
            started (the lookup is still running); otherwise the row windows
            to fold ``result`` with as the final result, empty when nothing
            is locked at the lookup's world size.
        """
        found, _ = fold_unfold_grouped(result.hit_cells, row_windows)
        if not state.in_source:
            # A partial primary hit skips the fallback even if a source TP
            # holds a longer prefix; see "Known limitation" in hetero_tp.md.
            stage = state.plan.next_source() if found == 0 and advance else None
            if stage is None:
                job.hetero = None
                return row_windows
            self._submit_source_stage(job, state, stage)
            return None
        if found > 0 and advance:
            state.source_result = result
            state.reshard = state.plan.convert(
                found, result, partial(self._submit_final_stage, job, state)
            )
            return None
        if found > 0:
            state.plan.release(found, result)
        stage = state.plan.next_source() if advance else None
        if stage is not None:
            self._submit_source_stage(job, state, stage)
            return None
        job.hetero = None
        return ()

    def _submit_final_stage(self, job: _PrefetchJob, state: _HeteroTPState) -> bool:
        """Submit the L1-only prefetch of a finished reshard.

        Runs on a reshard thread, also after a failed reshard: the final
        stage then reports whatever reached L1 (usually nothing).

        Args:
            job: The lookup's job.
            state: ``job.hetero``.

        Returns:
            Whether the final stage was submitted and swapped into ``job``.
        """
        try:
            # L1-only: the reshard just wrote these objects to L1; looking
            # past them in L2 would only repeat the primary stage's miss.
            handle, row_windows = self._submit_prefetch(
                state.key,
                state.suffix_hashes,
                state.group_layout_descs,
                job.attn_desc,
                state.key.require_num_kv_readers(),
                allowed_l2_memory_tags=frozenset(),
            )
        except Exception:
            logger.exception(
                "Cross-TP final stage failed for request %s; reporting a miss",
                job.request_id,
            )
            return False
        self._swap_prefetch(job, handle, row_windows, None)
        return True

    def _pending_reshard(self, job: _PrefetchJob) -> Future[bool] | None:
        """Return ``job``'s unconsumed reshard; call under ``_prefetch_job_lock``."""
        return job.hetero.reshard if job.hetero is not None else None

    def _submit_source_stage(
        self, job: _PrefetchJob, state: _HeteroTPState, stage: SourceStage
    ) -> None:
        """Submit a source stage's prefetch and make it the job's stage.

        Args:
            job: The lookup's job.
            state: ``job.hetero``.
            stage: The source stage from ``state.plan``.
        """
        state.in_source = True
        self._swap_prefetch(
            job,
            *self._submit_prefetch(
                stage.key,
                state.suffix_hashes,
                stage.group_layout_descs,
                stage.attn_desc,
                stage.num_kv_readers,
            ),
            state,
        )

    def _submit_prefetch(
        self,
        key: IPCCacheServerKey,
        suffix_hashes: list[bytes],
        group_layout_descs: dict[int, MemoryLayoutDesc],
        attn_desc: AttnWindowDesc,
        num_kv_readers: int,
        allowed_l2_memory_tags: frozenset[MemoryTags] | None = None,
    ) -> tuple[PrefetchHandle, tuple[int, ...]]:
        """Submit a prefetch over ``suffix_hashes`` for every object group.

        Args:
            key: Key whose model, world size and salt name the objects.
            suffix_hashes: Chunk hashes to prefetch.
            group_layout_descs: Per-object-group layouts at ``key``'s world
                size.
            attn_desc: Attention windows at ``key``'s world size.
            num_kv_readers: Read locks to take per object.
            allowed_l2_memory_tags: L2 tags the prefetch may load from;
                ``None`` allows all, an empty set restricts it to L1.

        Returns:
            The prefetch handle and the per-row windows to fold its result
            with.
        """
        spec = PrefetchTaskSpec(
            key_groups=ipc_key_to_grouped_object_keys(
                key,
                suffix_hashes,
                list(range(attn_desc.num_object_groups)),
                group_layout_descs,
                attn_desc,
            ),
            num_kv_readers=num_kv_readers,
            allowed_l2_memory_tags=allowed_l2_memory_tags,
        )
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec, external_request_id=key.request_id
        )
        return handle, tuple(row.sliding_window_size for row in spec.key_groups)

    def _swap_prefetch(
        self,
        job: _PrefetchJob,
        handle: PrefetchHandle,
        row_windows: tuple[int, ...],
        hetero: _HeteroTPState | None,
    ) -> None:
        """Point ``job`` at a newly submitted stage; readers see all or none.

        Args:
            job: The lookup's job.
            handle: The new stage's prefetch handle.
            row_windows: The new stage's per-row windows.
            hetero: The fallback state after this stage; ``None`` for the
                final stage.
        """
        with self._prefetch_job_lock:
            job.handle, job.row_windows, job.hetero = handle, row_windows, hetero

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
