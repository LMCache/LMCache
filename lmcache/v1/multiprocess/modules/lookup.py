# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
import threading
import time

# First Party
from lmcache.v1.common.attn_window import DEFAULT_ATTN_WINDOW_DESC, AttnWindowDesc
from lmcache.v1.common.custom_types import (
    READ_LIFETIME_V1,
    IPCCacheServerKey,
    SessionKey,
)
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.token_codec import unpack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.engine_context import (
    MPCacheServerContext,
    check_chunk_hash_key,
    finish_lookup_read,
)
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
from lmcache.v1.multiprocess.session import Session

logger = init_logger(__name__)

# Stateless, so one instance can serve every LookupModule.
_DEFAULT_LOAD_POLICY = DefaultKVLoadPolicy({})
# Bounded tombstone table preventing delayed LOOKUP RPCs from resurrecting
# a lookup attempt whose END_SESSION arrived first.
_MAX_CLOSED_LOOKUPS = 16_384


def _key_chunk_hashes(
    hasher: TokenHasher,
    key: IPCCacheServerKey,
    start: int = 0,
    end: int | None = None,
) -> list[bytes]:
    """Chunk hashes of ``key`` for ``[start, end)``.

    Hashes the key's tokens, or takes the hashes a client computed instead
    (see ``IPCCacheServerKey.chunk_hashes``); both give the same result.

    Args:
        hasher: The server's token hasher.
        key: Key carrying ``token_bytes`` or ``chunk_hashes``.
        start: Token-level start index (must be chunk-aligned).
        end: Token-level end index (must be chunk-aligned).

    Returns:
        List of ``bytes`` hash values for chunks in ``[start, end)``.
    """
    if not key.chunk_hashes:
        return hasher.compute_packed_chunk_hashes(key.token_bytes, start=start, end=end)
    hashes = hasher.client_chunk_hashes(key)
    num_complete = len(hashes)
    # An arriving key holds exactly end // chunk_size hashes
    # (check_chunk_hash_key); copies narrowed to a lock range keep them all.
    if end is not None:
        num_complete = min(num_complete, end // hasher.chunk_size)
    return hashes[start // hasher.chunk_size : num_complete]


def resolve_prefetched_obj_keys(
    ctx: MPCacheServerContext,
    key: IPCCacheServerKey,
    hit_chunks: int,
    locked_gids: tuple,
) -> list[ObjectKey]:
    """Resolve the subset of a request range that lookup actually locked.

    ``key.worker_id=None`` resolves every KV rank for scheduler-owned cleanup.
    The per-group windows and shard replication come from the registration for
    ``(key.model_name, key.world_size)``.
    """
    registry = ctx.layout_desc_registry
    chunk_size = registry.get_chunk_size(key.model_name, key.world_size)
    token_hasher = registry.get_token_hasher(key.model_name, key.world_size)
    chunk_hashes = _key_chunk_hashes(token_hasher, key, start=key.start, end=key.end)
    if not chunk_hashes:
        return []

    start_chunk = key.start // chunk_size
    end_chunk = start_chunk + len(chunk_hashes)
    group_windows = registry.find_attn_desc(
        key.model_name, key.world_size
    ).num_chunks_in_sw
    shard_replication_desc = registry.find_shard_replication_desc(
        key.model_name, key.world_size
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
        obj_keys.extend(
            ipc_key_to_object_keys(
                key, group_hashes, [group_idx], shard_replication_desc
            )[0]
        )
    return obj_keys


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    # Sliding window of each submitted key row, in row order; empty when no
    # prefetch was submitted.
    row_windows: tuple[int, ...]
    request_id: str
    # Tokens LMCache was asked to look up, past the engine-covered prefix
    # (hit-rate denominator): the full chunks, plus the sub-chunk tail when the
    # lookup probes the unfull chunk; 0 on early exits.
    requested_tokens: int
    chunk_size: int
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
    # The lookup key, for the unfull-chunk stage that runs once the hit is
    # known; None on early exits, which never probe an unfull chunk.
    unfull_chunk_lookup_key: IPCCacheServerKey | None = None
    # Named lookup_id and injective L1 read_owner (empty on legacy lookups).
    lookup_id: str = ""
    read_owner: str = ""
    job_key: SessionKey = ""
    consuming: bool = False


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
        build_ctx.add_experimental(READ_LIFETIME_V1)
        return cls(
            build_ctx.engine_context,
            create_kv_load_policy(
                build_ctx.mp_config.kv_load_policy,
                build_ctx.mp_config.runtime_plugin_config.extra_config,
            ),
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        load_policy: KVLoadPolicy = _DEFAULT_LOAD_POLICY,
    ) -> None:
        self._ctx = ctx
        self._load_policy = load_policy
        self._prefetch_jobs: dict[SessionKey, _PrefetchJob] = {}
        self._closed_lookups: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._prefetch_job_lock = threading.Lock()
        self._ctx.session_manager.destroy_listeners.append(self._on_session_destroyed)
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
        via query_prefetch_status. When the KV load policy declines the
        request, no prefetch is submitted and the job reports zero hits.

        Before a prefetch that could load from an L2 adapter, publishes the
        token bindings of the looked-up chunks for cache-event subscribers.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Legacy wire field; ignored (kept for payload arity).

        Raises:
            ValueError: If ``key`` carries chunk hashes while cache events
                need token ids, or not one per complete chunk of
                ``[0, end)`` (see ``check_chunk_hash_key``).
        """
        check_chunk_hash_key(self._ctx, key)
        if key.lookup_id:
            with self._prefetch_job_lock:
                if key.session_key in self._closed_lookups:
                    return
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

        chunk_size = self._ctx.layout_desc_registry.get_chunk_size(
            model_name, world_size
        )
        token_hasher = self._ctx.layout_desc_registry.get_token_hasher(
            model_name, world_size
        )
        chunk_hashes = _key_chunk_hashes(token_hasher, key, end=key.end)
        # A prompt shorter than one chunk has no full-chunk hash. A connector
        # that loads unfull chunks still sends its unaligned length, so carry
        # on: the empty job below probes the tail stored inside chunk 0. An
        # aligned end (0 here) asks for no tail, so it returns now.
        if not chunk_hashes and key.end % chunk_size == 0:
            self._register_empty_prefetch_job(
                key,
                chunk_size=chunk_size,
                early_exit_reason="empty_chunk_hashes",
            )
            return

        # Chunk-aligned tokens submitted for lookup; the full-chunk prefetch
        # and the load policy work on these.
        requested_tokens = len(chunk_hashes) * chunk_size
        # Sub-chunk trailing tokens past the last full chunk. A lookup that
        # probes the unfull chunk can hit them (a tail saved by
        # ``save_unfull_chunk``), so they join its hit-rate denominator; a
        # connector that does not load unfull chunks sends an aligned end,
        # which leaves none.
        unfull_tail_tokens = max(0, min(key.end, key.num_tokens) - requested_tokens)

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
                        "chunk_size": chunk_size,
                        "seq_len": key.num_tokens,
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                        # Empty for a key carrying client-computed hashes.
                        "token_ids": unpack_token_ids(key.token_bytes),
                    },
                )
            )

        # Submit one key row per (object group, kv rank), each with its own
        # attention window.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        shard_replication_desc = (
            self._ctx.layout_desc_registry.find_shard_replication_desc(
                model_name, world_size
            )
        )
        session = self._ctx.session_manager.get_or_create(key.session_key, token_hasher)
        session.update_from_key(key)
        start_chunk = min(max(0, key.start // chunk_size), len(chunk_hashes))
        # The engine-covered prefix is never looked up, so it is excluded from
        # the hit-rate denominator and reported separately.
        looked_up_tokens = requested_tokens - start_chunk * chunk_size
        # The paths below that probe the unfull chunk also count its tokens,
        # so a tail hit can never push the hit rate above 1.
        probed_tokens = looked_up_tokens + unfull_tail_tokens
        session.begin_lookup(key)

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
                key,
                chunk_size=chunk_size,
                early_exit_reason="no_group_layout_descs",
            )
            return

        if not self._load_policy.should_load(
            KVLoadContext(
                request_id=key.request_id,
                model_name=model_name,
                chunk_size=chunk_size,
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
                chunk_size=chunk_size,
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
                        shard_replication_desc,
                    )
                    for obj_key in group_keys
                ]
            )

        # Prefetch only [start, end); status offsets hits back to absolute.
        suffix_hashes = chunk_hashes[start_chunk:]
        if not suffix_hashes:
            # Every full chunk is engine-covered, or there is none: nothing to
            # lock or prefetch. The empty job still reports the covered prefix
            # as the hit and probes the unfull chunk after it.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=probed_tokens,
                chunk_size=chunk_size,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
                probe_unfull_chunk=True,
            )
            return

        # The prefetch can commit chunks to L1 that this server's engine never
        # stored: copies loaded from a shared L2 or a P2P peer. Bind their
        # tokens first, as STORE does, so the cache events can announce them.
        # Without an L2 adapter nothing can be loaded.
        if (
            self._ctx.event_bus.has_subscribers(EventType.MP_TOKENS)
            and self._ctx.storage_manager.l2_adapters()
        ):
            token_ids = unpack_token_ids(key.token_bytes)
            token_offsets = [
                (start_chunk + index) * chunk_size
                for index in range(len(suffix_hashes))
            ]
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_TOKENS,
                    session_id=key.request_id,
                    metadata={
                        "chunk_hashes": suffix_hashes,
                        "token_chunks": [
                            token_ids[offset : offset + chunk_size]
                            for offset in token_offsets
                        ],
                        "token_offsets": token_offsets,
                        "parent_hashes": [
                            chunk_hashes[start_chunk - 1] if start_chunk else None,
                            *suffix_hashes[:-1],
                        ],
                    },
                )
            )

        spec = PrefetchTaskSpec(
            key_groups=ipc_key_to_grouped_object_keys(
                key,
                suffix_hashes,
                list(range(attn_desc.num_object_groups)),
                group_layout_descs,
                attn_desc,
                shard_replication_desc,
            ),
            read_owner=key.read_owner,
        )
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec,
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                row_windows=tuple(row.sliding_window_size for row in spec.key_groups),
                request_id=key.request_id,
                requested_tokens=probed_tokens,
                chunk_size=chunk_size,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
                prompt_tokens=key.num_tokens,
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=model_name,
                cache_salt=key.cache_salt,
                unfull_chunk_lookup_key=key,
                lookup_id=key.lookup_id,
                read_owner=key.read_owner,
                job_key=key.session_key,
            )
        )

    @request_handler(HandlerType.BLOCKING)
    def query_prefetch_status(
        self,
        request_id: str,
        lookup_id: str = "",
    ) -> int | None:
        """Poll the status of a prefetch job by request_id (and optional lookup_id)."""
        job_key: SessionKey = (request_id, lookup_id) if lookup_id else request_id
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(job_key)
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
        lookup_id: str = "",
    ) -> int | None:
        """Block until a prefetch job completes, then return its hit tokens."""
        job_key: SessionKey = (request_id, lookup_id) if lookup_id else request_id
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(job_key)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if not self._ctx.storage_manager.wait_prefetch_status(job.handle, timeout):
            return None
        return self.query_prefetch_status(request_id, lookup_id)

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
        caller's responsibility to align the boundaries as desired. An
        unaligned ``end`` releases the full chunks below it, plus the unfull
        chunk ending exactly there if the lookup locked it.

        Only the keys the prefetch actually read-locked are released: the
        key's range is floored to the current lookup's ``start``, and each
        extra range resolves against its own boundaries.

        Releases the same per-object count the lookup reserved (each
        group's :meth:`ShardReplicationDesc.num_kv_readers`).

        The release is derived from the lookup result recorded on the
        session (hit length, lock model, L1 owners). If that result has not
        been consumed yet it is handed over here first. While the prefetch
        is still running nothing is released: its locks still belong to the
        prefetch controller and expire with the read-lock TTL.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Legacy wire field; ignored (kept for payload arity).
            extra_ranges: Lock ranges left over from this request's earlier,
                abandoned lookups, as ``(start, end)`` token pairs. Each end
                must be that lookup's hit boundary, so the range resolves to
                exactly the keys it locked (sliding-window groups translate
                from the hit).

        Raises:
            ValueError: If ``key`` carries chunk hashes the server cannot use
                (see ``check_chunk_hash_key``).
        """
        if key.start >= key.end and not extra_ranges:
            return

        check_chunk_hash_key(self._ctx, key)
        token_hasher = self._ctx.layout_desc_registry.get_token_hasher(
            key.model_name, key.world_size
        )
        chunk_size = self._ctx.layout_desc_registry.get_chunk_size(
            key.model_name, key.world_size
        )
        if key.lookup_id:
            session = self._ctx.session_manager.get(key.session_key)
            if session is None:
                return
        else:
            session = self._ctx.session_manager.get_or_create(
                key.request_id, token_hasher
            )
        if session.prefetch_hit_chunks < 0:
            # The lookup result has not reached the session yet (no
            # QUERY/WAIT_PREFETCH_STATUS consumed it). Hand it over now so
            # the release below sees the real hit length, lock model and L1
            # owners. While the prefetch is still running its locks belong
            # to the prefetch controller, which settles them itself in
            # _finish_request; releasing them here would double-decrement,
            # so leave them to the controller and the read-lock TTL.
            with self._prefetch_job_lock:
                job = self._prefetch_jobs.get(key.session_key)
            if job is not None and self._consume_prefetch_result(job) is None:
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
                    range_end // chunk_size,
                    session.prefetch_locked_gids,
                )
            )
        # A key ending at the unfull chunk the lookup locked releases it too.
        # The lookup's retained owners name every key it locked, so a key
        # ending anywhere else releases no unfull chunk.
        if key.has_unfull_chunk(chunk_size):
            unfull_obj_keys = self._all_object_keys(
                key,
                [TokenHasher.hash_to_bytes(session.get_unfull_chunk_hash(key.end))],
                chunk_size,
            )
            locked_keys = session.get_prefetch_owners()
            if all(obj_key in locked_keys for obj_key in unfull_obj_keys):
                obj_keys.extend(unfull_obj_keys)
        if not obj_keys:
            return

        finish_lookup_read(
            self._ctx,
            key,
            obj_keys,
            self._ctx.get_read_owners(key.session_key),
            read_owner=key.read_owner,
        )

    @request_handler(HandlerType.BLOCKING)
    def end_session(self, request_id: str, lookup_id: str = "") -> None:
        """Remove a request session or tear down a single named lookup attempt."""
        if lookup_id:
            session_key = (request_id, lookup_id)
            with self._prefetch_job_lock:
                self._mark_lookup_closed_locked(session_key)
                popped_job = self._prefetch_jobs.pop(session_key, None)
            if popped_job is not None and popped_job.handle.prefetch_request_id >= 0:
                res = self._ctx.storage_manager.query_prefetch_status(popped_job.handle)
                if res is not None and res.l1_owners and popped_job.read_owner:
                    self._ctx.storage_manager.finish_read_prefetched(
                        list(res.l1_owners.keys()),
                        l1_owners=res.l1_owners,
                        read_owner=popped_job.read_owner,
                    )
            self._ctx.session_manager.remove(session_key)
            return

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
        chunk_size: int | None = None,
    ) -> list[ObjectKey]:
        """Resolve every object key of a request across all object groups.

        The object-group count and shard replication are read from the layout
        registry for ``key``'s ``(model_name, world_size)``. The order of the
        returned keys is unspecified.

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.
            chunk_size: Effective full chunk size when resolving an
                unfull chunk; adds it to object identity.

        Returns:
            The object keys of all groups and kv ranks for ``chunk_hashes``.
        """
        registry = self._ctx.layout_desc_registry
        attn_desc = registry.find_attn_desc(key.model_name, key.world_size)
        shard_replication_desc = registry.find_shard_replication_desc(
            key.model_name, key.world_size
        )
        per_group = ipc_key_to_object_keys(
            key,
            chunk_hashes,
            list(range(attn_desc.num_object_groups)),
            shard_replication_desc,
            chunk_size,
        )
        return [obj_key for group_keys in per_group for obj_key in group_keys]

    def _mark_lookup_closed_locked(self, session_key: tuple[str, str]) -> None:
        self._closed_lookups[session_key] = None
        self._closed_lookups.move_to_end(session_key)
        while len(self._closed_lookups) > _MAX_CLOSED_LOOKUPS:
            self._closed_lookups.popitem(last=False)

    def _on_session_destroyed(self, session: Session) -> None:
        """Release unclaimed named read leases when a lookup session is removed."""
        if not isinstance(session.request_id, tuple):
            return
        with self._prefetch_job_lock:
            self._mark_lookup_closed_locked(session.request_id)
            self._prefetch_jobs.pop(session.request_id, None)
        owner, l1_owners = session.close_read_ownership()
        if owner and l1_owners:
            self._ctx.storage_manager.finish_read_prefetched(
                list(l1_owners.keys()),
                l1_owners=l1_owners,
                read_owner=owner,
            )

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        job_key = job.job_key or job.request_id
        with self._prefetch_job_lock:
            if job.lookup_id and job_key in self._closed_lookups:
                closed = True
            else:
                closed = False
                self._prefetch_jobs[job_key] = job
        if closed:
            if job.handle.prefetch_request_id >= 0:
                res = self._ctx.storage_manager.query_prefetch_status(job.handle)
                if res is not None and res.l1_owners and job.read_owner:
                    self._ctx.storage_manager.finish_read_prefetched(
                        list(res.l1_owners.keys()),
                        l1_owners=res.l1_owners,
                        read_owner=job.read_owner,
                    )
            self._ctx.session_manager.remove(job_key)

    def _register_empty_prefetch_job(
        self,
        key: IPCCacheServerKey,
        requested_tokens: int = 0,
        chunk_size: int | None = None,
        num_object_groups: int = 1,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        start_chunk: int = 0,
        engine_covered_chunks: int = 0,
        early_exit_reason: str = "",
        probe_unfull_chunk: bool = False,
    ) -> None:
        """Register a job that resolves immediately without a prefetch.

        Args:
            key: The lookup's cache key.
            requested_tokens: Hit-rate denominator tokens; 0 on early exits.
            chunk_size: Effective chunk size, when registration is known.
            num_object_groups: Object groups of the request's layout.
            attn_desc: The request's attention window descriptor.
            start_chunk: First chunk of the looked-up range; reported as the
                hit when everything below it is engine-covered.
            engine_covered_chunks: Engine-covered chunks below ``key.start``,
                reported as ``apc_covered_tokens``.
            early_exit_reason: The ``lookup()`` branch that returned before
                submitting a prefetch; empty when the lookup simply had
                nothing to fetch.
            probe_unfull_chunk: Probe the unfull chunk after the hit prefix;
                set when every full chunk is engine-covered or the prompt has
                no full chunk.
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
                chunk_size=(
                    chunk_size
                    if chunk_size is not None
                    else self._ctx.default_chunk_size
                ),
                num_object_groups=num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=engine_covered_chunks,
                prompt_tokens=key.num_tokens,
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=key.model_name,
                cache_salt=key.cache_salt,
                early_exit_reason=early_exit_reason,
                unfull_chunk_lookup_key=key if probe_unfull_chunk else None,
                lookup_id=key.lookup_id,
                read_owner=key.read_owner,
                job_key=key.session_key,
            )
        )

    def _consume_prefetch_result(self, job: _PrefetchJob) -> int | None:
        """Hand a finished prefetch result over to the request's session."""
        job_key = job.job_key or job.request_id
        with self._prefetch_job_lock:
            if job.consuming:
                return None
            job.consuming = True
        try:
            result = self._ctx.storage_manager.query_prefetch_status(job.handle)
            if result is None:
                return None
            if job.row_windows:
                found_count, _retain = fold_unfold_grouped(
                    result.hit_cells, job.row_windows
                )
                l1_found_count, _l1_retain = fold_unfold_grouped(
                    result.l1_hit_cells, job.row_windows
                )
            else:
                # Nothing was submitted (early exit), so nothing can be hit.
                found_count = 0
                l1_found_count = 0

            # The fold covers only the looked-up sub-range: the tier split is
            # attributed on it, so the covered prefix never counts as an L1 hit.
            l1_chunks = min(l1_found_count, found_count)
            l2_chunks = found_count - l1_chunks
            # The engine resolves the hit in absolute chunks; shift it back.
            found_count += job.start_chunk

            if job.lookup_id:
                session = self._ctx.session_manager.get(job_key)
                if session is None:
                    if result.l1_owners and job.read_owner:
                        self._ctx.storage_manager.finish_read_prefetched(
                            list(result.l1_owners.keys()),
                            l1_owners=result.l1_owners,
                            read_owner=job.read_owner,
                        )
                    with self._prefetch_job_lock:
                        self._prefetch_jobs.pop(job_key, None)
                    return 0
            else:
                session = self._ctx.session_manager.get_or_create(job.request_id)

            unfull_chunk_boundary, unfull_chunk_probes, unfull_chunk_owners = (
                self._lock_unfull_chunk(
                    job.unfull_chunk_lookup_key, found_count, job.attn_desc, session
                )
                if job.unfull_chunk_lookup_key is not None
                else (0, 0, {})
            )
            combined_owners = {**result.l1_owners, **unfull_chunk_owners}
            if not session.record_prefetch_result(
                found_count,
                tuple(range(job.attn_desc.num_object_groups)),
                combined_owners,
            ):
                if combined_owners and job.read_owner:
                    self._ctx.storage_manager.finish_read_prefetched(
                        list(combined_owners.keys()),
                        l1_owners=combined_owners,
                        read_owner=job.read_owner,
                    )
                with self._prefetch_job_lock:
                    self._prefetch_jobs.pop(job_key, None)
                return 0

            chunk_size = job.chunk_size
            covered_tokens = job.engine_covered_chunks * chunk_size
            unfull_hit_tokens = (
                unfull_chunk_boundary - found_count * chunk_size
                if unfull_chunk_boundary
                else 0
            )
            l1_hit_tokens = l1_chunks * chunk_size + unfull_hit_tokens
            hit_tokens = (l1_chunks + l2_chunks) * chunk_size + unfull_hit_tokens
            request_hit_tokens = max(job.engine_hit_tokens, covered_tokens + hit_tokens)
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP_PREFETCH_END,
                    session_id=job.request_id,
                    metadata={
                        "found_count": found_count,
                        "requested_tokens": job.requested_tokens,
                        "apc_covered_tokens": covered_tokens,
                        "hit_tokens": hit_tokens,
                        "l1_hit_tokens": l1_hit_tokens,
                        "l2_hit_tokens": l2_chunks * chunk_size,
                        "prompt_tokens": job.prompt_tokens,
                        "engine_hit_tokens": job.engine_hit_tokens,
                        "request_hit_tokens": request_hit_tokens,
                        "l1_hit_keys": result.l1_hit_count,
                        "l2_hit_keys": result.l2_hit_count,
                        "early_exit_reason": job.early_exit_reason,
                        "unfull_chunk_boundary": unfull_chunk_boundary,
                        "unfull_chunk_probes": unfull_chunk_probes,
                        "model_name": job.model_name,
                        "cache_salt": job.cache_salt,
                    },
                )
            )

            with self._prefetch_job_lock:
                self._prefetch_jobs.pop(job_key, None)

            return unfull_chunk_boundary or found_count * job.chunk_size
        finally:
            with self._prefetch_job_lock:
                job.consuming = False

    def _lock_unfull_chunk(
        self,
        key: IPCCacheServerKey,
        hit_chunks: int,
        attn_desc: AttnWindowDesc,
        session: Session,
    ) -> tuple[int, int, dict[ObjectKey, int]]:
        """Read-lock the longest unfull chunk L1 holds after the hit prefix.

        This is the lookup's second stage: it runs once the full-chunk hit is
        known and only looks inside the next chunk, so a tail stored after
        ``hit_chunks`` chunks is found however far ``key.end`` reaches.
        Candidate boundaries are ``key.end`` (an exact tail) and the points of
        ``lmcache.hint.unfull_chunk_grid`` below it, longest first; without
        the hint only ``key.end``.

        Args:
            key: The lookup key.
            hit_chunks: Full chunks the lookup hit.
            attn_desc: The registration's attention windows.
            session: The request's session, for the unfull chunk's hash.

        Returns:
            The locked unfull chunk's boundary (token count, 0 when none was
            locked), the number of boundaries checked, and the L1 owner of
            each locked key.

        Note:
            Each candidate is tried with its own L1-only prefetch, longest
            first, until one hits. Shorter candidates are never locked, so a
            read-once (temporary) tail below the hit survives; a candidate
            some object group misses is released by the prefetch itself.
        """
        chunk_size = self._ctx.layout_desc_registry.get_chunk_size(
            key.model_name, key.world_size
        )
        hit_tokens = hit_chunks * chunk_size
        last_boundary = min(key.end, hit_tokens + chunk_size - 1)
        grid = int((key.request_configs or {}).get("lmcache.hint.unfull_chunk_grid", 0))
        # An exact hit has the lookup's own length (a disaggregated decode
        # resuming where its prefill stopped).
        candidate_boundaries: list[int] = (
            [key.end] if hit_tokens < key.end == last_boundary else []
        )
        if grid:
            candidate_boundaries.extend(
                boundary
                for boundary in range(last_boundary // grid * grid, hit_tokens, -grid)
                if boundary != key.end
            )
        # A connector that does not load unfull chunks sends an aligned end
        # and no grid, so it has no candidate. A key carrying chunk hashes
        # has no tokens to hash an unfull chunk from.
        if not candidate_boundaries or key.chunk_hashes:
            return 0, 0, {}
        group_layout_descs = (
            self._ctx.layout_desc_registry.find_group_unfull_chunk_layout_descs(
                key.model_name, key.world_size
            )
        )
        # The model may have unregistered since the lookup was submitted.
        if not group_layout_descs:
            return 0, 0, {}
        object_group_ids = list(range(attn_desc.num_object_groups))
        shard_replication_desc = (
            self._ctx.layout_desc_registry.find_shard_replication_desc(
                key.model_name, key.world_size
            )
        )
        for num_checked, boundary in enumerate(candidate_boundaries, start=1):
            key_groups = ipc_key_to_grouped_object_keys(
                key,
                [TokenHasher.hash_to_bytes(session.get_unfull_chunk_hash(boundary))],
                object_group_ids,
                group_layout_descs,
                attn_desc,
                shard_replication_desc,
                chunk_size,
            )
            handle = self._ctx.storage_manager.submit_prefetch_task(
                PrefetchTaskSpec(
                    key_groups=key_groups,
                    # No L2 adapter: served from L1, complete on submit.
                    allowed_l2_memory_tags=frozenset(),
                    read_owner=key.read_owner,
                ),
                external_request_id=key.request_id,
            )
            result = self._ctx.storage_manager.query_prefetch_status(handle)
            assert result is not None, "an L1-only prefetch completes on submit"
            # A partial hit keeps no lock, so a miss here holds nothing.
            hit, _retain = fold_unfold_grouped(
                result.hit_cells, [row.sliding_window_size for row in key_groups]
            )
            if hit:
                return boundary, num_checked, result.l1_owners
        return 0, len(candidate_boundaries), {}

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
