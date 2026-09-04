# SPDX-License-Identifier: Apache-2.0
"""Blend unified lookup FSM: prefix, local-match, coordinator, and sparse legs."""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import threading
import time

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.api import BlendMatch
    from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.blend.matcher import (
        BlendTokenRangeMatcher,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_ranked
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.mp_coordinator.blend_client import PENDING
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules.blend.matcher import _unique_token_coverage
from lmcache.v1.multiprocess.modules.blend.read_set import (
    _CBReadGroups,
    _cb_chunk_major_object_keys,
    _classify_cb_read_groups,
    _narrow_attn_desc,
)

logger = init_logger(__name__)


#: ``_submit_prefix_leg`` result: ``(handle, world_size, prefix_gids, windows,
#: n_chunks, no_gpu_context)`` -- see that method's docstring.
_PrefixLegSubmit = tuple[
    PrefetchHandle | None, int, tuple[int, ...], tuple[int, ...], int, bool
]


@dataclass
class _CBUnifiedJob:
    """Per-request poll state for non-blocking cb_unified_lookup.

    Stashed across polls because the underlying status/found polls are
    consume-once.
    """

    matches: list[CBMatchResult]
    num_tokens: int = 0
    # Prefix leg (blend-module-owned submit/poll). ``prefix_handle`` is None when
    # there is no GPU context / no full chunk (poll reports 0 coverage).
    prefix_handle: PrefetchHandle | None = None
    prefix_world_size: int = 1
    prefix_lock_gids: tuple = ()  # the gids the prefix keys cover (lock model)
    prefix_windows: tuple = ()  # per-gid cross-chunk windows (fold input)
    prefix_num_chunks: int = 0  # chunks in the submitted prefix key list
    prefix_chunks: int | None = None  # stashed when the prefix poll completes
    retained_chunks: list[int] | None = None  # SEGMENTED_PREFIX: full gapped set
    sparse_started: bool = False  # prefix done -> sparse leg submitted/skipped
    handle: PrefetchHandle | None = None  # sparse handle, None if no sparse leg
    non_prefix: list[CBMatchResult] | None = None
    per_hash_obj_keys: dict | None = None
    expanded_uidx: list[int] | None = None
    found_uidx: set[int] | None = None  # stashed when the sparse poll completes
    l2_keys: int = 0  # sparse keys needing an L2 load (0 => no L2 read, span skipped)
    coord_submitted: bool = False  # coordinator match query was issued
    coord_deadline: float = 0.0  # time.monotonic() wall-clock cutoff for the leg
    segmented: bool = False  # SEGMENTED_PREFIX active for THIS registration
    # Either leg found no CB KV-cache layout for (model, world_size), so the
    # request cannot blend at all. Reported on CB_LOOKUP_END.
    no_gpu_context: bool = False


class LookupMixin:
    """The CB_UNIFIED_LOOKUP handler and its legs (methods of
    :class:`~lmcache.v1.multiprocess.modules.blend.module.BlendModule`,
    moved verbatim; state lives on the composed instance)."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__ (module.py) and the sibling
        # mixins; declared so the mixin type-checks standalone.
        UNRETRIEVED_KEYS_EXTRA: str
        _ctx: "MPCacheServerContext"
        _event_bus: Any
        _cb_jobs: dict[str, "_CBUnifiedJob"]
        _cb_jobs_lock: threading.Lock
        _coordinator: "BlendCoordinatorClient | None"
        _segmented_prefix: bool
        _token_range_matcher: "BlendTokenRangeMatcher"
        _pending_fp_hashes: set[bytes]
        _pending_fp_lock: threading.Lock
        _stale_strike: dict[bytes, int]
        _STALE_STRIKE_THRESHOLD: int

        def _drain_fingerprints_sync(self) -> None: ...

    def _match_fingerprints(self, key: IPCCacheServerKey) -> list[CBMatchResult]:
        """Drain pending registrations and fingerprint-match sub-sequences.

        Returns the raw matches (any order, possibly overlapping); the caller
        applies the prefix filter + overlap dedup once via
        :meth:`_non_overlapping_after_prefix`.
        """
        self._drain_fingerprints_sync()
        return self._token_range_matcher.match_sub_sequence(list(key.token_ids))

    @staticmethod
    def _non_overlapping_after_prefix(
        matches: list[CBMatchResult], prefix_tokens: int
    ) -> list[CBMatchResult]:
        """Matches outside the prefix coverage, leftmost-greedy overlap-deduped.

        Drops matches the prefix leg already covers (``cur_st < prefix_tokens``),
        then keeps a left-to-right non-overlapping subset -- two matches over the
        same request range can't both scatter. Filtering precedes the dedup so a
        prefix-covered match cannot suppress a usable one in the greedy pass.

        Args:
            matches: Candidate matches in any order; ``cur_st``/``cur_ed`` are
                request token positions.
            prefix_tokens: Contiguous prefix coverage in tokens; matches starting
                before it are dropped. Pass ``0`` to keep all (dedup only).

        Returns:
            Non-overlapping matches in ascending ``cur_st`` order.
        """
        kept: list[CBMatchResult] = []
        covered_end = -1
        for r in sorted(
            (r for r in matches if r.cur_st >= prefix_tokens),
            key=lambda r: r.cur_st,
        ):
            if r.cur_st >= covered_end:
                kept.append(r)
                covered_end = r.cur_ed
        return kept

    def _resolve_cb_read_layouts(
        self, model_name: str, world_size: int
    ) -> "tuple[_CBReadGroups, dict[int, MemoryLayoutDesc], AttnWindowDesc] | None":
        """Resolve blend's read set and layouts for ``(model_name, world_size)``.

        Args:
            model_name (str): Model name to match.
            world_size (int): Tensor-parallel world size to match.

        Returns:
            ``(read_groups, group_layout_descs, attn_desc)`` — the FULL
            registration descriptor; each leg narrows it to its own gids via
            :func:`_narrow_attn_desc` — or ``None`` when no registered CB
            context matches.

        Raises:
            RuntimeError: If the layout has no resolvable blend read set.
        """
        registry = self._ctx.layout_desc_registry
        layouts = registry.find_group_layout_descs(model_name, world_size)
        if not layouts:
            return None
        attn_desc = registry.find_attn_desc(model_name, world_size)
        read = _classify_cb_read_groups(
            attn_desc.num_object_groups, attn_desc.group_kinds
        )
        return read, layouts, attn_desc

    def _sparse_prefetch_submit(
        self,
        key: IPCCacheServerKey,
        resolved: "tuple[_CBReadGroups, dict[int, MemoryLayoutDesc], AttnWindowDesc]",
        matches: list[CBMatchResult],
    ) -> "tuple[PrefetchHandle, dict[bytes, list], list[int]]":
        """Coalesce all matches into one sparse L2->L1 prefetch and submit it.

        Non-blocking. Dedups object keys before submit (sparse keeps one read
        lock per loaded key, so a duplicate would leak). The caller polls
        ``query_prefetch_status(handle)`` then calls :meth:`_sparse_classify`
        with the found set.

        Args:
            key (IPCCacheServerKey): The request key.
            resolved: The read set + layouts from
                :meth:`_resolve_cb_read_layouts`.
            matches (list[CBMatchResult]): Non-prefix matches to prefetch.

        Returns:
            tuple[PrefetchHandle, dict[bytes, list], list[int]]: the prefetch
            handle, per-hash object keys (chunk-major: every read group's
            rank-expanded keys for that hash, group-major within the hash),
            and each expanded position's deduped-key index (maps the per-key
            found set back to every chunk).
        """
        read, layouts, attn_desc = resolved
        per_hash_obj_keys: dict[bytes, list] = {}
        all_hashes = [r.hash for r in matches]
        all_obj_keys = _cb_chunk_major_object_keys(key, all_hashes, read.gids)
        per_chunk = len(all_obj_keys) // len(all_hashes) if all_hashes else 0
        for i, h in enumerate(all_hashes):
            per_hash_obj_keys[h] = all_obj_keys[i * per_chunk : (i + 1) * per_chunk]

        # Dedup keys before submit (sparse keeps one read lock per loaded key;
        # a duplicate would leak). Map each expanded position to its deduped
        # index so the per-key found set resolves back to every chunk.
        uniq_keys: list = []
        key_to_uidx: dict = {}
        expanded_uidx: list[int] = []
        for k in all_obj_keys:
            uidx = key_to_uidx.get(k)
            if uidx is None:
                uidx = len(uniq_keys)
                key_to_uidx[k] = uidx
                uniq_keys.append(k)
            expanded_uidx.append(uidx)

        handle: PrefetchHandle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=uniq_keys,
                group_layout_descs=layouts,
                num_kv_readers=key.require_num_kv_readers(),
                policy=TrimPolicy.SPARSE,
                attn_desc=_narrow_attn_desc(attn_desc, read.gids),
            ),
            external_request_id=key.request_id,
        )
        return handle, per_hash_obj_keys, expanded_uidx

    def _sparse_classify(
        self,
        key: IPCCacheServerKey,
        matches: list[CBMatchResult],
        found_uidx: set[int],
        per_hash_obj_keys: dict[bytes, list],
        expanded_uidx: list[int],
    ) -> list[CBMatchResult]:
        """Classify each prefetched chunk as found or stale, and finalize state.

        A chunk is found only if every TP rank's key loaded; stale chunks take
        an eviction strike (evicted at threshold, kept while still in-flight).
        Stashes the found chunks' obj_keys for the retrieve path.

        Args:
            key (IPCCacheServerKey): The request key.
            matches (list[CBMatchResult]): The submitted non-prefix matches.
            found_uidx (set[int]): Deduped-key indices that loaded.
            per_hash_obj_keys (dict[bytes, list]): Per-hash TP-expanded keys.
            expanded_uidx (list[int]): Each expanded position's deduped index.

        Returns:
            list[CBMatchResult]: The found subset, in cur_st order.
        """
        # Per-chunk key stride: a chunk counts only when every read group's
        # rank-expanded keys loaded.
        per_chunk = len(expanded_uidx) // len(matches) if matches else 0
        found_cb_match_result: list[CBMatchResult] = []
        stale_hashes: list[bytes] = []
        for j, r in enumerate(matches):
            base = j * per_chunk
            if all(expanded_uidx[base + t] in found_uidx for t in range(per_chunk)):
                found_cb_match_result.append(r)
            else:
                stale_hashes.append(r.hash)
        # Stale drops silently shrink coverage (a fully-stale classify turns a
        # matched request into a full recompute) — log so it is diagnosable.
        if stale_hashes:
            logger.warning(
                "CB sparse classify for %s: %d found, %d stale of %d submitted",
                key.request_id,
                len(found_cb_match_result),
                len(stale_hashes),
                len(matches),
            )

        # Reset strikes for confirmed hashes.
        if found_cb_match_result:
            with self._pending_fp_lock:
                for r in found_cb_match_result:
                    self._stale_strike.pop(r.hash, None)
        # Stale: in-flight keep; >= threshold strikes -> evict.
        if stale_hashes:
            with self._pending_fp_lock:
                truly_evict: list[bytes] = []
                for h in stale_hashes:
                    if h in self._pending_fp_hashes:
                        continue
                    n = self._stale_strike.get(h, 0) + 1
                    if n >= self._STALE_STRIKE_THRESHOLD:
                        truly_evict.append(h)
                        self._stale_strike.pop(h, None)
                    else:
                        self._stale_strike[h] = n
            if truly_evict:
                self._token_range_matcher.remove_chunks(truly_evict)
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_CHUNKS_EVICTED,
                    metadata={"num_chunks": len(stale_hashes)},
                )
            )

        # Stash per-hash obj_keys for the retrieve (L2 opt); whatever it
        # never consumes is released by _release_unretrieved_locks.
        if found_cb_match_result:
            cache_entry = {
                r.hash: per_hash_obj_keys[r.hash]
                for r in found_cb_match_result
                if r.hash in per_hash_obj_keys
            }
            session = self._ctx.session_manager.get_or_create(key.request_id)
            # A repeat lookup for the same request (e.g. re-issued after a
            # preemption) replaces the stash; release the superseded
            # reservation first or its locks leak.
            prev = session.extras.pop(self.UNRETRIEVED_KEYS_EXTRA, None)
            if prev:
                self._ctx.storage_manager.finish_read_prefetched(
                    [k for ks in prev["per_hash"].values() for k in ks],
                    read_locks=prev["read_locks"],
                )
            session.extras[self.UNRETRIEVED_KEYS_EXTRA] = {
                "read_locks": key.require_num_kv_readers(),
                "per_hash": cache_entry,
            }

        return found_cb_match_result

    def _submit_prefix_leg(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
        policy: TrimPolicy,
    ) -> _PrefixLegSubmit:
        """Submit the CB prefix prefetch (non-blocking).

        Opens the ``cb.prefix_lookup`` span (CB namespace — CB requests no longer
        feed the MP request / mp.lookup_prefetch spans or the MP hit-rate
        aggregate; the CB hit-rate metric carries prefix tokens via
        CB_LOOKUP_END) and writes the shared session (``set_tokens`` +
        ``lookup_ipc_key``) so ``end_session``'s L1 keep-alive touch still
        resolves the request's keys.

        Args:
            key (IPCCacheServerKey): Request key (token IDs, request_id, model,
                world_size).
            tp_size (int): Tensor-parallel size for MLA multi-reader locking.
            policy (TrimPolicy): ``PREFIX`` or ``SEGMENTED_PREFIX``.

        Returns:
            tuple: ``(handle, world_size, prefix_gids, windows, n_chunks,
            no_gpu_context)`` — the object groups each chunk's keys cover (the
            leg's lock model), their cross-chunk windows, and the chunk count
            (fold inputs for the poll). ``handle`` is None when there is no GPU
            context or no full chunk (the poll then reports 0 coverage);
            ``no_gpu_context`` is True only for the former, and is reported on
            CB_LOOKUP_END.
        """
        rid = key.request_id
        model_name, world_size = key.model_name, key.world_size
        self._event_bus.publish(
            Event(event_type=EventType.CB_PREFIX_LOOKUP_START, session_id=rid)
        )

        resolved = self._resolve_cb_read_layouts(model_name, world_size)
        if resolved is None:
            logger.error(
                "No CB GPU context for model %s ws %d during prefix lookup!",
                model_name,
                world_size,
            )
            return None, world_size, (), (), 0, True
        read, layouts, attn_desc = resolved
        layout_desc = layouts[read.attn_gid]

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            return None, world_size, (), (), 0, False

        # Lookup-hash logger (chunk hashes, for debug); guarded so the metadata
        # dict is built only when a subscriber is listening.
        if self._event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=rid,
                    metadata={
                        "request_id": rid,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # Shared session: end_session reads lookup_ipc_key + the rolling hashes
        # to keep the request's KV alive in L1.
        num_kv_readers = key.require_num_kv_readers()
        # PREFIX leg set: recurrent + attention (the planes a prefix restore
        # consumes), never aux. Chunk-major so count_leading_ones() stays
        # prefix-aligned with _poll_prefix_leg's divisor.
        obj_keys = _cb_chunk_major_object_keys(key, chunk_hashes, read.prefix_gids)
        prefix_desc = _narrow_attn_desc(attn_desc, read.prefix_gids)
        session = self._ctx.session_manager.get_or_create(rid)
        session.set_tokens(list(key.token_ids))
        session.begin_lookup(key, tuple(attn_desc.num_chunks_in_sw))
        handle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=obj_keys,
                group_layout_descs=layouts,
                num_kv_readers=num_kv_readers,
                policy=policy,
                attn_desc=prefix_desc,
            ),
            external_request_id=rid,
        )
        return (
            handle,
            world_size,
            read.prefix_gids,
            tuple(prefix_desc.num_chunks_in_sw),
            len(chunk_hashes),
            False,
        )

    def _poll_prefix_leg(
        self, job: "_CBUnifiedJob", rid: str, segmented: bool
    ) -> "tuple[int, list[int] | None] | None":
        """Poll the CB prefix handle; on completion close the cb.prefix_lookup span.

        Consume-once: publishes CB_PREFIX_LOOKUP_END exactly once when the
        prefetch lands. The prefix hit tokens are accounted on the CB hit-rate
        metric at CB_LOOKUP_END, not here. For SEGMENTED_PREFIX also surfaces the
        gapped retained chunk set.

        Args:
            job (_CBUnifiedJob): Poll state holding the prefix handle + world size.
            rid (str): Request ID (event session_id).
            segmented (bool): SEGMENTED_PREFIX active -> also surface the gapped
                retained chunk set.

        Returns:
            tuple | None: ``(leading_chunks, retained_or_None)`` when resident;
            ``None`` while still loading. ``retained`` is the full gapped chunk
            set for SEGMENTED_PREFIX, else None.
        """
        if job.prefix_handle is not None:
            bm = self._ctx.storage_manager.query_prefetch_status(job.prefix_handle)
            if bm is None:
                return None  # still loading
            # Window-aware fold (a live handle implies >= 1 chunk): a
            # windowed group's out-of-window keys are trimmed from the load
            # (bits legitimately unset), so use the server's own fold;
            # count_leading_ones would read those bits as a miss.
            leading, _ = fold_unfold_ranked(
                bm,
                job.prefix_num_chunks,
                job.prefix_world_size,
                list(job.prefix_windows),
            )
            # Retain a chunk only if EVERY key loaded (AND across the rank
            # shards of every read group); a chunk missing any is a gap.
            if segmented:
                # Keys are chunk-major: one chunk spans ws shards per
                # locked group.
                per_chunk = job.prefix_world_size * len(job.prefix_lock_gids)
                shard_counts: dict[int, int] = {}
                for ki in bm.get_indices_list():
                    c = ki // per_chunk
                    shard_counts[c] = shard_counts.get(c, 0) + 1
                retained = sorted(c for c, n in shard_counts.items() if n == per_chunk)
            else:
                retained = None
        else:
            # No GPU context / no full chunk: nothing loaded.
            leading, retained = 0, ([] if segmented else None)
        # Publish the lock model so free_lookup_locks (APC-shadowed hits,
        # aborts before load) releases exactly what this leg locked.
        session = self._ctx.session_manager.get_or_create(rid)
        session.record_prefetch_result(leading, job.prefix_lock_gids)
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_PREFIX_LOOKUP_END,
                session_id=rid,
                metadata={"prefix_chunks": leading},
            )
        )
        return leading, retained

    def cb_unified_lookup(
        self, key: IPCCacheServerKey, tp_size: int
    ) -> CBUnifiedLookupResult | None:
        """Non-blocking single-RPC CB lookup (submit-once, poll-on-recall).

        First call submits the prefix lookup + fingerprint match; later calls
        poll both legs, returning ``None`` until the prefix and the sparse
        non-prefix complement are both resident in L1 (so a worker thread never
        blocks on the L2->L1 loads). The prefix job's L1 read locks persist for
        the retrieve.

        Args:
            key (IPCCacheServerKey): Request key (token IDs, request_id, model,
                world_size).
            tp_size (int): Tensor-parallel size for the prefix lookup.

        Returns:
            CBUnifiedLookupResult | None: ``None`` while either leg is still
            loading (the caller re-issues to poll); on completion, the prefix
            coverage in tokens plus the found non-prefix segments. When that
            result carries nothing to retrieve, the request is also ended
            (``CB_REQUEST_END``) here, since no retrieve will follow.
        """
        rid = key.request_id
        chunk_size = self._ctx.chunk_size

        with self._cb_jobs_lock:
            job = self._cb_jobs.get(rid)
        if job is None:
            # First invocation: start events + submit prefix + fingerprint match.
            self._event_bus.publish(
                Event(event_type=EventType.CB_REQUEST_START, session_id=rid)
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id=rid,
                    metadata={"num_tokens": len(key.token_ids)},
                )
            )
            # SEGMENTED_PREFIX keeps post-gap chunks L1-resident after a
            # mid-prefix L2 failure. Forced OFF for recurrent registrations:
            # its pure-load post-gap rows would hole the recurrence scan, so
            # a gap must truncate the prefix.
            resolved_pre = self._resolve_cb_read_layouts(key.model_name, key.world_size)
            use_segmented = self._segmented_prefix and not (
                resolved_pre is not None and resolved_pre[0].recurrent_gids
            )
            prefix_policy = (
                TrimPolicy.SEGMENTED_PREFIX if use_segmented else TrimPolicy.PREFIX
            )
            # Prefix leg: this module owns the submit + the cb.prefix_lookup span.
            (
                prefix_handle,
                prefix_ws,
                prefix_gids,
                prefix_windows,
                prefix_n_chunks,
                prefix_no_ctx,
            ) = self._submit_prefix_leg(key, tp_size, prefix_policy)
            # With a coordinator the fleet directory is the only match source;
            # skip the local matcher. The coordinator leg resolves at poll.
            matches: list[CBMatchResult]
            if self._coordinator is not None:
                matches = []
            else:
                # Local fingerprint match: CPU-bound, tight span.
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINT_MATCH_START,
                        session_id=rid,
                    )
                )
                matches = self._match_fingerprints(key)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINT_MATCH_END,
                        session_id=rid,
                        metadata={"matches": len(matches)},
                    )
                )
            job = _CBUnifiedJob(
                matches=matches,
                num_tokens=len(key.token_ids),
                prefix_handle=prefix_handle,
                prefix_world_size=prefix_ws,
                prefix_lock_gids=prefix_gids,
                prefix_windows=prefix_windows,
                prefix_num_chunks=prefix_n_chunks,
                no_gpu_context=prefix_no_ctx,
            )
            job.segmented = use_segmented
            job.coord_submitted = self._submit_coordinator_match(key)
            if job.coord_submitted and self._coordinator is not None:
                job.coord_deadline = time.monotonic() + self._coordinator.match_budget_s
                # Coordinator match leg: async span, ended on the
                # resolving poll (or deadline) in _poll_coordinator_match.
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_COORDINATOR_MATCH_START,
                        session_id=rid,
                    )
                )
            with self._cb_jobs_lock:
                self._cb_jobs[rid] = job

        assert job is not None
        segmented = job.segmented

        # --- Prefix leg: poll (consume-once) until the L1+L2 prefix lands. ---
        if job.prefix_chunks is None:
            res = self._poll_prefix_leg(job, rid, segmented)
            if res is None:
                return None  # prefix still loading -> defer
            job.prefix_chunks, prefix_retained = res
            if segmented:
                job.retained_chunks = prefix_retained
        # Poll above set it (or we returned); narrow for the arithmetic below.
        assert job.prefix_chunks is not None
        prefix_chunks: int = job.prefix_chunks

        # Prefix done: reconcile the complement outside the prefix coverage and
        # submit one sparse prefetch for it (once). Prefix-covered chunks never
        # enter the sparse prefetch, so they cannot leak a read lock.
        if not job.sparse_started:
            prefix_tokens = prefix_chunks * chunk_size
            if self._coordinator is not None:
                candidates = self._poll_coordinator_match(job, rid)
                if candidates is None:
                    return None  # coordinator still in flight (bounded by deadline)
            else:
                candidates = job.matches
            # Under SEGMENTED_PREFIX, a same-position match the prefix leg already
            # retained rides the segmented tail (prefix-class: pure load, no CHECK)
            # -- drop it here so it is not scattered twice. A same-position match
            # the tail does NOT cover is a genuine cross-context hit: keep it as
            # non-prefix (re-RoPE no-ops at delta 0, but it still needs CHECK).
            # Shifted (cur != old) matches are always kept. Then the single
            # prefix-filter + overlap dedup over the rest.
            if segmented:
                retained = set(job.retained_chunks or [])
                candidates = [
                    c
                    for c in candidates
                    if c.old_st != c.cur_st or (c.cur_st // chunk_size) not in retained
                ]
            # Prefix-filter only; the overlap dedup runs AFTER the prefetch, where
            # the winner is chosen among chunks that can actually be fetched.
            job.non_prefix = [c for c in candidates if c.cur_st >= prefix_tokens]
            if job.non_prefix:
                resolved = self._resolve_cb_read_layouts(key.model_name, key.world_size)
                if resolved is not None:
                    (
                        job.handle,
                        job.per_hash_obj_keys,
                        job.expanded_uidx,
                    ) = self._sparse_prefetch_submit(key, resolved, job.non_prefix)
                    # Only trace the span when the prefetch actually reads L2;
                    # all-L1-resident matches do no L2 work worth a span.
                    job.l2_keys = len(job.handle.l2_orig_indices)
                    if job.l2_keys > 0:
                        self._event_bus.publish(
                            Event(
                                event_type=EventType.CB_SPARSE_PREFETCH_START,
                                session_id=rid,
                                metadata={
                                    "n_chunks": len(job.non_prefix),
                                    "world_size": key.world_size,
                                    "n_keys": len(job.non_prefix) * key.world_size,
                                    "l2_keys": job.l2_keys,
                                },
                            )
                        )
                else:
                    logger.error(
                        "No CB GPU context for model %s ws %d during cb_unified_lookup",
                        key.model_name,
                        key.world_size,
                    )
                    job.no_gpu_context = True
                    job.non_prefix = []
            job.sparse_started = True

        # --- Sparse leg: poll (consume-once) until the scattered chunks land. ---
        if job.handle is not None and job.found_uidx is None:
            bm = self._ctx.storage_manager.query_prefetch_status(job.handle)
            if bm is None:
                return None  # sparse still loading -> defer
            job.found_uidx = set(bm.get_indices_list())
            if job.l2_keys > 0:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_SPARSE_PREFETCH_END,
                        session_id=rid,
                        metadata={
                            "found_keys": len(job.found_uidx),
                            "l2_keys": job.l2_keys,
                        },
                    )
                )

        # --- BOTH legs ready: classify the complement + finalize. ---
        if job.handle is not None:
            fetched = self._sparse_classify(
                key,
                job.non_prefix or [],
                job.found_uidx or set(),
                job.per_hash_obj_keys or {},
                job.expanded_uidx or [],
            )
            # Overlap dedup over the RETRIEVABLE candidates; dropped candidates'
            # keys are released by the retrieve path's orphan sweep.
            found = self._non_overlapping_after_prefix(
                fetched, prefix_chunks * chunk_size
            )
            if len(found) != len(fetched):
                logger.debug(
                    "CB kept %d of %d retrievable matches after overlap dedup (req=%s)",
                    len(found),
                    len(fetched),
                    rid,
                )
        else:
            found = []

        prefix_tokens = prefix_chunks * chunk_size
        num_tokens = job.num_tokens

        # Segmented tail: post-gap chunks the SEGMENTED_PREFIX prefix leg kept
        # resident (retained index > the leading run). Delivered at their
        # original positions (old_st == cur_st) so the connector tags them
        # ``prefix`` (pure load, no recompute); only the gap is recomputed. The
        # storage key is the same prefix-chained chunk hash the prefix leg used,
        # so no fingerprint match is needed to retrieve them.
        segmented_tail: list[CBMatchResult] = []
        if segmented and job.retained_chunks:
            chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
                list(key.token_ids)
            )
            for i in job.retained_chunks:
                if i < prefix_chunks or i >= len(chunk_hashes):
                    continue  # leading run (already prefix) / sub-chunk tail
                st = i * chunk_size
                segmented_tail.append(
                    CBMatchResult(
                        old_st=st,
                        old_ed=st + chunk_size,
                        cur_st=st,
                        cur_ed=st + chunk_size,
                        hash=chunk_hashes[i],
                    )
                )

        seg_tail_tokens = _unique_token_coverage(segmented_tail)
        non_prefix_hit_tokens = _unique_token_coverage(found)
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=rid,
                metadata={
                    "num_tokens": num_tokens,
                    "fingerprint_hits": len(found),
                    "prefix_hits": job.prefix_chunks,
                    "prefix_chunks": job.prefix_chunks,
                    "storage_hits": len(found),
                    "stale_chunks": len(job.non_prefix or []) - len(found),
                    "no_gpu_context": job.no_gpu_context,
                    "prefix_hit_tokens": prefix_tokens,
                    "segmented_prefix_hit_tokens": seg_tail_tokens,
                    "non_prefix_hit_tokens": non_prefix_hit_tokens,
                    "hit_tokens": prefix_tokens
                    + _unique_token_coverage(found + segmented_tail),
                    "requested_tokens": (num_tokens // chunk_size) * chunk_size,
                },
            )
        )
        if not found and not segmented_tail:
            # Nothing for the connector to retrieve, so no worker will call
            # cb_retrieve_pre_computed -- the only other place this module ends the
            # request. Close cb.request here or a miss / prefix-only request
            # leaks its root span until shutdown.
            self._event_bus.publish(
                Event(event_type=EventType.CB_REQUEST_END, session_id=rid)
            )
        with self._cb_jobs_lock:
            self._cb_jobs.pop(rid, None)
        return CBUnifiedLookupResult(
            prefix_coverage_tokens=prefix_tokens,
            non_prefix_segments=found,
            segmented_prefix_segments=segmented_tail,
        )

    def _submit_coordinator_match(self, key: IPCCacheServerKey) -> bool:
        """Issue a fleet directory match query for this request (best-effort).

        Args:
            key: The lookup request key.

        Returns:
            ``True`` if a query was submitted (so the finalize step should poll
            for it), ``False`` when there is no coordinator or submission failed.
        """
        coordinator = self._coordinator
        if coordinator is None:
            return False
        try:
            tokens = list(key.token_ids)
            if len(tokens) < self._ctx.chunk_size:
                return False
            coordinator.submit_match(key.request_id, tokens)
            return True
        except Exception:
            logger.warning(
                "CB coordinator match submit failed for request %s", key.request_id
            )
            return False

    def _poll_coordinator_match(
        self, job: "_CBUnifiedJob", rid: str
    ) -> "list[CBMatchResult] | None":
        """Poll the coordinator match result, deferring until it resolves.

        Mirrors the prefix/sparse legs: ``return None`` to defer while pending.
        A per-lookup wall-clock deadline (``job.coord_deadline``) bounds the
        total wait, including queue/pool time. Past the deadline the leg is
        abandoned and the lookup proceeds local-only (the client's later fill,
        if any, is dropped via ``take_match``).

        Args:
            job: The per-request poll state.
            rid: Request id.

        Returns:
            The global segments (possibly empty) once resolved or timed out, or
            ``None`` to defer (still in flight and within the deadline).
        """
        coordinator = self._coordinator
        if coordinator is None or not job.coord_submitted:
            return []
        poll = coordinator.poll_match(rid)
        if poll is PENDING:
            if time.monotonic() < job.coord_deadline:
                return None  # defer; bounded by job.coord_deadline
            coordinator.take_match(rid)
            logger.warning(
                "CB coordinator match deadline exceeded for %s; local-only", rid
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_COORDINATOR_MATCH_END,
                    session_id=rid,
                    metadata={"matches": 0, "timed_out": True},
                )
            )
            return []
        coordinator.take_match(rid)
        segments = self._build_global_segments(poll) if isinstance(poll, list) else []
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_COORDINATOR_MATCH_END,
                session_id=rid,
                metadata={"matches": len(segments), "timed_out": False},
            )
        )
        return segments

    def _build_global_segments(
        self, matches: "list[BlendMatch]"
    ) -> list[CBMatchResult]:
        """Convert coordinator matches into chunk-granular retrievable segments.

        Each coordinator ``chunk_hash`` is the hex of the chunk's content hash
        (the same ``th`` a local ``CBMatchResult.hash`` holds), so the matches
        are returned as ``CBMatchResult`` directly: the retrieve path then
        expands ``hash`` to per-rank object keys via
        ``ipc_key_to_object_keys`` using *this* server's model, salt, and world
        size, identical to local matches. A match on content another model or
        tenant stored therefore confirmed-misses at prefetch rather than being
        filtered coordinator-side.

        Args:
            matches: Matched chunks returned by the coordinator client.

        Returns:
            One :class:`CBMatchResult` per matched chunk (request order).
        """
        chunk_size = self._ctx.chunk_size
        return [
            CBMatchResult(
                old_st=m.old_st,
                old_ed=m.old_st + chunk_size,
                cur_st=m.cur_st,
                cur_ed=m.cur_st + chunk_size,
                hash=m.chunk_hash,
            )
            for m in matches
        ]
