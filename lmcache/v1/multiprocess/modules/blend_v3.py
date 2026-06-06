# SPDX-License-Identifier: Apache-2.0
"""Blend V3: paged-aware CacheBlend as an :class:`EngineModule`.

Plugs into the unified :class:`MPCacheEngine`; standard ``REGISTER_KV_CACHE``
+ ``CB_REGISTER_ROPE_V3`` for setup; STORE wrapper registers fingerprints;
retrieve scatters into the request's paged blocks.
"""

# Standard
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Any
import threading
import time

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import check_interprocess_event_support
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    TrimPolicy,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    CudaIPCWrapper,
    IPCCacheEngineKey,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.modules.blend import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import (
    TokenHasher,
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    update_table_id_numba,
)
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


@dataclass
class _CBRopeState:
    """Per-instance RoPE state IPC-shared from vLLM; dangles on reallocate."""

    head_size: int
    is_neox_style: bool  # NeoX = contiguous halves; else GPT-J.
    cos_sin_cache: torch.Tensor


@dataclass
class _CBUnifiedJob:
    """Per-request poll state for non-blocking ``cb_unified_lookup``.

    Stashed across polls because the underlying status/found polls are
    consume-once.
    """

    matches: list[CBMatchResult]
    num_tokens: int = 0
    prefix_chunks: int | None = None  # stashed when the prefix poll completes
    sparse_started: bool = False  # prefix done -> sparse leg submitted/skipped
    handle: PrefetchHandle | None = None  # sparse handle, None if no sparse leg
    non_prefix: list[CBMatchResult] | None = None
    per_hash_obj_keys: dict | None = None
    expanded_uidx: list[int] | None = None
    found_uidx: set[int] | None = None  # stashed when the sparse poll completes


class BlendTokenRangeMatcherV3(BlendTokenRangeMatcher):
    """V3 matcher: full-hash collision rejection + block-aligned probe stride.

    Probes every ``probe_stride`` positions; lossless because retrieve drops
    non-block-aligned ``cur_st`` anyway.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 16):
        super().__init__(chunk_size)
        self._chunk_poly_hash: list[int] = []
        self._probe_stride: int = probe_stride

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        start_chunk_idx: int = 0,
        position_offset: int = 0,
    ) -> None:
        """Index non-overlapping chunks; ``start_chunk_idx=1`` skips pos-0
        (handled by the standard prefix lookup); ``position_offset`` is
        added to recorded positions for tail-slices."""
        arr = np.array(token_ids, dtype=np.uint64)
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0 or start_chunk_idx >= n:
            return

        with self._lock:
            new_idxs = [
                i
                for i in range(start_chunk_idx, n)
                if token_hashes[i] not in self._token_hash_to_compact_id
            ]
            if not new_idxs:
                return
            n_new = len(new_idxs)
            new_chunk_hashes = chunk_hashes[new_idxs]

            base_id = len(self._chunk_token_hash)
            if base_id + n_new > self._TABLE_SIZE:
                logger.error(
                    "BlendTokenRangeMatcherV3 compact-ID overflow: %d chunks "
                    "registered, cannot add %d more (limit %d). Skipping.",
                    base_id,
                    n_new,
                    self._TABLE_SIZE,
                )
                return
            if base_id + n_new > int(self._TABLE_SIZE * 0.8):
                logger.warning(
                    "BlendTokenRangeMatcherV3 nearing capacity: %d/%d "
                    "compact IDs used. Hash collision rate is rising; "
                    "hit rate will degrade.",
                    base_id + n_new,
                    self._TABLE_SIZE,
                )
            compact_ids = np.arange(base_id, base_id + n_new, dtype=np.int64)

            update_table_id_numba(new_chunk_hashes, self._table_id, compact_ids)

            for k, orig_i in enumerate(new_idxs):
                th = token_hashes[orig_i]
                cid = int(compact_ids[k])
                poly_hash = int(new_chunk_hashes[k])
                slot = poly_hash & int(self._mask)
                self._chunk_token_hash.append(th)
                self._chunk_poly_hash.append(poly_hash)
                self._token_hash_to_start[th] = (
                    position_offset + orig_i * self.chunk_size
                )
                self._compact_id_to_slot[cid] = slot
                self._token_hash_to_compact_id[th] = cid

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Probe rolling-hash array every ``probe_stride`` positions; skips
        bucket-only collisions and evicted entries. One result per unique
        match; ``cur_st`` is the first block-aligned hit."""
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n_positions = int(rolling.shape[0])

        with self._lock:
            if not self._chunk_token_hash:
                logger.info(
                    "[match_probe] empty fingerprint table; n_tok=%d", len(token_ids)
                )
                return []

            mask = int(self._mask)
            stride = self._probe_stride
            seen_cids: set[int] = set()
            results: list[CBMatchResult] = []
            n_probes = 0
            n_table_hit = 0
            n_collision = 0
            n_evicted = 0
            n_no_old_st = 0
            for q_pos in range(0, n_positions, stride):
                n_probes += 1
                r = int(rolling[q_pos])
                cid = int(self._table_id[r & mask])
                if cid < 0 or cid in seen_cids:
                    continue
                n_table_hit += 1
                if r != self._chunk_poly_hash[cid]:
                    n_collision += 1
                    continue
                th = self._chunk_token_hash[cid]
                if th is None:
                    n_evicted += 1
                    continue
                old_st = self._token_hash_to_start.get(th)
                if old_st is None:
                    n_no_old_st += 1
                    continue
                seen_cids.add(cid)
                results.append(
                    CBMatchResult(
                        old_st=old_st,
                        old_ed=old_st + self.chunk_size,
                        cur_st=q_pos,
                        cur_ed=q_pos + self.chunk_size,
                        hash=th,
                    )
                )
            logger.info(
                "[match_probe] n_tok=%d stride=%d n_probes=%d "
                "table_hit=%d collisions=%d evicted=%d no_old_st=%d "
                "→ matches=%d (sample old_st=%s cur_st=%s)",
                len(token_ids),
                stride,
                n_probes,
                n_table_hit,
                n_collision,
                n_evicted,
                n_no_old_st,
                len(results),
                [r.old_st for r in results[:3]],
                [r.cur_st for r in results[:3]],
            )
            return results

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict stale entries; clears poly_hash so re-probes can't match."""
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get(th)
                if cid is None:
                    continue
                slot = int(self._compact_id_to_slot[cid])
                if slot < 0:
                    logger.warning(
                        "compact_id %d has no valid table slot; "
                        "entry may have been evicted twice",
                        cid,
                    )
                    continue
                self._table_id[slot] = -1
                self._compact_id_to_slot[cid] = -1
                self._chunk_token_hash[cid] = None
                self._chunk_poly_hash[cid] = 0
                self._token_hash_to_start.pop(th, None)
                del self._token_hash_to_compact_id[th]


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Total token coverage, merging overlapping ranges (sliding-window probe
    can return overlaps; naive sum would double-count)."""
    if not results:
        return 0
    intervals = sorted((r.cur_st, r.cur_ed) for r in results)
    coverage = 0
    cur_end = -1
    for st, ed in intervals:
        if st >= cur_end:
            coverage += ed - st
        elif ed > cur_end:
            coverage += ed - cur_end
        cur_end = max(cur_end, ed)
    return coverage


class BlendV3Module:
    """Paged-aware V3 CacheBlend. Wraps GPUTransfer STORE to register
    fingerprints; serves CB rope/lookup/retrieve RPCs; reads cross-module
    GPU state via :class:`GPUTransferModule.gpu_contexts`."""

    def __init__(
        self,
        ctx: MPCacheEngineContext,
        gpu_transfer: GPUTransferModule,
        lookup_module: LookupModule,
    ):
        self._ctx = ctx
        self._gpu_transfer = gpu_transfer
        # Reused by cb_unified_lookup to run the standard prefix lookup
        # (registers the prefix prefetch job + session state the retrieve
        # path depends on) inside the single unified RPC.
        self._lookup_module = lookup_module

        # Populated by cb_register_rope; mirrors gpu_transfer.gpu_contexts.
        self._cb_gpu_contexts: dict[int, GPUCacheContext] = {}
        self._cb_gpu_context_meta: dict[int, tuple[str, int]] = {}

        self._token_range_matcher = BlendTokenRangeMatcherV3(ctx.chunk_size)
        self._event_bus = ctx.event_bus
        self._cb_rope_state: dict[int, _CBRopeState] = {}

        # L2 opt: cache TP-expanded obj_keys at lookup, pop at retrieve.
        self._lookup_obj_keys_cache: dict[str, dict[bytes, list]] = {}
        self._lookup_obj_keys_lock = threading.Lock()

        # Non-blocking cb_unified_lookup poll state (submit-once, poll-on-recall)
        # so the handler never holds a worker thread across the L2->L1 loads.
        self._cb_jobs: dict[str, _CBUnifiedJob] = {}
        self._cb_jobs_lock = threading.Lock()

        # Async fingerprint registration: store enqueues, worker drains.
        _FpJob = tuple[list[int], list[bytes], int, int]
        self._fingerprint_queue: "Queue[_FpJob]" = Queue()
        self._fingerprint_stop = threading.Event()
        self._fingerprint_worker = threading.Thread(
            target=self._drain_fingerprint_queue,
            name="cb-fingerprint-worker",
            daemon=True,
        )
        self._fingerprint_worker.start()

        # In-flight fingerprint hashes; storage_gate keeps these from eviction.
        self._pending_fp_hashes: set[bytes] = set()
        self._pending_fp_lock = threading.Lock()

        # Lazy eviction strikes; evict only at threshold so async re-store
        # can refresh the bucket first.
        self._stale_strike: dict[bytes, int] = {}
        self._STALE_STRIKE_THRESHOLD = 2

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    @property
    def context(self) -> MPCacheEngineContext:
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        # STORE shadows GPUTransfer's; compositor registers V3 last.
        return [
            HandlerSpec(RequestType.STORE, self.store, ThreadPoolType.AFFINITY),
            HandlerSpec(
                RequestType.CB_REGISTER_ROPE_V3,
                self.cb_register_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNREGISTER_ROPE_V3,
                self.cb_unregister_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNIFIED_LOOKUP,
                self.cb_unified_lookup,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.CB_RETRIEVE_PRE_COMPUTED_V3,
                self.cb_retrieve_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        return {
            "registered_cb_rope_instances": list(self._cb_rope_state.keys()),
            "cb_rope_meta": {
                str(instance_id): self._cb_gpu_context_meta.get(instance_id)
                for instance_id in self._cb_rope_state.keys()
            },
        }

    def close(self) -> None:
        self._fingerprint_stop.set()
        self._cb_rope_state.clear()
        self._cb_gpu_contexts.clear()
        self._cb_gpu_context_meta.clear()

    # ------------------------------------------------------------------
    # V3 RPCs
    # ------------------------------------------------------------------

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_cache_ipc: CudaIPCWrapper,
        head_size: int,
        is_neox_style: bool,
    ) -> None:
        """Bolt rope state onto an already-registered ``cache_contexts`` entry;
        idempotent. ``REGISTER_KV_CACHE`` must precede this."""
        cache_contexts = self._gpu_transfer.cache_contexts
        if instance_id not in cache_contexts:
            raise ValueError(
                f"Instance {instance_id} has no paged KV cache registered; "
                "send REGISTER_KV_CACHE before CB_REGISTER_ROPE_V3."
            )
        entry = cache_contexts[instance_id]
        gpu_context = entry.cache_context

        cos_sin_cache = cos_sin_cache_ipc.to_tensor()
        # YaRN/longrope bake an mscale m into the rope cache (cos²+sin²=m²≠1).
        # vLLM already folds m into stored K, but CB re-RoPE assumes a pure
        # rotation, so an un-normalized m injects an m² error per K element
        # (gpt-oss m≈1.347 → degenerate output). Strip m; m≈1 models untouched.
        _c32 = cos_sin_cache.to(torch.float32)
        _half = _c32.shape[1] // 2
        _m = float((_c32[:, :_half] ** 2 + _c32[:, _half:] ** 2).mean().sqrt())
        if abs(_m - 1.0) >= 1e-3:
            logger.info(
                "CB re-RoPE: stripping rope-cache mscale=%.4f (m²=%.4f → K "
                "inflation if uncorrected) → unit magnitude",
                _m,
                _m * _m,
            )
            cos_sin_cache = (_c32 / _m).to(cos_sin_cache.dtype)
        else:
            logger.info(
                "CB re-RoPE: rope-cache magnitude≈%.4f (unit); no mscale "
                "normalization needed",
                _m,
            )
        self._cb_rope_state[instance_id] = _CBRopeState(
            head_size=head_size,
            is_neox_style=is_neox_style,
            cos_sin_cache=cos_sin_cache,
        )

        # cb_unified_lookup resolves (model, ws) → ctx via this mirror.
        self._cb_gpu_contexts[instance_id] = gpu_context
        self._cb_gpu_context_meta[instance_id] = (entry.model_name, entry.world_size)

        # Probe stride = ie block size; must divide chunk_size.
        ie_logical_block_size = (
            gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )
        if self._ctx.chunk_size % ie_logical_block_size == 0:
            self._token_range_matcher._probe_stride = ie_logical_block_size
        else:
            logger.warning(
                "CB matcher probe stride unchanged (%d): chunk_size %d is not "
                "a multiple of inference_engine_logical_block_size %d.",
                self._token_range_matcher._probe_stride,
                self._ctx.chunk_size,
                ie_logical_block_size,
            )

        logger.info(
            "Registered CB rope state for instance %d "
            "(cos_sin_cache shape=%s dtype=%s, head_size=%d, is_neox=%s, "
            "matcher_probe_stride=%d)",
            instance_id,
            tuple(cos_sin_cache.shape),
            cos_sin_cache.dtype,
            head_size,
            is_neox_style,
            self._token_range_matcher._probe_stride,
        )

    def cb_unregister_rope(self, instance_id: int) -> None:
        """Drop rope state. Paged KV cache stays (use UNREGISTER_KV_CACHE)."""
        self._cb_rope_state.pop(instance_id, None)
        self._cb_gpu_contexts.pop(instance_id, None)
        self._cb_gpu_context_meta.pop(instance_id, None)
        if instance_id not in self._gpu_transfer.cache_contexts:
            logger.warning(
                "cb_unregister_rope: instance %d not registered", instance_id
            )
            return
        logger.info("Unregistered CB rope state for instance %d", instance_id)

    # ------------------------------------------------------------------
    # Unified lookup (CB_UNIFIED_LOOKUP) + shared helpers
    # ------------------------------------------------------------------

    def _drain_fingerprints_sync(self) -> None:
        """Sync-drain pending fingerprint registrations (the async drainer
        races at low max_tokens)."""
        while True:
            try:
                job = self._fingerprint_queue.get_nowait()
            except QueueEmpty:
                break
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset = job
            try:
                self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
            except Exception:
                logger.exception("CB fingerprint registration failed (sync drain)")

    def _match_fingerprints(self, key: IPCCacheEngineKey) -> list[CBMatchResult]:
        """Drain pending registrations, fingerprint-match sub-sequences, then
        leftmost-greedy dedup over overlapping ranges. Returns matches sorted
        by ``cur_st`` (empty if none)."""
        self._drain_fingerprints_sync()
        matches = self._token_range_matcher.match_sub_sequence(list(key.token_ids))
        if not matches:
            return []
        matches.sort(key=lambda r: r.cur_st)
        deduped: list[CBMatchResult] = []
        covered_end = -1
        for r in matches:
            if r.cur_st >= covered_end:
                deduped.append(r)
                covered_end = r.cur_ed
        return deduped

    def _resolve_cb_layout_desc(
        self, model_name: str, world_size: int
    ) -> "MemoryLayoutDesc | None":
        """Find the CB KV buffer layout for (model, world_size), or None."""
        for gpu_id, (m_name, w_size) in self._cb_gpu_context_meta.items():
            if m_name == model_name and w_size == world_size:
                cb_ctx = self._cb_gpu_contexts[gpu_id]
                return MemoryLayoutDesc(
                    shapes=[cb_ctx.get_kv_buffer_shape(self._ctx.chunk_size)],
                    dtypes=[cb_ctx.dtype],
                )
        return None

    def _sparse_prefetch_submit(
        self,
        key: IPCCacheEngineKey,
        layout_desc: "MemoryLayoutDesc",
        matches: list[CBMatchResult],
    ) -> "tuple[PrefetchHandle, dict[bytes, list], list[int]]":
        """Coalesce all ``matches`` into one sparse prefetch and submit it
        (non-blocking). The caller polls ``query_prefetch_status(handle)`` then
        calls :meth:`_sparse_classify` with the found set."""
        world_size = key.world_size
        per_hash_obj_keys: dict[bytes, list] = {}
        all_hashes = [r.hash for r in matches]
        all_obj_keys = ipc_key_to_object_keys(key, all_hashes)
        for i, h in enumerate(all_hashes):
            per_hash_obj_keys[h] = all_obj_keys[i * world_size : (i + 1) * world_size]

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
            uniq_keys,
            layout_desc,
            external_request_id=key.request_id,
            policy=TrimPolicy.SPARSE,
        )
        return handle, per_hash_obj_keys, expanded_uidx

    def _sparse_classify(
        self,
        key: IPCCacheEngineKey,
        matches: list[CBMatchResult],
        found_uidx: set[int],
        per_hash_obj_keys: dict[bytes, list],
        expanded_uidx: list[int],
    ) -> list[CBMatchResult]:
        """Classify each chunk found/stale by whether every TP rank's key
        loaded, run stale-strike bookkeeping, and stash the obj_keys cache for
        retrieve. Returns the found subset (cur_st order)."""
        world_size = key.world_size
        found_cb_match_result: list[CBMatchResult] = []
        stale_hashes: list[bytes] = []
        for j, r in enumerate(matches):
            base = j * world_size
            if all(expanded_uidx[base + t] in found_uidx for t in range(world_size)):
                found_cb_match_result.append(r)
            else:
                stale_hashes.append(r.hash)

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

        # Stash per-hash obj_keys for retrieve (L2 opt).
        if found_cb_match_result:
            cache_entry = {
                r.hash: per_hash_obj_keys[r.hash]
                for r in found_cb_match_result
                if r.hash in per_hash_obj_keys
            }
            with self._lookup_obj_keys_lock:
                self._lookup_obj_keys_cache[key.request_id] = cache_entry

        return found_cb_match_result

    def cb_unified_lookup(
        self, key: IPCCacheEngineKey, tp_size: int
    ) -> CBUnifiedLookupResult | None:
        """Non-blocking single-RPC CB lookup (submit-once, poll-on-recall).

        First call submits the prefix lookup + fingerprint match; later calls
        poll both legs, returning ``None`` until the prefix and the sparse
        complement are both resident in L1 (so a worker thread never blocks on
        the L2->L1 loads). The prefix job's L1 read locks persist for the
        retrieve.
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
            self._lookup_module.lookup(key, tp_size)  # submit prefix (non-blocking)
            job = _CBUnifiedJob(
                matches=self._match_fingerprints(key),
                num_tokens=len(key.token_ids),
            )
            with self._cb_jobs_lock:
                self._cb_jobs[rid] = job

        # --- Prefix leg: poll (consume-once) until the L1+L2 prefix lands. ---
        if job.prefix_chunks is None:
            p = self._lookup_module.query_prefetch_status(rid)
            if p is None:
                return None  # prefix still loading -> defer
            job.prefix_chunks = p

        # Prefix done: reconcile the complement outside the prefix coverage and
        # submit one sparse prefetch for it (once). Prefix-covered chunks never
        # enter the sparse prefetch, so they cannot leak a read lock.
        if not job.sparse_started:
            prefix_tokens = job.prefix_chunks * chunk_size
            job.non_prefix = [
                r
                for r in job.matches
                if r.cur_st >= prefix_tokens and r.cur_st % chunk_size == 0
            ]
            if job.non_prefix:
                layout_desc = self._resolve_cb_layout_desc(
                    key.model_name, key.world_size
                )
                if layout_desc is not None:
                    (
                        job.handle,
                        job.per_hash_obj_keys,
                        job.expanded_uidx,
                    ) = self._sparse_prefetch_submit(key, layout_desc, job.non_prefix)
                else:
                    logger.error(
                        "No CB GPU context for model %s ws %d during cb_unified_lookup",
                        key.model_name,
                        key.world_size,
                    )
                    job.non_prefix = []
            job.sparse_started = True

        # --- Sparse leg: poll (consume-once) until the scattered chunks land. ---
        if job.handle is not None and job.found_uidx is None:
            bm = self._ctx.storage_manager.query_prefetch_status(job.handle)
            if bm is None:
                return None  # sparse still loading -> defer
            job.found_uidx = set(bm.get_indices_list())

        # --- BOTH legs ready: classify the complement + finalize. ---
        if job.handle is not None:
            found = self._sparse_classify(
                key,
                job.non_prefix or [],
                job.found_uidx or set(),
                job.per_hash_obj_keys or {},
                job.expanded_uidx or [],
            )
        else:
            found = []

        prefix_tokens = job.prefix_chunks * chunk_size
        num_tokens = job.num_tokens
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=rid,
                metadata={
                    "num_tokens": num_tokens,
                    "fingerprint_hits": len(found),
                    "prefix_hits": job.prefix_chunks,
                    "storage_hits": len(found),
                    "stale_chunks": len(job.non_prefix or []) - len(found),
                    "no_gpu_context": False,
                    "hit_tokens": _unique_token_coverage(found),
                    "requested_tokens": (num_tokens // chunk_size) * chunk_size,
                },
            )
        )
        with self._cb_jobs_lock:
            self._cb_jobs.pop(rid, None)
        return CBUnifiedLookupResult(
            prefix_coverage_tokens=prefix_tokens,
            non_prefix_segments=found,
        )

    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Paged store + matcher fingerprint registration (skips pos-0
        chunks; fingerprint failures logged, never raised)."""
        result = self._gpu_transfer.store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

        # The matcher is engine-shared; only worker 0 registers.
        if key.worker_id not in (0, None):
            return result

        # Enqueue on cupy_stream so CUDA FIFO ordering puts registration
        # after the L1-commit callback; otherwise lookups see the chunk as
        # not-yet-committed and drop the whole group as stale.
        try:
            session = self._ctx.session_manager.get_or_create(key.request_id)
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h)
                for h in session.get_hashes(key.start, key.end)
            ]
            if not chunk_hashes:
                return result
            tokens_in_range = list(key.token_ids)[key.start : key.end]
            start_chunk_idx = 1 if key.start == 0 else 0
            job = (tokens_in_range, chunk_hashes, start_chunk_idx, key.start)
            with self._pending_fp_lock:
                self._pending_fp_hashes.update(chunk_hashes[start_chunk_idx:])
            entry = self._gpu_transfer.cache_contexts.get(instance_id)
            gpu_ctx = entry.cache_context if entry is not None else None
            if gpu_ctx is not None and gpu_ctx.cupy_stream is not None:
                gpu_ctx.cupy_stream.launch_host_func(
                    self._fingerprint_queue.put_nowait, job
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

    def _drain_fingerprint_queue(self) -> None:
        """Best-effort background drainer for ``_fingerprint_queue``."""
        while not self._fingerprint_stop.is_set():
            try:
                job = self._fingerprint_queue.get(timeout=0.1)
            except QueueEmpty:
                continue
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset = job
            try:
                self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
            except Exception:
                logger.exception("CB fingerprint registration failed (async)")
            finally:
                with self._pending_fp_lock:
                    self._pending_fp_hashes.difference_update(
                        chunk_hashes[start_chunk_idx:]
                    )

    def _apply_cb_rope_batched(
        self,
        gpu_context: GPUCacheContext,
        rope_state: _CBRopeState,
        batch_len: int,
        slots_to_rope: list[tuple[int, int, int]],
    ) -> None:
        """Re-RoPE tmp-pool slots in-place (K-only, per group); list of
        ``(slot_idx, old_st, cur_st)``."""
        if not slots_to_rope:
            return
        num_groups = gpu_context.kv_layer_groups_manager.num_groups
        for group_idx in range(num_groups):
            group = gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx]
            if group.compress_ratio != 1:
                raise RuntimeError(
                    f"CB v3: group {group_idx} has compress_ratio="
                    f"{group.compress_ratio}; compressed layouts unsupported."
                )
            all_slots = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                batch_size=batch_len, group_idx=group_idx
            )
            if all_slots[0].shape[0] != 2:
                raise RuntimeError(
                    f"CB v3: group {group_idx} has kv_size={all_slots[0].shape[0]}; "
                    "MLA layouts unsupported."
                )
            num_layers, slots, hidden_dim = all_slots[0].shape[1:]
            n_heads = hidden_dim // rope_state.head_size
            if n_heads * rope_state.head_size != hidden_dim:
                raise RuntimeError(
                    f"CB rope: group {group_idx} hidden_dim ({hidden_dim}) "
                    f"not a multiple of head_size ({rope_state.head_size})."
                )
            slot_positions = torch.arange(
                slots, device=all_slots[0].device, dtype=torch.long
            )
            for slot_idx, old_st, cur_st in slots_to_rope:
                tmp = all_slots[slot_idx]
                k_flat = tmp[0].reshape(num_layers * slots, hidden_dim)
                old_positions = (old_st + slot_positions).repeat(num_layers)
                new_positions = (cur_st + slot_positions).repeat(num_layers)
                k_view = k_flat.view(-1, n_heads, rope_state.head_size)
                lmc_ops.rotary_embedding_k_fused(
                    old_positions,
                    new_positions,
                    k_view,
                    rope_state.head_size,
                    rope_state.cos_sin_cache,
                    rope_state.is_neox_style,
                )

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheEngineKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[int],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Scatter EVERY matched chunk into paged KV (prefix-hit + shifted);
        K-only re-RoPE on the shifted subset. Drops misaligned matches;
        MLA layouts unsupported."""
        cache_contexts = self._gpu_transfer.cache_contexts
        if instance_id not in cache_contexts:
            raise ValueError(
                f"Instance {instance_id} not registered for paged KV cache"
            )
        if instance_id not in self._cb_rope_state:
            raise ValueError(
                f"Instance {instance_id} has no CB rope state; "
                "send CB_REGISTER_ROPE_V3 before CB_RETRIEVE_PRE_COMPUTED_V3."
            )
        gpu_context = cache_contexts[instance_id].cache_context
        rope_state = self._cb_rope_state[instance_id]
        chunk_size = self._ctx.chunk_size

        _retrieve_t0 = time.perf_counter()
        cb_match_result = sorted(cb_match_result, key=lambda r: r.cur_st)
        # L2 opt: reuse lookup's obj_keys cache; fall back to re-resolve.
        with self._lookup_obj_keys_lock:
            cached = self._lookup_obj_keys_cache.pop(key.request_id, None)
        if cached is not None and all(r.hash in cached for r in cb_match_result):
            all_obj_keys = [k for r in cb_match_result for k in cached[r.hash]]
        else:
            all_obj_keys = ipc_key_to_object_keys(
                key, [r.hash for r in cb_match_result]
            )

        # Lookup read-locked the full found set, but the connector may have
        # dropped some matches (parent-covered / misaligned) before retrieve,
        # leaking their per-key read locks. Release those orphans now (disjoint
        # from all_obj_keys, which retrieve still consumes; needs the key cache).
        if cached is not None:
            retrieved_hashes = {r.hash for r in cb_match_result}
            orphan_keys = [
                k for h, ks in cached.items() if h not in retrieved_hashes for k in ks
            ]
            if orphan_keys:
                self._ctx.storage_manager.finish_read_prefetched(orphan_keys)
                logger.debug(
                    "CB V3 released %d prefetched-but-unretrieved keys (req=%s)",
                    len(orphan_keys),
                    key.request_id,
                )

        # prefix (no re-rope) vs shifted (re-rope), for logging.
        n_prefix = sum(1 for r in cb_match_result if r.old_st == r.cur_st)
        n_shifted = len(cb_match_result) - n_prefix

        if not all_obj_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_REQUEST_END,
                    session_id=key.request_id,
                )
            )
            return event_ipc_handle, True

        logger.debug("CB V3 retrieving object keys: %s", all_obj_keys)

        ie_logical_block_size = (
            gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )
        if chunk_size % ie_logical_block_size != 0:
            raise ValueError(
                f"chunk_size {chunk_size} must be a multiple of "
                f"inference_engine_logical_block_size {ie_logical_block_size}"
            )
        blocks_per_chunk = chunk_size // ie_logical_block_size
        num_groups = gpu_context.kv_layer_groups_manager.num_groups

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            # Staged once (single group), sliced per chunk inside the loop.
            all_block_ids_gpu = gpu_context.copy_view_block_ids_to_gpu([gpu_block_ids])[
                0
            ]

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id=key.request_id,
                    metadata={
                        "num_chunks": len(cb_match_result),
                        "model_name": key.model_name,
                    },
                ),
            )

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC "
                    "event handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    all_obj_keys
                ) as memory_objs:
                    if memory_objs is None:
                        return event_ipc_handle, False

                    # Drop malformed matches up front.
                    pairs: list[tuple[CBMatchResult, Any]] = []
                    for r, memory_obj in zip(
                        cb_match_result, memory_objs, strict=False
                    ):
                        if r.cur_st % ie_logical_block_size != 0:
                            logger.warning(
                                "Dropping CB match cur_st=%d: not aligned to "
                                "ie_logical_block_size=%d.",
                                r.cur_st,
                                ie_logical_block_size,
                            )
                            continue
                        cbs = r.cur_st // ie_logical_block_size
                        if cbs + blocks_per_chunk > int(all_block_ids_gpu.numel()):
                            logger.warning(
                                "Dropping CB match cur_st=%d old_st=%d: needs "
                                "blocks [%d:%d) but gpu_block_ids has %d. "
                                "Request %s.",
                                r.cur_st,
                                r.old_st,
                                cbs,
                                cbs + blocks_per_chunk,
                                int(all_block_ids_gpu.numel()),
                                key.request_id,
                            )
                            continue
                        pairs.append((r, memory_obj))

                    # Consecutive matches → one batched scatter per group.
                    runs: list[list[tuple[CBMatchResult, Any]]] = []
                    for r_obj in pairs:
                        r = r_obj[0]
                        if runs and runs[-1][-1][0].cur_ed == r.cur_st:
                            runs[-1].append(r_obj)
                        else:
                            runs.append([r_obj])

                    max_batch = gpu_context.max_batch_size
                    for run in runs:
                        for batch_start in range(0, len(run), max_batch):
                            batch = run[batch_start : batch_start + max_batch]
                            batch_len = len(batch)
                            first_cur_st = batch[0][0].cur_st

                            # (a) H2D fill into per-chunk tmp slots.
                            for slot_idx, (_, memory_obj) in enumerate(batch):
                                flat_slot = gpu_context.get_tmp_gpu_buffer_flat(
                                    chunk_idx=slot_idx
                                )
                                lmcache_memcpy_async_h2d(memory_obj, flat_slot)

                            # (b) Re-RoPE shifted (non-prefix) slots in place.
                            slots_to_rope = [
                                (slot_idx, r.old_st, r.cur_st)
                                for slot_idx, (r, _) in enumerate(batch)
                                if r.old_st != r.cur_st
                            ]
                            self._apply_cb_rope_batched(
                                gpu_context, rope_state, batch_len, slots_to_rope
                            )

                            # (c) One batched scatter per group.
                            chunk_block_start = first_cur_st // ie_logical_block_size
                            chunk_block_end = (
                                chunk_block_start + batch_len * blocks_per_chunk
                            )
                            chunk_block_ids_gpu = all_block_ids_gpu[
                                chunk_block_start:chunk_block_end
                            ]
                            for group_idx in range(num_groups):
                                tmp_buffers = (
                                    gpu_context.get_tmp_chunk_gpu_buffer_batched(
                                        batch_size=batch_len,
                                        group_idx=group_idx,
                                    )
                                )
                                group_kv_pointers = gpu_context.get_group_kv_pointers(
                                    group_idx
                                )
                                group_lmcache_chunk_size = (
                                    gpu_context.get_physical_chunk_size(group_idx)
                                )

                                lmc_ops.multi_layer_block_kv_transfer(
                                    group_kv_pointers,
                                    [tb.data_ptr() for tb in tmp_buffers],
                                    chunk_block_ids_gpu,
                                    gpu_context.device,
                                    lmc_ops.TransferDirection.H2D,
                                    gpu_context.get_shape_desc(group_idx),
                                    group_lmcache_chunk_size,
                                    gpu_context.gpu_kv_format_,
                                    0,  # skip_blocks_in_chunk
                                )
            except Exception:
                logger.exception("Error during retrieving prefetched results")
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={"success": False},
                    ),
                )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_REQUEST_END,
                        session_id=key.request_id,
                    ),
                )
                return event_ipc_handle, False

            event.record()
            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id=key.request_id,
                    metadata={"success": True},
                ),
            )

        _scatter_ms = (time.perf_counter() - _retrieve_t0) * 1000
        logger.info(
            "Retrieved pre-computed for %d match results into request %s "
            "paged blocks (scatter_ms=%.2f, prefix=%d shifted=%d)",
            len(cb_match_result),
            key.request_id,
            _scatter_ms,
            n_prefix,
            n_shifted,
        )
        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=key.request_id,
            ),
        )
        return event.ipc_handle(), True
