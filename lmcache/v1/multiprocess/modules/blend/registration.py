# SPDX-License-Identifier: Apache-2.0
"""Blend registration: rope state, STORE fingerprint hook, async drainer."""

# Standard
from queue import Empty as QueueEmpty
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Standard
    from queue import Queue
    import threading
    import weakref

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.blend.matcher import (
        BlendTokenRangeMatcher,
    )
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


# Default for cb_register_rope's wire-typed group_rot parameter (legacy: no
# declared windows). Never mutated — the handler only iterates it.
_EMPTY_GROUP_ROT: list[list[int]] = []


class RegistrationMixin:
    """Rope/fingerprint registration handlers (methods of
    :class:`~lmcache.v1.multiprocess.modules.blend.module.BlendModule`,
    moved verbatim; state lives on the composed instance)."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__ (module.py) and the sibling
        # mixins; declared so the mixin type-checks standalone.
        _ctx: "MPCacheServerContext"
        _event_bus: Any
        _transfer_module: "LMCacheDrivenTransferModule"
        _cb_rope_state: dict[int, _CBRopeState]
        _cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]"
        _token_range_matcher: "BlendTokenRangeMatcher"
        _fingerprint_queue: "Queue"
        _fingerprint_stop: "threading.Event"
        _pending_fp_hashes: set[bytes]
        _pending_fp_lock: "threading.Lock"

        def _cb_slot_buffers(
            self, gpu_context: Any, num_groups: int, n_pos: int
        ) -> Any: ...

        def _resolve_cb_plan_invariants(
            self, gpu_context: Any, rope_state: _CBRopeState, max_batch: int
        ) -> Any: ...

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        # Annotation must equal the protocol payload class exactly — the MQ
        # server's add_handler signature check (mq.py same_type) is strict.
        # Direct (non-wire) callers may still pass tuples/None entries; the
        # normalization below accepts them.
        group_rot: list[list[int]] = _EMPTY_GROUP_ROT,
    ) -> None:
        """Bolt CB re-RoPE state onto an already-registered KV-cache instance.

        Idempotent; ``REGISTER_KV_CACHE`` must precede this. Strips any
        YaRN/longrope mscale baked into each rope cache so re-RoPE stays a
        pure rotation.

        Args:
            instance_id (int): KV-cache instance to attach rope state to.
            cos_sin_caches_ipc (list[DeviceIPCWrapper]): IPC handles to vLLM's
                cos/sin rope cache(s) — one per distinct rope (dual-RoPE
                models send local/global); single-rope models send one.
            head_size (int): Rotary head dimension.
            is_neox_style (bool): True for NeoX (contiguous halves), else GPT-J.
            group_to_cache (list[int]): Per-engine-group index into the
                caches list; empty means every group uses cache 0.
            group_rot: Per-engine-group rotation window ``(offset_elems,
                width_elems)`` into each token row, or ``None`` per entry to
                skip that group's re-RoPE. Empty/omitted = legacy inference
                (rotate ``head_size`` dims at offset 0). MLA models must
                declare this — e.g. GLM/DeepSeek latents are
                ``(kv_lora_rank, qk_rope_head_dim)`` — because a single-plane
                MLA row is indistinguishable from a key-only cache to the
                legacy inference and would get its content dims rotated.

        Raises:
            ValueError: If ``instance_id`` has no registered KV cache,
                ``group_to_cache`` references a missing cache or does not
                cover every engine group of the registered model, or a
                ``group_rot`` entry is malformed.
        """
        entry = self._transfer_module.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(
                f"Instance {instance_id} has no paged KV cache registered; "
                "send REGISTER_KV_CACHE before CB_REGISTER_ROPE."
            )
        # Zero caches is legal (NoPE model): rope state still carries the head
        # layout for scatter geometry; every re-RoPE consumer skips.
        if group_to_cache:
            if min(group_to_cache) < 0 or max(group_to_cache) >= len(
                cos_sin_caches_ipc
            ):
                raise ValueError(
                    f"group_to_cache {group_to_cache} contains indices outside "
                    f"[0, {len(cos_sin_caches_ipc)}) for the sent cache(s)."
                )
            # Fail at registration, not mid-retrieve: every engine group of
            # the registered model must have a cache mapping.
            max_eg_idx = max(
                (
                    g.engine_group_idx
                    for g in entry.cache_context.kv_layer_groups_manager.kernel_groups
                ),
                default=-1,
            )
            if len(group_to_cache) <= max_eg_idx:
                raise ValueError(
                    f"group_to_cache covers {len(group_to_cache)} engine "
                    f"group(s) but the registered model has engine groups up "
                    f"to index {max_eg_idx}."
                )

        # Normalize declared rope windows (serialization turns tuples into
        # lists); validate here so a bad registration fails loudly instead of
        # mid-retrieve.
        norm_rot: "list[tuple[int, int] | None]" = []
        for eg_idx, rot_entry in enumerate(group_rot or []):
            if rot_entry is None or len(rot_entry) == 0:
                # None (direct call) / [] (wire encoding): skip this group.
                norm_rot.append(None)
                continue
            if len(rot_entry) != 2 or int(rot_entry[0]) < 0 or int(rot_entry[1]) <= 0:
                raise ValueError(
                    f"group_rot[{eg_idx}] = {rot_entry!r}: expected "
                    "(offset >= 0, width > 0) or None."
                )
            norm_rot.append((int(rot_entry[0]), int(rot_entry[1])))

        cos_sin_caches: list[torch.Tensor] = []
        for cache_idx, cache_ipc in enumerate(cos_sin_caches_ipc):
            cos_sin_cache = cache_ipc.to_tensor()
            # YaRN/longrope bake an mscale m into the rope cache
            # (cos²+sin²=m²≠1). vLLM already folds m into stored K, but CB
            # re-RoPE assumes a pure rotation, so an un-normalized m injects
            # an m² error per K element.
            _c32 = cos_sin_cache.to(torch.float32)
            _half = _c32.shape[1] // 2
            _m = float((_c32[:, :_half] ** 2 + _c32[:, _half:] ** 2).mean().sqrt())
            if abs(_m - 1.0) >= 1e-3:
                logger.info(
                    "CB re-RoPE: cache %d: stripping rope-cache mscale=%.4f "
                    "(m²=%.4f → K inflation if uncorrected) → unit magnitude",
                    cache_idx,
                    _m,
                    _m * _m,
                )
                cos_sin_cache = (_c32 / _m).to(cos_sin_cache.dtype)
            cos_sin_caches.append(cos_sin_cache)

        self._cb_rope_state[instance_id] = _CBRopeState(
            head_size=head_size,
            is_neox_style=is_neox_style,
            cos_sin_caches=cos_sin_caches,
            group_to_cache=list(group_to_cache),
            group_rot=norm_rot,
        )

        logger.info(
            "Registered CB rope state for instance %d "
            "(%d cache(s), shapes=%s dtype=%s, head_size=%d, is_neox=%s, "
            "group_map=%s, group_rot=%s)",
            instance_id,
            len(cos_sin_caches),
            [tuple(c.shape) for c in cos_sin_caches],
            cos_sin_caches[0].dtype if cos_sin_caches else "n/a (NoPE)",
            head_size,
            is_neox_style,
            "uniform" if not group_to_cache else str(group_to_cache),
            "legacy" if not norm_rot else str(norm_rot),
        )

        # Pre-warm the retrieve-plan invariants + slot-mapping staging for
        # this instance, off the retrieve critical path.
        try:
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            ctx = entry.cache_context if entry is not None else None
            if ctx is not None:
                self._cb_slot_buffers(
                    ctx, ctx.kv_layer_groups_manager.num_kernel_groups, 1 << 16
                )
                rope_state = self._cb_rope_state[instance_id]
                max_batch = ctx.max_batch_size
                if self._cb_plan_invariants.get(ctx) is None:
                    resolved = self._resolve_cb_plan_invariants(
                        ctx, rope_state, max_batch
                    )
                    if resolved is not None:
                        self._cb_plan_invariants[ctx] = (
                            rope_state,
                            max_batch,
                            resolved,
                        )
        except Exception:
            logger.debug("CB plan pre-warm skipped", exc_info=True)

    def cb_unregister_rope(self, instance_id: int) -> None:
        """Drop the instance's CB rope state; the paged KV cache is left intact.

        Args:
            instance_id (int): Instance whose rope state to remove (use
                ``UNREGISTER_KV_CACHE`` to free the KV cache itself).
        """
        self._cb_rope_state.pop(instance_id, None)
        if self._transfer_module.get_and_touch_context_entry(instance_id) is None:
            logger.warning(
                "cb_unregister_rope: instance %d not registered", instance_id
            )
            return
        logger.info("Unregistered CB rope state for instance %d", instance_id)

    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Paged store, then register the stored chunks as match fingerprints.

        Delegates the KV write to ``LMCacheDrivenTransfer.store``, then (worker 0 only)
        enqueues the chunk hashes for async fingerprint registration ordered
        after the L1 commit. Chunk 0 of a position-0 store is skipped (owned by
        the standard prefix path). Fingerprint failures are logged, never
        raised — they do not affect store correctness.

        Args:
            key (IPCCacheServerKey): Store key (token IDs + ``[start, end)``).
            instance_id (int): Target KV-cache instance.
            gpu_block_ids (list[list[int]]): Per-layer-group paged block IDs.
            event_ipc_handle (bytes): IPC handle to the producer's CUDA event.

        Returns:
            tuple[bytes, bool]: The underlying ``LMCacheDrivenTransfer.store`` result
            (event handle, success).
        """
        result = self._transfer_module.store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

        # The matcher is engine-shared; only worker 0 registers.
        if key.worker_id not in (0, None):
            return result

        # Enqueue on cupy_stream so CUDA FIFO ordering puts registration
        # after the L1-commit callback; otherwise lookups see the chunk as
        # not-yet-committed and drop the whole group as stale.
        chunk_hashes: list[bytes] = []
        tokens_in_range: list[int] = []
        try:
            session = self._ctx.session_manager.get_or_create(key.request_id)
            # Request-end cleanup may have deleted the session; get_or_create
            # then returns a fresh one whose hash chain is garbage. Re-set
            # tokens: idempotent if the session survived, corrective if not.
            session.set_tokens(list(key.token_ids))
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h)
                for h in session.get_hashes(key.start, key.end)
            ]
            if not chunk_hashes:
                return result
            tokens_in_range = list(key.token_ids)[key.start : key.end]
            # Chunk 0 is owned by the prefix lookup leg; its fingerprint
            # would be redundant.
            start_chunk_idx = 0 if key.start != 0 else 1
            job = (
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

    def _drain_fingerprints_sync(self) -> None:
        """Sync-drain pending fingerprint registrations (the async drainer
        races at low max_tokens)."""
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

        ``num_chunks`` is what ``on_new_token_hashes`` returned, so the event
        counts only chunks actually indexed into the match table: it excludes
        the ``start_chunk_idx`` chunks the store skipped *and* chunks the
        matcher deduplicated (already registered by an earlier store). Nothing
        is published when no chunk was indexed -- a re-store of known content
        is not a registration.

        Args:
            rid: Request ID of the store that enqueued the job.
            num_chunks: Chunks newly indexed by the matcher for this job.
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
