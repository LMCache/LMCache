# SPDX-License-Identifier: Apache-2.0
"""Blend retrieve: plan-then-execute H2D + re-RoPE + per-token scatter."""

# Standard
from typing import TYPE_CHECKING, Any, Protocol
import enum
import time

if TYPE_CHECKING:
    # Standard
    from collections import OrderedDict
    import weakref

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )

# Third Party
import numpy as np
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules.blend.read_set import (
    _BlendReadGroups,
    _cb_chunk_major_object_keys,
    _classify_cb_read_groups,
)
from lmcache.v1.multiprocess.modules.blend.rope import (
    _TORCH_TO_AT_SCALAR,
    _cb_group_rope_geometry,
    _CBRopeState,
)
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.platform.base.cache_context import BaseCacheContext

logger = init_logger(__name__)


#: No-op reasons already reported (bounded set); each is logged only once.
_NOOP_REASONS_SEEN: set[str] = set()


class RetrieveReason(enum.Enum):
    """Retrieve outcome taxonomy.

    ``scatter_ran`` is the wire bool: ``True`` means every matched row the
    client forwards this step is backed by scattered KV (or nothing was
    lost); ``False`` means the client must degrade the request itself
    (TP-consensus, never a raise). ``publish`` gates ``CB_RETRIEVE_NOOP``:
    ``True`` only when reuse was actually lost.
    """

    OK = "ok"
    ALREADY_APPLIED = "already_applied"
    MATCHES_BEYOND_ALLOC = "matches_beyond_alloc"
    MATCHES_STRADDLE_ALLOC = "matches_straddle_alloc"
    NO_OBJECT_KEYS = "no_object_keys"

    @property
    def scatter_ran(self) -> bool:
        return self is not RetrieveReason.MATCHES_STRADDLE_ALLOC

    @property
    def publish(self) -> bool:
        return self in (
            RetrieveReason.MATCHES_STRADDLE_ALLOC,
            RetrieveReason.NO_OBJECT_KEYS,
        )


class _DeviceEvent(Protocol):
    """Device-event surface the retrieve barrier needs; ``torch_dev.Event``
    satisfies it on every platform without naming a device-specific class."""

    def record(self) -> None: ...

    def synchronize(self) -> None: ...


# Plan-then-execute: one native call enqueues all fill/rope/scatter (plan as
# numpy int64 tables, one pybind crossing). Only retrieve path — if no plan
# can be built the retrieve fails and the client degrades to full recompute.
_HAS_NATIVE_RETRIEVE_PLAN = hasattr(device_ops, "execute_cb_retrieve_plan_flat")


class RetrieveMixin:
    """The CB_RETRIEVE_PRE_COMPUTED handler of ``BlendModule``: planning
    helpers first, then the handler; state lives on the composed instance."""

    if TYPE_CHECKING:
        # State owned by BlendModule.__init__; declared so the mixin type-checks.
        UNRETRIEVED_KEYS_EXTRA: str
        _ctx: "MPCacheServerContext"
        _event_bus: Any
        _transfer_module: "LMCacheDrivenTransferModule"
        _cb_rope_state: dict[int, _CBRopeState]
        _cb_applied_match_ranges: (
            "OrderedDict[tuple[str, int | None], set[tuple[bytes, int, int, tuple]]]"
        )
        _cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]"
        _cb_slot_staging: "weakref.WeakKeyDictionary[Any, tuple]"
        _cb_plan_done_events: "weakref.WeakKeyDictionary[Any, _DeviceEvent]"
        _cb_retrieve_streams: "weakref.WeakKeyDictionary[Any, Any]"
        _cb_retrieve_cupy_streams: "weakref.WeakKeyDictionary[Any, Any]"

    def _cb_staged_groups(
        self, gpu_context: BaseCacheContext
    ) -> "tuple[_BlendReadGroups, list[int]]":
        """Resolve the blend read set and staged kernel groups for a context.

        Returns:
            ``(read_groups, staged_kernel_indices)`` in read-group order;
            legacy fused layout: every kernel group in registration order.

        Raises:
            RuntimeError: If the layout has no resolvable blend read set.
        """
        kgm = gpu_context.kv_layer_groups_manager
        attn_desc = kgm.get_attn_desc()
        read = _classify_cb_read_groups(
            attn_desc.num_object_groups, attn_desc.group_kinds
        )
        staged = [
            ki
            for gid in read.blend_gids
            for ki in kgm.object_groups[gid].kernel_group_indices
        ]
        return read, staged

    def _cb_slot_buffers(
        self, gpu_context: BaseCacheContext, num_groups: int, n_pos: int
    ) -> "tuple[torch.Tensor, Any, torch.Tensor]":
        """Return ``(pinned, pinned_np, device)`` slot-mapping staging of at
        least ``(num_groups, n_pos)``, reused across requests per context."""
        entry = self._cb_slot_staging.get(gpu_context)
        if entry is None or entry[0].shape[0] < num_groups or entry[0].shape[1] < n_pos:
            cap = max(n_pos, 1 << 16)
            if entry is not None:
                cap = max(cap, int(entry[0].shape[1]))
            # Pin only for CUDA contexts (CPU-device unit tests have no CUDA).
            pinned = torch.empty(
                (num_groups, cap),
                dtype=torch.int64,
                pin_memory=(gpu_context.device.type == "cuda"),
            )
            dev = torch.empty(
                (num_groups, cap), dtype=torch.int64, device=gpu_context.device
            )
            entry = (pinned, pinned.numpy(), dev)
            self._cb_slot_staging[gpu_context] = entry
        return entry

    def _resolve_cb_plan_invariants(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        max_batch: int,
    ) -> "tuple[list[Any], list[list[torch.Tensor]], list[int]] | None":
        """Resolve the request-invariant plan half (cached per context).

        Returns ``(group_specs, slot_group_buffers, og_sizes)`` with
        slot-mapping fields left for the per-request stamp, or ``None`` on an
        unsupported layout. ``group_specs`` covers the staged kernel groups
        in staged order; ``slot_group_buffers`` is one destination view per
        (batch slot, read object group); ``og_sizes`` is each read group's
        per-slot byte size.
        """
        kgm = gpu_context.kv_layer_groups_manager
        cb_group_spec = device_ops.CBGroupSpec
        read, staged_kernel = self._cb_staged_groups(gpu_context)
        # Private staging pool: sharing the store path's gather staging forced
        # a device-wide sync per plan. Slots are compact over the read object
        # groups, so kernel-group offsets are rebuilt here.
        og_sizes: list[int] = []
        og_off: list[int] = []
        _off = 0
        for gid in read.blend_gids:
            og_buf = gpu_context.get_temp_object_group_buffer(0, gid)
            og_off.append(_off)
            og_sizes.append(int(og_buf.nbytes))
            _off += int(og_buf.nbytes)
        slot_bytes = _off
        slot_stride = (slot_bytes + 255) & ~255
        private_pool = torch.empty(
            max_batch * slot_stride, dtype=torch.uint8, device=gpu_context.device
        )
        private_base = int(private_pool.data_ptr())
        gid_of_kernel = {
            ki: pos
            for pos, gid in enumerate(read.blend_gids)
            for ki in kgm.object_groups[gid].kernel_group_indices
        }
        group_specs: list[Any] = []
        for group_idx in staged_kernel:
            group = kgm.kernel_groups[group_idx]
            buf0 = gpu_context.get_temp_kernel_group_buffer(0, group_idx)
            gid_pos = gid_of_kernel[group_idx]
            og_base = int(
                gpu_context.get_temp_object_group_buffer(
                    0, read.blend_gids[gid_pos]
                ).data_ptr()
            )
            group_off = og_off[gid_pos] + (int(buf0.data_ptr()) - og_base)
            num_layers, slot_tokens, hidden_dim = (
                int(buf0.shape[1]),
                int(buf0.shape[2]),
                int(buf0.shape[3]),
            )
            group_bs = group.tokens_per_block
            spec_common = dict(
                paged_kv_ptrs=gpu_context.get_kernel_group_kv_pointers(
                    group_idx
                ).data_ptr(),
                temp_buffer_ptrs=[
                    private_base + slot * slot_stride + group_off
                    for slot in range(max_batch)
                ],
                num_layers=num_layers,
                slot_tokens=slot_tokens,
                hidden_elems=hidden_dim,
                element_size=buf0.element_size(),
                engine_kv_format=gpu_context.get_engine_kv_format(group_idx),
                page_buffer_size=group.shape_desc.nb * group_bs,
                block_size=group_bs,
                head_size=rope_state.head_size,
                slot_mapping_base=0,
                slot_mapping_capacity=0,
                is_neox=rope_state.is_neox_style,
            )
            rot = rope_state.rot_for_group(group.engine_group_idx, buf0.dtype)
            if rot is None or not rope_state.cos_sin_caches:
                # Skipped group ([], quantized, or NoPE): staging + scatter
                # only. cos_sin_cache == 0 disables native rope, so the dtype
                # gate below must not run for it.
                group_specs.append(
                    cb_group_spec(
                        cos_sin_cache=0,
                        rot_dim=0,
                        rope_num_kv_heads=1,
                        rope_head_stride=hidden_dim,
                        key_scalar_type=0,
                        **spec_common,
                    )
                )
                continue
            at_scalar = _TORCH_TO_AT_SCALAR.get(buf0.dtype)
            if at_scalar is None:
                return None
            try:
                # Shared batched-rope geometry rules; the planner declines
                # (returns None) instead of raising on unsupported layouts.
                _fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
                    group,
                    int(buf0.shape[0]),
                    hidden_dim,
                    rope_state.head_size,
                    group_idx,
                    rot,
                )
            except RuntimeError:
                return None
            # NoPE took the skipped-group branch above, so non-None here.
            group_cos_sin = rope_state.cache_for_group(group.engine_group_idx)
            assert group_cos_sin is not None
            if rot_offset > 0 and int(group_cos_sin.shape[1]) != rot[1]:
                # Registration-level inconsistency: rotating with the wrong
                # width would corrupt KV.
                raise RuntimeError(
                    f"CB re-RoPE: group {group_idx} declares rope width "
                    f"{rot[1]} but the cos/sin cache has rot_dim "
                    f"{int(group_cos_sin.shape[1])}."
                )

            group_specs.append(
                cb_group_spec(
                    cos_sin_cache=group_cos_sin.data_ptr(),
                    rot_dim=int(group_cos_sin.shape[1]),
                    rope_num_kv_heads=n_heads,
                    rope_head_stride=per_head,
                    key_scalar_type=at_scalar,
                    rope_base_offset=rot_offset * buf0.element_size(),
                    **spec_common,
                )
            )
        slot_group_buffers = [
            [
                private_pool[
                    slot * slot_stride + og_off[g] : slot * slot_stride
                    + og_off[g]
                    + og_sizes[g]
                ]
                for g in range(len(read.blend_gids))
            ]
            for slot in range(max_batch)
        ]
        return group_specs, slot_group_buffers, og_sizes

    def _build_cb_retrieve_plan_flat(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        cpu_block_tables: "list[tuple[np.ndarray, int]]",
        runs: "list[list[tuple[CBMatchResult, Any]]]",
        max_batch: int,
    ) -> "tuple[list[Any], tuple[Any, Any, Any, Any], list[torch.Tensor]] | None":
        """Build the native retrieve plan for ``execute_cb_retrieve_plan_flat``
        (table layouts in the pybind docstring).

        Returns:
            ``(group_specs, (staging, ropes, scatters, step_offsets),
            keepalive)``, or ``None`` when the native plan cannot serve this
            request (old cuda_ops, non-lazy host objects, ``max_batch < 2``,
            unsupported geometry) — the caller then fails the retrieve and
            the client degrades to full recompute.
        """
        if not _HAS_NATIVE_RETRIEVE_PLAN or max_batch < 2:
            return None
        pairs = [pair for run in runs for pair in run]
        if not pairs:
            return None
        # Native staging requires the lazy-allocator (pin-chunked) host path.
        for _, chunk_objs in pairs:
            for memory_obj in chunk_objs:
                if not isinstance(memory_obj.parent(), LazyMemoryAllocator):
                    return None

        # Specs are invariant per registration except the slot-mapping fields:
        # cache per context, re-stamp per request. The cached rope_state
        # reference is the validity check — a re-registration swaps the
        # object, so identity comparison is sound.
        cached = self._cb_plan_invariants.get(gpu_context)
        if cached is not None and not (
            cached[0] is rope_state and cached[1] == max_batch
        ):
            cached = None
        if cached is None:
            resolved = self._resolve_cb_plan_invariants(
                gpu_context, rope_state, max_batch
            )
            if resolved is None:
                return None
            cached = (rope_state, max_batch, resolved)
            self._cb_plan_invariants[gpu_context] = cached
        group_specs, slot_group_buffers, og_sizes = cached[2]
        num_groups = len(group_specs)
        n_read = len(og_sizes)
        wave = max_batch // 2

        n = len(pairs)
        # Each chunk stages one H2D row per read object group.
        pos_table = np.array(
            [(r.cur_st, r.cur_ed, r.old_st) for r, _ in pairs], dtype=np.int64
        )
        cur_st, cur_ed, old_st = pos_table[:, 0], pos_table[:, 1], pos_table[:, 2]
        src = np.array(
            [[o.data_ptr for o in objs] for _, objs in pairs], dtype=np.int64
        )
        nbytes = np.array(
            [[o.get_size() for o in objs] for _, objs in pairs], dtype=np.int64
        )
        host_off = np.array(
            [[o.meta.address for o in objs] for _, objs in pairs], dtype=np.int64
        )
        for g in range(n_read):
            if (nbytes[:, g] != og_sizes[g]).any():
                bad = int(np.argmax(nbytes[:, g] != og_sizes[g]))
                raise ValueError(
                    f"CB retrieve: read group {g} object size "
                    f"{int(nbytes[bad, g])} != expected {int(og_sizes[g])} "
                    f"(chunk {bad}); staging it would read past the object."
                )

        # Slot mappings in numpy on pinned staging, one async H2D per group:
        # CPU math + async copies avoid device alloc/sync contention; the
        # copies ride the ambient stream (FIFO before the native exec).
        run_iter = [run for run in runs if run]
        if len(run_iter) == 1:
            pos_np = np.arange(
                run_iter[0][0][0].cur_st, run_iter[0][-1][0].cur_ed, dtype=np.int64
            )
        else:
            pos_np = np.concatenate(
                [
                    np.arange(run[0][0].cur_st, run[-1][0].cur_ed, dtype=np.int64)
                    for run in run_iter
                ]
            )
        n_pos = int(pos_np.shape[0])
        pinned, pinned_np, dev_buf = self._cb_slot_buffers(
            gpu_context, num_groups, n_pos
        )
        # Cross-request barrier: the previous retrieve may still read these
        # per-context buffers, and a HOST write cannot be stream-ordered --
        # host-sync the previous exec's event. Sound only with the
        # retrieve-owned staging pool; a device-wide synchronize would
        # serialize behind unrelated store traffic.
        prev_ev = self._cb_plan_done_events.pop(gpu_context, None)
        if prev_ev is not None:
            prev_ev.synchronize()
        div_mod: "dict[int, tuple[np.ndarray, np.ndarray]]" = {}
        for gi, ((block_ids_np, group_bs), spec) in enumerate(
            zip(cpu_block_tables, group_specs, strict=True)
        ):
            pair = div_mod.get(group_bs)
            if pair is None:
                pair = np.divmod(pos_np, group_bs)
                div_mod[group_bs] = pair
            q, rem = pair
            out = pinned_np[gi, :n_pos]
            np.multiply(block_ids_np[q], group_bs, out=out)
            out += rem
            dev_buf[gi, :n_pos].copy_(pinned[gi, :n_pos], non_blocking=True)
            # Safe to mutate: one handler per context, and the native call
            # copies spec contents at call time.
            spec.slot_mapping_base = int(dev_buf[gi].data_ptr())
            spec.slot_mapping_capacity = n_pos
        # The device staging must outlive the native call.
        keepalive = [dev_buf]
        # Waves of `wave` chunks per run, alternating slot halves.
        slot_of = np.empty(n, dtype=np.int64)
        slot_arange = np.arange(wave, dtype=np.int64)
        step_lens: list[int] = []
        i0 = 0
        for run in runs:
            m = len(run)
            for w0 in range(0, m, wave):
                batch_len = min(wave, m - w0)
                slot_base = (len(step_lens) % 2) * wave
                slot_of[i0 : i0 + batch_len] = slot_base + slot_arange[:batch_len]
                step_lens.append(batch_len)
                i0 += batch_len
        chunks_per_step = np.asarray(step_lens, dtype=np.int64)
        n_steps = len(step_lens)

        n_tok = cur_ed - cur_st
        tok_off = np.zeros(n, dtype=np.int64)
        np.cumsum(n_tok[:-1], out=tok_off[1:])

        # One staging row per (chunk, read object group), chunk-major; row
        # counts feed step_offsets column 0 (staging_end * n_read).
        obj_buf_ptrs = np.asarray(
            [[buf.data_ptr() for buf in bufs] for bufs in slot_group_buffers],
            dtype=np.int64,
        )
        dest = obj_buf_ptrs[slot_of]  # (n, n_read)
        staging = np.stack(
            [dest.ravel(), src.ravel(), nbytes.ravel(), host_off.ravel()], axis=1
        )

        groups_arr = np.arange(num_groups, dtype=np.int64)
        shifted = old_st != cur_st
        if not rope_state.cos_sin_caches:
            # NoPE: shifted matches need no re-RoPE; emit no rope rows.
            shifted = np.zeros_like(shifted)
        n_shifted = int(shifted.sum())
        ropes = np.stack(
            [
                np.tile(groups_arr, n_shifted),
                np.repeat(slot_of[shifted], num_groups),
                np.repeat(old_st[shifted], num_groups),
                np.repeat(cur_st[shifted], num_groups),
            ],
            axis=1,
        )
        scatters = np.stack(
            [
                np.tile(groups_arr, n),
                np.repeat(slot_of, num_groups),
                np.repeat(tok_off, num_groups),
                np.repeat(n_tok, num_groups),
            ],
            axis=1,
        )

        staging_end = np.cumsum(chunks_per_step)
        step_of_chunk = np.repeat(np.arange(n_steps, dtype=np.int64), chunks_per_step)
        shifted_per_step = np.bincount(
            step_of_chunk[shifted], minlength=n_steps
        ).astype(np.int64, copy=False)
        step_offsets = np.stack(
            [
                staging_end * n_read,
                np.cumsum(shifted_per_step) * num_groups,
                staging_end * num_groups,
            ],
            axis=1,
        )
        return (
            group_specs,
            (staging, ropes, scatters, step_offsets),
            keepalive,
        )

    def _release_applied_read_locks(
        self,
        cb_match_result: list[CBMatchResult],
        applied: list[CBMatchResult],
        all_obj_keys: list[ObjectKey],
        n_read: int,
        stream: Any,
    ) -> int:
        """Release the sparse-prefetch read locks of the scattered matches.

        Stream-ordered on ``stream`` so release fires after the scatter has
        read the objects. Matches not in ``applied`` stay locked for vLLM's
        full-alloc follow-up retrieve. ``all_obj_keys`` is chunk-major:
        ``n_read`` consecutive keys per match.

        Returns:
            The number of object keys whose release was enqueued.
        """
        applied_ids = {id(r) for r in applied}
        release_keys = [
            all_obj_keys[i * n_read + g]
            for i, r in enumerate(cb_match_result)
            if id(r) in applied_ids
            for g in range(n_read)
        ]
        if release_keys:
            submit_callback_to_stream(stream, "finish_read_prefetched", release_keys)
        return len(release_keys)

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Scatter every matched token range into the request's paged KV.

        Fills tmp slots from the lookup's prefetched chunks, K-only re-RoPEs
        the shifted subset, then writes per-token via the slot kernel — so
        non-block-aligned matches and partial vLLM blocks shared with
        recomputed tokens are written correctly. Scatter is all-or-nothing:
        defer while the allocation covers no matched token (vLLM calls this
        per block-alloc round), fail with scatter_ran=False when it covers
        only part. Never a partial scatter.

        Args:
            key: The request key.
            cb_match_result: Matched ranges to scatter, any order.
            gpu_block_ids: One block table per engine group, indexed by
                ``EngineGroupInfo.engine_group_id``.
            instance_id: Target KV-cache instance.
            event_ipc_handle: IPC handle to the forward's CUDA event.

        Note:
            ``gpu_block_ids`` may be partially allocated — some non-prefix
            matches can lie beyond the allocated bound; they defer or fail
            the scatter per the all-or-nothing rule above.

        Returns:
            The scatter-complete event handle and whether the scatter ran.

        Raises:
            ValueError: If the instance has no registered KV cache or rope
                state.
        """
        entry = self._transfer_module.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(
                f"Instance {instance_id} not registered for paged KV cache"
            )
        if instance_id not in self._cb_rope_state:
            raise ValueError(
                f"Instance {instance_id} has no CB rope state; "
                "send CB_REGISTER_ROPE before CB_RETRIEVE_PRE_COMPUTED."
            )
        gpu_context = entry.cache_context
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError(
                "Blend event backend is not initialized; register the KV cache "
                "before submitting retrieve requests"
            )
        rope_state = self._cb_rope_state[instance_id]
        chunk_size = self._ctx.chunk_size
        # Blend's read set: attention (+ connector-private aux), never
        # recurrent-state groups. Legacy fused layout: every kernel group.
        read_groups, staged_kernel = self._cb_staged_groups(gpu_context)
        n_read = len(read_groups.blend_gids)

        _retrieve_t0 = time.perf_counter()

        def _no_scatter(
            reason: RetrieveReason,
            detail: str = "",
        ) -> tuple[bytes, bool]:
            """Return ``(fresh scatter-complete handle, reason.scatter_ran)``.

            Records a NEW event in THIS process -- echoing the caller's own
            handle back makes the worker re-import it (CUDA "invalid device
            context"). Publishes CB_RETRIEVE_NOOP only when ``reason.publish``.
            ``detail`` is log-only, never a metric attribute.
            """
            if reason.value not in _NOOP_REASONS_SEEN:
                _NOOP_REASONS_SEEN.add(reason.value)
                logger.info(
                    "CB retrieve: no scatter (%s%s) — %d match(es) affected. "
                    "Logged once per distinct reason.",
                    reason.value,
                    f": {detail}" if detail else "",
                    len(cb_match_result),
                )

            with (
                torch_dev.device(gpu_context.device),
                torch_dev.stream(gpu_context.stream),
            ):
                done_event = event_backend.create_event(gpu_context.device)
                event_backend.record_event(done_event, gpu_context.stream)
                handle = event_backend.export_event(done_event, gpu_context.device)

            if reason.publish:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_RETRIEVE_NOOP,
                        session_id=key.request_id,
                        metadata={
                            "reason": reason.value,
                            "dropped_matches": len(cb_match_result),
                        },
                    )
                )
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_REQUEST_END,
                    session_id=key.request_id,
                )
            )
            return handle, reason.scatter_ran

        cb_match_result = sorted(cb_match_result, key=lambda r: r.cur_st)

        # vLLM re-calls retrieve as the block table grows. Key the applied set
        # by the blocks each range writes into: table growth keeps a range
        # applied, a reassigned destination re-scatters.
        def _dest(r: "CBMatchResult") -> tuple:
            """Destination blocks this range writes into.

            Empty on unavailable geometry, which never matches a prior applied
            entry, so the range re-scatters — a wrong skip would leave
            unpopulated KV; a redundant re-scatter only costs work.
            """
            out: list[int] = []
            try:
                kgm = gpu_context.kv_layer_groups_manager
                for kg in (kgm.kernel_groups[i] for i in staged_kernel):
                    tpb = kg.tokens_per_block
                    if not tpb:
                        return ()
                    blocks = gpu_block_ids[kg.engine_group_idx]
                    out.extend(blocks[r.cur_st // tpb : (r.cur_ed - 1) // tpb + 1])
            except (IndexError, TypeError, AttributeError):
                return ()
            return tuple(out)

        applied_ranges = self._cb_applied_match_ranges
        applied_key = (key.request_id, key.worker_id)
        prior_applied = applied_ranges.get(applied_key)
        if prior_applied:
            cb_match_result = [
                r
                for r in cb_match_result
                if (r.hash, r.cur_st, r.cur_ed, _dest(r)) not in prior_applied
            ]
            if not cb_match_result:
                return _no_scatter(RetrieveReason.ALREADY_APPLIED)
        applied_now: "set[tuple[bytes, int, int, tuple]]" = set()
        # Partial-alloc first call: matches can be beyond the allocated slots
        # -> settle before the obj-key machinery.
        if cb_match_result:
            try:
                slot_bound = min(
                    len(gpu_block_ids[kg.engine_group_idx]) * kg.tokens_per_block
                    for kg in (
                        gpu_context.kv_layer_groups_manager.kernel_groups[i]
                        for i in staged_kernel
                    )
                )
            except (IndexError, TypeError):
                slot_bound = None
            # All-or-nothing: never leave a request with partially
            # applied / partially released state.
            if slot_bound is not None and any(
                r.cur_ed > slot_bound for r in cb_match_result
            ):
                if all(r.cur_st >= slot_bound for r in cb_match_result):
                    # No matched token is forwarded this step; the follow-up
                    # full-alloc call scatters everything, locks stay held.
                    return _no_scatter(
                        RetrieveReason.MATCHES_BEYOND_ALLOC,
                        f"all {len(cb_match_result)} match(es) beyond "
                        f"slot_bound={slot_bound}",
                    )
                # Some matched tokens ARE forwarded this step, and the client
                # blends its own match list on any True return -- deferring
                # would read unpopulated KV. Fail -> full recompute.
                return _no_scatter(
                    RetrieveReason.MATCHES_STRADDLE_ALLOC,
                    f"{sum(1 for r in cb_match_result if r.cur_ed > slot_bound)}"
                    f"/{len(cb_match_result)} match(es) beyond "
                    f"slot_bound={slot_bound}",
                )
        # L2 opt: take the lookup's obj_keys stash (once; later calls
        # re-resolve). ``get``, not ``get_or_create``: a retrieve after
        # session end must not recreate ownership state.
        session = self._ctx.session_manager.get(key.request_id)
        _stash = (
            session.extras.pop(self.UNRETRIEVED_KEYS_EXTRA, None)
            if session is not None
            else None
        )
        cached = _stash["per_hash"] if _stash else None
        stash_read_locks = _stash["read_locks"] if _stash else 1
        if cached is not None and all(r.hash in cached for r in cb_match_result):
            # The lookup cached all-ranks obj keys (group-major, rank-minor);
            # select THIS rank's key per read group or TP>1 mispairs ranks.
            if key.worker_id is not None and key.world_size > 1:
                ws = key.world_size
                all_obj_keys = [
                    cached[r.hash][g * ws + key.worker_id]
                    for r in cb_match_result
                    for g in range(n_read)
                ]
            else:
                all_obj_keys = [k for r in cb_match_result for k in cached[r.hash]]
        else:
            # One key per (hash, read group), chunk-major like the cached path.
            all_obj_keys = _cb_chunk_major_object_keys(
                key, [r.hash for r in cb_match_result], read_groups.blend_gids
            )

        # The connector may have dropped matches after the lookup read-locked
        # the full found set; release those orphan locks now (disjoint from
        # all_obj_keys, which retrieve still consumes).
        if cached is not None:
            retrieved_hashes = {r.hash for r in cb_match_result}
            orphan_keys = [
                k for h, ks in cached.items() if h not in retrieved_hashes for k in ks
            ]
            if orphan_keys:
                # Nothing will read these keys: release every lock the lookup
                # took (N per key).
                self._ctx.storage_manager.finish_read_prefetched(
                    orphan_keys, read_locks=stash_read_locks
                )
                logger.debug(
                    "CB released %d prefetched-but-unretrieved keys (req=%s)",
                    len(orphan_keys),
                    key.request_id,
                )

        # Non-prefix sparse hits split by re-rope need (not prefix coverage).
        n_non_shifted = sum(1 for r in cb_match_result if r.old_st == r.cur_st)
        n_shifted = len(cb_match_result) - n_non_shifted

        if not all_obj_keys:
            return _no_scatter(RetrieveReason.NO_OBJECT_KEYS)

        logger.debug("CB retrieving object keys: %s", all_obj_keys)

        # CB supports only uncompressed single-block-id-space layouts, so the
        # first staged kernel group is representative.
        tokens_per_block = gpu_context.kv_layer_groups_manager.kernel_groups[
            staged_kernel[0]
        ].tokens_per_block
        if chunk_size % tokens_per_block != 0:
            raise ValueError(
                f"chunk_size {chunk_size} must be a multiple of "
                f"tokens_per_block {tokens_per_block}"
            )

        # Retrieve-owned stream: never the shared stream (FIFO behind store
        # copies re-creates the device-sync stall).
        retrieve_stream = self._cb_retrieve_streams.get(gpu_context)
        if retrieve_stream is None:
            if gpu_context.device.type == "cuda":
                retrieve_stream = torch_dev.Stream(device=gpu_context.device)
                # Third Party
                import cupy

                retrieve_cupy_stream = cupy.cuda.ExternalStream(
                    retrieve_stream.cuda_stream, gpu_context.device.index
                )
            else:
                retrieve_stream = gpu_context.stream
                retrieve_cupy_stream = gpu_context.cupy_stream
            self._cb_retrieve_streams[gpu_context] = retrieve_stream
            self._cb_retrieve_cupy_streams[gpu_context] = retrieve_cupy_stream
        retrieve_cupy_stream = self._cb_retrieve_cupy_streams[gpu_context]

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(retrieve_stream),
        ):
            event = event_backend.create_event(gpu_context.device)

            # Resolve each kernel group's block table + block size once by
            # engine_group_idx (kernel groups may share one). CPU tables only.
            kgm = gpu_context.kv_layer_groups_manager
            block_ids_np = [np.asarray(b, dtype=np.int64) for b in gpu_block_ids]
            cpu_block_tables: "list[tuple[np.ndarray, int]]" = []
            for group_idx in staged_kernel:
                eg_idx = kgm.kernel_groups[group_idx].engine_group_idx
                if eg_idx >= len(gpu_block_ids):
                    # Engine groups have independent block tables under HMA;
                    # substituting another's would silently corrupt KV.
                    raise ValueError(
                        f"CB retrieve: kernel group {group_idx} maps to engine "
                        f"group {eg_idx}, but only "
                        f"{len(gpu_block_ids)} block table(s) were "
                        "provided."
                    )
                group_bs = kgm.kernel_groups[group_idx].tokens_per_block
                cpu_block_tables.append((block_ids_np[eg_idx], group_bs))

            # CPU-synchronous sentinel holds cb.request open across the GPU
            # work so another worker's no-op CB_REQUEST_END cannot close the
            # root early; the root closes on the last CB_RETRIEVE_END.
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"instance_id": instance_id},
                )
            )

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id=key.request_id,
                    metadata={
                        "num_chunks": len(cb_match_result),
                        "model_name": key.model_name,
                        "worker_id": key.worker_id,
                    },
                ),
            )

            vllm_event = event_backend.import_event(
                event_ipc_handle, gpu_context.device
            )
            event_backend.wait_event(vllm_event, retrieve_stream)

            # Stage marks for the scatter_ms log line (CPU enqueue wall time):
            # fetch = prefetched read, plan = table build, exec = native enqueue.
            _stage_ms: dict[str, float] = {}
            _stage_t = time.perf_counter()
            # cb.scatter opens mid-try; track it so a failure closes the span.
            scatter_open = False
            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    all_obj_keys
                ) as memory_objs:
                    _stage_ms["fetch"] = (time.perf_counter() - _stage_t) * 1000
                    if memory_objs is None:
                        # Read failed: close the retrieve span and end the
                        # request, else cb.retrieve leaks. Return a fresh
                        # server event + False, never the client's own handle
                        # (self-import crashes TP).
                        self._event_bus.publish_on_stream(
                            gpu_context.cupy_stream,
                            Event(
                                event_type=EventType.CB_RETRIEVE_END,
                                session_id=key.request_id,
                                metadata={
                                    "success": False,
                                    "worker_id": key.worker_id,
                                },
                            ),
                        )
                        self._event_bus.publish_on_stream(
                            gpu_context.cupy_stream,
                            Event(
                                event_type=EventType.CB_REQUEST_END,
                                session_id=key.request_id,
                            ),
                        )
                        event_backend.record_event(event, retrieve_stream)
                        return (
                            event_backend.export_event(event, gpu_context.device),
                            False,
                        )

                    # Per-token scatter handles any cur_st. Each match owns n_read
                    # consecutive memory objects (chunk-major).
                    if len(memory_objs) != len(cb_match_result) * n_read:
                        raise ValueError(
                            f"CB retrieve: {len(memory_objs)} prefetched "
                            f"object(s) for {len(cb_match_result)} match(es) "
                            f"x {n_read} read group(s)."
                        )
                    grouped_objs = [
                        tuple(memory_objs[i * n_read : (i + 1) * n_read])
                        for i in range(len(cb_match_result))
                    ]
                    pairs: list[tuple[CBMatchResult, tuple[Any, ...]]] = []
                    # Bound by the smallest group: under HMA the sliding group
                    # has fewer blocks than the full group, so [0] isn't safe.
                    num_slots = min(
                        int(block_ids.shape[0]) * group_bs
                        for block_ids, group_bs in cpu_block_tables
                    )
                    for r, chunk_objs in zip(
                        cb_match_result, grouped_objs, strict=True
                    ):
                        if r.cur_ed > num_slots:
                            raise RuntimeError(
                                f"CB scatter: match cur_st={r.cur_st} "
                                f"cur_ed={r.cur_ed} exceeds {num_slots} slots "
                                f"for request {key.request_id}; the full-alloc "
                                f"gate should have deferred or failed this "
                                f"retrieve (gpu vs cpu block-table "
                                f"disagreement?)"
                            )
                        pairs.append((r, chunk_objs))

                    # cb.scatter span (GPU): re-RoPE is interleaved per-batch,
                    # so it is folded in rather than a separate span.
                    self._event_bus.publish_on_stream(
                        gpu_context.cupy_stream,
                        Event(
                            event_type=EventType.CB_SCATTER_START,
                            session_id=key.request_id,
                            metadata={
                                "scattered_tokens": sum(
                                    r.cur_ed - r.cur_st for r, _ in pairs
                                ),
                                "n_prefix": sum(
                                    1 for r, _ in pairs if r.old_st == r.cur_st
                                ),
                                "n_shifted": sum(
                                    1 for r, _ in pairs if r.old_st != r.cur_st
                                ),
                                "dropped": len(cb_match_result) - len(pairs),
                                "worker_id": key.worker_id,
                            },
                        ),
                    )
                    scatter_open = True

                    # Consecutive matches → one batched scatter per group.
                    runs: list[list[tuple[CBMatchResult, Any]]] = []
                    for r_obj in pairs:
                        r = r_obj[0]
                        if runs and runs[-1][-1][0].cur_ed == r.cur_st:
                            runs[-1].append(r_obj)
                        else:
                            runs.append([r_obj])

                    max_batch = gpu_context.max_batch_size

                    # One native call for the whole request. A None plan
                    # raises into the failure epilogue below: spans close, the
                    # client gets False and degrades to full recompute.
                    _stage_t = time.perf_counter()
                    native_flat = self._build_cb_retrieve_plan_flat(
                        gpu_context, rope_state, cpu_block_tables, runs, max_batch
                    )
                    _stage_ms["plan"] = (time.perf_counter() - _stage_t) * 1000
                    if native_flat is None:
                        raise RuntimeError(
                            "CB retrieve: native plan unavailable (cuda_ops "
                            "without execute_cb_retrieve_plan_flat, non-lazy "
                            "host objects, max_batch < 2, or unsupported "
                            "geometry); failing the retrieve — the client "
                            "degrades this request to full recompute."
                        )
                    plan_group_specs, plan_tables, _plan_keepalive = native_flat
                    _stage_t = time.perf_counter()
                    execute_cb_retrieve_plan_flat = (
                        device_ops.execute_cb_retrieve_plan_flat
                    )
                    execute_cb_retrieve_plan_flat(
                        gpu_context.device,
                        LazyMemoryAllocator.PIN_CHUNK_SIZE,
                        plan_group_specs,
                        *plan_tables,
                    )
                    _stage_ms["exec"] = (time.perf_counter() - _stage_t) * 1000

                    applied_now = {
                        (r.hash, r.cur_st, r.cur_ed, _dest(r)) for r, _ in pairs
                    }

                    self._release_applied_read_locks(
                        cb_match_result,
                        [r for r, _ in pairs],
                        all_obj_keys,
                        n_read,
                        retrieve_cupy_stream,
                    )

                    # Record this retrieve's device work for the next
                    # request's scoped barrier.
                    if gpu_context.device.type == "cuda":
                        done_ev = torch_dev.Event()
                        done_ev.record()
                        self._cb_plan_done_events[gpu_context] = done_ev

                    self._event_bus.publish_on_stream(
                        gpu_context.cupy_stream,
                        Event(
                            event_type=EventType.CB_SCATTER_END,
                            session_id=key.request_id,
                            metadata={"success": True, "worker_id": key.worker_id},
                        ),
                    )
                    scatter_open = False
            except Exception:
                logger.exception("Error during retrieving prefetched results")
                if scatter_open:
                    self._event_bus.publish_on_stream(
                        gpu_context.cupy_stream,
                        Event(
                            event_type=EventType.CB_SCATTER_END,
                            session_id=key.request_id,
                            metadata={"success": False, "worker_id": key.worker_id},
                        ),
                    )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={"success": False, "worker_id": key.worker_id},
                    ),
                )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_REQUEST_END,
                        session_id=key.request_id,
                    ),
                )
                # Fresh server event + False (never echo the client handle).
                event_backend.record_event(event, retrieve_stream)
                return event_backend.export_event(event, gpu_context.device), False

            event_backend.record_event(event, retrieve_stream)
            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id=key.request_id,
                    metadata={"success": True, "worker_id": key.worker_id},
                ),
            )

        # Record scattered ranges for the repeat-call guard (bounded LRU).
        if applied_now:
            applied_entry = applied_ranges.setdefault(applied_key, set())
            applied_entry.update(applied_now)
            applied_ranges.move_to_end(applied_key)
            # One entry per (request, worker): scale the cap by world size to
            # cover the same number of requests as at TP1.
            cap = 4096 * max(1, int(key.world_size or 1))
            while len(applied_ranges) > cap:
                applied_ranges.popitem(last=False)

        _scatter_ms = (time.perf_counter() - _retrieve_t0) * 1000
        logger.info(
            "Retrieved pre-computed for %d match results into request %s "
            "paged blocks (scatter_ms=%.2f, non_shifted=%d shifted=%d, "
            "stages_ms=%s)",
            len(cb_match_result),
            key.request_id,
            _scatter_ms,
            n_non_shifted,
            n_shifted,
            {k: round(v, 1) for k, v in _stage_ms.items()},
        )
        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=key.request_id,
            ),
        )
        return event_backend.export_event(event, gpu_context.device), True
