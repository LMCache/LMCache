# SPDX-License-Identifier: Apache-2.0
"""Layer-wise KV transfer module for the LMCache-driven MP path.

This module keeps the layer-wise (layer-major) retrieve path fully
separate from the default per-chunk path in
:mod:`lmcache.v1.multiprocess.modules.lmcache_driven_transfer`. Shared
plumbing is imported from that module rather than duplicated; the only
behaviour that differs is supplied through the three transfer hooks
declared by :class:`LMCacheDrivenTransferModule`.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Sequence, cast
import struct
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
    LMCacheDrivenTransferModule,
    _recalculate_blocks_to_skip,
    batched_iteration_with_skip,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.base.event_pool import EVENT_POOL_SIZE, EventPool
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Set once, the first time a layer batch cannot be merged into one native
# call and instead issues a separate H2D copy per layer. Still layer-wise:
# there is no per-chunk fallback on this path.
_warned_layerwise_fallback = False


def transfer_kv_layerwise(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    layer_events: list,
    event_backend: EventIPCBackend,
    batch_leader_map: dict[int, int] | None = None,
    layerwise_batch: int = 1,
    event_export_callback=None,
) -> None:
    """Transfer KV cache in layer-major order, recording per-layer events.

    Instead of copying all layers for each chunk batch (chunk-major),
    this function copies all chunks for each layer (layer-major).
    After each layer's data is fully on GPU, it records the corresponding
    event so vLLM can start that layer's attention immediately.

    The real pipeline overlap is cross-process: while this server does
    H2D(i+1)+scatter(i+1), vLLM's attention(i) runs concurrently via
    IPC event synchronization (same pattern as in-process per-layer).

    Args:
        cache_context: The GPU cache context containing KV cache info.
        block_ids_gpu: GPU block IDs indexed by LMCache KV group index.
        memory_objs: List of MemoryObj instances (one per chunk).
        object_group_id: Index of the object group being transferred.
        batch_size: Max chunks per batch (for temp buffer reuse).
        skip_first_n_tokens: Tokens to skip at the start of the range.
        layer_events: List of pre-created events, one per layer.
            Events are recorded as each layer completes.
        event_backend: The event backend for recording events.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id):
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)

    # Validate allocator type once (avoid per-chunk isinstance in hot loop)
    for mo in memory_objs[num_objects_to_skip:]:
        if mo is not None and not isinstance(mo.parent(), LazyMemoryAllocator):
            raise NotImplementedError(
                "Per-layer H2D for non-LazyMemoryAllocator not yet "
                "implemented. Use LazyMemoryAllocator for per-layer transfer."
            )

    # Cache per-kernel-group invariants on first call (avoids repeated
    # property lookups, dict accesses, and object construction on hot path).
    _lw_cache_attr = "_layerwise_invariants"
    if not hasattr(cache_context, _lw_cache_attr):
        setattr(cache_context, _lw_cache_attr, {})
    _lw_cache = getattr(cache_context, _lw_cache_attr)

    cache_key = object_group_id
    if cache_key in _lw_cache:
        kg_infos, all_layers = _lw_cache[cache_key]
    else:
        kg_infos = []
        for kernel_group_id in kernel_group_ids:
            kg = kv_groups_manager.kernel_groups[kernel_group_id]
            blocks_per_chunk = cache_context.calculate_num_blocks(
                lmcache_chunk_size, kernel_group_id
            )
            tokens_per_window = min(
                lmcache_chunk_size,
                kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
            )
            blocks_per_window = cache_context.calculate_num_blocks(
                tokens_per_window, kernel_group_id
            )
            sd = kg.shape_desc
            slots_per_chunk = cache_context.get_slots_per_chunk_in_sw(kernel_group_id)
            per_kv_bytes = slots_per_chunk * kg.hidden_dim_size * sd.element_size
            per_layer_bytes = sd.kv_size * per_kv_bytes

            _tb = cast(Any, cache_context)._temp_buffer
            kg_byte_offset_in_obj = _tb._offset_map.get(
                (0, object_group_id, kernel_group_id)
            )
            obj_group_offset = _tb._offset_map_object_group_only.get(
                (0, object_group_id)
            )
            if kg_byte_offset_in_obj is None or obj_group_offset is None:
                raise ValueError(
                    f"Cannot find temp buffer offset for "
                    f"kernel_group={kernel_group_id}, "
                    f"object_group={object_group_id}"
                )
            kg_byte_offset = kg_byte_offset_in_obj[0] - obj_group_offset[0]

            single_layer_sd = device_ops.PageBufferShapeDesc()
            single_layer_sd.kv_size = sd.kv_size
            single_layer_sd.nl = 1
            single_layer_sd.nb = sd.nb
            single_layer_sd.bs = sd.bs
            single_layer_sd.nh = sd.nh
            single_layer_sd.hs = sd.hs
            single_layer_sd.element_size = sd.element_size
            single_layer_sd.block_stride_elems = sd.block_stride_elems
            # Inert at nl=1: the staging slot this describes holds a single
            # layer, so layer_idx is pinned to 0 and the interleaved and
            # non-interleaved offset formulas are the same expression. Kept
            # True to read uniformly with the merged path's n_layer_sd.
            single_layer_sd.kv_interleaved = True

            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )

            # Flat buffer capacity: total temp buffer bytes.  Invariant
            # across requests (max_batch_size and the temp buffer shape are
            # fixed at cache-context construction), so compute it once.
            _temp_buf = cache_context.get_temp_kernel_group_buffer(0, kernel_group_id)
            total_buffer_bytes = (
                cache_context.max_batch_size
                * _temp_buf.nelement()
                * _temp_buf.element_size()
            )

            kg_infos.append(
                {
                    "kernel_group_id": kernel_group_id,
                    "kg": kg,
                    "blocks_per_chunk": blocks_per_chunk,
                    "blocks_per_window": blocks_per_window,
                    "slots_per_chunk": slots_per_chunk,
                    "per_layer_bytes": per_layer_bytes,
                    "kg_byte_offset": kg_byte_offset,
                    "sd": sd,
                    "single_layer_sd": single_layer_sd,
                    "group_kv_pointers": group_kv_pointers,
                    "total_buffer_bytes": total_buffer_bytes,
                    "max_per_layer_slots": total_buffer_bytes // per_layer_bytes,
                }
            )

        all_layers = []
        for kg_info_idx, info in enumerate(kg_infos):
            kg = info["kg"]
            for local_idx, global_layer_idx in enumerate(kg.layer_indices):
                all_layers.append((kg_info_idx, local_idx, global_layer_idx))

        # Sort by global layer index to ensure layer-major order.  Done
        # once here (not per request) since the cached list is reused.
        all_layers.sort(key=lambda x: x[2])

        _lw_cache[cache_key] = (kg_infos, all_layers)

    if not all_layers:
        return

    # Pre-compute batch plan (identical for every layer, avoid re-iteration)
    batch_plan: list[tuple[int, tuple]] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            raise ValueError(
                "MemoryObj is None for some objects in the batch during "
                "layerwise H2D transfer."
            )
        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue
        batch_plan.append((start_object_idx, memory_object_batch))

    if not batch_plan:
        # Nothing to transfer; still record 1 event so consumer doesn't hang.
        first_gl = all_layers[0][2] if all_layers else 0
        if first_gl < len(layer_events):
            event_backend.record_event(layer_events[first_gl], cache_context.stream)
        if batch_leader_map is not None:
            for _, _, global_layer_idx in all_layers:
                batch_leader_map[global_layer_idx] = first_gl
        return

    main_stream = cache_context.stream
    pin_chunk_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
    # Batch N layers per IPC event to reduce cross-process sync
    # overhead.  0 = layerwise disabled (caller should not reach here);
    # 1 = one event per layer (original behaviour); N>1 = N layers
    # transferred + scattered before events are recorded.
    layerwise_batch_size = max(1, layerwise_batch)
    use_native_layerwise_plan = _HAS_NATIVE_OBJECT_GROUP_TRANSFER

    # Flatten all valid chunks for single-launch-per-layer optimization.
    # In the existing temp buffer (4 full-chunk slots per kernel group),
    # each slot holds nl per-layer entries, giving 4*nl per-layer slots
    # total -- enough for hundreds of chunks on typical LLMs.
    all_chunks_flat: list[tuple[int, MemoryObj]] = []
    for start_object_idx, memory_object_batch in batch_plan:
        for local_idx, mo in enumerate(memory_object_batch):
            all_chunks_flat.append((start_object_idx + local_idx, mo))

    num_active_chunks = len(all_chunks_flat)

    # Pre-compute per-kernel-group block_ids and skip for single-launch path
    for info in kg_infos:
        sd = info["sd"]
        kernel_group_id = info["kernel_group_id"]
        blocks_per_window = info["blocks_per_window"]
        blocks_per_chunk = info["blocks_per_chunk"]

        # total_buffer_bytes / max_per_layer_slots are cached invariants;
        # only the chunk-count comparison depends on this request.
        info["single_launch"] = num_active_chunks <= info["max_per_layer_slots"]

        if num_active_chunks > 0:
            first_obj_idx = all_chunks_flat[0][0]
            last_obj_idx = all_chunks_flat[-1][0]
            first_block = first_obj_idx * blocks_per_window
            end_block = (last_obj_idx + 1) * blocks_per_window
            info["all_block_ids"] = block_ids_gpu[kernel_group_id][
                first_block:end_block
            ]
            first_start_token = first_obj_idx * lmcache_chunk_size
            skip_tokens_first = (
                max(first_start_token, skip_first_n_tokens) - first_start_token
            )
            orig_skip = cache_context.calculate_num_blocks(
                skip_tokens_first, kernel_group_id
            )
            info["all_skip_blocks"] = _recalculate_blocks_to_skip(
                blocks_per_chunk, blocks_per_window, orig_skip
            )

    # --- Layer-major loop with N-layer merged H2D + scatter ---
    # When layerwise_batch_size > 1, N consecutive same-kernel-group layers
    # are copied with one contiguous H2D per chunk and one scatter kernel
    # with nl=N + interleaved LMCache offset, exploiting the L1 layout
    # [K0,V0,K1,V1,...] where consecutive layers are contiguous.
    num_all_layers = len(all_layers)
    layer_batch_start = 0

    while layer_batch_start < num_all_layers:
        # Determine batch: consecutive same-kg layers, up to layerwise_batch_size
        first_kg_idx = all_layers[layer_batch_start][0]
        first_local = all_layers[layer_batch_start][1]
        batch_end = layer_batch_start + 1
        while (
            batch_end < num_all_layers
            and batch_end - layer_batch_start < layerwise_batch_size
            and all_layers[batch_end][0] == first_kg_idx
            and all_layers[batch_end][1]
            == first_local + (batch_end - layer_batch_start)
        ):
            batch_end += 1

        n_in_batch = batch_end - layer_batch_start
        info = kg_infos[first_kg_idx]
        kernel_group_id = info["kernel_group_id"]
        per_layer_bytes = info["per_layer_bytes"]
        kg_byte_offset = info["kg_byte_offset"]
        group_kv_pointers = info["group_kv_pointers"]
        slots_per_chunk = info["slots_per_chunk"]
        sd = info["sd"]

        # Check if N-layer merged path can be used:
        # - Native ops available
        # - At least 1 chunk fits in staging buffer for N layers
        n_bytes = n_in_batch * per_layer_bytes
        max_chunks_per_pass = (
            info["total_buffer_bytes"] // n_bytes if n_bytes > 0 else 0
        )
        can_merge = (
            use_native_layerwise_plan
            and n_in_batch > 1
            and info.get("all_block_ids") is not None
            and max_chunks_per_pass >= 1
        )

        if can_merge:
            # --- N-layer merged path: single execute call, multiple BatchSteps
            # Same pattern as _run_object_group_transfer_plan: build one
            # KernelGroupSpec with reusable buffer slots + full block_ids,
            # then one BatchStep per chunk sub-pass (buffer reused across
            # passes, serialized on the stream).
            n_layer_sd = device_ops.PageBufferShapeDesc()
            n_layer_sd.kv_size = sd.kv_size
            n_layer_sd.nl = n_in_batch
            n_layer_sd.nb = sd.nb
            n_layer_sd.bs = sd.bs
            n_layer_sd.nh = sd.nh
            n_layer_sd.hs = sd.hs
            n_layer_sd.element_size = sd.element_size
            n_layer_sd.block_stride_elems = sd.block_stride_elems
            n_layer_sd.kv_interleaved = True

            n_layer_kv_ptrs = group_kv_pointers[first_local : first_local + n_in_batch]

            buffer_base = cache_context.get_temp_kernel_group_buffer(
                0, kernel_group_id
            ).data_ptr()
            src_layer_offset = kg_byte_offset + first_local * per_layer_bytes
            blocks_per_window = info["blocks_per_window"]

            # Fixed gpu_ptrs for max_chunks_per_pass buffer slots (reused)
            slot_gpu_ptrs = [
                buffer_base + slot * n_bytes for slot in range(max_chunks_per_pass)
            ]

            # One KernelGroupSpec for this kg batch, referencing full
            # block_ids and reusable buffer slot pointers.
            all_block_ids = info["all_block_ids"]
            layer_spec = device_ops.KernelGroupSpec(
                n_layer_kv_ptrs.data_ptr(),
                slot_gpu_ptrs,
                n_layer_sd,
                slots_per_chunk,
                cache_context.get_engine_kv_format(kernel_group_id),
                all_block_ids.data_ptr(),
                all_block_ids.numel(),
            )

            # Build one BatchStep per chunk sub-pass
            batch_steps: list = []
            chunk_pass_start = 0
            while chunk_pass_start < num_active_chunks:
                chunk_pass_end = min(
                    chunk_pass_start + max_chunks_per_pass, num_active_chunks
                )
                pass_chunks = all_chunks_flat[chunk_pass_start:chunk_pass_end]
                pass_count = len(pass_chunks)

                staging: list = []
                for buf_idx, (_, memory_obj) in enumerate(pass_chunks):
                    gpu_dst = slot_gpu_ptrs[buf_idx]
                    src_ptr = memory_obj.data_ptr + src_layer_offset
                    staging.append(
                        device_ops.StagingCopy(
                            gpu_dst,
                            src_ptr,
                            n_bytes,
                            memory_obj.meta.address + src_layer_offset,
                        )
                    )

                # Block offset within sliced block_ids for this sub-pass
                first_obj_idx = pass_chunks[0][0]
                last_obj_idx = pass_chunks[-1][0]
                base_obj_idx = all_chunks_flat[0][0]
                block_ids_offset = (first_obj_idx - base_obj_idx) * blocks_per_window
                total_blocks = (last_obj_idx + 1 - first_obj_idx) * blocks_per_window

                first_start_token = first_obj_idx * lmcache_chunk_size
                skip_tokens_first = (
                    max(first_start_token, skip_first_n_tokens) - first_start_token
                )
                pass_skip = _recalculate_blocks_to_skip(
                    info["blocks_per_chunk"],
                    blocks_per_window,
                    cache_context.calculate_num_blocks(
                        skip_tokens_first, kernel_group_id
                    ),
                )

                layer_launch = device_ops.LaunchVar(
                    0,
                    block_ids_offset,
                    total_blocks,
                    pass_count,
                    pass_skip,
                )
                batch_steps.append(device_ops.BatchStep(staging, [layer_launch]))
                chunk_pass_start = chunk_pass_end

            # Single native call for all chunk sub-passes
            device_ops.execute_object_group_transfer_layerwise(
                lmcache_native.TransferDirection.H2D,
                cache_context.device,
                pin_chunk_size,
                [layer_spec],
                batch_steps,
            )
        else:
            # --- Per-layer fallback (N=1, no native ops, or buffer overflow) ---
            if n_in_batch > 1:
                global _warned_layerwise_fallback
                if not _warned_layerwise_fallback:
                    _warned_layerwise_fallback = True
                    reason = (
                        "native ops unavailable"
                        if not use_native_layerwise_plan
                        else f"staging buffer too small ({max_chunks_per_pass}"
                        f" chunks fit, need >=1)"
                    )
                    logger.warning(
                        "Layerwise merged H2D path unavailable (batch=%d, kg=%d): %s. "
                        "Falling back to %d separate per-layer H2D copies.",
                        n_in_batch,
                        kernel_group_id,
                        reason,
                        n_in_batch,
                    )
            for sub_idx in range(n_in_batch):
                layer_local_idx = first_local + sub_idx
                global_layer_idx = all_layers[layer_batch_start + sub_idx][2]
                single_layer_sd = info["single_layer_sd"]
                single_layer_kv_ptr = group_kv_pointers[
                    layer_local_idx : layer_local_idx + 1
                ]

                if info["single_launch"] and use_native_layerwise_plan:
                    buffer_base = cache_context.get_temp_kernel_group_buffer(
                        0, kernel_group_id
                    ).data_ptr()
                    all_gpu_ptrs_fb: list[int] = []
                    src_offset = kg_byte_offset + layer_local_idx * per_layer_bytes
                    staging_fb: list = []

                    for chunk_local_idx, (_, memory_obj) in enumerate(all_chunks_flat):
                        gpu_dst = buffer_base + chunk_local_idx * per_layer_bytes
                        src_ptr = memory_obj.data_ptr + src_offset
                        all_gpu_ptrs_fb.append(gpu_dst)
                        staging_fb.append(
                            device_ops.StagingCopy(
                                gpu_dst,
                                src_ptr,
                                per_layer_bytes,
                                memory_obj.meta.address + src_offset,
                            )
                        )

                    all_block_ids = info["all_block_ids"]
                    layer_spec = device_ops.KernelGroupSpec(
                        single_layer_kv_ptr.data_ptr(),
                        all_gpu_ptrs_fb,
                        single_layer_sd,
                        slots_per_chunk,
                        cache_context.get_engine_kv_format(kernel_group_id),
                        all_block_ids.data_ptr(),
                        all_block_ids.numel(),
                    )
                    layer_launch = device_ops.LaunchVar(
                        0,
                        0,
                        all_block_ids.numel(),
                        len(all_gpu_ptrs_fb),
                        info["all_skip_blocks"],
                    )
                    device_ops.execute_object_group_transfer_layerwise(
                        lmcache_native.TransferDirection.H2D,
                        cache_context.device,
                        pin_chunk_size,
                        [layer_spec],
                        [device_ops.BatchStep(staging_fb, [layer_launch])],
                    )
                elif info["single_launch"]:
                    buffer_base = cache_context.get_temp_kernel_group_buffer(
                        0, kernel_group_id
                    ).data_ptr()
                    all_gpu_ptrs_fb = []
                    src_offset = kg_byte_offset + layer_local_idx * per_layer_bytes

                    for chunk_local_idx, (_, memory_obj) in enumerate(all_chunks_flat):
                        gpu_dst = buffer_base + chunk_local_idx * per_layer_bytes
                        src_ptr = memory_obj.data_ptr + src_offset
                        device_ops.lmcache_memcpy_async(
                            gpu_dst,
                            src_ptr,
                            per_layer_bytes,
                            lmcache_native.TransferDirection.H2D,
                            memory_obj.meta.address + src_offset,
                            pin_chunk_size,
                        )
                        all_gpu_ptrs_fb.append(gpu_dst)

                    # CUDA-only entry point, resolved at runtime through
                    # DeviceOps.__getattr__ (see the guard above).
                    device_ops.multi_layer_block_kv_transfer_layerwise(
                        single_layer_kv_ptr,
                        all_gpu_ptrs_fb,
                        info["all_block_ids"],
                        cache_context.device,
                        lmcache_native.TransferDirection.H2D,
                        single_layer_sd,
                        slots_per_chunk,
                        cache_context.get_engine_kv_format(kernel_group_id),
                        info["all_skip_blocks"],
                    )
                else:
                    # Batched fallback for very long sequences
                    blocks_per_chunk = info["blocks_per_chunk"]
                    blocks_per_window = info["blocks_per_window"]
                    for start_object_idx, memory_object_batch in batch_plan:
                        batch_len = len(memory_object_batch)
                        batch_start_token = start_object_idx * lmcache_chunk_size
                        skip_tokens_in_chunk = (
                            max(batch_start_token, skip_first_n_tokens)
                            - batch_start_token
                        )
                        tmp_gpu_buffers: list[int] = []
                        for chunk_idx, memory_obj in enumerate(memory_object_batch):
                            full_kg_buffer = cache_context.get_temp_kernel_group_buffer(
                                chunk_idx, kernel_group_id
                            )
                            gpu_dst = full_kg_buffer.data_ptr()
                            src_offset = (
                                kg_byte_offset + layer_local_idx * per_layer_bytes
                            )
                            src_ptr = memory_obj.data_ptr + src_offset
                            device_ops.lmcache_memcpy_async(
                                gpu_dst,
                                src_ptr,
                                per_layer_bytes,
                                lmcache_native.TransferDirection.H2D,
                                memory_obj.meta.address + src_offset,
                                pin_chunk_size,
                            )
                            tmp_gpu_buffers.append(gpu_dst)

                        orig_skip_blocks = cache_context.calculate_num_blocks(
                            skip_tokens_in_chunk, kernel_group_id
                        )
                        recalculated_skip_blocks = _recalculate_blocks_to_skip(
                            blocks_per_chunk,
                            blocks_per_window,
                            orig_skip_blocks,
                        )
                        start_block_pos = start_object_idx * blocks_per_window
                        end_block_pos = (
                            start_object_idx + batch_len
                        ) * blocks_per_window
                        block_ids_curr = block_ids_gpu[kernel_group_id][
                            start_block_pos:end_block_pos
                        ]

                        device_ops.multi_layer_block_kv_transfer_layerwise(
                            single_layer_kv_ptr,
                            tmp_gpu_buffers,
                            block_ids_curr,
                            cache_context.device,
                            lmcache_native.TransferDirection.H2D,
                            single_layer_sd,
                            slots_per_chunk,
                            cache_context.get_engine_kv_format(kernel_group_id),
                            recalculated_skip_blocks,
                        )

        # Record 1 IPC event per batch (all layers in batch share it)
        first_gl = all_layers[layer_batch_start][2]
        if first_gl < len(layer_events):
            event_backend.record_event(layer_events[first_gl], main_stream)
        if batch_leader_map is not None:
            for sub_idx in range(n_in_batch):
                gl = all_layers[layer_batch_start + sub_idx][2]
                batch_leader_map[gl] = first_gl

        # Stream the event handle to the worker immediately so it can
        # start attention on these layers while later batches transfer.
        if event_export_callback is not None and first_gl < len(layer_events):
            event_export_callback(first_gl, n_in_batch, layer_events[first_gl])

        layer_batch_start = batch_end


@dataclass
class _LayerwiseSession:
    """Per-request layer-wise transfer state.

    Created lazily by :meth:`LMCacheLayerwiseTransferModule._ensure_session`
    on the first object-group copy and reached through the thread-local bound
    by :meth:`LMCacheLayerwiseTransferModule.retrieve_layerwise`.

    Args:
        layer_events: Pre-allocated pool events, one per global layer.
        event_backend: Event backend for the context device.
        batch_leader_map: Maps a global layer index to the index of the
            event that actually signals its batch.
        channel: Callable that emits one response frame, or None when
            the worker asked for a single closing response.
        export_cb: Callback invoked as each layer batch is enqueued.
    """

    layer_events: list
    event_backend: EventIPCBackend
    channel: Any = None
    export_cb: Any = None
    batch_leader_map: dict[int, int] = field(default_factory=dict)


@dataclass
class _PendingRequest:
    """Layer-wise request context handed to the hooks via thread-local."""

    instance_id: int
    channel: Any


class LMCacheLayerwiseTransferModule(LMCacheDrivenTransferModule):
    """LMCache-driven transfer module with a layer-wise retrieve path.

    Serves ``RETRIEVE_LAYERWISE``, which copies KV data in layer-major order
    and signals a pool event as each layer batch lands on the device.

    A server node started with ``--layerwise-batch > 0`` loads this module and
    serves the layer-wise retrieve path *exclusively*: plain ``RETRIEVE`` is
    still routed here so the mismatch can be reported, but it is rejected. See
    :meth:`retrieve`. Every other base request type is served unchanged.
    """

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # Keyed by instance_id. Held here rather than on ContextEntry so
        # the base module stays unaware of the layer-wise path.
        self._event_pools: dict[int, EventPool] = {}
        self._tls = threading.local()

    def get_handlers(self) -> list[HandlerSpec]:
        """Base handlers plus the layer-wise handlers.

        ``RETRIEVE`` stays registered on purpose even though this module
        rejects it: an unregistered request type is only logged server-side
        and never answered, so dropping it would strand a misconfigured worker
        for the full ``mq_timeout`` instead of failing it immediately.
        """
        return super().get_handlers() + [
            HandlerSpec(
                RequestType.REGISTER_LAYERWISE_IPC_EVENT_POOL,
                self.register_layerwise_ipc_event_pool,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.RETRIEVE_LAYERWISE,
                self.retrieve_layerwise,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def register_layerwise_ipc_event_pool(
        self, instance_id: int
    ) -> tuple[int, list[bytes]]:
        """Handle ``REGISTER_LAYERWISE_IPC_EVENT_POOL``.

        Issued by the worker right after ``REGISTER_KV_CACHE`` so that
        registration itself keeps its plain ``None`` response for every
        deployment, layer-wise or not.

        Args:
            instance_id: The GPU instance ID (such as PID).

        Returns:
            The configured ``layerwise_batch`` and the exported IPC handles
            of the per-layer event pool. The handle list is empty when
            layer-wise mode is disabled.
        """
        pool = self._ensure_event_pool(instance_id)
        if pool is None:
            return 0, []
        return self._ctx.layerwise_batch, pool.handles

    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Reject plain ``RETRIEVE``: this node serves layer-wise only.

        ``REGISTER_KV_CACHE`` is byte-identical for both connectors, so a
        worker that loaded the per-chunk connector against a layer-wise server
        registers successfully and only reveals the mismatch here, on its
        first retrieve. Report it with an actionable message rather than
        letting it fail deep inside the transfer hooks.

        The rejection is returned, not raised: the MQ server only logs
        exceptions escaping a blocking handler and never sends a reply, which
        would strand the worker for the full ``mq_timeout``. Mirrors the
        unregistered-instance rejection in :meth:`LMCacheDrivenTransferModule.
        retrieve` -- no device work was submitted, so the event handle is
        empty and the result is ``False``.

        Args:
            key: The IPC key for the KV cache blocks.
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: Unused; the request is rejected before any copy.
            event_ipc_handle: Unused; no producer event is imported.
            skip_first_n_tokens: Unused; the request is rejected.

        Returns:
            ``(b"", False)``: no completion event, retrieve not served.
        """
        del gpu_block_ids, event_ipc_handle, skip_first_n_tokens
        logger.error(
            "Rejecting per-chunk RETRIEVE for GPU instance ID %d: this MP "
            "server was started with --layerwise-batch=%d and serves "
            "RETRIEVE_LAYERWISE only. Start the worker with the layer-wise "
            "connector -- kv_transfer_config needs both "
            '"kv_connector": "LMCacheLayerwiseMPConnector" and '
            '"kv_connector_module_path": '
            '"lmcache.integration.vllm.lmcache_mp_connector_layerwise" -- '
            "or restart the server with --layerwise-batch 0 to serve the "
            "per-chunk path instead.",
            instance_id,
            self._ctx.layerwise_batch,
        )
        try:
            self._release_failed_retrieve_locks(key, instance_id)
        except Exception:
            # A cleanup failure must never suppress the terminal response.
            logger.exception(
                "Failed to release RETRIEVE locks for the rejected per-chunk "
                "request from GPU instance ID %d",
                instance_id,
            )
        return b"", False

    def close(self) -> None:
        """Drop event pools, then run the base teardown."""
        self._event_pools.clear()
        super().close()

    def _ensure_event_pool(self, instance_id: int) -> EventPool | None:
        """Create (once) the IPC event pool for a registered instance.

        Also latches the interleaved host-buffer layout, which is a
        deployment-wide invariant derived from ``layerwise_batch``. Doing
        it at registration also covers cold-start retrieves that read
        chunks written by a previous run.

        Args:
            instance_id: The GPU instance ID (such as PID).

        Returns:
            The pool for this instance, or None when layer-wise mode is
            disabled or the instance is not registered.
        """
        if self._ctx.layerwise_batch <= 0:
            return None
        pool = self._event_pools.get(instance_id)
        if pool is not None:
            return pool

        with self._lock:
            entry = self._cache_contexts.get(instance_id)
        if entry is None or entry.event_backend is None:
            return None

        cache_context = entry.cache_context
        kernel_groups = cache_context.kv_layer_groups_manager.kernel_groups
        for kgr in kernel_groups:
            kgr.shape_desc.kv_interleaved = True

        num_total_layers = sum(kgr.num_layers for kgr in kernel_groups)
        if num_total_layers > EVENT_POOL_SIZE:
            raise ValueError(
                f"Model has {num_total_layers} total layers but "
                f"EVENT_POOL_SIZE={EVENT_POOL_SIZE}. Increase "
                f"EVENT_POOL_SIZE or disable layerwise mode."
            )
        pool = EventPool(entry.event_backend, cache_context.device)
        self._event_pools[instance_id] = pool
        logger.info(
            "Allocated layerwise event pool for GPU ID %d (size=%d)",
            instance_id,
            pool.size,
        )
        return pool

    def retrieve_layerwise(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
        *,
        response_channel=None,
    ) -> tuple[bytes, bool, bool]:
        """Handle ``RETRIEVE_LAYERWISE``.

        Marks the calling thread as layer-wise for the duration of the
        inherited :meth:`retrieve`, whose per-object-group copies then route
        through this class's :meth:`_transfer_object_group`. See
        :meth:`LMCacheDrivenTransferModule.retrieve` for the argument
        semantics.

        Args:
            response_channel: Callable used to answer with one frame per
                layer batch, or None to report every event index in a single
                closing frame.

        Returns:
            ``(payload, is_final, succeeded)``. ``payload`` is empty when the
            indices were already reported frame by frame.
        """
        self._tls.request = _PendingRequest(instance_id, response_channel)
        self._tls.session = None
        try:
            # ``super()``, not ``self``: this class overrides ``retrieve`` to
            # reject the per-chunk request type, but the retrieve *loop* is
            # exactly what this handler reuses -- ``_transfer_object_group``
            # is the seam that makes it layer-wise.
            handle, succeeded = super().retrieve(
                key,
                instance_id,
                gpu_block_ids,
                event_ipc_handle,
                skip_first_n_tokens,
            )
            session = self._tls.session
        finally:
            self._tls.request = None
            self._tls.session = None
        # The inherited ``retrieve`` answers with the base ``(handle,
        # succeeded)`` pair from every exit it has; widen it here to the
        # ``(payload, is_final, succeeded)`` triple RETRIEVE_LAYERWISE
        # declares. ``session`` stays None when the request exited before any
        # copy was enqueued -- unregistered instance, block-ID underflow, or
        # nothing to transfer -- so there are no per-layer events to report.
        if session is None or not session.layer_events:
            return handle, True, succeeded
        if session.channel is not None:
            # The indices were already reported frame by frame; the closing
            # frame only has to carry the success flag.
            return b"", True, succeeded
        indices = [
            session.batch_leader_map.get(gl, gl)
            for gl in range(len(session.layer_events))
        ]
        return struct.pack(f"<{len(indices)}i", *indices), True, succeeded

    def _ensure_session(
        self,
        cache_context: BaseCacheContext,
        request: _PendingRequest,
    ) -> _LayerwiseSession:
        """Bind the pooled per-layer events for the in-flight request."""
        pool = self._ensure_event_pool(request.instance_id)
        if pool is None:
            raise RuntimeError(
                "Layerwise retrieve requested but no event pool exists for "
                f"instance {request.instance_id}"
            )
        with self._lock:
            entry = self._cache_contexts.get(request.instance_id)
        if entry is None or entry.event_backend is None:
            raise RuntimeError(
                "Layerwise retrieve requested but instance "
                f"{request.instance_id} has no event backend"
            )

        kernel_groups = cache_context.kv_layer_groups_manager.kernel_groups
        num_total_layers = sum(kgr.num_layers for kgr in kernel_groups)
        session = _LayerwiseSession(
            layer_events=[pool.event_at(i) for i in range(num_total_layers)],
            event_backend=entry.event_backend,
            channel=request.channel,
        )
        if request.channel is not None:
            # Built once per request, not per object group: the closure
            # only captures the channel. Pool mode sends the index, so there
            # is no export_event call on the hot path.
            def _export_cb(first_layer, count, event, _send=request.channel):
                _send(
                    (
                        struct.pack("<3i", first_layer, count, first_layer),
                        False,
                        False,
                    )
                )

            session.export_cb = _export_cb
        return session

    def _transfer_object_group(
        self,
        cache_context: BaseCacheContext,
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        *,
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: lmcache_native.TransferDirection,
    ) -> None:
        """Copy one object group in layer-major order."""
        request = getattr(self._tls, "request", None)
        if request is None:
            # ``retrieve`` rejects plain RETRIEVE outright, so the only way
            # into the inherited retrieve loop is RETRIEVE_LAYERWISE, which
            # binds the thread-local request first. Getting here means a new
            # copy site was added that bypasses that binding.
            raise RuntimeError(
                "Layerwise object-group transfer has no bound request; "
                "_transfer_object_group was reached outside retrieve_layerwise"
            )

        if direction != lmcache_native.TransferDirection.H2D:
            # transfer_kv_layerwise implements the retrieve direction only:
            # it records a per-layer event after each layer batch lands in
            # the engine's KV cache. Raise rather than assert so the check
            # survives `python -O`.
            raise ValueError(
                f"Layerwise object-group transfer supports H2D only, got {direction}"
            )

        session = getattr(self._tls, "session", None)
        if session is None:
            # Bound on the first copy of the request, not once per group.
            session = self._ensure_session(cache_context, request)
            self._tls.session = session

        transfer_kv_layerwise(
            cache_context,
            block_ids_gpu,
            memory_objs,
            object_group_id=object_group_id,
            batch_size=batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            layer_events=session.layer_events,
            event_backend=session.event_backend,
            batch_leader_map=session.batch_leader_map,
            layerwise_batch=self._ctx.layerwise_batch,
            event_export_callback=session.export_cb,
        )
