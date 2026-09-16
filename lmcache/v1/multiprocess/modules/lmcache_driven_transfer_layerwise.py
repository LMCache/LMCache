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
from functools import partial
from typing import Any, Sequence
import os
import struct
import threading

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_ops import build_staging_copies
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.layer_major_plan import (
    build_layer_major_specs,
    build_run_shape_desc,
    set_layer_major_staging_enabled,
    uniform_stride_runs,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    _HAS_NATIVE_OBJECT_GROUP_TRANSFER,
    batched_iteration_with_skip,
    recalculate_blocks_to_skip,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.base.event_pool import EVENT_POOL_SIZE, EventPool
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


# Set once, the first time a layer batch cannot be merged into one native
# call and instead issues a separate H2D copy per layer. Still layer-wise:
# there is no per-chunk fallback on this path.
_warned_layerwise_fallback = False

# Diagnostic instrumentation for the layer-wise batching path. Off by
# default; set LMCACHE_LAYERWISE_DEBUG=1 to have each object group report
# its kernel-group structure and the layer-batch sizes it actually
# achieved. Both are logged once per group so a steady-state benchmark
# is not flooded.
_LAYERWISE_DEBUG = os.getenv("LMCACHE_LAYERWISE_DEBUG", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)
_dbg_logged_groups: set[tuple[int, int]] = set()


def _dbg_run_lengths(all_layers: list) -> list[tuple[int, int]]:
    """Run-length encode the kernel-group id sequence of ``all_layers``.

    A merged layer batch cannot span a kernel-group boundary, so these run
    lengths are the hard ceiling on the achievable batch size regardless of
    the configured ``layerwise_batch_size``.

    Args:
        all_layers: ``(kg_info_idx, local_idx, global_layer_idx)`` triples
            in global layer order.

    Returns:
        ``(kernel_group_index, run_length)`` pairs in layer order.
    """
    runs: list[list[int]] = []
    for kg_idx, _, _ in all_layers:
        if runs and runs[-1][0] == kg_idx:
            runs[-1][1] += 1
        else:
            runs.append([kg_idx, 1])
    return [(kg, n) for kg, n in runs]


def _object_group_unit_ptr(
    slot_base_ptrs: Sequence[int],
    unit_idx: int,
    *,
    unit_bytes: int,
    units_per_slot: int,
) -> int:
    """Device address of one staging unit inside the object-group buffer.

    The buffer is used here as a flat arena rather than as a mirror of the
    object, since each launch is told where its own run starts. Slots are
    separate allocations, so the caller passes one base address per slot
    rather than a single arena base: walking flat past the end of one would
    land in the next slot.
    """
    slot, sub = divmod(unit_idx, units_per_slot)
    return slot_base_ptrs[slot] + sub * unit_bytes


def _transfer_layer_major(
    cache_context: BaseCacheContext,
    *,
    object_group_id: int,
    kg_infos: list[dict],
    req_infos: list[dict],
    all_layers: list[tuple[int, int, int]],
    all_chunks_flat: list[tuple[int, "MemoryObj"]],
    lmcache_chunk_size: int,
    skip_first_n_tokens: int,
    layerwise_batch_size: int,
    pin_chunk_size: int,
    main_stream: Any,
    event_backend: EventIPCBackend,
    layer_events: list,
    batch_leader_map: dict[int, int] | None,
    event_export_callback: Any,
) -> None:
    """H2D + scatter for an object laid out by model depth.

    The kernel-group-major path batches N layers of ONE kernel group, which
    is one contiguous range only because such an object stores a group's
    layers back to back. A layer-major object stores a depth's caches back to
    back instead, so the contiguous range -- and therefore the unit of a
    single H2D copy -- is N consecutive model depths, spanning every kernel
    group those depths own.

    Each depth batch is then scattered by one launch per (kernel group,
    constant-stride run), with ``layer_stride_elems`` telling the kernel how
    far apart that group's layers sit. Merging is what keeps the launch count
    near the number of depth batches instead of the number of caches.

    Batching by whole depths also keeps the event bookkeeping honest: the
    consumer waits per model layer, so every cache of a depth has to be
    covered by the same event.
    """
    num_active_chunks = len(all_chunks_flat)
    if num_active_chunks == 0 or not all_layers:
        return

    def depth_of(entry: tuple[int, int, int]) -> int:
        return kg_infos[entry[0]]["model_depths"][entry[1]]

    def offset_of(entry: tuple[int, int, int]) -> int:
        return kg_infos[entry[0]]["layer_offsets"][entry[1]]

    def size_of(entry: tuple[int, int, int]) -> int:
        return kg_infos[entry[0]]["per_layer_bytes"]

    # The staging destination is scratch, not a mirror of the object: the
    # launch is told where each run starts, so the whole object-group slot can
    # be used as a flat arena and packed with as many chunks as it holds.
    probe = cache_context.get_temp_object_group_buffer(0, object_group_id)
    slot_bytes = probe.nelement() * probe.element_size()
    # One base address per batch slot, resolved once for the whole object
    # group. A slot's address depends only on the slot and the object group,
    # so resolving it per pointer-table entry re-sliced the same device
    # tensors on every depth batch of every request.
    slot_base_ptrs = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id).data_ptr()
        for slot in range(cache_context.max_batch_size)
    ]

    base_obj_idx = all_chunks_flat[0][0]
    num_entries = len(all_layers)
    batch_start = 0
    dbg_runs = 0
    dbg_launches = 0

    while batch_start < num_entries:
        # Take up to layerwise_batch_size whole depths.
        batch_end = batch_start
        depths_taken = 0
        current_depth = None
        while batch_end < num_entries:
            depth = depth_of(all_layers[batch_end])
            if depth != current_depth:
                if depths_taken == layerwise_batch_size:
                    break
                depths_taken += 1
                current_depth = depth
            batch_end += 1

        entries = all_layers[batch_start:batch_end]
        run_offset = offset_of(entries[0])
        run_bytes = sum(size_of(entry) for entry in entries)

        # The single H2D below assumes these entries tile one byte range with
        # no hole and no overlap. That is what the layer-major placement
        # produces; check it rather than trust it, because the failure mode is
        # silently transferred garbage.
        last = entries[-1]
        if offset_of(last) + size_of(last) != run_offset + run_bytes:
            raise ValueError(
                f"Layer-major batch for object group {object_group_id} is not "
                f"contiguous: entries {entries[0]}..{last} span "
                f"[{run_offset}, {offset_of(last) + size_of(last)}) but hold "
                f"{run_bytes} bytes"
            )
        if run_bytes > slot_bytes:
            raise ValueError(
                f"Layer-major batch of {depths_taken} depth(s) needs "
                f"{run_bytes} staging bytes, more than the {slot_bytes}-byte "
                f"object-group slot"
            )

        units_per_slot = slot_bytes // run_bytes
        max_chunks_per_pass = units_per_slot * cache_context.max_batch_size
        # The spec's pointer table is indexed by a chunk's position within a
        # pass, so it only needs to cover the largest pass this request will
        # actually run. Filling it to the arena's capacity instead resolves a
        # staging buffer per unreachable slot, and the native side truncates
        # the table to LaunchVar.num_objects regardless.
        units_needed = min(max_chunks_per_pass, num_active_chunks)

        unit_ptr = partial(
            _object_group_unit_ptr,
            slot_base_ptrs,
            unit_bytes=run_bytes,
            units_per_slot=units_per_slot,
        )

        # One spec per (kernel group, constant-stride run) in this batch.
        specs: list = []
        launch_meta: list[tuple[int, int]] = []  # (spec index, kg_info_idx)
        kv_ptr_views: list[torch.Tensor] = []  # keep slices alive
        by_kg: dict[int, list[tuple[int, int, int]]] = {}
        for entry in entries:
            by_kg.setdefault(entry[0], []).append(entry)

        for kg_info_idx, kg_entries in by_kg.items():
            info = kg_infos[kg_info_idx]
            req_info = req_infos[kg_info_idx]
            all_block_ids = req_info["all_block_ids"]
            if all_block_ids is None:
                continue
            sd = info["sd"]
            kg_offsets = [offset_of(entry) for entry in kg_entries]

            for run_start, run_len, stride_bytes in uniform_stride_runs(kg_offsets):
                first_local = kg_entries[run_start][1]
                rel = kg_offsets[run_start] - run_offset
                run_sd = build_run_shape_desc(sd, nl=run_len, stride_bytes=stride_bytes)

                kv_ptrs = info["group_kv_pointers"][first_local : first_local + run_len]
                kv_ptr_views.append(kv_ptrs)

                launch_meta.append((len(specs), kg_info_idx))
                specs.append(
                    device_ops.KernelGroupSpec(
                        kv_ptrs.data_ptr(),
                        [unit_ptr(unit) + rel for unit in range(units_needed)],
                        run_sd,
                        info["slots_per_chunk"],
                        cache_context.get_engine_kv_format(info["kernel_group_id"]),
                        all_block_ids.data_ptr(),
                        all_block_ids.numel(),
                    )
                )

        if specs:
            batch_steps: list = []
            pass_start = 0
            while pass_start < num_active_chunks:
                pass_end = min(pass_start + max_chunks_per_pass, num_active_chunks)
                pass_chunks = all_chunks_flat[pass_start:pass_end]

                staging: list = []
                for buf_idx, (_, memory_obj) in enumerate(pass_chunks):
                    staging.append(
                        device_ops.StagingCopy(
                            unit_ptr(buf_idx),
                            memory_obj.data_ptr + run_offset,
                            run_bytes,
                            memory_obj.meta.address + run_offset,
                        )
                    )

                first_obj_idx = pass_chunks[0][0]
                last_obj_idx = pass_chunks[-1][0]
                first_start_token = first_obj_idx * lmcache_chunk_size
                skip_tokens_first = (
                    max(first_start_token, skip_first_n_tokens) - first_start_token
                )

                launches: list = []
                for spec_index, kg_info_idx in launch_meta:
                    info = kg_infos[kg_info_idx]
                    kernel_group_id = info["kernel_group_id"]
                    blocks_per_window = info["blocks_per_window"]
                    launches.append(
                        device_ops.LaunchVar(
                            spec_index,
                            (first_obj_idx - base_obj_idx) * blocks_per_window,
                            (last_obj_idx + 1 - first_obj_idx) * blocks_per_window,
                            len(pass_chunks),
                            recalculate_blocks_to_skip(
                                info["blocks_per_chunk"],
                                blocks_per_window,
                                cache_context.calculate_num_blocks(
                                    skip_tokens_first, kernel_group_id
                                ),
                            ),
                        )
                    )

                batch_steps.append(device_ops.BatchStep(staging, launches))
                pass_start = pass_end

            device_ops.execute_object_group_transfer_layerwise(
                lmcache_native.TransferDirection.H2D,
                cache_context.device,
                pin_chunk_size,
                specs,
                batch_steps,
            )
            dbg_runs += 1
            dbg_launches += len(specs)

        # One event per depth batch. The registration indices it covers are
        # scattered, so they are sent explicitly rather than as a range.
        covered = [entry[2] for entry in entries]
        leader = covered[0]
        if leader < len(layer_events):
            event_backend.record_event(layer_events[leader], main_stream)
        if batch_leader_map is not None:
            for global_layer_idx in covered:
                batch_leader_map[global_layer_idx] = leader
        if event_export_callback is not None and leader < len(layer_events):
            event_export_callback(leader, len(covered), layer_events[leader], covered)

        batch_start = batch_end

    if _LAYERWISE_DEBUG:
        dbg_key = (object_group_id, layerwise_batch_size)
        if dbg_key not in _dbg_logged_groups:
            _dbg_logged_groups.add(dbg_key)
            logger.info(
                "layerwise[group %s]: layer-major, requested batch=%d -> "
                "%d depth batch(es), %d scatter launch(es) for %d cache(s)",
                object_group_id,
                layerwise_batch_size,
                dbg_runs,
                dbg_launches,
                len(all_layers),
            )


def transfer_kv_layer_major(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    *,
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Copy one object group whose staging slot is ordered by model depth.

    This is the store side of the layer-major layout, and it mirrors
    ``_run_object_group_transfer_plan`` in the per-chunk module: same batched
    iteration, same skip accounting, same staging copies. The one difference
    is what a kernel group costs. Layer-major scatters a group through the
    object, so a group resolves to one spec per constant-stride run rather
    than to a single spec, and a batch therefore emits one launch per run.

    Keeping a copy here rather than parameterising the original is
    deliberate: it leaves the per-chunk path with no layout branch at all.
    The price is that a fix to the batch walk below belongs in both places.

    Args:
        cache_context: The GPU cache context holding the KV cache geometry.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The MemoryObj instances to copy. None entries are only
            valid for D2H (the batch is skipped); H2D raises.
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).

    Raises:
        NotImplementedError: If the native object-group transfer is missing,
            or the batch holds a GDS-backed object.
        ValueError: If a None entry is found in memory_objs when direction is
            H2D.
    """
    if not _HAS_NATIVE_OBJECT_GROUP_TRANSFER or any(
        isinstance(mo, GDSMemoryObject) for mo in memory_objs
    ):
        # The per-kernel-group fallback addresses a whole kernel group as one
        # contiguous staging region, which this layout does not provide.
        #
        # TODO: supporting GDS here means driving the GDS reads off the
        # per-(layer, kernel group) placement instead of the per-kernel-group
        # one, the same way the native path below does.
        raise NotImplementedError(
            "The layer-major staging layout requires the native object-group "
            "transfer path; the per-kernel-group fallback cannot address a "
            "layer-major buffer."
        )

    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    # --- Per-kernel-group invariants, resolved once ---
    blocks_per_chunk_by_kg: dict[int, int] = {}
    blocks_per_window_by_kg: dict[int, int] = {}
    for kernel_group_id in kernel_group_ids:
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
        blocks_per_chunk_by_kg[kernel_group_id] = blocks_per_chunk
        blocks_per_window_by_kg[kernel_group_id] = blocks_per_window

    # A launch unit is one (spec index, kernel group) pair to dispatch. The
    # whole object is staged here, which is the case that merges best.
    kernel_group_specs, launch_units = build_layer_major_specs(
        cache_context,
        object_group_id,
        block_ids_gpu,
        kernel_group_ids,
        max_batch_size,
    )

    # Temp object-group staging buffers (reused per batch slot).
    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id)
        for slot in range(max_batch_size)
    ]

    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = 0
    if not attn_desc.is_full_attention(object_group_id) and is_h2d:
        sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
        num_objects_to_skip = max(0, len(memory_objs) - sw_size_chunks)
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    # --- Walk the batches in order, emitting staging + launch work per step ---
    batch_steps: list[Any] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        staging = build_staging_copies(
            memory_object_batch,
            object_group_buffers[:batch_len],
            is_h2d,
        )

        launches: list[Any] = []
        for spec_index, kernel_group_id in launch_units:
            blocks_per_chunk = blocks_per_chunk_by_kg[kernel_group_id]
            blocks_per_window = blocks_per_window_by_kg[kernel_group_id]

            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            recalculated_skip_blocks = recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            launches.append(
                device_ops.LaunchVar(
                    spec_index,
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    recalculated_skip_blocks,
                )
            )

        batch_steps.append(device_ops.BatchStep(staging, launches))

    if not batch_steps:
        return

    device_ops.execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
    )


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

            # Every layer is placed on its own, so its offset comes from
            # the per-layer map rather than from a group base plus a
            # multiple of per_layer_bytes. That map is filled under both
            # geometries, so this reads the same either way.
            layer_offsets = [
                cache_context.get_layer_offset_in_object(
                    object_group_id, kernel_group_id, local
                )
                for local in range(kg.num_layers)
            ]

            group_kv_pointers = cache_context.get_kernel_group_kv_pointers(
                kernel_group_id
            )

            info = {
                "kernel_group_id": kernel_group_id,
                "kg": kg,
                "blocks_per_chunk": blocks_per_chunk,
                "blocks_per_window": blocks_per_window,
                "slots_per_chunk": slots_per_chunk,
                "per_layer_bytes": per_layer_bytes,
                "layer_offsets": layer_offsets,
                "model_depths": list(kg.model_depths or kg.layer_indices),
                "sd": sd,
                "group_kv_pointers": group_kv_pointers,
            }

            kg_infos.append(info)

        all_layers = []
        for kg_info_idx, info in enumerate(kg_infos):
            kg = info["kg"]
            for local_idx, global_layer_idx in enumerate(kg.layer_indices):
                all_layers.append((kg_info_idx, local_idx, global_layer_idx))

        # Order the entries the way the object is laid out, so a batch of
        # adjacent entries is a contiguous byte range. Done once here (not per
        # request) since the cached list is reused.
        #
        # The object is ordered by model depth, ties broken by kernel
        # group. For a model with one cache per layer that is the same
        # order as the registration index; for one whose layers span
        # several kernel groups the two differ, and depth is the one the
        # bytes follow.
        all_layers.sort(key=lambda x: (kg_infos[x[0]]["model_depths"][x[1]], x[0]))

        if _LAYERWISE_DEBUG:
            logger.info(
                "layerwise[group %s]: %d entries across %d kernel group(s), "
                "%d bytes/chunk total; kg runs (kg, len)=%s",
                object_group_id,
                len(all_layers),
                len(kg_infos),
                sum(
                    i["per_layer_bytes"] * len(i["kg"].layer_indices) for i in kg_infos
                ),
                _dbg_run_lengths(all_layers),
            )
            for dbg_idx, dbg_info in enumerate(kg_infos):
                dbg_sd = dbg_info["sd"]
                dbg_kg = dbg_info["kg"]
                logger.info(
                    "layerwise[group %s]:   kg %d (id=%s): %d layer(s), "
                    "kv_size=%d nh=%d hs=%d hidden=%d elt=%dB "
                    "slots/chunk=%d per_layer_bytes=%d "
                    "layer_indices=%s",
                    object_group_id,
                    dbg_idx,
                    dbg_info["kernel_group_id"],
                    len(dbg_kg.layer_indices),
                    dbg_sd.kv_size,
                    dbg_sd.nh,
                    dbg_sd.hs,
                    dbg_kg.hidden_dim_size,
                    dbg_sd.element_size,
                    dbg_info["slots_per_chunk"],
                    dbg_info["per_layer_bytes"],
                    list(dbg_kg.layer_indices),
                )

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

    # Flatten all valid chunks for single-launch-per-layer optimization.
    # In the existing temp buffer (4 full-chunk slots per kernel group),
    # each slot holds nl per-layer entries, giving 4*nl per-layer slots
    # total -- enough for hundreds of chunks on typical LLMs.
    all_chunks_flat: list[tuple[int, MemoryObj]] = []
    for start_object_idx, memory_object_batch in batch_plan:
        for local_idx, mo in enumerate(memory_object_batch):
            all_chunks_flat.append((start_object_idx + local_idx, mo))

    num_active_chunks = len(all_chunks_flat)

    # Pre-compute per-kernel-group block_ids and skip for single-launch path.
    #
    # These three values are REQUEST-SCOPED and must not be written back into
    # ``kg_infos``: that list is cached on the cache_context and reused by
    # later requests, while ``all_block_ids`` is a view into *this* request's
    # staged block-id tensor. Storing it there let a subsequent request read a
    # stale slice (wrong block ids -> silent corruption) or one whose backing
    # allocation had been freed (dangling device pointer -> illegal access).
    # The per-chunk path keeps the equivalent state in locals for exactly this
    # reason -- see ``transfer_kv_per_object_group`` in
    # lmcache_driven_transfer.py, which rebuilds ``block_ids_tensor`` and its
    # KernelGroupSpec on every call and caches nothing on the context.
    req_infos: list[dict] = [{} for _ in kg_infos]
    for info, req_info in zip(kg_infos, req_infos, strict=True):
        sd = info["sd"]
        kernel_group_id = info["kernel_group_id"]
        blocks_per_window = info["blocks_per_window"]
        blocks_per_chunk = info["blocks_per_chunk"]

        if num_active_chunks > 0:
            first_obj_idx = all_chunks_flat[0][0]
            last_obj_idx = all_chunks_flat[-1][0]
            first_block = first_obj_idx * blocks_per_window
            end_block = (last_obj_idx + 1) * blocks_per_window
            req_info["all_block_ids"] = block_ids_gpu[kernel_group_id][
                first_block:end_block
            ]
            first_start_token = first_obj_idx * lmcache_chunk_size
            skip_tokens_first = (
                max(first_start_token, skip_first_n_tokens) - first_start_token
            )
            orig_skip = cache_context.calculate_num_blocks(
                skip_tokens_first, kernel_group_id
            )
            req_info["all_skip_blocks"] = recalculate_blocks_to_skip(
                blocks_per_chunk, blocks_per_window, orig_skip
            )

    _transfer_layer_major(
        cache_context,
        object_group_id=object_group_id,
        kg_infos=kg_infos,
        req_infos=req_infos,
        all_layers=all_layers,
        all_chunks_flat=all_chunks_flat,
        lmcache_chunk_size=lmcache_chunk_size,
        skip_first_n_tokens=skip_first_n_tokens,
        layerwise_batch_size=layerwise_batch_size,
        pin_chunk_size=pin_chunk_size,
        main_stream=main_stream,
        event_backend=event_backend,
        layer_events=layer_events,
        batch_leader_map=batch_leader_map,
        event_export_callback=event_export_callback,
    )


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
        # Layer-major staging only pays off on this path, and both paths in a
        # deployment must agree on the object layout, so the opt-in belongs to
        # the module that wants it rather than to the server or the staging
        # buffer. Loading this module is itself the opt-in, and it happens
        # before any cache context is created, which is when the layout is
        # baked in. The server loads this class only for --layerwise-batch
        # > 0, so that flag is what the layout really follows. Every geometry
        # this module accepts is then staged by model depth, including the one
        # whose layers each own a single kernel group, where depth order and
        # kernel-group-major order happen to be the same order.
        set_layer_major_staging_enabled(True)
        # Keyed by instance_id. Held here rather than on ContextEntry so
        # the base module stays unaware of the layer-wise path.
        self._event_pools: dict[int, EventPool] = {}
        self._tls = threading.local()

    @request_handler(
        RequestType.REGISTER_LAYERWISE_IPC_EVENT_POOL,
        HandlerType.SYNC,
    )
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

    # Registered on purpose even though this override rejects it: an
    # unregistered request type is only logged server-side and never
    # answered, so dropping it would strand a misconfigured worker for the
    # full ``mq_timeout`` instead of failing it immediately.
    @request_handler(
        RequestType.RETRIEVE,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
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

    @request_handler(
        RequestType.RETRIEVE_LAYERWISE,
        HandlerType.STREAMING,
        requires_client_affinity=True,
    )
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
            def _export_cb(
                first_layer, count, event, layer_indices=None, _send=request.channel
            ):
                if layer_indices is None:
                    frame = struct.pack("<3i", first_layer, count, first_layer)
                else:
                    # Layer-major batches cover a set of registration indices
                    # that is not a contiguous range, so send them verbatim.
                    # The negative first field and the extra length are what
                    # tell the two frame shapes apart on the worker side.
                    frame = struct.pack(
                        "<3i", -1, len(layer_indices), first_layer
                    ) + struct.pack(f"<{len(layer_indices)}i", *layer_indices)
                _send((frame, False, False))

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
        transfer_key: str,
    ) -> None:
        """Copy one object group, layer-aware in both directions.

        Retrieve is layer-*wise*: :func:`transfer_kv_layerwise` records a
        per-layer event after each layer batch lands in the engine's KV
        cache, so the consumer can start a layer before the object is whole.

        Store has no such staging, but it is what writes the depth order
        the retrieve above depends on, so every store goes to
        :func:`transfer_kv_layer_major`. The inherited per-chunk copy
        writes the other object layout and is never correct here.
        """
        if direction != lmcache_native.TransferDirection.H2D:
            transfer_kv_layer_major(
                cache_context,
                block_ids_gpu,
                memory_objs,
                object_group_id=object_group_id,
                batch_size=batch_size,
                skip_first_n_tokens=skip_first_n_tokens,
                direction=direction,
            )
            return

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
        # ``transfer_key`` is accepted to match the base seam but not used
        # yet: neither layer-aware native entry point takes the phase-timing
        # arguments, so this path records no samples in either direction.
