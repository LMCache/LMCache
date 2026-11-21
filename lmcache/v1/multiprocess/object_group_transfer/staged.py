# SPDX-License-Identifier: Apache-2.0
"""Staged copy path: staging buffer plus the block transfer kernel, issued
as one native plan (host-memory objects only) or one batch at a time."""

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, cast

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.v1.common.attn_window import recalculate_blocks_to_skip
from lmcache.v1.common.logging import init_logger
from lmcache.v1.gpu_connector.gds_context import (
    SlabDirection,
    get_l1_gds_context,
)
from lmcache.v1.gpu_connector.gpu_ops import build_staging_copies
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    stage,
)
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    batched_iteration_with_skip,
    resolve_chunk_range,
)
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    device_ops, "execute_object_group_transfer"
)
_HAS_TRANSFER_PHASE_TIMING: bool = hasattr(device_ops, "pop_completed_phase_timings")


def native_plan_available() -> bool:
    """Return whether the extension can run a whole plan in one native call."""
    return _HAS_NATIVE_OBJECT_GROUP_TRANSFER


def run_object_group_transfer_plan(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    transfer_key: str,
    chunk_range: tuple[int, int] | None = None,
) -> None:
    """Plan and execute one object group's transfer in a single native call.

    This is the fast path of :func:`transfer_kv_per_object_group`: it runs the
    same batched-iteration / skip logic, but instead of issuing each staging
    copy and kernel launch immediately (each a GIL release/re-acquire), it
    resolves every argument to plain pointers/scalars (the "planner", GIL held
    throughout) and hands the whole plan to ``execute_object_group_transfer``,
    which issues all of it on the stream within a single GIL release.

    Requires every object in the range to be host memory (staged through
    the lazy-allocator path).

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        block_ids_gpu: GPU block IDs, indexed by LMCache KV group index.
        memory_objs: The MemoryObj instances to copy. None entries are only
            valid for D2H (the batch is skipped); H2D raises.
        object_group_id: Index of the object group being copied.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the range.
        direction: H2D (retrieve) or D2H (store).
        transfer_key: Identity of this store/retrieve operation, echoed back
            on every phase-timing sample (a request issues several transfers,
            so the request id cannot identify one).
        chunk_range: ``[start, end)`` chunks to move; None means all.

    Raises:
        ValueError: If a None entry is found in memory_objs when direction is
            H2D, or if an object's size does not match its GPU staging buffer.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    # --- Per-kernel-group invariants, resolved once (vs. every batch before) ---
    kernel_group_specs: list[Any] = []
    spec_index_by_kg: dict[int, int] = {}
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

        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        block_ids_tensor = block_ids_gpu[kernel_group_id]
        temp_buffers = [
            cache_context.get_temp_kernel_group_buffer(slot, kernel_group_id)
            for slot in range(max_batch_size)
        ]

        spec_index_by_kg[kernel_group_id] = len(kernel_group_specs)
        kernel_group_specs.append(
            device_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                block_ids_tensor.data_ptr(),
                block_ids_tensor.numel(),
            )
        )

    # Temp object-group staging buffers (reused per batch slot, like above).
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
    start_chunk, end_chunk = resolve_chunk_range(
        len(memory_objs), chunk_range, num_objects_to_skip
    )
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs[:end_chunk], batch_size, skip_count=start_chunk
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
        for kernel_group_id in kernel_group_ids:
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
                    spec_index_by_kg[kernel_group_id],
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    recalculated_skip_blocks,
                )
            )

        batch_steps.append(device_ops.BatchStep(staging, launches))

    if not batch_steps:
        return

    # Time the phases only when a subscriber consumes the samples. An older
    # compiled extension has neither the keywords nor anything to consume
    # them, so fall back to the untimed legacy signature.
    timing_kwargs = (
        {
            "phase_timing_enabled": is_observability_enabled()
            and get_event_bus().has_subscribers(EventType.MP_TRANSFER_PHASE_SAMPLES),
            # Echoed back verbatim on each sample; the transfer's identity.
            "session_id": transfer_key,
        }
        if _HAS_TRANSFER_PHASE_TIMING
        else {}
    )
    device_ops.execute_object_group_transfer(
        direction,
        cache_context.device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
        **timing_kwargs,
    )


def _launch_batch_kernels(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    object_group_id: int,
    start_object_idx: int,
    batch_len: int,
    skip_tokens_in_chunk: int,
    direction: "lmcache_native.TransferDirection",
    slot_start: int = 0,
) -> None:
    """Launch the block kernel for one staged batch, once per kernel group.

    Args:
        cache_context: Cache context of the KV cache.
        block_ids_gpu: Block IDs on the GPU, per kernel group.
        object_group_id: Object group index.
        slot_start: First staging slot used by this batch.
        start_object_idx: Chunk index of the batch's first chunk.
        batch_len: Chunks in the batch; chunk ``i`` is in staging slot ``i``.
        skip_tokens_in_chunk: Leading tokens of the batch not written.
        direction: H2D (retrieve) or D2H (store).
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    kernel_group_ids = kv_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices
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

        # Get the block ids for this chunk
        start_block_pos = start_object_idx * blocks_per_window
        end_block_pos = (start_object_idx + batch_len) * blocks_per_window

        block_ids_curr_batch = block_ids_gpu[kernel_group_id][
            start_block_pos:end_block_pos
        ]

        # Re-calculate the skip blocks for this kernel group
        orig_skip_blocks = cache_context.calculate_num_blocks(
            skip_tokens_in_chunk, kernel_group_id
        )
        recalculated_skip_blocks = recalculate_blocks_to_skip(
            blocks_per_chunk,
            blocks_per_window,
            orig_skip_blocks,
        )

        # Launch kernel
        group_kv_pointers = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        group_lmcache_chunk_size = cache_context.get_slots_per_chunk_in_sw(
            kernel_group_id
        )
        tmp_gpu_buffers_batched = [
            cache_context.get_temp_kernel_group_buffer(
                slot_start + i, kernel_group_id
            ).data_ptr()
            for i in range(batch_len)
        ]
        device_ops.multi_layer_block_kv_transfer(
            group_kv_pointers,
            tmp_gpu_buffers_batched,
            block_ids_curr_batch,
            cache_context.device,
            direction,
            cache_context.get_shape_desc(kernel_group_id),
            group_lmcache_chunk_size,
            cache_context.get_engine_kv_format(kernel_group_id),
            recalculated_skip_blocks,
        )


class _GDSStream(Protocol):
    """CUDA slot stream ordering work behind an already-recorded input event."""

    def wait_event(self, event: object) -> None: ...

    def query(self) -> bool: ...


@dataclass
class _GDSSlot:
    slot: int
    object_index: int
    skip_tokens: int
    nbytes: int
    scatter_submitted: bool = False


def _run_gds_concurrent(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    skip_first_n_tokens: int,
    skip_objects: int,
    end_chunk: int,
    direction: "lmcache_native.TransferDirection",
    block_ids_host: Sequence[Sequence[int]],
) -> bool:
    """Reap/refill at most four independent stream-ordered GDS slots.

    A shared slab scope serializes groups from different cache contexts. Within
    that scope, the pending queue bounds both operations and useful bytes. Reads
    are validated before scatter; stores gather before I/O on the same stream.
    Every slot remains owned until its final stream wait completes. This runs on
    the existing blocking affinity handler, never the transport event loop.
    """
    chunk_size = cache_context.lmcache_tokens_per_chunk
    objects = [
        mo
        for index, mo in enumerate(memory_objs[:end_chunk])
        if mo is not None
        and index >= skip_objects
        and (index + 1) * chunk_size > skip_first_n_tokens
    ]
    if not objects or cache_context.device.type != "cuda":
        return False
    if not all(isinstance(mo, GDSMemoryObject) for mo in objects):
        return False
    owners = {mo.get_l1_manager() for mo in objects}
    if len(owners) != 1:
        return False
    context = get_l1_gds_context(owners.pop())
    if not context.io_concurrency:
        return False
    is_read = direction == lmcache_native.TransferDirection.H2D
    if is_read:
        # Concurrent scatter must preserve the serial ordering of aliased blocks.
        if not block_ids_host:
            return False
        manager = cache_context.kv_layer_groups_manager
        first_object = max(skip_objects, skip_first_n_tokens // chunk_size)
        for group_id in manager.object_groups[object_group_id].kernel_group_indices:
            window_tokens = min(
                chunk_size, manager.get_subchunk_sw_size_tokens(group_id)
            )
            blocks = cache_context.calculate_num_blocks(window_tokens, group_id)
            destinations = block_ids_host[group_id][
                first_object * blocks : end_chunk * blocks
            ]
            if len(destinations) != len(set(destinations)):
                return False
    budget = context.max_inflight_bytes
    if any(mo.get_size() > budget for mo in objects):
        raise ValueError("GDS object exceeds the configured in-flight byte budget")

    with cache_context.gds_transfer_scope(), context.io_scope():
        count = min(cache_context.max_batch_size, context.io_concurrency)
        streams = [cache_context.get_gds_transfer_stream(slot) for slot in range(count)]
        for slot, stream in enumerate(streams):
            context.register_stream(
                cache_context.device,
                stream,
                staging_buffer=cache_context.get_temp_object_group_buffer(
                    slot, object_group_id
                ),
            )
        ready = cache_context.record_gds_transfer_ready()
        free_slots = list(range(count))
        pending: deque[_GDSSlot] = deque()
        inflight_bytes = 0

        def finish_oldest() -> int:
            """Validate I/O and finish scatter before this slot can be reused."""
            if is_read:
                for entry in pending:
                    stream = streams[entry.slot]
                    if entry.scatter_submitted:
                        continue
                    if entry is not pending[0] and not cast(_GDSStream, stream).query():
                        continue
                    context.synchronize_and_check(cache_context.device, stream)
                    with torch_dev.stream(stream):
                        _launch_batch_kernels(
                            cache_context,
                            block_ids_gpu,
                            object_group_id,
                            entry.object_index,
                            1,
                            entry.skip_tokens,
                            direction,
                            entry.slot,
                        )
                    entry.scatter_submitted = True
                entry = pending[0]
                platform_stream.synchronize_stream(
                    cache_context.device, streams[entry.slot]
                )
            else:
                entry = pending[0]
                context.synchronize_and_check(cache_context.device, streams[entry.slot])
            pending.popleft()
            free_slots.append(entry.slot)
            return entry.nbytes

        try:
            for index in range(skip_objects, end_chunk):
                obj = memory_objs[index]
                skip = max(0, skip_first_n_tokens - index * chunk_size)
                if skip >= chunk_size:
                    continue
                if obj is None:
                    if is_read:
                        raise ValueError("Missing GDS object during retrieve")
                    continue
                assert isinstance(obj, GDSMemoryObject)  # Eligibility checked above.
                nbytes = obj.get_size()
                while not free_slots or inflight_bytes + nbytes > budget:
                    inflight_bytes -= finish_oldest()
                entry = _GDSSlot(free_slots.pop(), index, skip, nbytes)
                # Track ownership before gather or I/O can enqueue partially and fail.
                pending.append(entry)
                inflight_bytes += nbytes
                stream = streams[entry.slot]
                with torch_dev.stream(stream):
                    cast(_GDSStream, stream).wait_event(ready)
                    if not is_read:
                        _launch_batch_kernels(
                            cache_context,
                            block_ids_gpu,
                            object_group_id,
                            index,
                            1,
                            skip,
                            direction,
                            entry.slot,
                        )
                    context.transfer_async(
                        obj,
                        cache_context.get_temp_object_group_buffer(
                            entry.slot, object_group_id
                        ),
                        SlabDirection.READ if is_read else SlabDirection.WRITE,
                    )
            while pending:
                inflight_bytes -= finish_oldest()
        finally:
            # Keep the original error; every partially submitted owner/stream drains.
            for entry in pending:
                try:
                    context.synchronize_and_check(
                        cache_context.device, streams[entry.slot]
                    )
                except Exception:
                    logger.exception("GDS slot cleanup failed after transfer error")
    return True


def run_staged_loop(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmcache_native.TransferDirection",
    *,
    engine: TransferEngine,
    block_ids_host: Sequence[Sequence[int]] = (),
    chunk_range: tuple[int, int] | None = None,
) -> None:
    """Stage and transfer chunks one batch at a time, with any engine.

    Args:
        cache_context: Cache context of the KV cache.
        block_ids_gpu: Block IDs on the GPU, per kernel group.
        memory_objs: One object per chunk; a None entry skips its batch on
            store.
        object_group_id: Object group index.
        batch_size: Chunks per batch.
        skip_first_n_tokens: Leading tokens not written.
        direction: H2D (retrieve) or D2H (store).
        block_ids_host: Host block IDs for checking concurrent scatter destinations.
        engine: The engine of every object in the range.
        chunk_range: ``[start, end)`` chunks to move; None means all.

    Raises:
        ValueError: If a None entry is found on retrieve.
        NotImplementedError: If an object is owned by a peer.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    is_h2d = direction == lmcache_native.TransferDirection.H2D

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

    start_chunk, end_chunk = resolve_chunk_range(
        len(memory_objs), chunk_range, num_objects_to_skip
    )
    if engine is TransferEngine.NVME_DMA and _run_gds_concurrent(
        cache_context,
        block_ids_gpu,
        memory_objs,
        object_group_id,
        skip_first_n_tokens,
        start_chunk,
        end_chunk,
        direction,
        block_ids_host,
    ):
        return

    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs[:end_chunk], batch_size, skip_count=start_chunk
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

        # On retrieve the kernel skips the chunks before the first one it
        # writes, so they are not staged.
        first = skip_tokens_in_chunk // lmcache_chunk_size if is_h2d else 0
        staged_objs = memory_object_batch[first:]
        staged_buffers = [
            cache_context.get_temp_object_group_buffer(chunk_idx, object_group_id)
            for chunk_idx in range(first, batch_len)
        ]

        if is_h2d:
            stage(engine, staged_objs, staged_buffers, direction)

        _launch_batch_kernels(
            cache_context,
            block_ids_gpu,
            object_group_id,
            start_object_idx,
            batch_len,
            skip_tokens_in_chunk,
            direction,
        )

        if not is_h2d:
            stage(engine, staged_objs, staged_buffers, direction)
