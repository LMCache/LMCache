# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass, field
from typing import Any, Sequence
import threading
import time

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.storage_manager import L1WriteCompletion
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def all_null_chunk_masks(
    block_ids: Sequence[Sequence[int]],
    object_groups: Sequence[ObjectGroupInfo],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
    null_block_id: int = 0,
) -> list[list[bool]]:
    """Mark, per object group, the chunks whose engine block ids are all null.

    A chunk is null for an object group when every block ID of every kernel
    group equals the server's null marker. Align-mode Mamba/linear
    layers produce such chunks: only the block holding the last recurrent state
    is real, so every earlier chunk is null. These chunks must not be stored --
    the null block carries no valid KV, and object keys are content hashes, so
    committing them would serve garbage to a later prefix hit.

    Args:
        block_ids: Raw per-kernel-group engine block ids (before any downsample),
            indexed by kernel-group index.
        object_groups: The object groups, indexed by object-group id.
        blocks_per_chunk: Blocks in one chunk per kernel group, indexed by
            kernel-group index.
        num_chunks: Number of chunks in the request.
        null_block_id: Server-wide block ID denoting absent data. Defaults to
            the historical vLLM null block zero.

    Returns:
        ``mask[g][i]`` is True iff chunk ``i`` is all-null for object group ``g``.
    """
    masks: list[list[bool]] = []
    for group in object_groups:
        chunk_null: list[bool] = []
        for i in range(num_chunks):
            is_null = True
            for kg in group.kernel_group_indices:
                bpc = blocks_per_chunk[kg]
                if any(
                    block != null_block_id
                    for block in block_ids[kg][i * bpc : (i + 1) * bpc]
                ):
                    is_null = False
                    break
            chunk_null.append(is_null)
        masks.append(chunk_null)
    return masks


def _stage_sparse_layer(
    cache_context: BaseCacheContext,
    memory_objs: Sequence[MemoryObj],
    selected_block_ids: list[list[int]],
    object_group_id: int,
    layer_id: int,
) -> None:
    """Copy one serving layer from the current unified SGLang MHA layout.

    The unified connector registers flat K tensors followed by flat V tensors.
    Each host object stores those physical tensors in its kernel-group order.
    Copy only the two tensors of this serving layer; the job retains the source
    lease until all queued H2D copies and the consumer have completed.
    """
    tensors = cache_context.kv_tensors
    if len(tensors) % 2 or not 0 <= layer_id < len(tensors) // 2:
        raise ValueError("sparse transfer requires paired K/V layers")
    manager = cache_context.kv_layer_groups_manager
    object_group = manager.object_groups[object_group_id]
    for physical_layer in (layer_id, layer_id + len(tensors) // 2):
        group_id = next(
            (
                g
                for g in object_group.kernel_group_indices
                if physical_layer in manager.kernel_groups[g].layer_indices
            ),
            None,
        )
        if group_id is None:
            raise ValueError("sparse layer is not in the requested object group")
        group = manager.kernel_groups[group_id]
        if cache_context.get_engine_kv_format(group_id) not in (
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS,
        ):
            raise ValueError("sparse transfer requires unified dense SGLang MHA")
        block_size = group.shape_desc.bs
        blocks_per_key = cache_context.calculate_num_blocks(
            cache_context.lmcache_tokens_per_chunk, group_id
        )
        destinations = selected_block_ids[group_id]
        if block_size <= 0 or len(destinations) != len(memory_objs) * blocks_per_key:
            raise ValueError("sparse transfer has an invalid block mapping")
        target = tensors[physical_layer].view(
            tensors[physical_layer].shape[0], block_size, -1
        )
        group_position = object_group.kernel_group_indices.index(group_id)
        local_layer = group.layer_indices.index(physical_layer)
        for object_index, memory_obj in enumerate(memory_objs):
            source = memory_obj.get_tensor(group_position)
            if source is None or source.ndim not in (4, 5) or source.shape[0] != 1:
                raise ValueError(
                    "sparse transfer requires single-plane layer/token MHA objects"
                )
            if (
                local_layer >= source.shape[1]
                or source.shape[2] != blocks_per_key * block_size
            ):
                raise ValueError("sparse source layer/token geometry is invalid")
            source_layer = source[0, local_layer].flatten(start_dim=1)
            if (
                source_layer.shape[1] != target.shape[2]
                or source_layer.dtype != target.dtype
            ):
                raise ValueError("sparse source dtype/hidden geometry does not match")
            start = object_index * blocks_per_key
            for offset, block in enumerate(
                destinations[start : start + blocks_per_key]
            ):
                if not 0 <= block < target.shape[0]:
                    raise ValueError("sparse destination block is out of bounds")
                target[block].copy_(
                    source_layer[offset * block_size : (offset + 1) * block_size],
                    non_blocking=True,
                )


@dataclass
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None


@dataclass
class _SparsePrefetchJob:
    """Server-side state for one logical sparse prefetch lease."""

    handle: PrefetchHandle
    keys: tuple[ObjectKey, ...]
    instance_id: int
    request_id: str
    generation: int
    layer_id: int
    condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )
    found_indices: tuple[int, ...] | None = None
    status_inflight: bool = False
    retrieving: bool = False
    completion_submitted: bool = False
    completed: bool = False
    copy_submitted: bool = False
    copy_synchronized: bool = False
    copy_sync_inflight: bool = False
    copy_stream: Any = field(default=None, repr=False)
    last_error: BaseException | None = field(default=None, repr=False)
    cancel_requested: bool = False


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()
        self._sparse_jobs: dict[tuple[int, str, int, int], _SparsePrefetchJob] = {}
        self._sparse_jobs_lock = threading.Lock()
        self._sparse_orphan_handles: dict[tuple[int, int], PrefetchHandle] = {}

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_write_by_owner",
            self._ctx.storage_manager.finish_write_by_owner,
            payload_type=L1WriteCompletion,
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "complete_sparse_prefetch",
            self._complete_sparse_prefetch,
            payload_type=tuple[int, str, int, int],
        )
        self._device_host_func_dispatcher.start()

    def _release_imported_event(self, payload: tuple[int, int]) -> None:
        """Drop an imported worker event; the stream wait queued on it has run.

        Args:
            payload: ``(instance_id, import_token)`` of the imported event.
        """
        instance_id, import_token = payload
        entry = self.get_and_touch_context_entry(instance_id)
        if entry is not None:
            entry.cache_context.release_imported_event(import_token)

    def register_host_func(self, kind: str, handler: Any, payload_type: Any) -> None:
        """Register *handler* for *kind* on the per-process device host-func
        dispatcher (stream-ordered callbacks without a driver-thread GIL
        acquire); pair with ``submit_callback_to_stream``."""
        self._device_host_func_dispatcher.register(kind, handler, payload_type)

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def _release_failed_retrieve_locks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> None:
        """Release only one failed instance's unconsumed lookup locks.

        The lookup session is the ownership record.  If it is absent or does
        not match the RETRIEVE range, no release is attempted: L1 locks are
        anonymous refcounts, so guessing could consume a concurrent reader's
        lock.  ``claim_failed_retrieve_release`` also makes duplicate failure
        responses idempotent.
        """
        session = self._ctx.session_manager.get(key.request_id)
        if session is None:
            logger.warning(
                "Cannot release RETRIEVE locks for unregistered instance %d: "
                "request %s has no lookup session",
                instance_id,
                key.request_id,
            )
            return

        lock_state = session.prepare_failed_retrieve_release(key)
        if lock_state is None:
            return
        hit_chunks, locked_gids, group_windows, lookup_generation = lock_state
        obj_keys = resolve_prefetched_obj_keys(
            self._ctx,
            key,
            hit_chunks,
            locked_gids,
            group_windows=group_windows,
        )
        if not session.claim_failed_retrieve_release(
            instance_id, key, lookup_generation
        ):
            return
        if obj_keys:
            # One failed RETRIEVE owns one read lock per key.  In
            # particular, do not release the scheduler's whole MLA
            # reservation here: the remaining TP workers and concurrent
            # requests still own their independent read locks.
            self._ctx.storage_manager.finish_read_prefetched(obj_keys, read_locks=1)

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        stale_candidates: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                stale_candidates.append((iid, self._cache_contexts[iid]))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        candidate = current = None
        for iid, candidate in stale_candidates:
            if not self._cleanup_sparse_instance(iid):
                logger.error(
                    "Keeping stale GPU instance %d registered because sparse "
                    "prefetch cleanup did not complete",
                    iid,
                )
                continue
            with self._lock:
                current = self._cache_contexts.get(iid)
                if current is not candidate:
                    continue
                timeout = (
                    reap_timeout_s
                    if current.has_liveness_signal
                    else registration_grace_s
                )
                if time.monotonic() - current.last_seen <= timeout:
                    continue
                e = self._cache_contexts.pop(iid)
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        stale_candidates.clear()
        candidate = current = None
        if entries:
            del e  # a bound name would pin the final entry (see _release_entries)
            self._release_entries(entries)
        return reaped_ids

    def _release_entries(self, entries: list[ContextEntry]) -> None:
        """Release a batch of entries and reclaim their device memory.

        Args:
            entries: The only remaining references to the released entries.
                The list is cleared before memory is reclaimed.
        """
        if not entries:
            return
        for entry in entries:
            entry.cache_context.close()
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
        del entry
        entries.clear()
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
            "active_sparse_prefetches": self._active_sparse_count(),
        }

    def _active_sparse_count(self) -> int:
        with self._sparse_jobs_lock:
            return len(self._sparse_jobs) + len(
                getattr(self, "_sparse_orphan_handles", {})
            )

    def _sparse_job_key(
        self, instance_id: int, request_id: str, generation: int, layer_id: int
    ) -> tuple[int, str, int, int]:
        return instance_id, request_id, generation, layer_id

    def _get_sparse_job(
        self, instance_id: int, request_id: str, generation: int, layer_id: int
    ) -> _SparsePrefetchJob | None:
        with self._sparse_jobs_lock:
            return self._sparse_jobs.get(
                self._sparse_job_key(instance_id, request_id, generation, layer_id)
            )

    def _remove_sparse_job(self, job: _SparsePrefetchJob) -> None:
        key = self._sparse_job_key(
            job.instance_id, job.request_id, job.generation, job.layer_id
        )
        with self._sparse_jobs_lock:
            if self._sparse_jobs.get(key) is job:
                self._sparse_jobs.pop(key, None)

    def _remember_sparse_orphan_handle(
        self, instance_id: int, handle: PrefetchHandle
    ) -> None:
        """Retain a losing storage handle when its first cleanup fails."""
        with self._sparse_jobs_lock:
            orphan_handles = getattr(self, "_sparse_orphan_handles", None)
            if orphan_handles is None:
                orphan_handles = {}
                self._sparse_orphan_handles = orphan_handles
            orphan_handles[(instance_id, id(handle))] = handle

    def _cancel_or_remember_sparse_handle(
        self, instance_id: int, handle: PrefetchHandle
    ) -> bool:
        """Cancel a handle, retaining it when success is not confirmed."""
        try:
            result = self._ctx.storage_manager.cancel_prefetch_task(handle)
        except Exception:
            self._remember_sparse_orphan_handle(instance_id, handle)
            logger.exception("Failed to cancel a losing sparse prefetch handle")
            return False
        if result is False:
            self._remember_sparse_orphan_handle(instance_id, handle)
            logger.error("Sparse prefetch handle cancellation was not confirmed")
            return False
        return True

    def _cleanup_sparse_orphan_handles(self, instance_id: int | None = None) -> bool:
        """Retry cleanup for handles whose first cancellation was uncertain."""
        with self._sparse_jobs_lock:
            orphan_handles = {
                key: handle
                for key, handle in getattr(self, "_sparse_orphan_handles", {}).items()
                if instance_id is None or key[0] == instance_id
            }
        all_clean = True
        for handle_key, handle in orphan_handles.items():
            if not self._cancel_or_remember_sparse_handle(handle_key[0], handle):
                all_clean = False
                continue
            with self._sparse_jobs_lock:
                current = getattr(self, "_sparse_orphan_handles", {}).get(handle_key)
                if current is handle:
                    self._sparse_orphan_handles.pop(handle_key, None)
        return all_clean

    def _finish_sparse_job(self, job: _SparsePrefetchJob) -> None:
        with job.condition:
            job.completed = True
            job.retrieving = False
            job.condition.notify_all()
        self._remove_sparse_job(job)

    def _complete_sparse_prefetch(self, payload: tuple[int, str, int, int]) -> None:
        """Release a sparse lease after the GPU transfer stream completes."""
        instance_id, request_id, generation, layer_id = payload
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if job is None:
            return
        with job.condition:
            found_indices = job.found_indices
            retrieving = job.retrieving
        if not retrieving or found_indices is None:
            return
        found_keys = [job.keys[index] for index in found_indices]
        with job.condition:
            # This callback runs on the copy stream, so all H2D work is
            # complete before releasing the host read locks.
            job.copy_synchronized = True
            job.copy_sync_inflight = False
            job.condition.notify_all()
        try:
            # The PrefetchHandle owns the storage-manager read lease. Release
            # it only from this completion callback, after the copy stream has
            # synchronized; releasing the same keys separately here would
            # double-decrement the handle's read locks.
            self._ctx.storage_manager.release_prefetch_task(job.handle, keys=found_keys)
        except Exception:
            logger.exception(
                "Failed to release sparse prefetch lease after transfer: "
                "request_id=%s generation=%d",
                request_id,
                generation,
            )
            with job.condition:
                # The device callback already proves the copy is finished.
                # Leave the job reachable, but allow an explicit cancel or
                # release to retry the logical lease without waiting for a
                # completion callback that will never run again.
                job.retrieving = False
                job.condition.notify_all()
            return
        self._finish_sparse_job(job)

    def _cleanup_sparse_job(self, job: _SparsePrefetchJob) -> bool:
        """Cancel/release one job, waiting for an in-flight GPU copy."""
        deadline = time.monotonic() + 60.0
        with job.condition:
            if job.completed:
                return True
            job.cancel_requested = True
            while job.retrieving and not job.completed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Timed out waiting for sparse prefetch transfer cleanup: "
                        "request_id=%s generation=%d",
                        job.request_id,
                        job.generation,
                    )
                    return False
                job.condition.wait(timeout=min(remaining, 0.5))
            if job.completed:
                return True

        if not self._synchronize_sparse_copy(job):
            logger.error(
                "Cannot prove sparse retrieve copy completion; retaining "
                "request resources: request_id=%s generation=%d",
                job.request_id,
                job.generation,
            )
            return False

        with job.condition:
            found_indices = job.found_indices
        found_keys = (
            None
            if found_indices is None
            else [job.keys[index] for index in found_indices]
        )
        try:
            # A sparse retrieve may have acquired an L2 read lease after the
            # initial prefetch response.  Pass the resolved keys through so
            # cleanup releases both the original L1 lease and those L2 locks.
            self._ctx.storage_manager.release_prefetch_task(job.handle, keys=found_keys)
        except Exception:
            logger.exception(
                "Failed to release sparse prefetch job: request_id=%s generation=%d",
                job.request_id,
                job.generation,
            )
            return False
        self._finish_sparse_job(job)
        return True

    def _synchronize_sparse_copy(self, job: _SparsePrefetchJob) -> bool:
        """Synchronize a partially submitted copy before releasing its lease."""
        with job.condition:
            if not job.copy_submitted or job.copy_synchronized:
                return True
            while job.copy_sync_inflight:
                job.condition.wait()
                if job.copy_synchronized:
                    return True
            job.copy_sync_inflight = True
            copy_stream = job.copy_stream

        try:
            synchronize = getattr(copy_stream, "synchronize", None)
            if synchronize is None:
                raise RuntimeError("sparse retrieve copy stream has no synchronize()")
            synchronize()
        except Exception as exc:
            with job.condition:
                job.copy_sync_inflight = False
                job.last_error = exc
                job.condition.notify_all()
            logger.exception(
                "Could not confirm sparse retrieve copy completion; "
                "resources remain leased: request_id=%s generation=%d",
                job.request_id,
                job.generation,
            )
            return False

        with job.condition:
            job.copy_sync_inflight = False
            job.copy_synchronized = True
            job.condition.notify_all()
        return True

    def _cleanup_sparse_instance(self, instance_id: int) -> bool:
        with self._sparse_jobs_lock:
            jobs = [
                job
                for job in self._sparse_jobs.values()
                if job.instance_id == instance_id
            ]
        all_clean = True
        for job in jobs:
            all_clean = self._cleanup_sparse_job(job) and all_clean
        all_clean = self._cleanup_sparse_orphan_handles(instance_id) and all_clean
        return all_clean

    def _cleanup_all_sparse_jobs(self) -> bool:
        with self._sparse_jobs_lock:
            jobs = list(self._sparse_jobs.values())
        all_clean = True
        for job in jobs:
            all_clean = self._cleanup_sparse_job(job) and all_clean
        all_clean = self._cleanup_sparse_orphan_handles() and all_clean
        return all_clean

    def _resolve_sparse_status(
        self, job: _SparsePrefetchJob, timeout: float | None
    ) -> list[int] | None:
        """Return a stable found-index list without consuming it twice."""
        with job.condition:
            if job.found_indices is not None:
                return list(job.found_indices)
            if job.status_inflight:
                deadline = None if timeout is None else time.monotonic() + timeout
                while job.status_inflight and job.found_indices is None:
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        return None
                    job.condition.wait(timeout=remaining)
                return None if job.found_indices is None else list(job.found_indices)
            job.status_inflight = True

        found = None
        try:
            if not self._ctx.storage_manager.wait_prefetch_lease(job.handle, timeout):
                return None
            result = self._ctx.storage_manager.query_prefetch_lease(job.handle)
            if result is not None:
                found = tuple(index for index in result.hit_cells[0].get_indices_list())
        finally:
            with job.condition:
                if found is not None:
                    job.found_indices = found
                job.status_inflight = False
                job.condition.notify_all()
        return None if found is None else list(found)

    @request_handler(HandlerType.BLOCKING)
    def sparse_prefetch(
        self,
        instance_id: int,
        request_id: str,
        generation: int,
        layer_id: int,
        keys: list[ObjectKey],
    ) -> bool:
        """Submit a logical SPARSE prefetch and retain its read lease."""
        if generation < 0 or layer_id < 0 or not request_id:
            return False
        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None or not keys or len(set(keys)) != len(keys):
            return False
        if any(key.model_name != entry.model_name for key in keys):
            logger.warning(
                "Rejecting sparse prefetch with a mismatched model name: request_id=%s",
                request_id,
            )
            return False
        if any(key.object_group_id < 0 for key in keys):
            return False
        if len({(key.object_group_id, key.kv_rank) for key in keys}) != 1:
            return False
        job_key = self._sparse_job_key(instance_id, request_id, generation, layer_id)
        with self._sparse_jobs_lock:
            existing = self._sparse_jobs.get(job_key)
            if existing is not None:
                return existing.keys == tuple(keys)
            if any(
                job.instance_id == instance_id
                and job.request_id == request_id
                and job.generation != generation
                for job in self._sparse_jobs.values()
            ):
                # The caller must explicitly cancel the old generation before
                # reusing a request id. This makes stale slot reuse visible
                # instead of silently sharing a lease between generations.
                return False

        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(
            entry.model_name, entry.world_size
        )
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            entry.model_name, entry.world_size
        )
        if not group_layout_descs or attn_desc is None:
            return False
        if any(key.object_group_id >= attn_desc.num_object_groups for key in keys):
            return False

        handle = self._ctx.storage_manager.submit_prefetch_lease(
            PrefetchTaskSpec(
                key_groups=[
                    GroupedObjectKeys(
                        keys=list(keys),
                        object_group_id=keys[0].object_group_id,
                        layout_desc=group_layout_descs[keys[0].object_group_id],
                    )
                ],
                fetching_policy="full",
            ),
            external_request_id=f"{request_id}:{generation}:{layer_id}",
        )
        job = _SparsePrefetchJob(
            handle=handle,
            keys=tuple(keys),
            instance_id=instance_id,
            request_id=request_id,
            generation=generation,
            layer_id=layer_id,
        )
        duplicate = False
        conflict = False
        generation_conflict = False
        with self._sparse_jobs_lock:
            existing = self._sparse_jobs.get(job_key)
            if existing is not None:
                duplicate = existing.keys == job.keys
                conflict = not duplicate
            elif any(
                other.instance_id == instance_id
                and other.request_id == request_id
                and other.generation != generation
                for other in self._sparse_jobs.values()
            ):
                generation_conflict = True
            else:
                self._sparse_jobs[job_key] = job

        if generation_conflict:
            # The caller must explicitly cancel the old generation before
            # reusing a request id. This makes stale slot reuse visible
            # instead of silently sharing a lease between generations.
            self._cancel_or_remember_sparse_handle(instance_id, handle)
            return False

        if duplicate or conflict:
            # The first submitter owns the logical job. This caller still
            # owns a real storage-manager handle, so release that losing
            # handle before reporting a duplicate or conflicting submit.
            self._cancel_or_remember_sparse_handle(instance_id, handle)
            return duplicate
        return True

    @request_handler(HandlerType.BLOCKING)
    def sparse_query_prefetch(
        self, instance_id: int, request_id: str, generation: int, layer_id: int
    ) -> list[int] | None:
        """Query a sparse prefetch without releasing its lease."""
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if job is None:
            return None
        return self._resolve_sparse_status(job, timeout=0.0)

    @request_handler(HandlerType.BLOCKING)
    def sparse_wait_prefetch(
        self,
        instance_id: int,
        request_id: str,
        generation: int,
        layer_id: int,
        timeout: float,
    ) -> list[int] | None:
        """Wait for a sparse prefetch without releasing its lease."""
        if timeout < 0:
            return None
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if job is None:
            return None
        return self._resolve_sparse_status(job, timeout=timeout)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def sparse_retrieve(
        self,
        instance_id: int,
        request_id: str,
        generation: int,
        layer_id: int,
        keys: list[ObjectKey],
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, tuple[bool, list[int]]]:
        """Copy retained sparse objects into the supplied GPU page mapping."""
        entry = self.get_and_touch_context_entry(instance_id)
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if entry is None or job is None or tuple(keys) != job.keys:
            if job is not None:
                self._cleanup_sparse_job(job)
            return b"", (False, [])

        found_indices = self._resolve_sparse_status(job, timeout=None)
        if found_indices is None or not found_indices:
            self._cleanup_sparse_job(job)
            return b"", (False, [])

        cache_context = entry.cache_context
        event_backend = entry.event_backend
        if event_backend is None:
            self._cleanup_sparse_job(job)
            return b"", (False, found_indices)
        object_groups = cache_context.kv_layer_groups_manager.object_groups
        group_ids = {key.object_group_id for key in keys}
        if len(group_ids) != 1:
            logger.warning(
                "Sparse retrieve currently accepts one object group per request"
            )
            self._cleanup_sparse_job(job)
            return b"", (False, found_indices)
        object_group_id = next(iter(group_ids))
        object_group = object_groups[object_group_id]
        num_kernel_groups = cache_context.kv_layer_groups_manager.num_kernel_groups
        if len(gpu_block_ids) != num_kernel_groups:
            self._cleanup_sparse_job(job)
            return b"", (False, found_indices)

        blocks_per_chunk = {
            kernel_group_id: cache_context.calculate_num_blocks(
                self._ctx.chunk_size, kernel_group_id
            )
            for kernel_group_id in object_group.kernel_group_indices
        }
        for kernel_group_id, blocks_per_key in blocks_per_chunk.items():
            if len(gpu_block_ids[kernel_group_id]) != len(keys) * blocks_per_key:
                logger.warning(
                    "Sparse retrieve block mapping has wrong length: "
                    "request_id=%s group=%d",
                    request_id,
                    object_group_id,
                )
                self._cleanup_sparse_job(job)
                return b"", (False, found_indices)

        with job.condition:
            if job.retrieving or job.completed or job.cancel_requested:
                return b"", (False, found_indices)
            job.retrieving = True
            job.copy_stream = cache_context.stream

        found_keys = [keys[index] for index in found_indices]
        selected_block_ids: list[list[int]] = [[] for _ in range(num_kernel_groups)]
        for kernel_group_id, blocks_per_key in blocks_per_chunk.items():
            source_ids = gpu_block_ids[kernel_group_id]
            selected_block_ids[kernel_group_id] = [
                block_id
                for index in found_indices
                for block_id in source_ids[
                    index * blocks_per_key : (index + 1) * blocks_per_key
                ]
            ]

        try:
            with (
                torch_dev.device(cache_context.device),
                torch_dev.stream(cache_context.stream),
            ):
                event = event_backend.create_event(cache_context.device)
                producer_event = event_backend.import_event(
                    event_ipc_handle, cache_context.device
                )
                event_backend.wait_event(producer_event, cache_context.stream)
                stage_error: BaseException | None = None
                with self._ctx.storage_manager.read_prefetched_results(
                    found_keys, release_on_error=False
                ) as memory_objs:
                    if memory_objs is None or len(memory_objs) != len(found_keys):
                        raise RuntimeError(
                            "Sparse retrieve found keys changed before read"
                        )
                    with job.condition:
                        # Mark conservatively before entering the layer
                        # transfer. A native call can enqueue an H2D copy and
                        # then raise while processing a later object.
                        job.copy_submitted = True
                    try:
                        _stage_sparse_layer(
                            cache_context,
                            memory_objs,
                            selected_block_ids,
                            object_group_id=object_group_id,
                            layer_id=layer_id,
                        )
                    except BaseException as exc:
                        # Keep the context manager on its normal-exit path so
                        # deferred read locks are not released before the
                        # partially submitted copy is synchronized below.
                        stage_error = exc
                if stage_error is not None:
                    raise stage_error
                submit_callback_to_stream(
                    cache_context.cupy_stream,
                    "complete_sparse_prefetch",
                    (instance_id, request_id, generation, layer_id),
                )
                with job.condition:
                    job.completion_submitted = True
                    job.condition.notify_all()
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), (
                    True,
                    found_indices,
                )
        except Exception:
            logger.exception(
                "Sparse retrieve failed: request_id=%s generation=%d",
                request_id,
                generation,
            )
            with job.condition:
                if not job.completion_submitted:
                    job.retrieving = False
                job.condition.notify_all()
            if not job.completion_submitted and self._synchronize_sparse_copy(job):
                self._cleanup_sparse_job(job)
            return b"", (False, found_indices)

    @request_handler(HandlerType.BLOCKING)
    def sparse_cancel_prefetch(
        self, instance_id: int, request_id: str, generation: int, layer_id: int
    ) -> bool:
        """Cancel one generation and release its logical cache lease."""
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if job is None:
            # Cancellation is an idempotent cleanup operation.  The stream
            # completion callback or an earlier cancel may already have
            # removed this exact-generation job.
            return True
        return self._cleanup_sparse_job(job)

    @request_handler(HandlerType.BLOCKING)
    def sparse_release_prefetch(
        self, instance_id: int, request_id: str, generation: int, layer_id: int
    ) -> bool:
        """Release one generation after its destination pages are consumed."""
        job = self._get_sparse_job(instance_id, request_id, generation, layer_id)
        if job is None:
            # The stream callback may already have performed the idempotent
            # release; an absent exact-generation job is terminal success.
            return True
        return self._cleanup_sparse_job(job)

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        if not self._cleanup_all_sparse_jobs():
            raise RuntimeError(
                "Cannot close LMCache transfer module while sparse prefetch "
                "resources are still in flight"
            )
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        with self._lock:
            entries = list(self._cache_contexts.values())
            self._cache_contexts.clear()
        self._release_entries(entries)

    @request_handler()
    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        kv_groups_manager = cache_context.kv_layer_groups_manager
        num_object_groups = kv_groups_manager.num_object_groups
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        # One layout per object group, also in the single-group case: no
        # None special-casing downstream (group 0 maps to the merged layout).
        group_layout_descs = {
            gid: get_layout_desc(
                cache_context, self._ctx.chunk_size, object_group_id=gid
            )
            for gid in range(num_object_groups)
        }
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs=group_layout_descs,
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        if not self._cleanup_sparse_instance(instance_id):
            raise RuntimeError(
                "Cannot unregister GPU context while sparse prefetch resources "
                "are still in flight"
            )
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU; see store_with_chunk_mask."""
        handle, ok, _ = self.store_with_chunk_mask(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )
        return handle, ok

    def store_with_chunk_mask(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool, list[bool]]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes). The event handle is empty when no device work was submitted.
            The third element marks per chunk whether every object group
            committed it.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), a copy fails, or completion ownership is invalid, the whole
            store is skipped and nothing is committed; a subsequent retrieve misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
            Failed copies or completion preparation retain staging reservations
            under the existing write-TTL rules; queued GPU writes may still
            reference those buffers.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # The worker can reconnect to a replacement server before its next
            # registration probe. No device work was submitted in that window,
            # so return an empty completion-event handle and a terminal False
            # response instead of leaving the MQ future unanswered. Echoing the
            # producer handle would make the originating process import its own
            # IPC event, which is invalid on HIP.
            logger.warning(
                "Rejecting STORE for unregistered GPU instance ID %d",
                instance_id,
            )
            return b"", False, []
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return (
                    event_backend.export_event(event, cache_context.device),
                    False,
                    [],
                )

            # Chunks whose block ids are all the null block (e.g. align-mode
            # Mamba chunks holding no real state) carry no valid KV and must not
            # be committed. Computed on the raw block ids before downsampling
            # mutates them.
            null_block_id = self._ctx.null_block_id
            skipped_chunks = all_null_chunk_masks(
                gpu_block_ids,
                cache_context.kv_layer_groups_manager.object_groups,
                blocks_per_chunk,
                num_chunks,
                null_block_id,
            )

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)
            import_token = cache_context.hold_imported_event(producer_event)
            submit_callback_to_stream(
                cache_context.cupy_stream,
                "release_imported_event",
                (instance_id, import_token),
            )

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            # Worker 0 only: bindings depend on token content alone, so one
            # report covers every rank's keys. Published before finish_write
            # is enqueued so the token bindings precede the write-finished
            # events on the bus.
            if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                EventType.MP_TOKENS
            ):
                self._publish_token_bindings(key, obj_keys_per_obj_group[0])

            transfer_key = next_transfer_key(key.request_id)
            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                        "transfer_key": transfer_key,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    skip_mask = skipped_chunks[obj_group_id]
                    keys_to_reserve = [
                        k for i, k in enumerate(obj_keys) if not skip_mask[i]
                    ]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        keys_to_reserve, layout_desc
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (all-null chunks skipped above, or
                    # skipped by the storage manager) become None entries; the
                    # helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmcache_native.TransferDirection.D2H,
                        transfer_key=transfer_key,
                        block_ids_host=gpu_block_ids,
                    )

                completion = (
                    self._ctx.storage_manager.prepare_write_completion(all_dict)
                    if all_dict
                    else []
                )
                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write_by_owner",
                        completion,
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )

        # A chunk is stored only when every object group committed its key.
        stored_mask = [
            store_succeeded
            and all(keys[i] in all_dict for keys in obj_keys_per_obj_group)
            for i in range(num_chunks)
        ]
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
            stored_mask,
        )

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.
            The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            # See store(): there is no completion event because no device work
            # was submitted. The False result lets the caller recover or
            # recompute without importing its own producer event.
            logger.warning(
                "Rejecting RETRIEVE for unregistered GPU instance ID %d",
                instance_id,
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                # A cleanup failure must never suppress the terminal response:
                # the client otherwise waits forever because blocking-handler
                # exceptions are only logged by the MQ server.
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered "
                    "GPU instance ID %d",
                    instance_id,
                )
            return b"", False
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        transfer_key = next_transfer_key(key.request_id)
        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                    "transfer_key": transfer_key,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)
            import_token = cache_context.hold_imported_event(producer_event)
            submit_callback_to_stream(
                cache_context.cupy_stream,
                "release_imported_event",
                (instance_id, import_token),
            )

            # Per object group, the prefetch only locked the in-window suffix
            # (the last ``num_chunks_in_sw`` chunks; the whole prefix for full
            # attention, where the value is < 0). Read and transfer only those.
            # Aux (connector-private) groups are never served by the
            # std retrieve: the lookup does not lock their keys and their
            # block-id entry is a placeholder -- reading them would be an
            # unlocked read of a plane nobody consumes here.
            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            skipped_groups = {
                g for g, kind in enumerate(attn_desc.group_kinds) if kind == "aux"
            }
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(
                num_chunks - skip
                for g, skip in enumerate(group_skips)
                if g not in skipped_groups
            )

            prefetched_keys: list[ObjectKey] = []
            read_locked_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    if obj_group_id in skipped_groups:
                        continue
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    with self._ctx.storage_manager.read_prefetched_results(
                        in_window_keys
                    ) as window_objs:
                        if not window_objs or len(window_objs) != len(in_window_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in window_objs)

                        # None-pad the skipped prefix to full length so the
                        # transfer's ``num_objects_to_skip`` and block-id slicing
                        # line up unchanged; the None entries are never read.
                        memory_objs: list[MemoryObj | None] = [None] * skip + list(
                            window_objs
                        )

                        # Keep these locks owned by this retrieve until the
                        # stream copy is known to have completed.  If the
                        # native transfer submits one object and fails on the
                        # next, the deferred read context must not release
                        # these keys before the failure cleanup synchronizes
                        # the copy stream.
                        read_locked_keys.extend(in_window_keys)
                        transfer_kv_per_object_group(
                            cache_context,
                            block_ids_per_group_gpu,
                            memory_objs,
                            object_group_id=obj_group_id,
                            batch_size=cache_context.max_batch_size,
                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmcache_native.TransferDirection.H2D,
                            transfer_key=transfer_key,
                            block_ids_host=gpu_block_ids,
                        )
                    prefetched_keys.extend(in_window_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if retrieve_succeeded and prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                elif read_locked_keys:
                    try:
                        cache_context.stream.synchronize()
                    except Exception:
                        logger.exception(
                            "Cannot confirm failed retrieve copy completion; "
                            "retaining read locks"
                        )
                    else:
                        self._ctx.storage_manager.finish_read_prefetched(
                            read_locked_keys
                        )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == expected_retained
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Publish one ``MP_TOKENS`` event for ``key``'s chunks.

        Pairs each complete chunk in ``[key.start, key.end)`` with its
        ObjectKey chunk hash and token position. Must be called at store
        submission, before the write-finished events reach the bus, so the
        cache-event subscriber can stamp them onto the STORE entries. A
        store that later fails leaves only unused cache entries.

        Args:
            key: The IPC key of the store being submitted.
            obj_keys: One ObjectKey per complete chunk, in chunk order.
        """
        # Complete chunks in [key.start, key.end) paired with the absolute
        # position of each chunk's first token. Prefix-chained chunk hashes
        # imply a position without revealing it, so it is reported here. A
        # trailing partial chunk has no stored KV to bind to.
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [
            token_ids[offset : offset + chunk_size] for offset in token_offsets
        ]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys "
                "vs %d complete chunks in [%d, %d)",
                key.request_id,
                len(obj_keys),
                len(token_chunks),
                key.start,
                key.end,
            )
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [obj_key.chunk_hash for obj_key in obj_keys],
                    "token_chunks": token_chunks,
                    "token_offsets": token_offsets,
                },
            )
        )
