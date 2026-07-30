# SPDX-License-Identifier: Apache-2.0
"""Query-tensor (Q ring buffer) cache operations for the MPCacheServer.

The paged Q ring buffer (QRingBuffer) is a temporary buffer for query tensors,
made to be paged in GPU such that it's compatible with the existing LMCache
paged KV store/retrieve machinery. This implementation is copied and modified
from the LMCache-driven KV transfer module.
"""

# Standard
import threading
import time

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
    check_interprocess_event_support,
)
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.transfer_plan import (
    TransferPlanDirection,
    build_object_group_layout_desc,
    build_transfer_plan,
    export_kv_transfer_metadata,
    map_kernel_group_block_ids_to_engine_groups,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class QStoreModule(InstanceLivenessTarget):
    """Handles paged Q ring registration and store operations.

    Owns Q context registrations and provides handlers for register,
    unregister, and store of the paged Q ring.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._q_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _q_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

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
            entry = self._q_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._q_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._q_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._q_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap Q ring registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._q_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._q_contexts.pop(iid)))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped Q ring for instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            reaped.clear()
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
            # Non-CUDA device modules (xpu / musa) do not expose ipc_collect.
            ipc_collect()

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_Q_CACHE,
                self.register_q_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_Q_CACHE,
                self.unregister_q_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.STORE_Q,
                self.store_q,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return Q transfer module status information.

        Returns:
            A dict containing registered Q instance IDs and
            per-instance Q ring layout metadata.
        """
        registered_q_ids: list[int] = []
        q_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_q_ids.append(instance_id)
            ctx = entry.cache_context
            q_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "q_ring_layout": ctx.report_status(),
            }

        return {
            "registered_q_ids": registered_q_ids,
            "q_context_meta": q_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        with self._lock:
            entries = list(self._q_contexts.values())
            self._q_contexts.clear()
        self._release_entries(entries)

    def register_q_cache(
        self,
        instance_id: int,
        q_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the paged Q ring tensors for a given worker instance ID.

        Args:
            instance_id: The worker instance ID (shared with its KV cache).
            q_caches: The Q ring tensor wrappers from the serving engine.
            model_name: The query-specific model name (e.g. "<model>##query").
            world_size: The world size associated with the Q ring.
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
        # time so a stale entry is not reaped right after recovery.
        # REGISTER_Q_CACHE is SYNC-serialized on the MQ main loop, so it is
        # the sole inserter.
        with self._lock:
            existing = self._q_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Q ring for instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            q_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        transfer_metadata = export_kv_transfer_metadata(
            cache_context.kv_layer_groups_manager,
            self._ctx.chunk_size,
        )
        layout_desc = build_object_group_layout_desc(
            transfer_metadata,
            self._ctx.chunk_size,
            object_group_id=0,
        )
        attn_desc = transfer_metadata.build_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name, world_size, layout_desc, attn_desc
        )

        with self._lock:
            self._q_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                transfer_metadata=transfer_metadata,
                last_seen=now,
                has_liveness_signal=False,
            )

        logger.info(
            "Registered Q ring for instance %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def unregister_q_cache(self, instance_id: int) -> None:
        """Unregister the paged Q ring tensors for a given worker instance ID.

        Args:
            instance_id: The worker instance ID (shared with its KV cache).
        """
        with self._lock:
            popped = [
                e for e in (self._q_contexts.pop(instance_id, None),) if e is not None
            ]
        if not popped:
            logger.warning("No registered Q ring found for instance ID %d", instance_id)
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered Q ring for instance ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def store_q(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the paged Q ring blocks to CPU.

        Args:
            key: The IPC key for the Q ring blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The worker instance ID (such as PID).
            gpu_block_ids: Q ring block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes).

        Raises:
            ValueError: If no Q ring is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), or a copy fails, the whole store is skipped and nothing is
            committed (logged at WARNING); a subsequent retrieve simply misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No Q ring registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        transfer_metadata = entry.transfer_metadata

        num_object_groups = len(transfer_metadata.object_groups)
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            try:
                transfer_plan = build_transfer_plan(
                    transfer_metadata,
                    map_kernel_group_block_ids_to_engine_groups(
                        transfer_metadata, gpu_block_ids
                    ),
                    num_chunks,
                    TransferPlanDirection.STORE,
                )
            except ValueError as exc:
                logger.warning(
                    "Invalid STORE_Q block IDs for request_id=%s; "
                    "skipping the store: %s",
                    key.request_id,
                    exc,
                )
                event.record()
                return event.ipc_handle(), False

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC event "
                    "handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                cache_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=cache_context.stream)

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

            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for object_group_plan, obj_keys in zip(
                    transfer_plan.object_groups,
                    obj_keys_per_obj_group,
                    strict=True,
                ):
                    obj_group_id = object_group_plan.object_group_id
                    layout_desc = build_object_group_layout_desc(
                        transfer_metadata,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        obj_keys, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (skipped by the storage manager)
                    # become None entries; the helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    planned_memory_objs = [
                        memory_objs[chunk_idx]
                        for chunk_idx in object_group_plan.chunk_indices
                    ]
                    if planned_memory_objs:
                        # batch_size must stay 1 for store.
                        transfer_kv_per_object_group(
                            cache_context,
                            transfer_metadata,
                            object_group_plan,
                            cache_context.stage_block_ids(
                                [
                                    list(kernel_group_plan.block_ids)
                                    for kernel_group_plan in (
                                        object_group_plan.kernel_groups
                                    )
                                ]
                            ),
                            planned_memory_objs,
                            batch_size=1,
                            direction=lmc_ops.TransferDirection.D2H,
                        )

                store_succeeded = True
            except Exception:
                logger.exception("Cannot store Q keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        list(all_dict.keys()),
                    )
                else:
                    total_bytes = 0
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
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d Q tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True
