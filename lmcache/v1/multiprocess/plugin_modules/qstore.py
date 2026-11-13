# SPDX-License-Identifier: Apache-2.0
"""Query-tensor (Q ring buffer) cache operations for the MPCacheServer.

The paged Q ring buffer (QRingBuffer) is a temporary buffer for query tensors,
made to be paged in GPU such that it's compatible with the existing LMCache
paged KV store/retrieve machinery. This implementation is copied and modified
from the LMCache-driven KV transfer module.

This module is an opt-in plugin: it is not auto-discovered and is built only
when the server is started with
``--server-module '{"module_path": "lmcache.v1.multiprocess.plugin_modules.qstore"}'``.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any
import math
import threading
import time

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.common.custom_types import TRANSFER_QUERY, IPCCacheServerKey
from lmcache.v1.common.group_view import EngineGroupInfo
from lmcache.v1.common.utils import LayoutHints
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_manager import l1_owners_of
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    InstanceLivenessTarget,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
    get_layout_desc,
)
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.base.ipc_wrapper import KVCache
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.transport.qstore import QRingInfo
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _ring_pieces(cache_context: Any) -> list[torch.Tensor]:
    """View a Q ring's per-layer tensors as ``[num_blocks, piece_bytes]`` bytes.

    A piece is one layer's slice of one ring block, the unit SDK transfers
    address: piece ``p`` is layer ``p % num_layers`` of block
    ``p // num_layers``.
    """
    return [
        layer.reshape(layer.shape[0], -1).view(torch.uint8)
        for layer in cache_context.kv_tensors
    ]


def _copy_object(
    pieces: list[torch.Tensor],
    piece_ids: list[int],
    memory_obj: MemoryObj,
    to_ring: bool,
) -> None:
    """Copy one L1 object's bytes to or from its ring pieces, in order.

    Must run on the ring's stream; the copies are asynchronous.

    Args:
        pieces: The ring's pieces (see ``_ring_pieces``).
        piece_ids: The object's pieces; piece ``i`` holds bytes
            ``[i * piece_bytes, (i + 1) * piece_bytes)`` of the object.
        memory_obj: The object in L1.
        to_ring: Copy L1 into the ring if True, else the ring into L1.

    Raises:
        RuntimeError: If the object has no backing memory.
    """
    num_layers = len(pieces)
    piece_bytes = pieces[0].shape[1]
    size = memory_obj.get_size()
    # The lazy allocator pins L1 in separately registered chunks, and one
    # cudaMemcpy cannot span two: copy through the native helper, which splits
    # at their boundaries (as lmcache_memcpy_async_h2d/d2h do).
    lazy = isinstance(memory_obj.parent(), LazyMemoryAllocator)
    raw = memory_obj.raw_tensor
    if raw is None:
        raise RuntimeError("memory object has no backing memory")
    obj_bytes = raw.reshape(-1).view(torch.uint8)[:size]
    for index, piece_id in enumerate(piece_ids):
        block, layer = divmod(piece_id, num_layers)
        lo = index * piece_bytes
        nbytes = min(piece_bytes, size - lo)
        piece = pieces[layer][block, :nbytes]
        if lazy:
            host = memory_obj.data_ptr + lo
            device_ops.lmcache_memcpy_async(
                piece.data_ptr() if to_ring else host,
                host if to_ring else piece.data_ptr(),
                nbytes,
                lmcache_native.TransferDirection.H2D
                if to_ring
                else lmcache_native.TransferDirection.D2H,
                memory_obj.meta.address + lo,
                LazyMemoryAllocator.PIN_CHUNK_SIZE,
            )
        elif to_ring:
            piece.copy_(obj_bytes[lo : lo + nbytes], non_blocking=True)
        else:
            obj_bytes[lo : lo + nbytes].copy_(piece, non_blocking=True)


@dataclass(frozen=True)
class _SDKObjects:
    """One object group's share of a SDK transfer.

    Attributes:
        object_group_id: The engine's object group.
        keys: The objects moved, in chunk order.
        num_pieces: Ring pieces each object occupies.
    """

    object_group_id: int
    keys: list[ObjectKey]
    num_pieces: int


@dataclass
class _QRingState:
    """What an SDK needs to transfer through one worker's Q ring.

    The used view is vLLM's used blocks as of the worker's last report, minus
    the blocks of stores that completed since: those are no longer read, and
    vLLM writes them again only after freeing and reallocating them in a later
    step. It never misses a block vLLM holds while vLLM is idle.

    Attributes:
        q_caches: The ring's IPC wrappers as registered (original handles).
        model_name: The query model name (``<model>##query``).
        world_size: The world size of the registration.
        used_blocks: The used view; every other block is free.
        worker_id: The worker's rank, from its reports; -1 until the first.
        num_alloc: The ring's allocation count at the worker's last report.
        pending_stores: ``(completion event, ring block ids)`` of stores whose
            copies may still be running.
    """

    q_caches: KVCache
    model_name: str
    world_size: int
    used_blocks: set[int] = field(default_factory=set)
    worker_id: int = -1
    num_alloc: int = 0
    pending_stores: list[tuple[Any, list[int]]] = field(default_factory=list)

    def reconcile_completed_stores(self) -> None:
        """Drop the blocks of completed stores from the used view."""
        still_running = []
        for done, block_ids in self.pending_stores:
            if done.query():
                self.used_blocks.difference_update(block_ids)
            else:
                still_running.append((done, block_ids))
        self.pending_stores = still_running


class QStoreModule(DiscoverableModule, InstanceLivenessTarget):
    """Handles paged Q ring registration and store operations.

    Owns Q context registrations and provides handlers for register,
    unregister, and store of the paged Q ring.

    Args:
        ctx: The shared engine context.
    """

    module_name = "qstore"
    # Requires the LMCache-driven transfer path to exist.
    module_dependencies = (LMCacheDrivenTransferModule,)

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build whenever this module is loaded through ``--server-module``.

        Requires the LMCache-driven transfer path and advertises the
        ``transfer_query`` experimental feature to clients.

        Args:
            build_ctx: Shared build context; the LMCache-driven transfer
                module must already be built.

        Returns:
            The constructed module.

        Raises:
            ValueError: If the LMCache-driven transfer module is not built
                (``supported_transfer_mode`` is neither ``lmcache_driven`` nor
                ``auto``).
        """
        try:
            build_ctx.require(LMCacheDrivenTransferModule)
        except ValueError:
            raise ValueError(
                f"Server module '{cls.module_name}' ({TRANSFER_QUERY}) requires "
                "supported_transfer_mode='lmcache_driven' or 'auto'."
            ) from None
        build_ctx.add_experimental(TRANSFER_QUERY)
        return cls(
            build_ctx.engine_context, build_ctx.require(LMCacheDrivenTransferModule)
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        kv_module: LMCacheDrivenTransferModule | None = None,
    ) -> None:
        self._ctx = ctx
        # The engine's KV registrations, whose L1 objects SDK transfers move.
        self._kv_module = kv_module
        self._q_contexts: dict[int, ContextEntry] = {}
        # Per-ring state for SDK retrieves into the ring, keyed like
        # _q_contexts and guarded by the same lock.
        self._q_rings: dict[int, _QRingState] = {}
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
                self._q_rings.pop(iid, None)
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
            "object_layouts": self._object_layouts(),
        }

    def _object_layouts(self) -> list[dict]:
        """Describe the L1 objects of every registered model, for SDK transfers.

        One object per (chunk, object group) holds one tensor per kernel
        group, in ascending kernel-group order, packed back to back. The
        layouts are the ones the server stores and reads with (its layout
        registry), for the engine's KV models and the Q rings alike.

        Returns:
            One entry per registered ``(model_name, world_size)``: its object
            groups' kind, window in chunks, and one chunk's tensor shapes and
            dtypes.
        """
        entries = list(self.context_entries_snapshot().values())
        if self._kv_module is not None:
            entries += self._kv_module.context_entries_snapshot().values()
        registry = self._ctx.layout_desc_registry
        layouts = []
        for model_name, world_size in sorted(
            {(e.model_name, e.world_size) for e in entries}
        ):
            group_layouts = registry.find_group_layout_descs(model_name, world_size)
            if group_layouts is None:
                continue
            attn_desc = registry.find_attn_desc(model_name, world_size)
            kinds = attn_desc.group_kinds or ("attention",) * len(group_layouts)
            layouts.append(
                {
                    "model_name": model_name,
                    "world_size": world_size,
                    "object_groups": [
                        {
                            "object_group_idx": group_id,
                            "kind": kinds[group_id],
                            "num_chunks_in_sw": attn_desc.num_chunks_in_sw[group_id],
                            "chunk_object_shapes": [
                                list(shape) for shape in layout.shapes
                            ],
                            "chunk_object_dtypes": [
                                str(dtype) for dtype in layout.dtypes
                            ],
                        }
                        for group_id, layout in sorted(group_layouts.items())
                    ],
                }
            )
        return layouts

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        with self._lock:
            entries = list(self._q_contexts.values())
            self._q_contexts.clear()
            self._q_rings.clear()
        self._release_entries(entries)

    @request_handler()
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
        # REGISTER_Q_CACHE is SERIALIZED on the MQ main loop, so it is
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
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        kv_groups_manager = cache_context.kv_layer_groups_manager
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name, world_size, layout_desc, attn_desc
        )

        with self._lock:
            self._q_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )
            self._q_rings[instance_id] = _QRingState(
                q_caches=list(q_caches),
                model_name=model_name,
                world_size=world_size,
            )

        logger.info(
            "Registered Q ring for instance %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    @request_handler()
    def unregister_q_cache(self, instance_id: int) -> None:
        """Unregister the paged Q ring tensors for a given worker instance ID.

        Args:
            instance_id: The worker instance ID (shared with its KV cache).
        """
        if not self._drop_instance_state(instance_id):
            logger.warning("No registered Q ring found for instance ID %d", instance_id)

    def drop_instance_state(self, instance_id: int) -> None:
        """Release this module's state during unified instance cleanup."""
        self._drop_instance_state(instance_id)

    def _drop_instance_state(self, instance_id: int) -> bool:
        """Pop and release one Q-ring registration if present."""
        with self._lock:
            popped = [
                e for e in (self._q_contexts.pop(instance_id, None),) if e is not None
            ]
            self._q_rings.pop(instance_id, None)
        if not popped:
            return False

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered Q ring for instance ID %d", instance_id)
        return True

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
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
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError(
                "Q ring event backend is not initialized; register the Q cache "
                "before submitting store requests"
            )
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
                    "STORE_Q block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            vllm_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(vllm_event, cache_context.stream)

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

            transfer_key = next_transfer_key(key.request_id)
            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "transfer_key": transfer_key,
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
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        obj_keys, layout_desc
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
                    )

                l1_owners = l1_owners_of(all_dict)
                store_succeeded = True
            except Exception:
                logger.exception("Cannot store Q keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                self._track_store_completion(instance_id, gpu_block_ids)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        l1_owners,
                    )
                else:
                    total_bytes = 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "transfer_key": transfer_key,
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
        return event_backend.export_event(event, cache_context.device), store_succeeded

    def _track_store_completion(
        self, instance_id: int, gpu_block_ids: list[list[int]]
    ) -> None:
        """Remember a store's ring blocks until its copies have completed.

        Must be called on the ring's stream, after the store's copies are
        enqueued: the recorded event completes when they have.

        Args:
            instance_id: The worker instance the ring is registered under.
            gpu_block_ids: The store's ring block ids, per group.
        """
        done = torch_dev.Event()
        done.record()
        block_ids = [block_id for group in gpu_block_ids for block_id in group]
        with self._lock:
            state = self._q_rings.get(instance_id)
            if state is not None:
                state.pending_stores.append((done, block_ids))

    @request_handler()
    def report_q_ring_state(
        self,
        instance_id: int,
        worker_id: int,
        used_blocks: list[int],
        num_alloc: int,
    ) -> None:
        """At the end of step, report the allocated blocks in the Q ring.

        Args:
            instance_id: The worker instance the ring is registered under.
            worker_id: The worker's rank.
            used_blocks: The ring's used blocks at the end of the step.
            num_alloc: The ring's allocation count; a report with fewer
                allocations is older and ignored.
        """
        with self._lock:
            state = self._q_rings.get(instance_id)
            if state is None or num_alloc < state.num_alloc:
                return
            state.reconcile_completed_stores()
            state.used_blocks = set(used_blocks)
            state.worker_id = worker_id
            state.num_alloc = num_alloc

    @request_handler()
    def get_q_ring(self, model_name: str, world_size: int) -> list[QRingInfo]:
        """Return the Q rings registered for a model, one per worker.

        Args:
            model_name: The query model name (``<model>##query``).
            world_size: The world size of the registration.

        Returns:
            One entry per registered worker, sorted by rank.
        """
        rings: list[QRingInfo] = []
        with self._lock:
            for instance_id, state in self._q_rings.items():
                if state.model_name != model_name or state.world_size != world_size:
                    continue
                state.reconcile_completed_stores()
                rings.append(
                    QRingInfo(
                        instance_id=instance_id,
                        worker_id=state.worker_id,
                        q_caches=list(state.q_caches),
                        used_blocks=sorted(state.used_blocks),
                        num_alloc=state.num_alloc,
                    )
                )
        return sorted(rings, key=lambda ring: (ring.worker_id, ring.instance_id))

    # SDK transfers: the engine's L1 objects through the Q ring, as bytes.

    def _layout_entry(self, model_name: str, world_size: int) -> ContextEntry | None:
        """Return a registration of the model, for its object layout.

        KV objects follow the engine's KV registration; query objects
        (``<model>##query``) follow the Q ring's own, which ``store_q`` stored
        them with. Every registration of a model has the same layout (the
        server keeps one per model), so any matching one serves.
        """
        kv_entries = (
            self._kv_module.context_entries_snapshot().values()
            if self._kv_module is not None
            else ()
        )
        for entry in [*kv_entries, *self.context_entries_snapshot().values()]:
            if entry.model_name == model_name and entry.world_size == world_size:
                return entry
        return None

    def _sdk_targets(
        self,
        op: str,
        key: IPCCacheServerKey,
        instance_id: int,
        piece_ids: list[int],
    ) -> tuple[ContextEntry, ContextEntry, list[torch.Tensor]] | None:
        """Resolve a SDK transfer's ring and KV registration, checking pieces.

        Args:
            op: The request name, for the log.
            key: The KV model's key.
            instance_id: The worker instance whose ring is used.
            piece_ids: The pieces the transfer uses.

        Returns:
            ``(ring entry, KV entry, ring pieces)``, or None (logged) if the
            ring or the model's KV registration is unknown, or a piece is out
            of range, repeated, or in a block vLLM holds.
        """
        ring = self.get_and_touch_context_entry(instance_id)
        kv = self._layout_entry(key.model_name, key.world_size)
        used: set[int] | None = None
        with self._lock:
            state = self._q_rings.get(instance_id)
            if state is not None:
                state.reconcile_completed_stores()
                used = set(state.used_blocks)
        if ring is None or used is None:
            logger.warning(
                "Rejecting %s for request_id=%s: no Q ring registered for instance %d",
                op,
                key.request_id,
                instance_id,
            )
            return None
        if kv is None:
            logger.warning(
                "Rejecting %s for request_id=%s: no registration of %r",
                op,
                key.request_id,
                key.model_name,
            )
            return None
        pieces = _ring_pieces(ring.cache_context)
        num_layers, num_blocks = len(pieces), pieces[0].shape[0]
        bad = sorted(
            piece_id
            for piece_id in set(piece_ids)
            if not 0 <= piece_id < num_layers * num_blocks
            or piece_id // num_layers in used
        )
        if bad or len(set(piece_ids)) != len(piece_ids):
            logger.error(
                "Rejecting %s for request_id=%s: pieces %s of instance %d are "
                "out of range or in blocks vLLM holds, or pieces are repeated",
                op,
                key.request_id,
                bad,
                instance_id,
            )
            return None
        return ring, kv, pieces

    def _sdk_plan(
        self,
        kv_context: Any,
        key: IPCCacheServerKey,
        object_group_ids: list[int],
        piece_bytes: int,
    ) -> list[_SDKObjects]:
        """Return the objects a SDK transfer moves, in transfer order.

        Every chunk of each object group, read or written whole. Only
        full-attention groups (no sliding window) are transferred.

        Args:
            kv_context: The KV registration's cache context.
            key: The KV model's key.
            object_group_ids: The object groups to move.
            piece_bytes: Bytes per ring piece.

        Returns:
            One entry per object group, in ascending group order.

        Raises:
            ValueError: If a group is unknown, not full attention, or has a
                sliding window.
        """
        manager = kv_context.kv_layer_groups_manager
        num_object_groups = manager.num_object_groups
        attn_desc = manager.get_attn_desc()
        kinds = attn_desc.group_kinds or ("attention",) * num_object_groups
        keys_per_group = self._ctx.resolve_obj_keys(key, list(range(num_object_groups)))
        plan: list[_SDKObjects] = []
        for group_id in sorted(set(object_group_ids)):
            if (
                not 0 <= group_id < num_object_groups
                or kinds[group_id] != "attention"
                or attn_desc.num_chunks_in_sw[group_id] >= 0
            ):
                raise ValueError(f"object group {group_id} cannot be transferred")
            keys = list(keys_per_group[group_id])
            layout = get_layout_desc(
                kv_context, self._ctx.chunk_size, object_group_id=group_id
            )
            num_bytes = sum(
                shape.numel() * dtype.itemsize
                for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
            )
            plan.append(_SDKObjects(group_id, keys, math.ceil(num_bytes / piece_bytes)))
        return plan

    def _prepare_sdk_transfer(
        self,
        op: str,
        key: IPCCacheServerKey,
        instance_id: int,
        object_group_ids: list[int],
        piece_ids: list[int],
    ) -> (
        tuple[ContextEntry, ContextEntry, list[torch.Tensor], list[_SDKObjects]] | None
    ):
        """Validate a SDK transfer and plan its objects.

        Returns:
            ``(ring entry, KV entry, ring pieces, plan)``, or None (logged) if
            the transfer is rejected.
        """
        targets = self._sdk_targets(op, key, instance_id, piece_ids)
        if targets is None:
            return None
        ring, kv, pieces = targets
        try:
            plan = self._sdk_plan(
                kv.cache_context, key, object_group_ids, pieces[0].shape[1]
            )
        except ValueError:
            logger.exception("Rejecting %s for request_id=%s", op, key.request_id)
            return None
        needed = sum(len(group.keys) * group.num_pieces for group in plan)
        if needed != len(piece_ids):
            logger.error(
                "Rejecting %s for request_id=%s: %d pieces given, the objects need %d",
                op,
                key.request_id,
                len(piece_ids),
                needed,
            )
            return None
        return ring, kv, pieces, plan

    def _release_rejected_retrieve(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> None:
        """Release the read locks of a rejected ``sdk_retrieve``'s range.

        The lookup read-locked the range's objects; a retrieve that reads them
        releases the locks once its copies are done, so a rejected one must
        release them itself. Without a registration of the model the keys
        cannot be resolved, and the locks expire with the read-lock TTL.

        Args:
            key: The rejected retrieve's key.
            object_group_ids: The object groups it asked for.
        """
        kv = self._layout_entry(key.model_name, key.world_size)
        if kv is None:
            return
        num_object_groups = kv.cache_context.kv_layer_groups_manager.num_object_groups
        groups = sorted(g for g in set(object_group_ids) if 0 <= g < num_object_groups)
        if not groups:
            return
        keys = [
            obj_key
            for group_keys in self._ctx.resolve_obj_keys(key, groups)
            for obj_key in group_keys
        ]
        storage_manager = self._ctx.storage_manager
        storage_manager.finish_read_by_owner(
            storage_manager.prepare_read_completion(
                keys, self._ctx.get_read_owners(key.request_id)
            )
        )

    def _wait_for_sdk_gpu_copy(
        self, ring: ContextEntry, instance_id: int, event_ipc_handle: bytes
    ) -> None:
        """Make the ring's stream wait for the SDK's GPU copies of the pieces.

        The SDK records its event after the copies it queued on the ring: the
        object bytes it wrote for a store, the previous batch's reads for a
        retrieve. The server's copies, queued after this wait, then neither
        read unwritten pieces nor overwrite unread ones.
        """
        cache_context = ring.cache_context
        event_backend = ring.event_backend
        assert event_backend is not None
        producer_event = event_backend.import_event(
            event_ipc_handle, cache_context.device
        )
        event_backend.wait_event(producer_event, cache_context.stream)
        kv = (
            self._kv_module.get_and_touch_context_entry(instance_id)
            if self._kv_module is not None
            else None
        )
        if kv is None:
            cache_context.stream.synchronize()
            return
        import_token = kv.cache_context.hold_imported_event(producer_event)
        submit_callback_to_stream(
            cache_context.cupy_stream,
            "release_imported_event",
            (instance_id, import_token),
        )

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def sdk_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        object_group_ids: list[int],
        piece_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Copy the engine's L1 objects, as bytes, into free Q ring pieces in
        QRingBuffer. Retrieve and store (QKV transfer) must not happen when
        vLLM is decoding a request.

        Args:
            key: The KV model's key; its range's objects are read.
            instance_id: The worker instance whose ring is written.
            object_group_ids: The object groups to read, every chunk of each.
            piece_ids: Pieces to write, ``block * num_layers + layer``: the
                objects of each group in ascending group order, chunk after
                chunk, each object's bytes over its pieces in order.
            event_ipc_handle: The IPC handle of the caller's event to wait on.

        Returns:
            The completion event's IPC handle and whether every object was
            copied. The handle is empty when no device work was submitted.
        """
        st = time.perf_counter()
        prepared = self._prepare_sdk_transfer(
            "SDK_RETRIEVE", key, instance_id, object_group_ids, piece_ids
        )
        if prepared is None:
            try:
                self._release_rejected_retrieve(key, object_group_ids)
            except Exception:
                # Never suppress the terminal response (see RETRIEVE).
                logger.exception(
                    "Failed to release SDK_RETRIEVE locks for request_id=%s",
                    key.request_id,
                )
            return b"", False
        ring, _, pieces, plan = prepared
        cache_context = ring.cache_context
        event_backend = ring.event_backend
        assert event_backend is not None
        storage_manager = self._ctx.storage_manager
        read_owners = self._ctx.get_read_owners(key.request_id)
        retained = [obj_key for group in plan for obj_key in group.keys]
        read_completion = storage_manager.prepare_read_completion(retained, read_owners)
        succeeded = False
        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)
            self._wait_for_sdk_gpu_copy(ring, instance_id, event_ipc_handle)
            try:
                cursor = 0
                for group in plan:
                    found, objs = storage_manager.unsafe_read(
                        group.keys, l1_owners=read_owners
                    )
                    if found != group.keys:
                        logger.error(
                            "SDK_RETRIEVE for request_id=%s: object group %d has "
                            "%d of %d objects",
                            key.request_id,
                            group.object_group_id,
                            len(found),
                            len(group.keys),
                        )
                        break
                    for memory_obj in objs:
                        _copy_object(
                            pieces,
                            piece_ids[cursor : cursor + group.num_pieces],
                            memory_obj,
                            to_ring=True,
                        )
                        cursor += group.num_pieces
                else:
                    succeeded = True
            except Exception:
                logger.exception("Cannot retrieve raw objects due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                if retained:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_by_owner",
                        read_completion,
                    )
        if succeeded:
            logger.info(
                "Retrieved %d raw objects into the Q ring in %.3f seconds",
                len(retained),
                time.perf_counter() - st,
            )
        return event_backend.export_event(event, cache_context.device), succeeded

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def sdk_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        object_group_ids: list[int],
        piece_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the engine's objects, written as bytes into Q ring pieces in
        the QRingBuffer.

        Args:
            key: The KV model's key; its range's objects are written.
            instance_id: The worker instance whose ring is read.
            object_group_ids: The object groups to write, every chunk of each.
            piece_ids: The objects' pieces, in ``sdk_retrieve``'s order.
            event_ipc_handle: The IPC handle of the caller's event to wait on.

        Returns:
            The completion event's IPC handle and whether the store completed
            without a failure. The handle is empty when no device work was
            submitted.
        """
        st = time.perf_counter()
        prepared = self._prepare_sdk_transfer(
            "SDK_STORE", key, instance_id, object_group_ids, piece_ids
        )
        if prepared is None:
            return b"", False
        ring, kv, pieces, plan = prepared
        cache_context = ring.cache_context
        event_backend = ring.event_backend
        assert event_backend is not None
        storage_manager = self._ctx.storage_manager
        all_dict: dict[ObjectKey, MemoryObj] = {}
        l1_owners: dict[ObjectKey, int] = {}
        succeeded = False
        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)
            self._wait_for_sdk_gpu_copy(ring, instance_id, event_ipc_handle)
            try:
                cursor = 0
                for group in plan:
                    layout_desc = get_layout_desc(
                        kv.cache_context,
                        self._ctx.chunk_size,
                        object_group_id=group.object_group_id,
                    )
                    reserved = storage_manager.reserve_write(group.keys, layout_desc)
                    all_dict.update(reserved)
                    for obj_key in group.keys:
                        # Keys the storage manager skipped (e.g. already
                        # stored) leave their pieces unread.
                        memory_obj = reserved.get(obj_key)
                        if memory_obj is not None:
                            _copy_object(
                                pieces,
                                piece_ids[cursor : cursor + group.num_pieces],
                                memory_obj,
                                to_ring=False,
                            )
                        cursor += group.num_pieces
                l1_owners = l1_owners_of(all_dict)
                succeeded = True
            except Exception:
                logger.exception("Cannot store raw objects due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit only when every object copied.
                if succeeded and all_dict:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        l1_owners,
                    )
        if succeeded:
            logger.info(
                "Stored %d raw objects from the Q ring in %.3f seconds",
                len(all_dict),
                time.perf_counter() - st,
            )
        return event_backend.export_event(event, cache_context.device), succeeded
