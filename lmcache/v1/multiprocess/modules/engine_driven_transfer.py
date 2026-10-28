# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    InstanceLivenessTarget,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)

# In-flight prepare->commit timings tracked per server. A prepared transfer is
# not guaranteed to commit, so this is a cap rather than a precise bound: it
# sits far above the transfers a deployment can have genuinely in flight
# (workers x ranks x chunks), so eviction only ever discards abandoned entries.
_MAX_TRACKED_TRANSFERS = 4096


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
        device: Label identifying the worker's KV cache device, used to
            correlate and label its transfer events. Falls back to the
            instance id for clients that do not report a device.
        chunk_bytes: Size of one single-group chunk object in bytes. Only the
            single-group fallback uses it; a multi-group transfer's volume is
            read off the SHM slots it actually reserved, because its groups
            have different per-chunk sizes.
        last_seen: ``time.monotonic()`` of the most recent activity from this
            instance (register, PING, prepare/commit). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window. Latched only by PING.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int
    device: str
    chunk_bytes: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False


class EngineDrivenTransferModule(DiscoverableModule, InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

    module_name = "engine_driven_transfer"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build unless the transfer mode excludes the engine-driven path."""
        mode = build_ctx.mp_config.supported_transfer_mode
        if mode == "lmcache_driven":
            return None
        return cls(build_ctx.engine_context)

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._engine_driven_contexts: dict[int, EngineDrivenContextEntry] = {}
        self._strategies: dict[int, TransferStrategy] = {}
        # Guards _engine_driven_contexts and _strategies together (the reaper
        # mutates them off the MQ main loop). Leaf lock, never held with
        # _pending_shm_lock.
        self._lock = threading.Lock()
        self._pending_shm_writes: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_reads: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_lock = threading.Lock()
        # Transfer timing state; see _timing_key for the key. Bounded, because
        # a prepared transfer is not guaranteed to commit: a multi-group store
        # whose gather fails returns without committing, and a live worker is
        # never reaped, so its abandoned entries would otherwise accumulate for
        # the process lifetime. Insertion order is start order, so the oldest
        # entry is the front one; over the cap the oldest loses its timing
        # sample, which degrades a metric instead of leaking. Leaf lock, never
        # held with _lock.
        self._transfer_starts: dict[
            tuple[str, int, str, int, int], tuple[float, int | None, int | None]
        ] = {}
        self._transfer_starts_lock = threading.Lock()
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        with self._lock:
            entries = dict(self._engine_driven_contexts)
        for instance_id, entry in entries.items():
            registered_non_cuda_ids.append(instance_id)
            non_cuda_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "block_size": entry.metadata.block_size,
                "use_mla": entry.metadata.use_mla,
            }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
        }

    def close(self) -> None:
        """Release resources owned by this module."""
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered non-GPU instances."""
        with self._lock:
            return len(self._engine_driven_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap non-GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one that
        has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, EngineDrivenContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._engine_driven_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                entry = self._engine_driven_contexts.pop(iid)
                self._strategies.pop(iid, None)
                reaped.append((iid, entry))
        for iid, entry in reaped:
            self._release_entry(iid, entry)
            logger.warning(
                "Reaped non-GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
        return [iid for iid, _ in reaped]

    def _resolve_for_transfer(
        self, instance_id: int
    ) -> tuple[EngineDrivenContextEntry, TransferStrategy]:
        """Return (entry, strategy) for a transfer, refreshing last_seen.

        Pair-atomicity guarantees the entry exists whenever the strategy
        does. Refreshes last_seen (no latch) so an active worker is not
        reaped mid-transfer.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry and its transfer strategy.

        Raises:
            ValueError: If the instance is not registered (or was reaped).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            strategy = self._strategies.get(instance_id)
            if entry is None or strategy is None:
                raise ValueError(
                    "non-GPU context not registered (or reaped) for "
                    f"instance ID {instance_id}"
                )
            entry.last_seen = now
            return entry, strategy

    def _release_entry(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        """Release resources for a popped entry (run outside the lock).

        Sweeps the instance's in-flight transfer timings and pending SHM
        transfers, then unregisters its layout descriptor.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
        with self._transfer_starts_lock:
            for state_key in [k for k in self._transfer_starts if k[1] == instance_id]:
                del self._transfer_starts[state_key]

        with self._pending_shm_lock:
            stale_writes = [k for k in self._pending_shm_writes if k[0] == instance_id]
            stale_reads = [k for k in self._pending_shm_reads if k[0] == instance_id]
            write_obj_keys = [self._pending_shm_writes.pop(k) for k in stale_writes]
            read_obj_keys = [self._pending_shm_reads.pop(k) for k in stale_reads]

        for obj_keys in write_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_write(obj_keys)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_single_group_obj_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
        """Resolve object keys for the single object group used by
        non-GPU transfers."""
        return self._ctx.resolve_obj_keys(key, [0])[0]

    def _resolve_obj_keys_by_group(
        self, key: IPCCacheServerKey, num_groups: int
    ) -> list[list[ObjectKey]]:
        """Resolve object keys for every LMCache group, group-major.

        Args:
            key: Cache key for the token range.
            num_groups: Number of LMCache groups the worker registered.

        Returns:
            ``keys[g][c]`` is chunk ``c``'s key in group ``g``
            (``ObjectKey.object_group_id == g``).
        """
        return self._ctx.resolve_obj_keys(key, list(range(num_groups)))

    def _group_keys_for(
        self, entry: "EngineDrivenContextEntry", key: IPCCacheServerKey
    ) -> "list[list[ObjectKey]] | None":
        """Group-major keys when ``entry`` registered multi-group, else None."""
        layouts = entry.metadata.group_layouts
        if not layouts or len(layouts) < 2:
            return None
        return self._resolve_obj_keys_by_group(key, len(layouts))

    def _publish_transfer_event(
        self,
        event_type: EventType,
        key: IPCCacheServerKey,
        instance_id: int,
        entry: EngineDrivenContextEntry,
        **metadata: object,
    ) -> None:
        """Publish one engine-driven transfer END event.

        Only END is published, and only from ``commit_*``. It carries the
        server-measured ``duration_seconds`` -- the ``prepare_*`` to
        ``commit_*`` window, in which the worker drives the device copies, so
        it covers that copy plus the transport and is what the matching log
        line reports.

        There is deliberately no START. The GPU path needs one because its
        subscribers infer a duration by pairing START with END per request
        and device, which assumes one transfer at a time per rank; that does
        not hold here, where several chunks of one request can be in flight
        on a rank. Since this path reports the duration itself and both
        subscribers prefer a reported one, a START would contribute nothing
        but correlation state -- state that a prepare which never commits
        (a worker whose gather fails is never reaped) would leave behind in
        every subscriber, unbounded. Publishing nothing until the transfer is
        known to have completed leaves nothing to clean up.

        Each rank of a tensor-parallel worker reports its own ``device``, so a
        consumer can tell their transfers of one request apart.

        No ``transfer_key`` is reported: that key correlates an END with the
        native copy-engine phase samples, which only the lmcache-driven path
        records.

        Args:
            event_type: The ``MP_STORE_*`` / ``MP_RETRIEVE_*`` type to publish.
            key: Cache key of the transfer.
            instance_id: Worker instance identifier.
            entry: The worker's registered context.
            metadata: Event-type specific fields merged into the metadata,
                e.g. ``total_bytes`` and ``num_tokens`` on the END events.
                Fields passed as ``None`` are omitted, so consumers see an
                absent field rather than a null one.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=event_type,
                session_id=key.request_id,
                metadata={
                    "device": entry.device,
                    "engine_id": instance_id,
                    "model_name": entry.model_name,
                    **{k: v for k, v in metadata.items() if v is not None},
                },
            )
        )

    def _publish_transfer_submitted(
        self,
        event_type: EventType,
        key: IPCCacheServerKey,
        entry: EngineDrivenContextEntry,
    ) -> None:
        """Publish a ``MP_*_SUBMITTED`` sentinel for one transfer.

        Published from ``commit_*``, immediately before the END, rather than
        from ``prepare_*`` as the GPU path does. It exists only to keep the
        submitted/finished counters balanced: they are subtracted per device
        to give transfers in flight, and feeding only the END side would
        drive that difference negative. Pairing them at commit reports zero
        in flight, which is honest here -- the server learns of an
        engine-driven transfer only once it has completed -- and means a
        prepare that never commits publishes nothing to strand.

        Carries only ``device``, matching what the lmcache-driven path
        reports: a richer label set on one side would force a consumer to
        aggregate it away before subtracting.

        Args:
            event_type: ``MP_STORE_SUBMITTED`` or ``MP_RETRIEVE_SUBMITTED``.
            key: Cache key of the transfer.
            entry: The worker's registered context.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=event_type,
                session_id=key.request_id,
                metadata={"device": entry.device},
            )
        )

    @staticmethod
    def _timing_key(
        phase: str, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[str, int, str, int, int]:
        """Identify one in-flight transfer for timing purposes.

        One entry per transfer, not per request: every rank of a
        tensor-parallel worker transfers under one request_id, and a rank can
        have several of its chunks in flight, so a coarser key loses
        durations. ``IPCCacheServerKey`` cannot serve as the key on its own
        because it excludes ``request_id`` from equality by design, so two
        concurrent requests over the same token range would share one entry:
        the second prepare would overwrite the first's start and the first
        commit would consume the replacement. Within one request_id the token
        range pins the transfer, so the key's ``(start, end)`` completes the
        identity without hashing the request's whole token block.

        Args:
            phase: ``"store"`` or ``"retrieve"``.
            key: Cache key of the transfer.
            instance_id: Worker instance identifier.

        Returns:
            The timing-dict key for this transfer.
        """
        return (phase, instance_id, key.request_id, key.start, key.end)

    def _begin_transfer(
        self,
        phase: str,
        key: IPCCacheServerKey,
        instance_id: int,
        prepared: "tuple[int | None, int | None] | None",
    ) -> None:
        """Record the start of one prepared transfer.

        Args:
            phase: ``"store"`` or ``"retrieve"``.
            key: Cache key of the transfer.
            instance_id: Worker instance identifier.
            prepared: ``(chunks, bytes)`` the transfer will move. ``bytes`` is
                ``None`` for an inline store, whose payload -- and so its byte
                count -- only arrives at commit. ``None`` for an untracked
                transfer.
        """
        chunks, nbytes = prepared if prepared is not None else (None, None)
        with self._transfer_starts_lock:
            while len(self._transfer_starts) >= _MAX_TRACKED_TRANSFERS:
                self._transfer_starts.pop(next(iter(self._transfer_starts)))
            self._transfer_starts[self._timing_key(phase, key, instance_id)] = (
                time.perf_counter(),
                chunks,
                nbytes,
            )

    def _finish_transfer(
        self, phase: str, key: IPCCacheServerKey, instance_id: int
    ) -> "tuple[float, int | None, int | None] | None":
        """Close one transfer and return what it took.

        Args:
            phase: ``"store"`` or ``"retrieve"``.
            key: Cache key of the transfer.
            instance_id: Worker instance identifier.

        Returns:
            ``(elapsed_seconds, chunks, bytes)``, or ``None`` when the
            transfer has no recorded start: its instance was reaped, the commit
            arrived without a prepare, or the entry was evicted under the
            tracking cap. ``chunks``/``bytes`` are ``None`` within a found
            entry when the strategy does not report them, which is a different
            thing from the transfer being untracked -- callers must not
            conflate them, because the whole requested range is the right
            count only in the former case.
        """
        with self._transfer_starts_lock:
            state = self._transfer_starts.pop(
                self._timing_key(phase, key, instance_id), None
            )
        if state is None:
            return None
        start, chunks, nbytes = state
        return time.perf_counter() - start, chunks, nbytes

    @staticmethod
    def _committed_volume(
        timing: "tuple[float, int | None, int | None] | None",
        moved: bool,
        wire_bytes: int | None = None,
    ) -> "tuple[float | None, int | None, int | None]":
        """Resolve what a finished transfer moved, or ``None`` if unknowable.

        Both quantities are fixed at prepare, except the inline store's byte
        count, which is only known once its payload arrives at commit --
        ``wire_bytes`` supplies it there. An untracked transfer (its worker
        was reaped, the commit arrived without a prepare, or the entry was
        evicted under the tracking cap) reports nothing: the requested range
        is not a safe substitute, since an SHM store reserves only the chunks
        that were missing and counting the range would report already-cached
        chunks as newly stored. The completion event is still published so
        consumers see the transfer finish.

        Args:
            timing: Result of :meth:`_finish_transfer`.
            moved: Whether the transfer actually moved data.
            wire_bytes: Bytes the transport actually carried, for a strategy
                that only knows them at commit. Ignored when the prepare
                already reported a byte count.

        Returns:
            ``(elapsed_seconds, chunk_count, total_bytes)``. Every element is
            ``None`` when the transfer was not tracked.
        """
        if timing is None:
            return None, None, None
        elapsed, chunks, nbytes = timing
        if not moved:
            return elapsed, 0, 0
        return elapsed, chunks, nbytes if nbytes is not None else wire_bytes

    def _inline_store_chunks(
        self,
        key: IPCCacheServerKey,
        group_keys: "list[list[ObjectKey]] | None",
    ) -> int | None:
        """Count the distinct chunks an inline store will carry, if knowable.

        A single-group inline store carries every object key of the range, so
        the count is exact. A multi-group one does not: the worker leaves out
        chunks that are all-null in a group, and it does not tell the server
        which -- ``_prepare_multigroup_store`` calls ``prepare_store`` with
        only the key and instance id, so ``skipped_chunks`` arrives empty and
        counting the range would report chunks that never travel. The chunk
        list lives in the payload instead, which only ``commit_store``
        decodes, so report nothing rather than the range. ``total_bytes``
        still comes from the payload length, which is what the throughput
        histograms need.

        Args:
            key: Cache key of the transfer.
            group_keys: Group-major object keys, or ``None`` for single-group.

        Returns:
            The distinct chunk count, or ``None`` for a multi-group store.
        """
        if group_keys is None:
            return len(self._resolve_single_group_obj_keys(key))
        return None

    def _served_retrieve_chunks(
        self,
        key: IPCCacheServerKey,
        group_keys: "list[list[ObjectKey]] | None",
        group_first_chunk: "Sequence[int]",
    ) -> int:
        """Count the distinct chunk positions a successful retrieve serves.

        A retrieve is all-or-nothing over the keys resolved for it, so the
        positions are known before the payload exists. Single-group serves
        the whole resolved range. Multi-group serves each group's in-window
        suffix, which differ: a recurrent or sliding-window group may carry
        one trailing chunk while a full-attention group carries the whole
        prefix, so the answer is the union over groups -- not group 0's
        length, which would report that one trailing chunk for the lot.

        Args:
            key: Cache key of the transfer.
            group_keys: Per group, the narrowed keys served, or ``None`` for
                single-group.
            group_first_chunk: Per group, the absolute position of its first
                narrowed key. Empty means every group starts at 0.

        Returns:
            The number of distinct chunk positions served.
        """
        if group_keys is None:
            return len(self._resolve_single_group_obj_keys(key))
        firsts = list(group_first_chunk) or [0] * len(group_keys)
        served: set[int] = set()
        for first, g_keys in zip(firsts, group_keys, strict=True):
            served.update(range(first, first + len(g_keys)))
        return len(served)

    @staticmethod
    def _prepared_volume(
        context: dict[str, Any],
    ) -> "tuple[int, int] | None":
        """Return ``(chunks, bytes)`` the prepared transfer will move.

        The SHM strategy reserves only the chunks that still need the
        transfer and describes each as a slot in its response context; the
        inline strategy reserves nothing at prepare and reports no slots, so
        the caller derives the volume from the object keys instead.

        Args:
            context: ``PrepareStoreResponse`` / ``PrepareRetrieveResponse``
                context.

        Returns:
            The chunk and byte counts, or ``None`` when the strategy does not
            report slots.
        """
        slots = context.get("slots")
        if slots is None:
            return None
        # One slot per chunk per group, so a multi-group response has several
        # slots at the same chunk position; tokens follow the distinct chunks
        # (the response reports them only when it spans groups).
        chunk_indices = context.get("chunk_indices")
        chunks = len(set(chunk_indices)) if chunk_indices is not None else len(slots)
        return chunks, sum(int(slot["length"]) for slot in slots)

    @request_handler()
    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                num_physical_slots).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        now = time.monotonic()
        with self._lock:
            existing = self._engine_driven_contexts.get(payload.instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered (non-GPU); refreshing liveness",
                    payload.instance_id,
                )
                return RegisterEngineDrivenContextResponse(
                    shm_name=shm_name,
                    pool_size=pool_size,
                    null_block_id=self._ctx.null_block_id,
                )

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        num_physical_slots = payload.num_physical_slots
        if num_physical_slots is None:
            # Compatibility with clients from before the physical-slot field
            # was added. Those clients require one slot per logical token.
            num_physical_slots = self._ctx.chunk_size
        elif num_physical_slots <= 0:
            raise ValueError(
                f"num_physical_slots must be positive, got {num_physical_slots}"
            )

        shape = (
            torch.Size(
                [payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, num_physical_slots, payload.hidden_dim_size]
            )
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        group_layouts: list[MemoryLayoutDesc] | None = None
        if payload.group_layouts:
            group_layouts = []
            for gl in payload.group_layouts:
                g_dtype = getattr(torch, gl.dtype_str, None)
                if g_dtype is None or not isinstance(g_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid group dtype_str '{gl.dtype_str}' in "
                        "engine-driven registration"
                    )
                # Sliding-window groups store a reduced token window per chunk;
                # size the per-chunk object to that window (fall back to the full
                # chunk when the worker didn't report one, e.g. legacy payloads).
                g_tokens = gl.window_tokens or self._ctx.chunk_size
                # Plane count is per group: a two-plane K/V group can sit
                # beside a one-plane index cache. Legacy workers report 0, so
                # fall back to the top-level flag they were sized by.
                g_single_plane = gl.kv_size == 1 if gl.kv_size else payload.use_mla
                g_shape = (
                    torch.Size([gl.num_layers, g_tokens, gl.hidden_dim_size])
                    if g_single_plane
                    else torch.Size([2, gl.num_layers, g_tokens, gl.hidden_dim_size])
                )
                group_layouts.append(
                    MemoryLayoutDesc(shapes=[g_shape], dtypes=[g_dtype])
                )
        group_windows: list[int] | None = None
        if payload.group_layouts:
            group_windows = []
            for gl in payload.group_layouts:
                if gl.window_chunks < 0:
                    raise ValueError(
                        f"Invalid group window_chunks {gl.window_chunks} in "
                        "engine-driven registration"
                    )
                group_windows.append(gl.window_chunks or -1)
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
            group_layouts=group_layouts,
            group_windows=group_windows,
        )
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SERIALIZED, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            device=payload.device or f"instance:{payload.instance_id}",
            chunk_bytes=shape.numel() * dtype.itemsize,
            last_seen=now,
            has_liveness_signal=False,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
            read_owner_resolver=self._ctx.get_read_owners,
            shm_name=shm_name,
            pool_size=pool_size,
            pending_writes=self._pending_shm_writes,
            pending_reads=self._pending_shm_reads,
            pending_lock=self._pending_shm_lock,
            transfer_key_factory=self._make_transfer_key,
        )
        with self._lock:
            self._engine_driven_contexts[payload.instance_id] = entry
            self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
        )

        if group_layouts:
            # The lookup/prefetch path fans its object-group lookups out over
            # the registered AttnWindowDesc's group count. A hybrid-KV worker
            # stores and retrieves ``len(group_layouts)`` object groups, but the
            # default single-group desc makes lookup prefetch only group 0 --
            # every later group then misses at retrieve (unsafe_read finds no
            # read-locked objects) and vLLM consumes the unloaded blocks. Match
            # the desc's group count to the worker's, with each group's own
            # window and kind: sliding-window and recurrent-state groups need
            # only their trailing chunks for a hit, and the worker never stores
            # their all-null earlier chunks, so a full-prefix window would make
            # every lookup on such a model miss.
            assert group_windows is not None
            attn_desc = AttnWindowDesc(
                num_chunks_in_sw=group_windows,
                world_size=payload.world_size,
                group_kinds=tuple(
                    "recurrent" if gl.recurrent else "attention"
                    for gl in payload.group_layouts
                ),
            )
            self._ctx.layout_desc_registry.register(
                payload.model_name,
                payload.world_size,
                layout_desc,
                attn_desc=attn_desc,
                group_layout_descs=dict(enumerate(group_layouts)),
            )
        else:
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, layout_desc
            )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name,
            pool_size=pool_size,
            null_block_id=self._ctx.null_block_id,
        )

    @request_handler(operation="unregister_kv_cache_engine_driven_context")
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        with self._lock:
            entry = self._engine_driven_contexts.pop(instance_id, None)
            if entry is not None:
                self._strategies.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._release_entry(instance_id, entry)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        skipped_chunks: tuple[tuple[int, ...], ...] = (),
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            skipped_chunks: Per LMCache group, the chunk positions the worker
                will not store because every engine block in them is the null
                block. Empty means none; otherwise one entry per registered
                group. Only multi-group workers send it.

        Returns:
            PrepareStoreResponse with empty slots for inline mode.

        Raises:
            ValueError: If ``skipped_chunks`` does not have one entry per
                registered group or names a chunk outside the key's range.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        group_keys = self._group_keys_for(entry, key)
        if skipped_chunks:
            if group_keys is None or len(skipped_chunks) != len(group_keys):
                raise ValueError(
                    f"skipped_chunks has {len(skipped_chunks)} groups for "
                    f"{0 if group_keys is None else len(group_keys)} registered "
                    f"groups (instance_id={instance_id})"
                )
            for gid, (skips, g_keys) in enumerate(
                zip(skipped_chunks, group_keys, strict=True)
            ):
                if any(not 0 <= idx < len(g_keys) for idx in skips):
                    raise ValueError(
                        f"skipped_chunks for group {gid} is outside its "
                        f"{len(g_keys)} chunks (instance_id={instance_id})"
                    )
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
            group_keys=group_keys,
            skipped_chunks=skipped_chunks,
        )
        prepared: "tuple[int | None, int | None] | None" = self._prepared_volume(
            response.context
        )
        if prepared is None:
            # Inline: no slots to count, and its byte count only arrives with
            # the payload at commit.
            prepared = (self._inline_store_chunks(key, group_keys), None)
        # An unknown count (multi-group inline) is not zero: the worker only
        # prepares when it has something to store, so the transfer is tracked.
        if prepared[0] == 0:
            # Every chunk is already cached or all-null, so the strategy
            # reserved nothing and the worker returns without committing.
            # Recording a start here would leave an entry no commit ever pops.
            return response
        self._begin_transfer("store", key, instance_id, prepared)
        return response

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit serialized CPU chunks to storage.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Typed inline CPU tensor payload produced by the worker.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        timing = self._finish_transfer("store", key, instance_id)
        result = strategy.commit_store(
            key=key,
            instance_id=instance_id,
            cpu_data=cpu_data,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
            group_keys=self._group_keys_for(entry, key),
        )
        elapsed, stored_count, total_bytes = self._committed_volume(
            timing, moved=result, wire_bytes=len(cpu_data) or None
        )
        self._publish_transfer_submitted(EventType.MP_STORE_SUBMITTED, key, entry)
        self._publish_transfer_event(
            EventType.MP_STORE_END,
            key,
            instance_id,
            entry,
            stored_count=stored_count,
            total_bytes=total_bytes,
            num_tokens=None
            if stored_count is None
            else stored_count * self._ctx.chunk_size,
            duration_seconds=elapsed,
        )
        if elapsed is not None and result:
            # A multi-group inline store cannot count its chunks (see
            # _inline_store_chunks), but its byte count is exact -- report
            # that rather than logging nothing, so a completed store stays
            # visible on the one transport that has no alternative.
            if stored_count is not None:
                logger.info(
                    "Stored %d tokens in %.3f seconds",
                    stored_count * self._ctx.chunk_size,
                    elapsed,
                )
            elif total_bytes is not None:
                logger.info(
                    "Stored %.1f MiB in %.3f seconds",
                    total_bytes / (1 << 20),
                    elapsed,
                )
        return result

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Retrieve prefetched chunks and return serialized CPU tensors.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        group_keys = self._group_keys_for(entry, key)
        group_first_chunk: list[int] = []
        windows = entry.metadata.group_windows
        if group_keys is not None and windows is not None:
            # Serve each group's in-window suffix only, as prefetch locked it:
            # a sliding-window or recurrent group's earlier chunks are not
            # needed and were never stored.
            narrowed: list[list[ObjectKey]] = []
            for g_keys, window in zip(group_keys, windows, strict=True):
                first = 0 if window < 0 else max(0, len(g_keys) - window)
                group_first_chunk.append(first)
                narrowed.append(g_keys[first:])
            group_keys = narrowed
        response = strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
            group_keys=group_keys,
            group_first_chunk=group_first_chunk,
        )
        prepared: "tuple[int | None, int | None]" = (0, 0)
        if response.success:
            from_slots = self._prepared_volume(response.context)
            # Inline answers with the payload itself: its length is exactly
            # what moved, and a retrieve is all-or-nothing over the chunks
            # narrowed above, so both numbers are known here.
            prepared = from_slots or (
                self._served_retrieve_chunks(key, group_keys, group_first_chunk),
                len(response.data),
            )
        self._begin_transfer("retrieve", key, instance_id, prepared)
        return response

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Args:
            key: Cache key (unused for inline transfer).
            instance_id: Worker instance identifier (unused for inline transfer).

        Returns:
            Always ``True``.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        timing = self._finish_transfer("retrieve", key, instance_id)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        elapsed, retrieved_count, total_bytes = self._committed_volume(
            timing, moved=True
        )
        self._publish_transfer_submitted(EventType.MP_RETRIEVE_SUBMITTED, key, entry)
        self._publish_transfer_event(
            EventType.MP_RETRIEVE_END,
            key,
            instance_id,
            entry,
            retrieved_count=retrieved_count,
            cache_salt=key.cache_salt,
            total_bytes=total_bytes,
            num_tokens=None
            if retrieved_count is None
            else retrieved_count * self._ctx.chunk_size,
            duration_seconds=elapsed,
        )
        # A miss commits nothing: skip the log line rather than report zero
        # tokens on every lookup that found no cache, as the GPU path does.
        if elapsed is not None and retrieved_count:
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                retrieved_count * self._ctx.chunk_size,
                elapsed,
            )
        return result
