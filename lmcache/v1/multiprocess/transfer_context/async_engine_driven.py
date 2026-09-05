# SPDX-License-Identifier: Apache-2.0
"""Async engine-driven data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _single_group_block_ids,
)

logger = init_logger(__name__)


@dataclass
class _WorkerStagingArena:
    """Pinned staging slabs owned exclusively by one executor worker."""

    slabs: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = field(
        default_factory=dict
    )

    def views(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor] | None:
        """Return contiguous pinned chunk views, or ``None`` on allocation failure."""
        key = (tuple(shape), dtype)
        slab = self.slabs.get(key)
        if slab is None or slab.shape[0] < count:
            try:
                slab = torch.empty(
                    (count, *shape), dtype=dtype, device="cpu", pin_memory=True
                )
            except RuntimeError:
                logger.warning(
                    "Failed to allocate pinned CPU staging slab "
                    "(shape=%s, dtype=%s, chunks=%d)",
                    tuple(shape),
                    dtype,
                    count,
                )
                return None
            self.slabs[key] = slab
        return [slab[chunk_idx] for chunk_idx in range(count)]


class _WorkerStagingState(threading.local):
    """Thread-local binding between an executor thread and its staging arena."""

    def __init__(self) -> None:
        self.arena: _WorkerStagingArena | None = None


# TODO: async retrieve path TBD, but benefit might be very limited
class AsyncEngineDrivenTransferContext(EngineDrivenTransferContext):
    """Fully async engine-driven data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    while the deferred gather runs off the forward thread. For unpinned SHM
    transfers, the future resolves after GPU-to-staging completion so the
    engine can release source blocks while LMCache finishes the staging-to-SHM
    copy and commit internally.
    ``submit_retrieve`` stays synchronous and returns an already-resolved future
    exactly as on the base context.

    Inherits :class:`EngineDrivenTransferContext` and reuses its
    ``register()`` (layout / SHM registration, no stream dependency) and
    ``submit_retrieve()`` (this path does not change retrieve). Only the store
    is made async.

    Store is three-phase, all executed entirely in a background thread:

    1. prepare: call prepare_store() to negotiate buffers with the server
       (the costliest step in pickle mode due to the synchronous RPC round-trip).
    2. gather: wait for the forward event on the copy stream, then enqueue
       GPU->CPU copies. Unpinned SHM uses worker-owned pinned staging buffers;
       pickle lets the gather helper allocate its returned CPU chunks.
    3. commit: wait for gather completion (via a recorded CUDA event), then
       perform commit_store() and resolve the returned future.

    ``submit_store`` performs only O(1) work on the forward thread (registration
    check and block-id flattening) before submitting all three phases to the
    background ``commit_executor``, so the forward thread is never blocked by
    the RPC round-trip or gather kernel launch latency.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.
    """

    def __init__(self, commit_workers: int = 2) -> None:
        """Initialize the async context and create its async resources.

        Args:
            commit_workers: Retained for call compatibility. It is currently
                ignored and the context always uses two workers.

        TODO: Honor ``commit_workers`` if workload data establishes a need for
        a user-facing tuning knob.
        """
        super().__init__()
        del commit_workers
        self._commit_workers = 2
        logger.info(
            "Initializing async engine-driven store context with %d workers",
            self._commit_workers,
        )
        self._copy_stream: Any = torch_dev.Stream()
        self._worker_staging_state = _WorkerStagingState()
        self._commit_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_engine_driven_commit",
        )
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        # Tracks gather tasks that have been submitted to _commit_executor but
        # have not yet recorded their CUDA event. flush_inflight_stores waits
        # on all of these before synchronizing _inflight_gather_events, closing
        # the window where preemption could overwrite paged KV blocks before an
        # in-flight gather has had a chance to record its CUDA event.
        self._pending_stores: set[threading.Event] = set()
        # Serializes commit_store calls across worker threads, since the
        # underlying ZMQ socket is not thread-safe and there are two workers.
        self._commit_lock = threading.Lock()
        self._is_closing = False

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor] | None:
        """Allocate contiguous staging views from the current worker's arena.

        Each executor thread creates its arena on first use. An arena grows
        only when its worker encounters a larger store, then reuses its
        contiguous slab for later stores.

        Args:
            shape: Shape of one KV chunk.
            dtype: Data type of one KV chunk.
            count: Number of tensors needed.

        Returns:
            Contiguous chunk views backed by the worker's staging slab, or
            ``None`` when pinned allocation fails.
        """
        arena = self._worker_staging_state.arena
        if arena is None:
            arena = _WorkerStagingArena()
            self._worker_staging_state.arena = arena
        return arena.views(shape, dtype, count)

    def _release_staging(self, _chunks: list[torch.Tensor]) -> None:
        """Release staging chunks after a store completes.

        Worker-owned arenas retain their slabs for reuse, so no per-store
        action is needed.
        """

    def create_recorded_event(self) -> IPCEvent:
        """Create a local event that orders compute before the copy stream.

        Returns:
            A local device event recorded on the current stream. The event is
            never exported across processes.

        Raises:
            RuntimeError: If :meth:`register` has not completed.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Async engine-driven transfer context is not registered. "
                "Call register() before creating transfer events."
            )
        event = torch_dev.Event()
        event.record(torch_dev.current_stream())
        return event

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Three-phase async store (prepare, gather and commit all in background).

        Performs only O(1) work on the forward thread (registration check and
        block-id flattening), then submits all three phases — prepare_store,
        gather (GPU->CPU), and commit — to the background ``commit_executor``.
        Returns an unresolved future. Pickle transfers resolve after all three
        phases complete. Unpinned SHM transfers using staging resolve after the
        gather event; the staging-to-SHM copy and commit remain owned by the
        background executor and are drained by :meth:`close`.

        Args:
            _request_id: External request identifier (used for logging).
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            _event: Synchronization event; ``wait()`` is called in background.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            An unresolved :class:`MessagingFuture` that resolves to ``True``
            on success, ``False`` on failure.

        Raises:
            RuntimeError: If register() was not called first.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if _event is None:
            raise RuntimeError(
                "Async engine-driven transfer requires a local ordering event."
            )
        completion: MessagingFuture[bool] = MessagingFuture()
        engine_driven_context = self._engine_driven_context
        commit_executor = self._commit_executor

        # Signals when this task has recorded its CUDA event (or exited early),
        # allowing flush_inflight_stores to safely proceed.
        gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            full_block_ids = _single_group_block_ids(block_ids)

            def _prepare_gather_and_commit() -> None:
                gather_done: Any | None = None
                ok = False
                release_before_commit = False
                copy_staging_to_shm = False
                staging_destinations: list[torch.Tensor] = []
                staged_chunks: list[torch.Tensor] = []
                try:
                    # --- Phase 1: prepare_store ---
                    # In pickle mode this is the costliest step (sync RPC
                    # round-trip).  Running it here keeps the forward thread free.
                    result = engine_driven_context.prepare_store(key, instance_id)
                    out_buffers, chunk_indices = (
                        result if result is not None else (None, None)
                    )

                    if chunk_indices is not None and len(chunk_indices) == 0:
                        # All chunks are already in cache: no gather, no commit.
                        ok = True
                        return

                    # Determine gather target:
                    # - Pinned SHM path: gather directly into SHM views
                    # - Unpinned SHM path: gather into pinned staging then copy
                    # - Pickle path (no out_buffers): let gather allocate chunks
                    gather_target = out_buffers
                    if out_buffers is not None and not (
                        isinstance(engine_driven_context, EngineDrivenContextShm)
                        and engine_driven_context.is_pinned
                    ):
                        release_before_commit = True
                        first_buffer = out_buffers[0]
                        if any(
                            buffer.shape != first_buffer.shape
                            or buffer.dtype != first_buffer.dtype
                            for buffer in out_buffers
                        ):
                            raise ValueError(
                                "Pinned SHM staging requires uniform chunk "
                                "shapes and dtypes"
                            )
                        allocated_chunks = self._alloc_pinned_staging(
                            first_buffer.shape,
                            first_buffer.dtype,
                            len(out_buffers),
                        )
                        if allocated_chunks is None:
                            gather_target = out_buffers
                        else:
                            staged_chunks = allocated_chunks
                            gather_target = staged_chunks
                            staging_destinations = out_buffers
                            copy_staging_to_shm = True

                    # --- Phase 2: gather (GPU->CPU copy on copy stream) ---
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)

                        gathered_chunks = gather_paged_kv_to_cpu(
                            kv_caches,
                            full_block_ids,
                            blocks_in_chunk,
                            layout_hints=self._layout_hints,
                            engine_kv_format=self._engine_kv_format,
                            out=gather_target,
                            chunk_indices=chunk_indices,
                        )

                        if gather_target is None:
                            gather_target = gathered_chunks
                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()

                    if gather_done is not None:
                        gather_done.synchronize()

                    if release_before_commit:
                        # The source GPU KV is no longer needed. Let vLLM
                        # release its blocks while LMCache finishes the CPU-only
                        # work and commit in this executor task.
                        completion.set_result(True)
                    if copy_staging_to_shm:
                        for dst, src in zip(
                            staging_destinations, staged_chunks, strict=True
                        ):
                            dst.copy_(src)
                        gather_target = staging_destinations
                    # --- Phase 3: commit ---
                    with self._commit_lock:
                        ok = engine_driven_context.commit_store(
                            key,
                            instance_id,
                            gather_target,
                        )
                    if not ok:
                        logger.error(
                            "Async engine-driven commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    self._release_staging(staged_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    if not completion.query():
                        completion.set_result(ok)

            # Submitting the task is the ownership-transfer point: once it
            # succeeds, the closure is solely responsible for resolving the
            # future. The except below therefore only handles failures that
            # occur before this submit.
            commit_executor.submit(_prepare_gather_and_commit)
        except Exception:
            logger.exception("Failed to submit async engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
            gather_launched.set()
            completion.set_result(False)
            return completion

        return completion

    def flush_inflight_stores(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Called at preemption/eviction time so that vLLM cannot overwrite
        paged KV blocks before a deferred gather has finished reading them.

        Waits for all submitted-but-not-yet-launched stores to record their
        CUDA events before synchronizing those events, preventing a race where
        ``flush_inflight_stores`` returns before a background gather has
        started.
        """
        with self._inflight_lock:
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=False)

    def close(self) -> None:
        """Drain in-flight gather/commit work before closing the base context."""
        with self._inflight_lock:
            self._is_closing = True
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=True)
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        super().close()

    def _sync_gather_events(self, suppress_errors: bool = False) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Args:
            suppress_errors: If True, log exceptions instead of propagating.
        """
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                if not suppress_errors:
                    raise
                logger.exception("Failed while draining gather events")
