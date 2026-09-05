# SPDX-License-Identifier: Apache-2.0
"""Async engine-driven data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _single_group_block_ids,
)

logger = init_logger(__name__)

# Number of background threads used to run commit (CPU->server) work for the
# async engine-driven store path. >1 so that a slow gather for one store does
# not block the commit of another store whose gather already finished.
DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 2


# TODO: async retrieve path TBD, but benefit might be very limited
class AsyncEngineDrivenTransferContext(EngineDrivenTransferContext):
    """Fully async engine-driven data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    while the deferred gather runs off the forward thread. Normally the future
    resolves after gather and commit. With pinned SHM staging enabled, it
    resolves after GPU-to-staging completion so the engine can release source
    blocks while LMCache finishes the staging-to-SHM copy and commit internally.
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
       GPU->CPU copies. When SHM buffers are available, gather writes directly
       into SHM views (matching the synchronous path). Otherwise, gather
       targets pinned staging buffers.
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

    def __init__(
        self,
        commit_workers: int | None = None,
    ) -> None:
        """Initialize the async context and create its async resources.

        Args:
            commit_workers: Number of background threads used to run commit
                (CPU->server) work. When omitted, reads
                ``LMCACHE_ENGINE_DRIVEN_COMMIT_WORKERS`` and otherwise defaults
                to two.
        """
        super().__init__()
        if commit_workers is None:
            commit_workers = int(
                os.environ.get(
                    "LMCACHE_ENGINE_DRIVEN_COMMIT_WORKERS",
                    str(DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS),
                )
            )
        self._commit_workers = max(1, int(commit_workers))
        self._use_separate_copy_streams = (
            os.environ.get("LMCACHE_ENGINE_DRIVEN_SEPARATE_COPY_STREAMS") == "1"
        )
        self._use_pinned_shm_staging = (
            os.environ.get("LMCACHE_ENGINE_DRIVEN_PINNED_STAGING") == "1"
        )
        self._copy_stream: Any = torch_dev.Stream()
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
        # underlying ZMQ socket is not thread-safe and commit_workers defaults
        # to >1.
        self._commit_lock = threading.Lock()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._staging_slabs: list[torch.Tensor] = []
        self._staging_preallocation_lock = threading.Lock()
        self._staging_preallocated = False
        self._staging_prealloc_slots = int(
            os.environ.get("LMCACHE_ENGINE_DRIVEN_PINNED_STAGING_PREALLOC_SLOTS", "0")
        )
        self._staging_chunks_per_slot = int(
            os.environ.get("LMCACHE_ENGINE_DRIVEN_PINNED_STAGING_CHUNKS_PER_SLOT", "40")
        )
        self._is_closing = False

    def _ensure_pinned_staging_preallocated(self) -> None:
        """Preallocate configured pinned staging slots during registration."""
        if (
            not self._use_pinned_shm_staging
            or self._staging_prealloc_slots <= 0
            or self._staging_preallocated
        ):
            return
        if self._staging_chunks_per_slot <= 0:
            raise ValueError(
                "LMCACHE_ENGINE_DRIVEN_PINNED_STAGING_CHUNKS_PER_SLOT must be positive"
            )

        with self._staging_preallocation_lock:
            if self._staging_preallocated:
                return
            engine_driven_context = self.engine_driven_context
            layout_desc = engine_driven_context.layout_desc
            if not layout_desc.shapes or not layout_desc.dtypes:
                raise RuntimeError(
                    "Cannot preallocate pinned staging without layout metadata"
                )

            shape = layout_desc.shapes[0]
            dtype = layout_desc.dtypes[0]
            key = (tuple(shape), dtype)
            started_at = time.perf_counter()
            slabs = [
                torch.empty(
                    (self._staging_chunks_per_slot, *shape),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self._staging_prealloc_slots)
            ]
            chunks = [
                slab[chunk_idx]
                for slab in slabs
                for chunk_idx in range(self._staging_chunks_per_slot)
            ]
            with self._inflight_lock:
                self._staging_slabs.extend(slabs)
                self._staging_pool.setdefault(key, []).extend(chunks)
            self._staging_preallocated = True
            allocated_bytes = sum(slab.numel() * slab.element_size() for slab in slabs)
            logger.info(
                "Preallocated %d pinned staging slots (%d chunks/slot, %.2f GiB) "
                "in %.3f seconds",
                self._staging_prealloc_slots,
                self._staging_chunks_per_slot,
                allocated_bytes / (1024**3),
                time.perf_counter() - started_at,
            )

    def _after_register(self) -> None:
        """Allocate pinned staging after the registered KV layout is available."""
        self._ensure_pinned_staging_preallocated()

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor]:
        """Allocate pinned (page-locked) staging tensors for GPU->CPU copies.

        Tensors are reused from the pool when available to avoid repeated
        allocations on the hot path.

        Args:
            shape: Tensor shape to allocate.
            dtype: Tensor dtype to allocate.
            count: Number of tensors needed.

        Returns:
            List of ``count`` pinned CPU tensors.
        """
        key = (tuple(shape), dtype)
        with self._inflight_lock:
            pooled = self._staging_pool.setdefault(key, [])
            staged = [pooled.pop() for _ in range(min(len(pooled), count))]
        if len(staged) == count:
            return staged

        missing = count - len(staged)
        for _ in range(missing):
            try:
                staged.append(
                    torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                )
            except RuntimeError:
                # Graceful fallback for CPU-only / pin-memory-disabled setups.
                logger.warning(
                    "Falling back to non-pinned CPU staging buffer "
                    "(shape=%s, dtype=%s)",
                    tuple(shape),
                    dtype,
                )
                staged.append(torch.empty(shape, dtype=dtype, device="cpu"))
        return staged

    def _release_staging(self, chunks: list[torch.Tensor]) -> None:
        """Return staging tensors to the pool for reuse.

        Args:
            chunks: Tensors previously obtained from :meth:`_alloc_pinned_staging`.
        """
        if not chunks:
            return
        key = (tuple(chunks[0].shape), chunks[0].dtype)
        with self._inflight_lock:
            self._staging_pool.setdefault(key, []).extend(chunks)

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
        Returns an unresolved future. Normally it resolves after all three
        phases complete. When pinned SHM staging is enabled, it resolves after
        the gather event completes; the staging-to-SHM copy and commit remain
        owned by the background executor and are drained by :meth:`close`.

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
        profile_store = os.environ.get("LMCACHE_PROFILE_PAGED_GATHER") == "1"
        submitted_at = time.perf_counter()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            full_block_ids = _single_group_block_ids(block_ids)

            def _prepare_gather_and_commit() -> None:
                task_started = time.perf_counter()
                copy_stream = (
                    torch_dev.Stream()
                    if self._use_separate_copy_streams
                    else self._copy_stream
                )
                gather_done: Any | None = None
                ok = False
                # Whether we gathered directly into SHM views (True) or into
                # pinned staging buffers that need to be released later (False).
                used_shm_direct = False
                copy_staging_to_shm = False
                staged_chunks: list[torch.Tensor] = []
                try:
                    # --- Phase 1: prepare_store ---
                    # In pickle mode this is the costliest step (sync RPC
                    # round-trip).  Running it here keeps the forward thread free.
                    result = engine_driven_context.prepare_store(key, instance_id)
                    prepare_finished = time.perf_counter()
                    out_buffers, chunk_indices = (
                        result if result is not None else (None, None)
                    )

                    if chunk_indices is not None and len(chunk_indices) == 0:
                        # All chunks are already in cache: no gather, no commit.
                        ok = True
                        return

                    num_chunks = (
                        len(chunk_indices)
                        if chunk_indices is not None
                        else len(full_block_ids) // blocks_in_chunk
                    )

                    # Determine gather target:
                    # - SHM path (out_buffers available): gather into SHM views
                    # - Pickle path (no out_buffers): gather into pinned staging
                    if out_buffers is not None and self._use_pinned_shm_staging:
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
                        staged_chunks = self._alloc_pinned_staging(
                            first_buffer.shape,
                            first_buffer.dtype,
                            len(out_buffers),
                        )
                        gather_target = staged_chunks
                        copy_staging_to_shm = True
                    elif out_buffers is not None:
                        gather_target = out_buffers
                        used_shm_direct = True
                    else:
                        layout_desc = engine_driven_context.layout_desc
                        if not layout_desc.shapes:
                            raise RuntimeError(
                                "engine-driven layout_desc.shapes is empty"
                            )
                        if not layout_desc.dtypes:
                            raise RuntimeError(
                                "engine-driven layout_desc.dtypes is empty"
                            )
                        staged_chunks = self._alloc_pinned_staging(
                            layout_desc.shapes[0],
                            layout_desc.dtypes[0],
                            num_chunks,
                        )
                        gather_target = staged_chunks

                    # --- Phase 2: gather (GPU->CPU copy on copy stream) ---
                    with torch.inference_mode(), torch_dev.stream(copy_stream):
                        _event.wait(stream=copy_stream)

                        gather_paged_kv_to_cpu(
                            kv_caches,
                            full_block_ids,
                            blocks_in_chunk,
                            layout_hints=self._layout_hints,
                            engine_kv_format=self._engine_kv_format,
                            out=gather_target,
                            chunk_indices=chunk_indices,
                        )
                        gather_enqueued = time.perf_counter()

                        gather_done = torch_dev.Event()
                        gather_done.record(copy_stream)

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()

                    if gather_done is not None:
                        gather_done.synchronize()
                    gather_finished = time.perf_counter()

                    if copy_staging_to_shm:
                        # The source GPU KV is no longer needed. Let vLLM
                        # release its blocks while LMCache finishes the CPU-only
                        # SHM copy and commit in this executor task.
                        if out_buffers is None:
                            raise RuntimeError(
                                "Pinned SHM staging requires destination buffers"
                            )
                        completion.set_result(True)
                        for dst, src in zip(out_buffers, staged_chunks, strict=True):
                            dst.copy_(src)
                    staging_copy_finished = time.perf_counter()

                    # --- Phase 3: commit ---
                    commit_buffers = (
                        out_buffers if copy_staging_to_shm else gather_target
                    )
                    if commit_buffers is None:
                        raise RuntimeError("Engine-driven store has no commit buffers")
                    with self._commit_lock:
                        ok = engine_driven_context.commit_store(
                            key,
                            instance_id,
                            commit_buffers,
                        )
                    commit_finished = time.perf_counter()

                    if profile_store:
                        logger.info(
                            "Async store profile: request_id=%s queue=%.3fs "
                            "prepare=%.3fs gather_enqueue=%.3fs "
                            "gather_sync=%.3fs commit=%.3fs total=%.3fs "
                            "staging_copy=%.3fs",
                            _request_id,
                            task_started - submitted_at,
                            prepare_finished - task_started,
                            gather_enqueued - prepare_finished,
                            gather_finished - gather_enqueued,
                            commit_finished - staging_copy_finished,
                            commit_finished - submitted_at,
                            staging_copy_finished - gather_finished,
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
                    if not used_shm_direct:
                        self._release_staging(staged_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    if not completion.query():
                        completion.set_result(ok)

            # Submitting the task is the ownership-transfer point: once it
            # succeeds, the closure is solely responsible for releasing staging
            # buffers and resolving the future. The except below therefore only
            # handles failures that occur *before* this submit.
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
