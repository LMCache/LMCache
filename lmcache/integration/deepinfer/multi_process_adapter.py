# SPDX-License-Identifier: Apache-2.0
"""Native DeepInfer client adapters for LMCache multiprocess mode.

This module contains only the transport surface DeepInfer needs: scheduler lookup
and lock cleanup, plus worker registration and asynchronous store/retrieve.
DeepInfer-specific cache geometry stays in the DeepInfer connector; engine-neutral
transport and device-event futures stay under :mod:`lmcache.v1.multiprocess`.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import threading
import time
import uuid

# Third Party
import torch
import zmq

# First Party
from lmcache.utils import EngineType, init_logger
from lmcache.v1.client.heartbeat import HeartbeatThread
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.client.worker_transfer import (
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
)
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)

DEFAULT_MQ_TIMEOUT = 300.0
DEFAULT_HEARTBEAT_INTERVAL = 10.0


def _get_chunk_size(client: RequestClient, timeout: float) -> int:
    """Read the server's configured token chunk size."""
    return int(client.get_chunk_size().result(timeout))


@dataclass(frozen=True)
class DeepInferMPParallelConfig:
    """Map DeepInfer execution ranks onto logical LMCache KV ranks.

    ``engine_*`` describes the workers that execute inference. ``kv_*``
    identifies those workers in LMCache. Replicated MLA groups retain the full
    rank geometry and declare their sharing through ``tp_replicas`` at
    registration time.
    """

    engine_world_size: int
    engine_worker_id: int
    kv_world_size: int
    kv_worker_id: int
    tp_replicas: int

    def __post_init__(self) -> None:
        if self.engine_world_size <= 0:
            raise ValueError("DeepInfer engine world size must be positive")
        if not 0 <= self.engine_worker_id < self.engine_world_size:
            raise ValueError(
                "DeepInfer engine worker id must be in "
                f"[0, {self.engine_world_size}), got {self.engine_worker_id}"
            )
        if self.kv_world_size <= 0:
            raise ValueError("DeepInfer KV world size must be positive")
        if not 0 <= self.kv_worker_id < self.kv_world_size:
            raise ValueError(
                "DeepInfer KV worker id must be in "
                f"[0, {self.kv_world_size}), got {self.kv_worker_id}"
            )
        if self.tp_replicas <= 0:
            raise ValueError("DeepInfer TP replica count must be positive")

    @classmethod
    def for_model(
        cls,
        engine_world_size: int,
        engine_worker_id: int,
        *,
        is_mla: bool,
    ) -> "DeepInferMPParallelConfig":
        """Map execution ranks to KV objects for the active cache layout."""
        if not is_mla:
            return cls(
                engine_world_size=engine_world_size,
                engine_worker_id=engine_worker_id,
                kv_world_size=engine_world_size,
                kv_worker_id=engine_worker_id,
                tp_replicas=1,
            )
        return cls(
            engine_world_size=engine_world_size,
            engine_worker_id=engine_worker_id,
            kv_world_size=engine_world_size,
            kv_worker_id=engine_worker_id,
            tp_replicas=engine_world_size,
        )

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank is the canonical writer for its replicated group."""
        return self.kv_worker_id % self.tp_replicas == 0


@dataclass
class DeepInferMPTransferSpec:
    """One DeepInfer token range and its engine-side block IDs."""

    token_ids: list[int]
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0


@dataclass
class _DeepInferLookupState:
    """Non-blocking LOOKUP acknowledgement and prefetch-status RPCs."""

    key: IPCCacheServerKey
    lookup_future: MessagingFuture[Any]
    started_at: float
    status_future: MessagingFuture[Any] | None = None


class DeepInferMPSchedulerAdapter:
    """Scheduler-side lookup client for a DeepInfer deployment."""

    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        block_size: int,
        parallel_config: DeepInferMPParallelConfig,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
    ) -> None:
        client = RequestClientFactory.create(server_url, context=context)
        self._model_name = model_name
        self._parallel = parallel_config
        self._mq_timeout = mq_timeout
        try:
            self.lmcache_tokens_per_chunk = _get_chunk_size(client, mq_timeout)
            if self.lmcache_tokens_per_chunk % block_size:
                raise ValueError(
                    "LMCache chunk size must be divisible by DeepInfer block size, got "
                    f"{self.lmcache_tokens_per_chunk} and {block_size}"
                )
        except Exception:
            try:
                client.close()
            except Exception:
                logger.warning(
                    "Failed to close DeepInfer scheduler MQ client after init failure",
                    exc_info=True,
                )
            raise
        self._client = client
        self._closed = False
        self._pending_lookups: dict[str, _DeepInferLookupState] = {}
        self._lookup_results: dict[str, int] = {}

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
    ) -> None:
        """Submit a lookup once for ``request_id``.

        Args:
            request_id: DeepInfer request identifier.
            token_ids: Full prompt token sequence.
        """
        if request_id in self._pending_lookups:
            return
        aligned_end = (
            len(token_ids) // self.lmcache_tokens_per_chunk
        ) * self.lmcache_tokens_per_chunk
        if aligned_end == 0:
            self._lookup_results[request_id] = 0
            return
        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            worker_id=None,
        )
        self._pending_lookups[request_id] = _DeepInferLookupState(
            key=key,
            lookup_future=self._client.lookup(key, self._parallel.engine_world_size),
            started_at=time.monotonic(),
        )

    def check_lookup_result(self, request_id: str) -> int | None:
        """Return matched tokens, ``None`` while prefetch is still pending."""
        if request_id in self._lookup_results:
            return self._lookup_results[request_id]
        state = self._pending_lookups.get(request_id)
        if state is None:
            return 0

        if time.monotonic() - state.started_at >= self._mq_timeout:
            logger.warning("DeepInfer LMCache lookup timed out for %s", request_id)
            self._pending_lookups.pop(request_id, None)
            self._lookup_results[request_id] = 0
            return 0

        try:
            if not state.lookup_future.query():
                return None
            state.lookup_future.result(timeout=0)
            if state.status_future is None:
                state.status_future = self._client.query_prefetch_status(request_id)
                return None
            if not state.status_future.query():
                return None
            result = state.status_future.result(timeout=0)
        except Exception:
            logger.exception("DeepInfer LMCache lookup failed for %s", request_id)
            self._pending_lookups.pop(request_id, None)
            self._lookup_results[request_id] = 0
            return 0

        if result is None:
            state.status_future = self._client.query_prefetch_status(request_id)
            return None
        matched_tokens = int(result) * self.lmcache_tokens_per_chunk
        self._lookup_results[request_id] = matched_tokens
        return matched_tokens

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        """Release lookup locks for an DeepInfer token range."""
        if start >= end:
            return
        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            worker_id=None,
        )
        self._client.free_lookup_locks(key, self._parallel.engine_world_size)

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Discard client-side lookup state after handoff or cleanup."""
        self._pending_lookups.pop(request_id, None)
        self._lookup_results.pop(request_id, None)

    def end_session(self, request_id: str) -> None:
        """Ask the LMCache server to release request-scoped state."""
        self._client.end_session(request_id)

    def shutdown(self) -> None:
        """Close the scheduler-side request client."""
        if self._closed:
            return
        self._closed = True
        self._pending_lookups.clear()
        self._lookup_results.clear()
        try:
            self._client.close()
        except Exception:
            logger.warning(
                "Failed to close DeepInfer scheduler MQ client",
                exc_info=True,
            )

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        worker_id: int | None,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey.from_token_ids(
            model_name=self._model_name,
            world_size=self._parallel.kv_world_size,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
        )


class DeepInferMPWorkerAdapter:
    """Worker-side DeepInfer registration and asynchronous transfer client."""

    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        block_size: int,
        parallel_config: DeepInferMPParallelConfig,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        transfer_mode: str | None = None,
    ) -> None:
        client = RequestClientFactory.create(server_url, context=context)
        self._model_name = model_name
        self._parallel = parallel_config
        self._mq_timeout = mq_timeout
        self._heartbeat_interval = heartbeat_interval
        self._transfer_mode = transfer_mode
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        try:
            self.lmcache_tokens_per_chunk = _get_chunk_size(client, mq_timeout)
            if self.lmcache_tokens_per_chunk % block_size:
                raise ValueError(
                    "LMCache chunk size must be divisible by DeepInfer block size, got "
                    f"{self.lmcache_tokens_per_chunk} and {block_size}"
                )
        except Exception:
            try:
                client.close()
            except Exception:
                logger.warning(
                    "Failed to close DeepInfer worker MQ client after init failure",
                    exc_info=True,
                )
            raise
        self._client = client
        self.blocks_in_chunk = self.lmcache_tokens_per_chunk // block_size
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._transfer_context: TransferContext | None = None
        self._registered = False
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        self._closed = False
        self._lifecycle_generation = 0
        self._registrations_inflight = 0
        self._context_submission_leases: dict[TransferContext, int] = {}
        self._context_operation_futures: dict[
            TransferContext, set[MessagingFuture[bool]]
        ] = {}
        self._public_registration_started = False
        self._shutdown_complete = threading.Event()
        self._health_event = threading.Event()
        self._heartbeat: HeartbeatThread | None = None

    @property
    def is_healthy(self) -> bool:
        """Whether the server is reachable and this worker is registered."""
        with self._state_lock:
            return not self._closed and self._health_event.is_set()

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        *,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register DeepInfer cache views with the LMCache server."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("DeepInfer LMCache worker adapter is shut down")
            if self._public_registration_started:
                raise RuntimeError(
                    "DeepInfer KV caches are already being or were registered"
                )
            self._public_registration_started = True
            self._kv_caches = kv_caches
            self._engine_group_infos = list(engine_group_infos)
            generation = self._lifecycle_generation
        try:
            if not self._register_kv_caches(generation):
                raise RuntimeError("DeepInfer LMCache registration was cancelled")
            if not self._mark_healthy():
                raise RuntimeError("DeepInfer LMCache worker adapter is shutting down")

            with self._state_lock:
                if self._closed or generation != self._lifecycle_generation:
                    raise RuntimeError(
                        "DeepInfer LMCache worker adapter is shutting down"
                    )
                heartbeat = HeartbeatThread(
                    req_client=self._client,
                    health_event=self._health_event,
                    interval=self._heartbeat_interval,
                    instance_id=self.instance_id,
                    name="lmcache-deepinfer-heartbeat",
                )
                heartbeat.register_recover_callback(self._reregister_kv_caches)
                heartbeat.register_unhealthy_callback(self._mark_unhealthy)
                heartbeat.register_healthy_callback(self._mark_healthy)
                self._heartbeat = heartbeat
                # Publish and start under the same lifecycle critical section
                # so shutdown cannot detach an unstarted heartbeat. If start
                # raises, keep it published so shutdown can stop/join even an
                # implementation that launched a thread before raising.
                heartbeat.start()
        except Exception:
            self.shutdown()
            raise

    def submit_store_request(
        self,
        request_id: str,
        spec: DeepInferMPTransferSpec,
    ) -> MessagingFuture[bool] | None:
        """Submit an asynchronous DeepInfer store.

        ``None`` means the request was dropped before submission because the
        server was unhealthy. The DeepInfer connector treats that as a missed
        save opportunity and continues serving the request.
        """
        with self._state_lock:
            if self._closed or not self._health_event.is_set():
                return None
            context = self._require_transfer_context_locked()
            kv_caches = self._kv_caches
            block_ids = expand_engine_block_ids(
                self._engine_group_infos,
                spec.block_ids,
            )
            self._context_submission_leases[context] = (
                self._context_submission_leases.get(context, 0) + 1
            )
        try:
            event = context.create_recorded_event()
            future = context.submit_store(
                request_id,
                self._create_key(request_id, spec),
                kv_caches,
                block_ids,
                event,
                self.blocks_in_chunk,
            )
            if event is not None:
                future.retain_reference(event)
            self._track_operation_future(context, future)
        finally:
            self._release_submission_lease(context)
        return future

    def submit_retrieve_request(
        self,
        request_id: str,
        spec: DeepInferMPTransferSpec,
    ) -> MessagingFuture[bool] | None:
        """Submit an asynchronous DeepInfer retrieve.

        ``None`` means no transfer was submitted. The DeepInfer connector reports
        that load as failed so the scheduler recomputes into its existing
        allocated blocks.
        """
        with self._state_lock:
            if self._closed or not self._health_event.is_set():
                return None
            context = self._require_transfer_context_locked()
            kv_caches = self._kv_caches
            block_ids = expand_engine_block_ids(
                self._engine_group_infos,
                spec.block_ids,
            )
            self._context_submission_leases[context] = (
                self._context_submission_leases.get(context, 0) + 1
            )
        try:
            event = context.create_recorded_event()
            future = context.submit_retrieve(
                request_id,
                self._create_key(request_id, spec),
                kv_caches,
                block_ids,
                event,
                self.blocks_in_chunk,
            )
            if event is not None:
                future.retain_reference(event)
            self._track_operation_future(context, future)
        finally:
            self._release_submission_lease(context)
        return future

    def shutdown(self) -> None:
        """Stop heartbeat, unregister DeepInfer cache views, and close the client."""
        with self._state_lock:
            if self._closed:
                wait_for_shutdown = True
                heartbeat = None
            else:
                wait_for_shutdown = False
                self._closed = True
                self._lifecycle_generation += 1
                self._health_event.clear()
                heartbeat = self._heartbeat
                self._heartbeat = None
        if wait_for_shutdown:
            self._shutdown_complete.wait()
            return

        transfer_context: TransferContext | None = None
        registered = False
        try:
            if heartbeat is not None:
                try:
                    heartbeat.stop_and_wait()
                except Exception:
                    logger.warning(
                        "Failed while joining DeepInfer heartbeat during shutdown",
                        exc_info=True,
                    )

            with self._state_changed:
                while self._registrations_inflight or self._context_submission_leases:
                    self._state_changed.wait()

            self._drain_operation_futures()

            with self._state_changed:
                registered = self._registered
                transfer_context = self._transfer_context
                self._transfer_context = None
                self._registered = False

            if registered and transfer_context is not None:
                try:
                    future = transfer_context.unregister()
                    if future is not None:
                        future.result(timeout=self._mq_timeout)
                except Exception:
                    logger.warning(
                        "DeepInfer LMCache unregister failed during shutdown",
                        exc_info=True,
                    )
        finally:
            if transfer_context is not None:
                try:
                    transfer_context.close()
                except Exception:
                    logger.warning(
                        "Failed to close DeepInfer transfer context during shutdown",
                        exc_info=True,
                    )
            try:
                self._client.close()
            except Exception:
                logger.warning(
                    "Failed to close DeepInfer worker MQ client during shutdown",
                    exc_info=True,
                )
            finally:
                self._shutdown_complete.set()

    def _register_kv_caches(self, expected_generation: int) -> bool:
        """Build and publish a context if its lifecycle generation stays live."""
        with self._state_lock:
            if self._closed or expected_generation != self._lifecycle_generation:
                return False
            kv_caches = self._kv_caches
            engine_group_infos = list(self._engine_group_infos)
            self._registrations_inflight += 1

        transfer_context: TransferContext | None = None
        try:
            transfer_context = create_transfer_context(
                kv_caches,
                instance_id=self.instance_id,
                req_client=self._client,
                mode=self._transfer_mode,
            )
            transfer_context.register(
                kv_caches,
                self._model_name,
                self._parallel.kv_world_size,
                self.blocks_in_chunk,
                self._mq_timeout,
                layout_hints={},
                engine_group_infos=engine_group_infos,
                engine_type=EngineType.DEEPINFER,
                lmcache_chunk_tokens=self.lmcache_tokens_per_chunk,
            )
        except Exception:
            try:
                if transfer_context is not None:
                    # REGISTER completion is ambiguous on timeout. Roll back
                    # the candidate's wire registration before releasing local
                    # resources. During recovery the old local context remains
                    # published (but unhealthy), so a later heartbeat can
                    # register it again if this removes a pre-existing entry.
                    self._rollback_candidate_registration(transfer_context)
                    transfer_context.close()
            except Exception:
                logger.warning(
                    "Failed to close rejected DeepInfer transfer context",
                    exc_info=True,
                )
            finally:
                self._finish_registration_attempt()
            raise

        if transfer_context is None:
            self._finish_registration_attempt()
            raise RuntimeError("DeepInfer transfer context creation returned None")
        with self._state_lock:
            accepted = (
                not self._closed and expected_generation == self._lifecycle_generation
            )
            if accepted:
                previous_context = self._transfer_context
                if (
                    previous_context is not None
                    and previous_context is not transfer_context
                    and self._health_event.is_set()
                ):
                    # A context swap must stop new submissions to the old
                    # context before waiting for its existing leases.
                    self._health_event.clear()
                self._transfer_context = transfer_context
                self._registered = True
            else:
                previous_context = None

        if accepted:
            try:
                if (
                    previous_context is not None
                    and previous_context is not transfer_context
                ):
                    self._wait_for_context_leases(previous_context)
                    try:
                        previous_context.close()
                    except Exception:
                        logger.warning(
                            "Failed to close superseded DeepInfer transfer context",
                            exc_info=True,
                        )
                return True
            finally:
                self._finish_registration_attempt()

        # shutdown may finish before a blocking REGISTER returns. Remove the
        # late server registration and never publish its local context.
        try:
            try:
                future = transfer_context.unregister()
                if future is not None:
                    future.result(timeout=self._mq_timeout)
            except Exception:
                logger.warning(
                    "Failed to remove late DeepInfer LMCache registration",
                    exc_info=True,
                )
        finally:
            try:
                transfer_context.close()
            except Exception:
                logger.warning(
                    "Failed to close late DeepInfer transfer context",
                    exc_info=True,
                )
            finally:
                self._finish_registration_attempt()
        return False

    def _reregister_kv_caches(self) -> bool:
        """Re-register saved cache views before a recovered server is usable."""
        with self._state_lock:
            heartbeat = self._heartbeat
            has_caches = bool(self._kv_caches)
            generation = self._lifecycle_generation
            closed = self._closed
        if not has_caches:
            return not closed
        if closed or heartbeat is None or heartbeat.stop_requested:
            return False
        try:
            registered = self._register_kv_caches(generation)
        except Exception:
            logger.exception(
                "Failed to re-register DeepInfer KV caches after LMCache recovery"
            )
            return False
        if not registered:
            return False
        logger.warning("Re-registered DeepInfer KV caches after LMCache recovery")
        return True

    def _mark_unhealthy(self) -> None:
        """Clear worker health so new transfer submissions are dropped."""
        with self._state_lock:
            self._health_event.clear()

    def _mark_healthy(self) -> bool:
        """Publish health only for a live generation with a registered context."""
        with self._state_lock:
            if self._closed or not self._registered or self._transfer_context is None:
                return False
            self._health_event.set()
            return True

    def _finish_registration_attempt(self) -> None:
        """Release one registration barrier slot and wake shutdown."""
        with self._state_changed:
            self._registrations_inflight -= 1
            self._state_changed.notify_all()

    def _wait_for_context_leases(self, transfer_context: TransferContext) -> None:
        """Wait until no synchronous submission is using ``transfer_context``."""
        with self._state_changed:
            while self._context_submission_leases.get(transfer_context, 0):
                self._state_changed.wait()

    def _track_operation_future(
        self,
        transfer_context: TransferContext,
        future: MessagingFuture[bool],
    ) -> None:
        """Track an operation so shutdown can drain it before unregistering."""
        with self._state_changed:
            tracked = list(self._context_operation_futures.get(transfer_context, set()))

        completed: list[MessagingFuture[bool]] = []
        for pending in tracked:
            try:
                if pending.query():
                    completed.append(pending)
            except Exception:
                # Keep failures tracked so shutdown observes them via result().
                continue

        with self._state_changed:
            futures = self._context_operation_futures.setdefault(
                transfer_context, set()
            )
            futures.difference_update(completed)
            futures.add(future)

    def _drain_operation_futures(self) -> None:
        """Wait for every submitted operation to reach a terminal state."""
        with self._state_changed:
            futures = [
                future
                for context_futures in self._context_operation_futures.values()
                for future in context_futures
            ]

        for future in futures:
            try:
                # DeviceMessagingFuture.result() also synchronizes the exported
                # completion event, so the server no longer uses this worker's
                # GPUCacheContext when unregister is sent below.
                future.result()
            except Exception:
                logger.warning(
                    "DeepInfer transfer ended with an error during shutdown",
                    exc_info=True,
                )

        with self._state_changed:
            self._context_operation_futures.clear()
            self._state_changed.notify_all()

    def _rollback_candidate_registration(
        self,
        transfer_context: TransferContext,
    ) -> None:
        """Best-effort removal of an ambiguously registered candidate."""
        try:
            future = transfer_context.unregister()
            if future is not None:
                future.result(timeout=self._mq_timeout)
        except Exception:
            logger.warning(
                "Failed to roll back rejected DeepInfer LMCache registration",
                exc_info=True,
            )

    def _release_submission_lease(
        self,
        transfer_context: TransferContext,
    ) -> None:
        """Release one context-specific submission lease and wake waiters."""
        with self._state_changed:
            remaining = self._context_submission_leases[transfer_context] - 1
            if remaining:
                self._context_submission_leases[transfer_context] = remaining
            else:
                del self._context_submission_leases[transfer_context]
            self._state_changed.notify_all()

    def _require_transfer_context_locked(self) -> TransferContext:
        """Return the active context while the caller holds ``_state_lock``."""
        transfer_context = self._transfer_context
        if transfer_context is None:
            raise RuntimeError(
                "DeepInfer KV caches are not registered; call "
                "register_kv_caches() first"
            )
        return transfer_context

    def _create_key(
        self,
        request_id: str,
        spec: DeepInferMPTransferSpec,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey.from_token_ids(
            model_name=self._model_name,
            world_size=self._parallel.kv_world_size,
            worker_id=self._parallel.kv_worker_id,
            token_ids=spec.token_ids,
            start=spec.start,
            end=spec.end,
            request_id=request_id,
        )
