# SPDX-License-Identifier: Apache-2.0
"""Native ATOM client adapters for LMCache multiprocess mode.

This module contains only the transport surface ATOM needs: scheduler lookup
and lock cleanup, plus worker registration and asynchronous store/retrieve.
ATOM-specific cache geometry stays in the ATOM connector; engine-neutral
transport and device-event futures stay under :mod:`lmcache.v1.multiprocess`.
"""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import threading
import uuid

# Third Party
import torch
import zmq

# First Party
from lmcache.utils import EngineType, init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context import (
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)

DEFAULT_MQ_TIMEOUT = 300.0
DEFAULT_HEARTBEAT_INTERVAL = 10.0


class _IpcEvent(Protocol):
    """Device event accepted by the multiprocess transfer context."""

    def ipc_handle(self) -> Any: ...

    def wait(self, stream: object | None = None) -> None: ...


def _send_request(
    client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """Submit one typed request to an LMCache multiprocess server."""
    return client.submit_request(
        request_type,
        payloads,
        get_response_class(request_type),
    )


def _get_chunk_size(client: MessageQueueClient, timeout: float) -> int:
    """Read the server's configured token chunk size."""
    return int(_send_request(client, RequestType.GET_CHUNK_SIZE, []).result(timeout))


class _HeartbeatThread(PeriodicThread):
    """Keep track of the active LMCache server's health."""

    def __init__(
        self,
        client: MessageQueueClient,
        health_event: threading.Event,
        instance_id: int,
        interval: float,
    ) -> None:
        super().__init__(
            name="lmcache-atom-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._client = client
        self._health_event = health_event
        self._instance_id = instance_id
        self._timeout = interval
        self._recover_callback: Callable[[], bool] = lambda: True
        self._unhealthy_callback: Callable[[], None] = lambda: None
        self._healthy_callback: Callable[[], bool] = self._set_health_event

    def _set_health_event(self) -> bool:
        self._health_event.set()
        return True

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        """Run ``callback`` before publishing recovery from an outage."""
        self._recover_callback = callback

    def register_unhealthy_callback(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once when a healthy server becomes unhealthy."""
        self._unhealthy_callback = callback

    def register_healthy_callback(self, callback: Callable[[], bool]) -> None:
        """Publish health through ``callback`` after any needed recovery."""
        self._healthy_callback = callback

    def _publish_unhealthy(self) -> None:
        """Publish one healthy-to-unhealthy edge and clear the probe event."""
        self._unhealthy_callback()
        self._health_event.clear()

    def stop_and_wait(self) -> None:
        """Request stop and join even when one callback exceeds the soft timeout."""
        self.stop()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join()

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        try:
            healthy = bool(
                _send_request(
                    self._client,
                    RequestType.PING,
                    [self._instance_id],
                ).result(timeout=self._timeout)
            )
        except Exception:
            healthy = False
        if self.stop_requested:
            return ThreadRunSummary(success=True, message="stopping")

        if healthy and not was_healthy:
            try:
                healthy = self._recover_callback()
            except Exception:
                logger.exception("ATOM LMCache recovery callback failed")
                healthy = False

        if self.stop_requested:
            return ThreadRunSummary(success=True, message="stopping")
        if healthy:
            try:
                healthy = self._healthy_callback()
            except Exception:
                logger.exception("ATOM LMCache healthy callback failed")
                healthy = False
        if not healthy and self._health_event.is_set():
            self._publish_unhealthy()
        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass(frozen=True)
class AtomMPParallelConfig:
    """ATOM rank information used to construct LMCache keys."""

    world_size: int
    worker_id: int
    tp_size: int


@dataclass
class AtomMPTransferSpec:
    """One ATOM token range and its engine-side block IDs."""

    token_ids: list[int]
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0


class AtomMPSchedulerAdapter:
    """Scheduler-side lookup client for an ATOM deployment."""

    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        block_size: int,
        parallel_config: AtomMPParallelConfig,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
    ) -> None:
        client = MessageQueueClient(server_url, context)
        self._model_name = model_name
        self._parallel = parallel_config
        self._mq_timeout = mq_timeout
        try:
            self.lmcache_tokens_per_chunk = _get_chunk_size(client, mq_timeout)
            if self.lmcache_tokens_per_chunk % block_size:
                raise ValueError(
                    "LMCache chunk size must be divisible by ATOM block size, got "
                    f"{self.lmcache_tokens_per_chunk} and {block_size}"
                )
        except Exception:
            try:
                client.close()
            except Exception:
                logger.warning(
                    "Failed to close ATOM scheduler MQ client after init failure",
                    exc_info=True,
                )
            raise
        self._client = client
        self._closed = False
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
    ) -> None:
        """Submit a lookup once for ``request_id``.

        Args:
            request_id: ATOM request identifier.
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
        _send_request(
            self._client,
            RequestType.LOOKUP,
            [key, self._parallel.tp_size],
        ).result(timeout=self._mq_timeout)
        self._pending_lookups.add(request_id)

    def check_lookup_result(self, request_id: str) -> int | None:
        """Return matched tokens, ``None`` while prefetch is still pending."""
        if request_id in self._lookup_results:
            return self._lookup_results[request_id]
        if request_id not in self._pending_lookups:
            return 0
        result = _send_request(
            self._client,
            RequestType.QUERY_PREFETCH_STATUS,
            [request_id],
        ).result(timeout=self._mq_timeout)
        if result is None:
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
        """Release lookup locks for an ATOM token range."""
        if start >= end:
            return
        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            worker_id=None,
        )
        _send_request(
            self._client,
            RequestType.FREE_LOOKUP_LOCKS,
            [key, self._parallel.tp_size],
        )

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Discard client-side lookup state after handoff or cleanup."""
        self._pending_lookups.discard(request_id)
        self._lookup_results.pop(request_id, None)

    def end_session(self, request_id: str) -> None:
        """Ask the LMCache server to release request-scoped state."""
        _send_request(self._client, RequestType.END_SESSION, [request_id])

    def shutdown(self) -> None:
        """Close the scheduler-side message queue client."""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.close()
        except Exception:
            logger.warning("Failed to close ATOM scheduler MQ client", exc_info=True)

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        worker_id: int | None,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._parallel.world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )


class AtomMPWorkerAdapter:
    """Worker-side ATOM registration and asynchronous transfer client."""

    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        block_size: int,
        parallel_config: AtomMPParallelConfig,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        transfer_mode: str | None = None,
    ) -> None:
        client = MessageQueueClient(server_url, context)
        self._model_name = model_name
        self._block_size = block_size
        self._parallel = parallel_config
        self._mq_timeout = mq_timeout
        self._heartbeat_interval = heartbeat_interval
        self._transfer_mode = transfer_mode
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        try:
            self.lmcache_tokens_per_chunk = _get_chunk_size(client, mq_timeout)
            if self.lmcache_tokens_per_chunk % block_size:
                raise ValueError(
                    "LMCache chunk size must be divisible by ATOM block size, got "
                    f"{self.lmcache_tokens_per_chunk} and {block_size}"
                )
        except Exception:
            try:
                client.close()
            except Exception:
                logger.warning(
                    "Failed to close ATOM worker MQ client after init failure",
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
        self._heartbeat: _HeartbeatThread | None = None

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
        """Register ATOM cache views with the LMCache server."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("ATOM LMCache worker adapter is shut down")
            if self._public_registration_started:
                raise RuntimeError(
                    "ATOM KV caches are already being or were registered"
                )
            self._public_registration_started = True
            self._kv_caches = kv_caches
            self._engine_group_infos = list(engine_group_infos)
            generation = self._lifecycle_generation
        try:
            if not self._register_kv_caches(generation):
                raise RuntimeError("ATOM LMCache registration was cancelled")
            if not self._mark_healthy():
                raise RuntimeError("ATOM LMCache worker adapter is shutting down")

            with self._state_lock:
                if self._closed or generation != self._lifecycle_generation:
                    raise RuntimeError("ATOM LMCache worker adapter is shutting down")
                heartbeat = _HeartbeatThread(
                    self._client,
                    self._health_event,
                    self.instance_id,
                    self._heartbeat_interval,
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
        spec: AtomMPTransferSpec,
        event: _IpcEvent,
    ) -> MessagingFuture[bool] | None:
        """Submit an asynchronous ATOM store.

        ``None`` means the request was dropped before submission because the
        server was unhealthy. The ATOM connector treats that as a completed
        save opportunity, matching vLLM's degraded-mode behavior.
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
            future = context.submit_store(
                request_id,
                self._create_key(request_id, spec),
                self.instance_id,
                kv_caches,
                block_ids,
                event,
                self.blocks_in_chunk,
            )
            future.retain_reference(event)
            self._track_operation_future(context, future)
        finally:
            self._release_submission_lease(context)
        return future

    def submit_retrieve_request(
        self,
        request_id: str,
        spec: AtomMPTransferSpec,
        event: _IpcEvent,
    ) -> MessagingFuture[bool] | None:
        """Submit an asynchronous ATOM retrieve.

        ``None`` means no transfer was submitted. The ATOM connector reports
        that load as failed so the scheduler recomputes into its existing
        allocated blocks, as vLLM does for failed asynchronous loads.
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
            future = context.submit_retrieve(
                request_id,
                self._create_key(request_id, spec),
                self.instance_id,
                kv_caches,
                block_ids,
                event,
                self.blocks_in_chunk,
            )
            future.retain_reference(event)
            self._track_operation_future(context, future)
        finally:
            self._release_submission_lease(context)
        return future

    def shutdown(self) -> None:
        """Stop heartbeat, unregister ATOM cache views, and close the client."""
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
                        "Failed while joining ATOM heartbeat during shutdown",
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
                    _send_request(
                        self._client,
                        RequestType.UNREGISTER_KV_CACHE,
                        [self.instance_id],
                    ).result(timeout=self._mq_timeout)
                except Exception:
                    logger.warning(
                        "ATOM LMCache unregister failed during shutdown",
                        exc_info=True,
                    )
        finally:
            if transfer_context is not None:
                try:
                    transfer_context.close()
                except Exception:
                    logger.warning(
                        "Failed to close ATOM transfer context during shutdown",
                        exc_info=True,
                    )
            try:
                self._client.close()
            except Exception:
                logger.warning(
                    "Failed to close ATOM worker MQ client during shutdown",
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
                mode=self._transfer_mode,
            )
            transfer_context.register(
                self.instance_id,
                kv_caches,
                self._model_name,
                self._parallel.world_size,
                self.blocks_in_chunk,
                self._client,
                self._mq_timeout,
                _send_request,
                layout_hints={},
                engine_group_infos=engine_group_infos,
                engine_type=EngineType.ATOM,
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
                    "Failed to close rejected ATOM transfer context",
                    exc_info=True,
                )
            finally:
                self._finish_registration_attempt()
            raise

        assert transfer_context is not None
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
                            "Failed to close superseded ATOM transfer context",
                            exc_info=True,
                        )
                return True
            finally:
                self._finish_registration_attempt()

        # shutdown may finish before a blocking REGISTER returns. Remove the
        # late server registration and never publish its local context.
        try:
            try:
                _send_request(
                    self._client,
                    RequestType.UNREGISTER_KV_CACHE,
                    [self.instance_id],
                ).result(timeout=self._mq_timeout)
            except Exception:
                logger.warning(
                    "Failed to remove late ATOM LMCache registration",
                    exc_info=True,
                )
        finally:
            try:
                transfer_context.close()
            except Exception:
                logger.warning(
                    "Failed to close late ATOM transfer context",
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
                "Failed to re-register ATOM KV caches after LMCache recovery"
            )
            return False
        if not registered:
            return False
        logger.warning("Re-registered ATOM KV caches after LMCache recovery")
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
                    "ATOM transfer ended with an error during shutdown",
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
            _send_request(
                self._client,
                RequestType.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except Exception:
            logger.warning(
                "Failed to roll back rejected ATOM LMCache registration",
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
                "ATOM KV caches are not registered; call register_kv_caches() first"
            )
        return transfer_context

    def _create_key(
        self,
        request_id: str,
        spec: AtomMPTransferSpec,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._parallel.world_size,
            worker_id=self._parallel.worker_id,
            token_ids=tuple(spec.token_ids),
            start=spec.start,
            end=spec.end,
            request_id=request_id,
        )
