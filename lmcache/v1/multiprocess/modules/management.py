# SPDX-License-Identifier: Apache-2.0
"""Management and utility operations for the MPCacheServer."""

# Standard
from collections.abc import Sequence
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


class ManagementModule:
    """Handles management and utility operations for the cache engine.

    Owns the lock used during cache clearing and provides handlers for
    ping, chunk-size queries, clear, debug, and block-allocation reporting.
    Also owns the periodic reaper that evicts workers which have gone
    silent, driving the injected liveness targets.

    Args:
        ctx: The shared engine context.
        liveness_targets: Modules the reaper drives -- the transfer modules
            whose per-context registrations are refreshed on PING and scanned
            for staleness, plus any state mirror (e.g. ``BlendModule``)
            notified via ``drop_instance_state`` when an instance is reaped.
        mirror_state_owner: The liveness target whose reaps invalidate mirrored
            per-instance state. Reaps from other contexts with the same
            ``instance_id`` do not notify mirrors.
        worker_reap_timeout_seconds: Silence budget for a registered context
            after it observes both PING and real transfer activity; 0 disables
            reaping.
        worker_registration_grace_seconds: Per-context silence budget before
            both signals.
        experimental_transfer: Types of experimental intermediate tensor
            transfer built in the server.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        liveness_targets: Sequence[InstanceLivenessTarget] = (),
        mirror_state_owner: InstanceLivenessTarget | None = None,
        worker_reap_timeout_seconds: float = 0.0,
        worker_registration_grace_seconds: float = 0.0,
        experimental_transfer: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        self._clear_lock = threading.Lock()
        self._liveness_targets = tuple(liveness_targets)
        self._mirror_state_owner = mirror_state_owner
        self._reap_timeout = worker_reap_timeout_seconds
        self._reap_grace = worker_registration_grace_seconds
        self._experimental_transfer = tuple(experimental_transfer)

        # Periodic reaper, started only when reaping is enabled and there is
        # something to scan. Scans every reap_timeout/4, so an instance is
        # reaped between timeout and timeout + interval after its last signal.
        self._reaper: PeriodicThread | None = None
        if self._reap_timeout > 0 and self._liveness_targets:
            reaper = create_periodic_thread(
                name="lmcache-mp-worker-reaper",
                interval=self._reap_timeout / 4,
                execute_fn=self._reap_cycle,
                level=ThreadLevel.MEDIUM,
            )
            reaper.start()
            self._reaper = reaper

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(RequestType.CLEAR, self.clear, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.GET_CHUNK_SIZE,
                self.get_chunk_size,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.GET_EXPERIMENTAL,
                self.get_experimental,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.PING, self.ping, ThreadPoolType.NORMAL),
            HandlerSpec(RequestType.NOOP, self.debug, ThreadPoolType.SYNC),
            HandlerSpec(
                RequestType.REPORT_BLOCK_ALLOCATION,
                self.report_block_allocations,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict:
        """Return module-specific status information.

        Returns:
            A dict with a ``worker_liveness`` summary when reaping targets
            are present, otherwise empty. The compatibility key
            ``tracked_instances`` counts registered contexts and may include
            the same instance ID more than once.
        """
        if not self._liveness_targets:
            return {}
        # Preserve the existing status key for compatibility. The value counts
        # registered contexts, so one instance ID may contribute more than one.
        tracked_registrations = sum(
            t.tracked_instance_count() for t in self._liveness_targets
        )
        return {
            "worker_liveness": {
                "enabled": self._reaper is not None,
                "reap_timeout_seconds": self._reap_timeout,
                "registration_grace_seconds": self._reap_grace,
                "tracked_instances": tracked_registrations,
            }
        }

    def close(self) -> None:
        """Stop the reaper, if one is running."""
        if self._reaper is not None:
            self._reaper.stop()

    def ping(self, instance_id: int | None) -> bool:
        """Respond to a ping and refresh the sender's liveness.

        Args:
            instance_id: The sender's worker instance ID, or None for an
                untracked prober (the scheduler adapter). When not None, the
                worker's last-seen time is refreshed on every liveness target.

        Returns:
            Always True.
        """
        if instance_id is not None:
            for target in self._liveness_targets:
                target.touch_instance(instance_id)
        return True

    def _reap_cycle(self) -> ThreadRunSummary:
        """Run one reaper scan: reap stale contexts and drop owned mirrors.

        Only reaps from ``mirror_state_owner`` invalidate mirrored state. This
        keeps an independently activated context (for example QStore) from
        deleting state owned by a still-protected GPU KV context with the same
        instance ID.

        Returns:
            A summary recording how many registrations were reaped this scan.
        """
        reaped: list[int] = []
        mirror_reaped: list[int] = []
        for target in self._liveness_targets:
            target_reaped = target.reap_stale_instances(
                self._reap_timeout, self._reap_grace
            )
            reaped.extend(target_reaped)
            if target is self._mirror_state_owner:
                mirror_reaped.extend(target_reaped)
        for instance_id in mirror_reaped:
            for target in self._liveness_targets:
                target.drop_instance_state(instance_id)
        return ThreadRunSummary(success=True, message=f"reaped={len(reaped)}")

    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations.

        Returns:
            The chunk size.
        """
        return self._ctx.chunk_size

    def get_experimental(self) -> list[str]:
        """Return the experimental intermediate tensor transfer built in the
        server.

        Returns:
            The enabled experimental intermediate tensor transfer types.
            See ``lmcache.v1.multiprocess.modules.experimental.__init__``.
        """
        return list(self._experimental_transfer)

    def clear(self) -> None:
        """Clear all stored KV cache data from the storage manager."""
        with self._clear_lock:
            self._ctx.storage_manager.memcheck()
            self._ctx.storage_manager.clear(force=True)
            self._ctx.storage_manager.memcheck()

    def debug(self) -> str:
        """Return a simple health-check string.

        Returns:
            The literal string ``"OK"``.
        """
        return "OK"

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )
