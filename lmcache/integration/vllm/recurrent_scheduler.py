# SPDX-License-Identifier: Apache-2.0
"""Coordinate optional historical state restoration from the scheduler."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.integration.vllm.recurrent_allocate import allocate_recurrent_destinations
from lmcache.integration.vllm.recurrent_destinations import RecurrentDestinations
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.transport.recurrent_history import (
    HISTORY_CONFIG,
    HistoryCopy,
    HistoryRequest,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

logger = init_logger(__name__)


class RecurrentHistoryScheduler:
    """Own optional history plans until every worker finishes the receive."""

    def __init__(
        self, pool: BlockPool, adapter: LMCacheMPSchedulerAdapter, groups: list[int]
    ) -> None:
        self.pool, self.adapter, self.groups = pool, adapter, groups
        self.block_size = pool.hash_block_size
        self.pending: dict[str, RecurrentDestinations] = {}
        self.enabled = True

    def prepare(self, request: Request, tracker: LMCacheMPRequestTracker) -> None:
        """Plan an extra history restore when this request needs a primary load."""
        if not self.enabled or not tracker.is_ready_for_retrieving():
            return
        if request.request_id in self.pending:
            return
        try:
            self._prepare(request, tracker)
        except (RuntimeError, ValueError, TimeoutError):
            self.enabled = False
            logger.exception("Recurrent history planning disabled after failure")

    def settle(self, request_id: str, cancel: bool = False) -> None:
        """Settle a finished receive or cancel before request tracker removal."""
        pending = self.pending.get(request_id)
        if pending is None:
            return
        pending.canceled |= cancel
        pending.receive_completed |= not cancel
        action = "cancel" if pending.canceled else "status"
        try:
            outcome = self.adapter.recurrent_history(
                HistoryRequest(action=action, lease=pending.plan.lease)
            )
            if pending.settle(outcome):
                self.pending.pop(request_id)
                self.adapter.recurrent_history(
                    HistoryRequest(action="acknowledge", lease=pending.plan.lease)
                )
            elif not cancel:
                self.enabled = False
                logger.error("History copy is unfenced after receive %s", request_id)
        except (RuntimeError, ValueError, TimeoutError):
            self.enabled = False
            logger.exception("Recurrent history completion failed for %s", request_id)
            if pending.released:
                self.pending.pop(request_id, None)

    def poll(self) -> None:
        """Retry fenced cleanup for canceled or completed receives.

        Called at scheduler steps so a transient control timeout does not leave
        private blocks pinned permanently. Pending active receives are untouched.
        """
        for request_id, pending in list(self.pending.items()):
            if pending.canceled or pending.receive_completed:
                self.settle(request_id, cancel=pending.canceled)

    def _prepare(self, request: Request, tracker: LMCacheMPRequestTracker) -> None:
        end = tracker.num_lmcache_hit_tokens
        # A primary load can end in an unfull chunk. History requires an
        # aligned prefix key and can be skipped independently of that load.
        if end <= self.block_size or end % self.block_size:
            return
        already_local, local_pins = [], []
        for pos in range(self.block_size, end, self.block_size):
            blocks = self.pool.get_cached_block(
                request.block_hashes[pos // self.block_size - 1], self.groups
            )
            if blocks:
                already_local.append(pos)
                local_pins.extend(blocks)
        key = IPCCacheServerKey(
            model_name=self.adapter.model_name,
            world_size=self.adapter.world_size,
            worker_id=None,
            token_bytes=tracker.packed_token_ids(),
            start=0,
            end=end,
            request_id=request.request_id,
            cache_salt=tracker.cache_salt,
            request_configs=tracker.request_configs,
        )
        plan = self.adapter.recurrent_history(
            HistoryRequest(action="prepare", key=key, already_local=already_local)
        )
        if not plan.lease:
            return
        try:
            if plan.groups != self.groups:
                raise ValueError("Server recurrent engine groups differ from vLLM")
            if (
                not plan.positions
                or sorted(set(plan.positions)) != plan.positions
                or any(
                    pos <= 0 or pos >= end or pos % self.block_size
                    for pos in plan.positions
                )
            ):
                raise ValueError("Server checkpoints lie outside the restore prefix")
            pending = allocate_recurrent_destinations(
                self.pool, request, self.block_size, plan, local_pins
            )
        except (MemoryError, ValueError):
            self.adapter.recurrent_history(
                HistoryRequest(action="cancel", lease=plan.lease)
            )
            return
        self.pending[request.request_id] = pending
        tracker.request_configs = dict(tracker.request_configs or {})
        tracker.request_configs[HISTORY_CONFIG] = HistoryCopy(
            lease=plan.lease, destinations=pending.block_ids()
        )
