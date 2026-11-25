# SPDX-License-Identifier: Apache-2.0
"""Coordinate optional historical state restoration from the scheduler."""

# Standard
from typing import TYPE_CHECKING
import time

# Third Party
import msgspec

# First Party
from lmcache.integration.vllm.recurrent_allocate import allocate_recurrent_destinations
from lmcache.integration.vllm.recurrent_destinations import RecurrentDestinations
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.errors import LMCacheTimeoutError
from lmcache.v1.common.logging import init_logger
from lmcache.v1.transport.recurrent_history import (
    HistoryCopy,
    HistoryRequest,
    HistoryResponse,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )
    from lmcache.v1.transport.futures import MessagingFuture
    from lmcache.v1.transport.server_module import ServerModuleCallResponse

    HistoryFuture = MessagingFuture[ServerModuleCallResponse]
    CheckpointKey = tuple[str, int, BlockHash]

logger = init_logger(__name__)


class RecurrentHistoryScheduler:
    """Own optional history plans until every worker finishes the receive."""

    def __init__(
        self, pool: "BlockPool", adapter: "LMCacheMPSchedulerAdapter", groups: list[int]
    ) -> None:
        self.pool, self.adapter, self.groups = pool, adapter, groups
        self.block_size = pool.hash_block_size
        self.pending: dict[str, RecurrentDestinations] = {}
        self.enabled = True
        self.planning: "dict[str, tuple[HistoryFuture, float, int]]" = {}
        self.plans: dict[str, HistoryResponse] = {}
        self.abandoned: "list[tuple[HistoryFuture, float, int]]" = []
        self.controls: "dict[str, tuple[str, HistoryFuture | None, float]]" = {}
        self.retiring: dict[str, str] = {}
        self.background: "dict[str, tuple[Request, LMCacheMPRequestTracker, int]]" = {}
        self.primary_done: set[str] = set()
        self.dispatch: list[tuple[str, bytes, int, str, bytes]] = []
        self.owners: "dict[CheckpointKey, str]" = {}
        self.owned_keys: "dict[str, list[CheckpointKey]]" = {}

    def _checkpoint_key(self, request, tracker, pos):
        return (
            tracker.cache_salt,
            pos,
            request.block_hashes[pos // self.block_size - 1],
        )

    def _forget_owner(self, rid):
        for key in self.owned_keys.pop(rid, ()):
            self.owners.pop(key, None)

    def _other_owner(self, request, tracker, pos):
        return self.owners.get(self._checkpoint_key(request, tracker, pos))

    def begin_background(self, request, tracker):
        """Queue optional planning without deferring or changing the main receive."""
        rid = request.request_id
        if rid in self.background or rid in self.pending:
            return
        end = tracker.num_lmcache_hit_tokens
        if (
            self.enabled
            and end > self.block_size
            and len(self.background) + len(self.pending) < 128
        ):
            self.background[rid] = (request, tracker, end)

    def has_work(self):
        return bool(
            self.background
            or self.pending
            or self.abandoned
            or self.retiring
            or self.dispatch
        )

    def take_background_tasks(self):
        tasks, self.dispatch = self.dispatch, []
        return tasks

    def _poll_background(self):
        for rid, (request, tracker, end) in tuple(self.background.items()):
            if not self.plan_ready(request, tracker, end):
                continue
            plan = self.plans.get(rid)
            if plan is not None and rid not in self.owned_keys:
                if (
                    plan.groups != self.groups
                    or not plan.positions
                    or sorted(set(plan.positions)) != plan.positions
                    or any(
                        pos <= 0
                        or pos >= end
                        or pos % self.block_size
                        or pos // self.block_size > len(request.block_hashes)
                        for pos in plan.positions
                    )
                ):
                    self.plans.pop(rid)
                    self._retire(plan.lease, "cancel")
                    plan = None
                else:
                    keys = [
                        self._checkpoint_key(request, tracker, p)
                        for p in plan.positions
                    ]
                    if any(key in self.owners for key in keys) or any(
                        self.pool.get_cached_block(key[-1], self.groups) for key in keys
                    ):
                        # Concurrent prepares can return overlapping leases. Return the
                        # whole lease and rescan before allocating destinations.
                        self.plans.pop(rid)
                        self._retire(plan.lease, "cancel")
                        continue
                    self.owned_keys[rid] = keys
                    self.owners.update(dict.fromkeys(keys, rid))
            if plan is not None and rid not in self.primary_done:
                continue
            del self.background[rid]
            self.primary_done.discard(rid)
            if plan is None:
                continue
            self._allocate(request, end)
            pending = self.pending.get(rid)
            if pending is None:
                self._forget_owner(rid)
                continue
            pending.receive_completed = True
            pending.receipt_deadline = time.monotonic() + self.adapter._mq_timeout
            self.dispatch.append(
                (
                    rid,
                    tracker.packed_token_ids(),
                    end,
                    tracker.cache_salt,
                    msgspec.msgpack.encode(
                        HistoryCopy(
                            lease=pending.plan.lease, destinations=pending.block_ids()
                        )
                    ),
                )
            )

    def _retire(self, lease, action):
        self.retiring[lease] = action

    def plan_ready(self, request, tracker, end):
        """Submit on a confirmed external hit and let other requests advance."""
        rid = request.request_id
        if rid in self.plans:
            return True
        if rid in self.planning:
            future, deadline, planned_end = self.planning[rid]
            if not future.query() and time.monotonic() < deadline:
                return False
            del self.planning[rid]
            if not future.query():
                self.enabled = False
                return True  # server expires unclaimed plans
            try:
                plan = self.adapter.read_recurrent_history(future)
                if plan.lease and planned_end != end:
                    self._retire(plan.lease, "cancel")
                elif plan.lease:
                    self.plans[rid] = plan
            except Exception:
                self.enabled = False
                logger.exception("History prepare failed")
            return True
        if (
            not self.enabled
            or end <= self.block_size
            or end % self.block_size
            or len(self.planning)
            + len(self.pending)
            + len(self.retiring)
            + len(self.abandoned)
            >= 128
        ):
            return True
        positions = range(self.block_size, end, self.block_size)
        local = [
            pos
            for pos in positions
            if self.pool.get_cached_block(
                request.block_hashes[pos // self.block_size - 1], self.groups
            )
            or self._other_owner(request, tracker, pos)
        ]
        if len(local) == len(positions):
            return True
        key = IPCCacheServerKey(
            model_name=self.adapter.model_name,
            world_size=self.adapter.world_size,
            worker_id=None,
            token_bytes=tracker.packed_token_ids(),
            start=0,
            end=end,
            request_id=rid,
            cache_salt=tracker.cache_salt,
            request_configs=tracker.request_configs,
        )
        try:
            future = self.adapter.submit_recurrent_history(
                HistoryRequest(action="prepare", key=key, already_local=local)
            )
        except Exception:
            self.enabled = False
            logger.exception("Could not submit history prepare")
            return True
        self.planning[rid] = future, time.monotonic() + self.adapter._mq_timeout, end
        return False

    def settle(self, request_id: str, cancel: bool = False) -> None:
        """Settle a finished receive or cancel before request tracker removal."""
        if not cancel and request_id in self.background:
            self.primary_done.add(request_id)
        if cancel:
            self._forget_owner(request_id)
            self.background.pop(request_id, None)
            self.primary_done.discard(request_id)
            job = self.planning.pop(request_id, None)
            if job is not None:
                self.abandoned.append(job)
            plan = self.plans.pop(request_id, None)
            if plan is not None:
                self._retire(plan.lease, "cancel")
        pending = self.pending.get(request_id)
        if pending is not None:
            pending.canceled |= cancel
            pending.receive_completed |= not cancel

    def worker_finished(self, rid: str, lease: str, rank: int) -> None:
        """Accept each worker receipt once, for the currently owned lease."""
        pending = self.pending.get(rid)
        if (
            pending
            and pending.plan.lease == lease
            and 0 <= rank < self.adapter.world_size
        ):
            pending.receipt_ranks.add(rank)

    def finish_request(self, rid, normal_finish):
        pending = self.pending.get(rid)
        if normal_finish and rid in self.primary_done:
            return
        if normal_finish and pending is not None and pending.receive_completed:
            return  # primary finished normally; independent history can still publish
        self.settle(rid, cancel=True)

    def poll(self) -> None:
        """Advance completed control futures without blocking the scheduler."""
        self._poll_background()
        abandoned, self.abandoned = self.abandoned, []
        for future, deadline, end in abandoned:
            if future.query():
                try:
                    plan = self.adapter.read_recurrent_history(future)
                    if plan.lease:
                        self._retire(plan.lease, "cancel")
                except Exception:
                    logger.exception("Could not collect abandoned prepare")
            elif time.monotonic() < deadline:
                self.abandoned.append((future, deadline, end))
        for rid, pending in tuple(self.pending.items()):
            if not (pending.canceled or pending.receive_completed):
                continue
            if (
                not pending.canceled
                and len(pending.receipt_ranks) < self.adapter.world_size
            ):
                if time.monotonic() < pending.receipt_deadline:
                    continue
                pending.canceled = True
            lease = pending.plan.lease
            action = "cancel" if pending.canceled else "status"
            outcome = self._advance(lease, action)
            if outcome is not None and outcome.lease == lease and not outcome.settled:
                # A worker can complete with an error before submitting a device copy.
                # Only the server can confirm that these destinations are safe to free.
                pending.canceled = True
            if outcome is not None and pending.settle(outcome):
                self._forget_owner(rid)
                del self.pending[rid]
                self._retire(lease, "acknowledge")
        for lease, action in tuple(self.retiring.items()):
            response = self._advance(lease, action)
            if response is not None:
                if action == "acknowledge":
                    del self.retiring[lease]
                elif response.settled:
                    self._retire(lease, "acknowledge")

    def _advance(self, lease, action):
        now = time.monotonic()
        previous, future, when = self.controls.get(lease, (action, None, 0))
        try:
            if future is None:
                if now >= when:
                    future = self.adapter.submit_recurrent_history(
                        HistoryRequest(action=action, lease=lease)
                    )
                    self.controls[lease] = (
                        action,
                        future,
                        now + self.adapter._mq_timeout,
                    )
                return None
            if not future.query():
                if now >= when:
                    raise LMCacheTimeoutError("Recurrent history control timed out")
                return None
            outcome = self.adapter.read_recurrent_history(future)
            del self.controls[lease]
            if previous == action:
                if not outcome.settled and action != "acknowledge":
                    self.controls[lease] = action, None, now + 0.05
                return outcome
        except Exception:
            self.controls[lease] = action, None, now + 0.05
            logger.exception("Recurrent history control failed for %s", lease)
        return None

    def _allocate(self, request: "Request", end: int) -> None:
        plan = self.plans.pop(request.request_id, None)
        if plan is None:
            return
        local_pins = []
        for pos in range(self.block_size, end, self.block_size):
            blocks = self.pool.get_cached_block(
                request.block_hashes[pos // self.block_size - 1], self.groups
            )
            if blocks:
                local_pins.extend(blocks)
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
            self._retire(plan.lease, "cancel")
            return
        self.pending[request.request_id] = pending
