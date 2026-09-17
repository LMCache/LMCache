# SPDX-License-Identifier: Apache-2.0
"""Step driver: runs the vLLM scheduler + LMCacheMPConnector against the
``CacheModel`` and checks the correctness properties after every step.

Properties (see docs/design/integration/vllm/mp_preemption_correctness.md):

* P0  liveness / hygiene: everything finishes, trackers and locks drain.
* P1  structural op consistency against ``kv_cache_manager``.
* P2  no store reads a block after it was rewritten (poisoning), under the
      configured completion policy.
* P5  resume-load behaviour (checked by individual scenarios).
"""

# Standard
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from unittest import mock
import contextlib

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
)
import lmcache.integration.vllm.lmcache_mp_connector as connector_mod

# Local
from .fake_lmcache import CacheModel, CompletionPolicy, FakeSchedulerAdapter
from .vllm_factories import create_model_runner_output, create_scheduler


def sampled_token(request_id: str, index: int) -> int:
    """Deterministic, request-specific generated token."""
    h = 0
    for ch in f"{request_id}:{index}":
        h = (h * 131 + ord(ch)) % 1_000_003
    return 500_000 + h % 50_000


@dataclass
class StepRecord:
    step: int
    scheduled: dict[str, int]
    preempted: set[str]
    resumed: set[str]
    num_running: int
    num_waiting: int
    stores: list[tuple[str, int, int]]
    retrieves: list[tuple[str, int, int]]
    need_flush: bool


@dataclass
class Simulation:
    scheduler: Any
    connector: LMCacheMPConnector
    adapter: FakeSchedulerAdapter
    model: CacheModel
    block_size: int
    chunk_size: int
    step: int = 0
    records: list[StepRecord] = field(default_factory=list)
    # request_id -> generation (num_preemptions) -> next expected store start
    _store_cursor: dict[tuple[str, int], int] = field(default_factory=dict)
    # requests whose request_finished() returned delay_free=True and whose
    # blocks vLLM is holding until we report finished_sending.
    _pending_send: set[str] = field(default_factory=set)
    _delay_free_seen: dict[str, bool] = field(default_factory=dict)
    finished_requests: dict[str, Request] = field(default_factory=dict)
    max_preemptions: int = 0
    resumed_with_retrieve: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        num_blocks: int,
        block_size: int = 4,
        chunk_size: int = 8,
        completion_policy: CompletionPolicy = CompletionPolicy.FLUSH_ON_FLAG,
        store_latency: int = 1,
        load_latency: int = 1,
        lookup_defers: int = 0,
        evict_before_retrieve: Callable[..., bool] | None = None,
        lazy_offload: bool = False,
        **scheduler_kwargs: Any,
    ) -> "Simulation":
        model = CacheModel(
            chunk_size=chunk_size,
            block_size=block_size,
            completion_policy=completion_policy,
            store_latency=store_latency,
            load_latency=load_latency,
            lookup_defers=lookup_defers,
            evict_before_retrieve=evict_before_retrieve,
        )
        adapter = FakeSchedulerAdapter(model)
        extra = dict(scheduler_kwargs.pop("kv_connector_extra_config", {}) or {})
        if lazy_offload:
            extra["lmcache.mp.lazy_offload"] = True
        # The connector instantiates ``LMCacheMPSchedulerAdapter`` by name from
        # its own module; hand it our fake instead of a ZMQ client.
        with mock.patch.object(connector_mod, "LMCacheMPSchedulerAdapter", adapter):
            scheduler = create_scheduler(
                num_blocks=num_blocks,
                block_size=block_size,
                kv_connector_extra_config=extra,
                **scheduler_kwargs,
            )
        connector = scheduler.connector
        assert isinstance(connector, LMCacheMPConnector), type(connector)
        assert connector.scheduler_adapter is adapter
        sim = cls(
            scheduler=scheduler,
            connector=connector,
            adapter=adapter,
            model=model,
            block_size=block_size,
            chunk_size=chunk_size,
        )
        sim._spy_request_finished()
        return sim

    def _spy_request_finished(self) -> None:
        connector = self.connector
        original = connector.request_finished

        def spy(request: Request, block_ids: Any) -> tuple[bool, Any]:
            delay_free, params = original(request, block_ids)
            self._delay_free_seen[request.request_id] = delay_free
            if delay_free:
                self._pending_send.add(request.request_id)
            self.finished_requests[request.request_id] = request
            return delay_free, params

        connector.request_finished = spy  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------
    @property
    def usable_tokens(self) -> int:
        """Token capacity of the pool (vLLM reserves one null block)."""
        pool = self.scheduler.kv_cache_manager.block_pool
        return (pool.num_gpu_blocks - 1) * self.block_size

    def add_requests(self, requests: Iterable[Request]) -> None:
        for r in requests:
            # vLLM validates max_model_len against the pool at startup; a
            # request that can never fit alone would head-of-line block the
            # FCFS queue forever and make the scenario meaningless.
            need = r.num_prompt_tokens + r.sampling_params.max_tokens
            assert need <= self.usable_tokens, (
                f"request {r.request_id} needs {need} tokens but the pool holds "
                f"{self.usable_tokens}; shrink prompt/max_tokens"
            )
            self.scheduler.add_request(r)

    def abort(self, request_id: str) -> None:
        self.scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)

    def run_step(self) -> StepRecord:
        self.step += 1
        self.model.begin_step(self.step)
        scheduler = self.scheduler

        out: SchedulerOutput = scheduler.schedule()
        meta = out.kv_connector_metadata
        assert isinstance(meta, LMCacheMPConnectorMetadata)

        # --- P1: structural checks against the kv cache manager -----------
        self._check_ops(out, meta)
        self._check_tracker_block_lists()

        # --- forward pass writes -------------------------------------------
        writes = self._forward_writes(out)
        self.model.apply_forward_writes(writes, meta.need_flush_before_forward)

        # --- stores/retrieves submitted after forward ----------------------
        request_tokens = {
            rid: list(scheduler.requests[rid].all_token_ids)
            for rid in scheduler.requests
        }
        self.model.submit_ops(meta, request_tokens)

        finished_recving, invalid = self.model.end_step()
        finished_sending = {
            rid
            for rid in list(self._pending_send)
            if not self.model.has_inflight_stores_for(rid)
        }
        self._pending_send -= finished_sending

        # --- model runner output -------------------------------------------
        req_ids = list(out.num_scheduled_tokens.keys())
        sampled: list[list[int]] = []
        for rid in req_ids:
            req = scheduler.requests[rid]
            # After schedule(), num_computed_tokens already includes this
            # step's tokens; a request samples iff its whole sequence so far
            # is computed (i.e. this step finished the prefill or is decode).
            if req.num_computed_tokens >= req.num_tokens:
                sampled.append([sampled_token(rid, req.num_output_tokens)])
            else:
                sampled.append([])
        mro = create_model_runner_output(
            req_ids,
            sampled,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid,
        )
        scheduler.update_from_output(out, mro)

        preempted = set(out.preempted_req_ids or ())
        resumed = set(out.scheduled_cached_reqs.resumed_req_ids)
        for rid in preempted:
            req = scheduler.requests.get(rid)
            if req is not None:
                self.max_preemptions = max(self.max_preemptions, req.num_preemptions)
        rec = StepRecord(
            step=self.step,
            scheduled=dict(out.num_scheduled_tokens),
            preempted=preempted,
            resumed=resumed,
            num_running=len(scheduler.running),
            num_waiting=scheduler.get_request_counts()[1],
            stores=[
                (r.request_id, r.op.start, r.op.end)
                for r in meta.requests
                if r.direction == "STORE"
            ],
            retrieves=[
                (r.request_id, r.op.start, r.op.end)
                for r in meta.requests
                if r.direction == "RETRIEVE"
            ],
            need_flush=meta.need_flush_before_forward,
        )
        self.records.append(rec)
        return rec

    def run_until_idle(self, max_steps: int = 2000) -> None:
        for _ in range(max_steps):
            if (
                not self.scheduler.has_unfinished_requests()
                and not self._pending_send
                and not self.model.inflight_stores
                and not self.model.inflight_retrieves
            ):
                return
            self.run_step()
        raise AssertionError(
            f"simulation did not converge in {max_steps} steps: "
            f"running={len(self.scheduler.running)} "
            f"waiting={self.scheduler.get_request_counts()[1]} "
            f"pending_send={self._pending_send} "
            f"inflight_stores={len(self.model.inflight_stores)}"
        )

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def _block_ids(self, request_id: str) -> list[int]:
        return list(self.scheduler.kv_cache_manager.get_block_ids(request_id)[0])

    def _forward_writes(self, out: SchedulerOutput) -> dict[int, tuple[int, ...]]:
        """Blocks written by this step's forward pass, with their content."""
        writes: dict[int, tuple[int, ...]] = {}
        bs = self.block_size
        for rid, n in out.num_scheduled_tokens.items():
            req = self.scheduler.requests[rid]
            end = req.num_computed_tokens  # already advanced by schedule()
            start = end - n
            blocks = self._block_ids(rid)
            tokens = list(req.all_token_ids)
            for b in range(start // bs, (end + bs - 1) // bs):
                if b >= len(blocks):
                    break
                # A block still being filled holds only the tokens computed so
                # far; a full block holds all ``bs`` tokens.
                writes[blocks[b]] = tuple(tokens[b * bs : min((b + 1) * bs, end)])
        return writes

    def _check_ops(
        self, out: SchedulerOutput, meta: LMCacheMPConnectorMetadata
    ) -> None:
        bs = self.block_size
        chunk = self.chunk_size
        for r in meta.requests:
            assert isinstance(r, LMCacheMPRequestMetadata)
            rid = r.request_id
            op = r.op
            req = self.scheduler.requests[rid]
            tracker = self.connector.request_trackers[rid]
            blocks = self._block_ids(rid)
            assert op.start % chunk == 0 and op.end % chunk == 0, (r.direction, rid, op)
            assert op.end > op.start, (r.direction, rid, op)
            request_tokens = list(req.all_token_ids)[: len(op.token_ids)]
            assert list(op.token_ids) == request_tokens, (
                f"{r.direction} {rid}: op token ids diverge from request tokens"
            )
            expected_blocks = blocks[op.start // bs : op.end // bs]
            assert op.block_ids[0] == expected_blocks, (
                f"P1a {r.direction} {rid} step {self.step} [{op.start},{op.end}): "
                f"op blocks {op.block_ids[0]} != vLLM blocks {expected_blocks} "
                f"(all={blocks})"
            )
            if r.direction == "STORE":
                assert op.end <= req.num_computed_tokens, (
                    f"P1b STORE {rid} step {self.step}: stores "
                    f"[{op.start},{op.end}) but only {req.num_computed_tokens} "
                    f"tokens computed"
                )
                gen = req.num_preemptions
                cursor_key = (rid, gen)
                if cursor_key not in self._store_cursor:
                    # First store of this generation starts where the lookup
                    # said LMCache already had data.
                    self._store_cursor[cursor_key] = tracker.num_lmcache_hit_tokens
                assert op.start == self._store_cursor[cursor_key], (
                    f"P1b STORE {rid} gen {gen}: start {op.start} != cursor "
                    f"{self._store_cursor[cursor_key]}"
                )
                self._store_cursor[cursor_key] = op.end
            else:
                assert op.end == tracker.num_lmcache_hit_tokens, (rid, op, tracker)
                assert op.end <= len(req.all_token_ids)
                assert op.start == (tracker.num_vllm_hit_tokens // chunk) * chunk, (
                    rid,
                    op,
                    tracker,
                )
                assert op.skip_first_n_tokens == tracker.num_vllm_hit_tokens - op.start
                assert 0 <= op.skip_first_n_tokens < chunk
                if req.num_preemptions > 0:
                    self.resumed_with_retrieve.add(rid)

    def _check_tracker_block_lists(self) -> None:
        for rid, tracker in self.connector.request_trackers.items():
            req = self.scheduler.requests.get(rid)
            if req is None or req.status not in (
                RequestStatus.RUNNING,
                RequestStatus.WAITING_FOR_REMOTE_KVS,
            ):
                continue
            if tracker.state == LMCacheMPRequestState.PREFETCHING:
                continue
            assert tracker.allocated_block_ids.get(0, []) == self._block_ids(rid), (
                f"P1d {rid} step {self.step}: tracker blocks "
                f"{tracker.allocated_block_ids.get(0, [])} != vLLM "
                f"{self._block_ids(rid)}"
            )

    # ------------------------------------------------------------------
    # End-of-run assertions
    # ------------------------------------------------------------------
    def assert_clean_shutdown(self) -> None:
        self.model.drain()
        assert not self.scheduler.has_unfinished_requests()
        assert self.scheduler.requests == {}, list(self.scheduler.requests)
        assert self.connector.request_trackers == {}, list(
            self.connector.request_trackers
        )
        for rid, n in self.adapter.end_session_calls.items():
            assert n == 1, f"end_session called {n}x for {rid}"
        missing = set(self.finished_requests) - set(self.adapter.end_session_calls)
        assert not missing, f"no end_session for {missing}"
        # Every block must be back in the pool.
        pool = self.scheduler.kv_cache_manager.block_pool
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1, (
            pool.get_num_free_blocks(),
            pool.num_gpu_blocks,
        )

    def assert_no_leaked_locks(self) -> None:
        outstanding = self.model.locks.outstanding()
        assert not outstanding, {
            r: [k[1][-2:] for k in ks] for r, ks in outstanding.items()
        }

    def assert_no_poisoned_stores(self) -> None:
        assert not self.model.poisoned_stores, [
            (s.request_id, s.submitted_step, s.complete_step)
            for s in self.model.poisoned_stores
        ]
        assert not self.model.poisoned_retrieves

    def assert_no_missing_retrieves(self) -> None:
        assert not self.model.missing_retrieves, [
            (r.request_id, r.submitted_step) for r in self.model.missing_retrieves
        ]

    def assert_preemption_happened(self, at_least: int = 1) -> None:
        total = sum(len(r.preempted) for r in self.records)
        assert total >= at_least, f"only {total} preemptions; scenario is vacuous"

    def assert_all_invariants(self) -> None:
        self.assert_clean_shutdown()
        self.assert_no_leaked_locks()
        self.assert_no_poisoned_stores()
        self.assert_no_missing_retrieves()


@contextlib.contextmanager
def simulation(**kwargs: Any):
    sim = Simulation.create(**kwargs)
    try:
        yield sim
    finally:
        with contextlib.suppress(Exception):
            sim.connector.shutdown()
