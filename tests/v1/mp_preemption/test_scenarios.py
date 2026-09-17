# SPDX-License-Identifier: Apache-2.0
"""Scripted preemption scenarios (S1-S12 in the design doc).

Every scenario asserts that preemption actually happened (otherwise it is
vacuous) and then the full invariant set: clean shutdown, no leaked locks,
no poisoned stores, no missing retrieves.
"""

# Third Party
from vllm.v1.request import RequestStatus
import pytest

# Local
from .driver import simulation
from .fake_lmcache import CompletionPolicy
from .vllm_factories import create_request, make_prompt

BS = 4
CHUNK = 8

POLICIES = [CompletionPolicy.SYNC, CompletionPolicy.FLUSH_ON_FLAG]


def _req(rid: str, prompt: list[int], max_tokens: int, **kw) -> object:
    return create_request(rid, prompt, max_tokens=max_tokens, block_size=BS, **kw)


# ---------------------------------------------------------------------------
# S1: preempt during chunked prefill, before anything was stored
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_s1_preempt_during_chunked_prefill(policy):
    with simulation(
        num_blocks=9,  # 8 usable blocks = 32 tokens
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        max_num_batched_tokens=16,  # r1's 24-token prompt takes 3 steps
    ) as sim:
        # Both are admitted in step 1.  r1 prefills in chunks and is
        # running[-1], so when the 32-token pool runs out during its prefill
        # it is the preemption victim before it ever stored a chunk.
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 4, 0, 4), max_tokens=24),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=4),
            ]
        )
        preempted_in_prefill = False
        for _ in range(400):
            rec = sim.run_step()
            for rid in rec.preempted:
                req = sim.finished_requests.get(rid) or sim.scheduler.requests[rid]
                if req.num_output_tokens == 0:
                    preempted_in_prefill = True
            if not sim.scheduler.has_unfinished_requests():
                break
        sim.run_until_idle()
        sim.assert_preemption_happened()
        assert preempted_in_prefill, "r1 was never preempted while prefilling"
        sim.assert_all_invariants()


# ---------------------------------------------------------------------------
# S2: preempt during decode after a chunk was stored; resume with APC hit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_s2_preempt_after_store_resume_loads(policy, async_scheduling):
    with simulation(
        num_blocks=13,  # 12 usable = 48 tokens
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        async_scheduling=async_scheduling,
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=30),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=20),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()
        # P5: at least one resumed request loaded its prefix from LMCache.
        stored_before_preempt = any(s[1] == "r1" for s in sim.model.store_ops)
        assert stored_before_preempt
        assert sim.resumed_with_retrieve, (
            "no resumed request issued a RETRIEVE; resume-load is not working. "
            f"retrieves={sim.model.retrieve_ops}"
        )


# ---------------------------------------------------------------------------
# S4: same request preempted repeatedly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_s4_repeated_preemption(policy, async_scheduling):
    with simulation(
        num_blocks=13,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        async_scheduling=async_scheduling,
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=36),
                _req("r1", make_prompt(1, 8, 1, 4), max_tokens=24),
                _req("r2", make_prompt(2, 8, 2, 4), max_tokens=24),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened(at_least=2)
        assert sim.max_preemptions >= 2, sim.max_preemptions
        sim.assert_all_invariants()


# ---------------------------------------------------------------------------
# S5: the block-reuse race.  The model must (a) detect poisoning when nothing
# waits for the store, and (b) see none when the flush rule is honoured.
# ---------------------------------------------------------------------------
def _run_reuse_race(policy, store_latency):
    with simulation(
        num_blocks=9,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        store_latency=store_latency,
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 4, 0, 4), max_tokens=24),
                _req("r1", make_prompt(1, 8, 1, 8), max_tokens=16),
                _req("r2", make_prompt(2, 8, 2, 8), max_tokens=16),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_clean_shutdown()
        return sim


def test_s5_negative_control_detects_poisoning_without_flush():
    """Oracle sensitivity: with no wait at all, reuse after preemption poisons."""
    poisoned_any = False
    for latency in (1, 2, 3):
        sim = _run_reuse_race(CompletionPolicy.SERVER_ASYNC, latency)
        if sim.model.poisoned_stores:
            poisoned_any = True
            break
    assert poisoned_any, "model never observed a poisoned store; oracle is blind"


@pytest.mark.parametrize("latency", [1, 2, 3])
def test_s5_flush_rule_prevents_poisoning(latency):
    sim = _run_reuse_race(CompletionPolicy.FLUSH_ON_FLAG, latency)
    sim.assert_no_poisoned_stores()
    sim.assert_no_missing_retrieves()
    sim.assert_no_leaked_locks()


# ---------------------------------------------------------------------------
# S7: resume load fails (chunk evicted between lookup and retrieve)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure_policy", ["recompute", "fail"])
@pytest.mark.parametrize("policy", POLICIES)
def test_s7_resume_load_failure_recomputes(policy, failure_policy):
    evicted: list = []

    def evict(key):
        # Evict the first chunk requested by any retrieve of a resumed request.
        if not evicted:
            evicted.append(key)
            return True
        return False

    with simulation(
        num_blocks=13,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        evict_before_retrieve=evict,
        kv_load_failure_policy=failure_policy,
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=30),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=20),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened()
        assert sim.model.missing_retrieves, "eviction never triggered a failed load"
        sim.assert_clean_shutdown()
        sim.assert_no_leaked_locks()
        sim.assert_no_poisoned_stores()
        failed = [
            r
            for r in sim.finished_requests.values()
            if r.status.name == "FINISHED_ERROR"
        ]
        if failure_policy == "fail":
            assert failed, "fail policy should have errored the request"
        else:
            assert not failed, "recompute policy must not error requests"


# ---------------------------------------------------------------------------
# S8: abort a request while it sits in the waiting queue as PREEMPTED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_s8_abort_while_preempted(policy):
    with simulation(
        num_blocks=13, block_size=BS, chunk_size=CHUNK, completion_policy=policy
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=36),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=24),
            ]
        )
        aborted = False
        for _ in range(400):
            sim.run_step()
            r1 = sim.scheduler.requests.get("r1")
            if r1 is not None and r1.status == RequestStatus.PREEMPTED and not aborted:
                # Let the scheduler poll the connector once for the preempted
                # request (so a resume lookup is in flight), then abort.
                sim.run_step()
                if "r1" in sim.scheduler.requests:
                    sim.abort("r1")
                    aborted = True
            if not sim.scheduler.has_unfinished_requests():
                break
        assert aborted, "r1 was never observed in PREEMPTED state"
        sim.run_until_idle()
        sim.assert_all_invariants()


# ---------------------------------------------------------------------------
# S10: reset_prefix_cache preempts every running request
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_s10_reset_prefix_cache(policy):
    with simulation(
        num_blocks=33, block_size=BS, chunk_size=CHUNK, completion_policy=policy
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=20),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=20),
            ]
        )
        for _ in range(6):
            sim.run_step()
        assert sim.scheduler.reset_prefix_cache(reset_running_requests=True)
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()


# ---------------------------------------------------------------------------
# S11: priority policy; the victim may already be scheduled this step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_s11_priority_victim(policy):
    with simulation(
        num_blocks=13,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        scheduling_policy="priority",
    ) as sim:
        # Low-priority (high value) r0 arrives first and is running; r1/r2
        # with higher priority arrive later and evict it.
        sim.add_requests(
            [_req("r0", make_prompt(0, 8, 0, 4), max_tokens=30, priority=5)]
        )
        for _ in range(3):
            sim.run_step()
        sim.add_requests(
            [
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=20, priority=0),
                _req("r2", make_prompt(2, 8, 2, 8), max_tokens=20, priority=1),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()


# ---------------------------------------------------------------------------
# S12: deferred lookup on a resumed request (scheduler polls repeatedly)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_s12_deferred_lookup_on_resume(policy):
    with simulation(
        num_blocks=13,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        lookup_defers=2,
    ) as sim:
        sim.add_requests(
            [
                _req("r0", make_prompt(0, 8, 0, 4), max_tokens=30),
                _req("r1", make_prompt(1, 16, 1, 8), max_tokens=20),
            ]
        )
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()
        # Each generation of each request submits exactly one lookup.
        per_req: dict[str, int] = {}
        for rid, _n in sim.adapter.lookups_submitted:
            per_req[rid] = per_req.get(rid, 0) + 1
        r1 = sim.finished_requests["r1"]
        assert per_req["r1"] == 1 + r1.num_preemptions, (per_req, r1.num_preemptions)
