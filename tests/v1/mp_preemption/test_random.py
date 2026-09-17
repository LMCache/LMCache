# SPDX-License-Identifier: Apache-2.0
"""Seeded random workloads under a tiny block pool.

Runs thousands of scheduler steps per seed with arrivals, shared prefixes,
mixed lengths and occasional aborts, checking every invariant after every
step.  Also asserts the workload actually reached the interesting states
(same-step preempt+resume, repeated preemption, resume with retrieve) so a
green run is not vacuous.
"""

# Standard
import random

# Third Party
import pytest

# Local
from .driver import simulation
from .fake_lmcache import CompletionPolicy
from .vllm_factories import create_request, make_prompt

BS = 4
CHUNK = 8


def _workload(rng: random.Random, n: int, block_size: int, capacity: int):
    """Random requests, each small enough to run alone in ``capacity`` tokens."""
    reqs = []
    for i in range(n):
        prefix_id = rng.randrange(3)
        prefix_len = rng.choice([p for p in (0, 8, 16, 24) if p <= capacity // 2])
        unique_len = rng.randrange(1, max(2, min(20, capacity // 2 - prefix_len + 1)))
        max_prompt = prefix_len + unique_len
        max_tokens = rng.randrange(1, max(2, min(40, capacity - max_prompt)))
        reqs.append(
            (
                rng.randrange(0, 60),  # arrival step
                create_request(
                    f"q{i}",
                    make_prompt(prefix_id, prefix_len, i, unique_len),
                    max_tokens=max_tokens,
                    block_size=block_size,
                ),
            )
        )
    reqs.sort(key=lambda t: t[0])
    return reqs


@pytest.mark.parametrize(
    "policy", [CompletionPolicy.SYNC, CompletionPolicy.FLUSH_ON_FLAG]
)
@pytest.mark.parametrize("seed", list(range(8)))
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_random_workload(policy, seed, async_scheduling):
    rng = random.Random(seed)
    num_blocks = rng.choice([9, 13, 17, 25])
    with simulation(
        num_blocks=num_blocks,
        block_size=BS,
        chunk_size=CHUNK,
        completion_policy=policy,
        store_latency=rng.choice([1, 2]),
        load_latency=rng.choice([1, 2]),
        lookup_defers=rng.choice([0, 1]),
        max_num_batched_tokens=rng.choice([8, 16, 64]),
        max_num_seqs=8,
        async_scheduling=async_scheduling,
    ) as sim:
        pending = _workload(rng, 24, BS, (num_blocks - 1) * BS)
        same_step_resume = 0
        for step in range(3000):
            while pending and pending[0][0] <= step:
                sim.add_requests([pending.pop(0)[1]])
            if not sim.scheduler.has_unfinished_requests() and not pending:
                break
            rec = sim.run_step()
            if rec.preempted & rec.resumed:
                same_step_resume += 1
            # Occasionally abort a random live request.
            if rng.random() < 0.02 and sim.scheduler.requests:
                victim = rng.choice(list(sim.scheduler.requests))
                if not sim.scheduler.requests[victim].is_finished():
                    sim.abort(victim)
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()
        # Record coverage for the session summary.
        print(
            f"seed={seed} policy={policy.value} async={async_scheduling} "
            f"blocks={num_blocks} "
            f"preemptions={sum(len(r.preempted) for r in sim.records)} "
            f"max_gen={sim.max_preemptions} same_step_resume={same_step_resume} "
            f"resumed_with_retrieve={len(sim.resumed_with_retrieve)} "
            f"stores={len(sim.model.store_ops)} retrieves={len(sim.model.retrieve_ops)}"
        )


def test_random_coverage_reaches_interesting_states():
    """Across seeds the workload must hit resume-with-retrieve at least once."""
    hits = 0
    for seed in range(6):
        rng = random.Random(1000 + seed)
        with simulation(
            num_blocks=13,
            block_size=BS,
            chunk_size=CHUNK,
            completion_policy=CompletionPolicy.FLUSH_ON_FLAG,
            max_num_seqs=8,
        ) as sim:
            pending = _workload(rng, 16, BS, 12 * BS)
            for step in range(3000):
                while pending and pending[0][0] <= step:
                    sim.add_requests([pending.pop(0)[1]])
                if not sim.scheduler.has_unfinished_requests() and not pending:
                    break
                sim.run_step()
            sim.run_until_idle()
            sim.assert_all_invariants()
            hits += len(sim.resumed_with_retrieve)
    assert hits > 0, "no seed produced a resume that loaded from LMCache"
