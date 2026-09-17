# SPDX-License-Identifier: Apache-2.0
"""Harness smoke tests: no preemption, then the simplest forced preemption."""

# Third Party
import pytest

# Local
from .driver import simulation
from .fake_lmcache import CompletionPolicy
from .vllm_factories import create_request, make_prompt

BS = 4
CHUNK = 8


def test_no_preemption_store_and_replay_hit():
    """Two identical prompts: the second must hit what the first stored."""
    with simulation(num_blocks=64, block_size=BS, chunk_size=CHUNK) as sim:
        prompt = make_prompt(1, 16, 1, 8)  # 24 tokens = 3 chunks
        sim.add_requests([create_request("a", prompt, max_tokens=6, block_size=BS)])
        sim.run_until_idle()
        assert sim.model.num_stored_chunks >= 3
        # Replay: APC is warm too, so drop it by using a fresh scheduler? No:
        # vLLM's APC still holds the prefix; LMCache must still be consulted.
        sim.add_requests([create_request("b", prompt, max_tokens=6, block_size=BS)])
        sim.run_until_idle()
        assert sim.adapter.lookups_submitted[-1][0] == "b"
        sim.assert_all_invariants()


@pytest.mark.parametrize(
    "policy",
    [CompletionPolicy.SYNC, CompletionPolicy.FLUSH_ON_FLAG],
)
def test_forced_preemption_survives(policy):
    """Pool too small for both requests' decode: one gets preempted."""
    with simulation(
        num_blocks=9, block_size=BS, chunk_size=CHUNK, completion_policy=policy
    ) as sim:
        reqs = [
            create_request("r0", make_prompt(0, 8, 0, 4), max_tokens=12, block_size=BS),
            create_request("r1", make_prompt(0, 8, 1, 4), max_tokens=12, block_size=BS),
        ]
        sim.add_requests(reqs)
        sim.run_until_idle()
        sim.assert_preemption_happened()
        sim.assert_all_invariants()
