# SPDX-License-Identifier: Apache-2.0
"""CPU timings of LMCache's scheduler hook against the installed vLLM block pool.

Run with the LMCache source checkout (including its tests) on PYTHONPATH.
Run once with the default scan, then repeat with --indexed on a supported pool.
"""

# Standard
from statistics import median
from time import perf_counter_ns
import argparse
import json
import logging

# First Party
from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
from tests.v1.lazy_offload.block_pool_fake import make_real_pool
from tests.v1.lazy_offload.test_eviction_aware_policy import add_op
from tests.v1.lazy_offload.test_offload_manager import (
    _FakeCompletionTracker,
    _make_scheduler_output,
    _make_store_metadata,
)


def run(count: int, indexed: bool, samples: int) -> None:
    """Time drain with historical ops and optionally one at-risk prefix."""
    pool = make_real_pool(65540)
    manager = LazyOffloadManager(
        {"lmcache.mp.indexed_prefix_writes": indexed}, [256], _FakeCompletionTracker()
    )
    manager.bind_block_pool(pool)
    policy = manager._require_policy()
    if indexed and policy.prefix_index is None:
        raise RuntimeError("Indexed prefix selection did not activate")

    # Four contiguous operations for a probe; only its third block will be due.
    probe_blocks = pool.get_new_blocks(4)
    for block in [*probe_blocks, *pool.blocks[32:65536]]:
        pool._insert_block_hash(block.block_id.to_bytes(8, "little"), block, None)
    for i in range(count):
        req = str(i)
        manager.on_request_arrived(req)
        bids = [32 + (8 * i + j) % (65536 - 32) for j in range(8)]
        add_op(policy, pool, req, bids, 256)
        manager.on_request_finished(req)

    step = _make_scheduler_output(8)
    for emit in (False, True):
        measurements = []
        for _ in range(samples + 10):
            if emit:
                manager.on_request_arrived("probe")
                for i, block in enumerate(probe_blocks):
                    manager.add_store_candidate(
                        _make_store_metadata(
                            "probe", [[block.block_id]], i * 256, (i + 1) * 256
                        )
                    )
                manager.on_request_finished("probe")
                pool.free_blocks([probe_blocks[2]], evict_first=True)
            start = perf_counter_ns()
            actions = manager.on_scheduler_step(step)
            measurements.append((perf_counter_ns() - start) / 1e6)
            if emit:
                (store,) = actions.stores_to_submit
                assert store.op.start == 0 and store.op.end == 768
                assert store.op.block_ids == [[b.block_id for b in probe_blocks[:3]]]
                assert policy.has_pending_request("probe")
                assert probe_blocks[2].ref_cnt == 1
                manager.on_store_results(set(), {"probe": 1})
                assert probe_blocks[2].ref_cnt == 0
                pool.touch([probe_blocks[2]])
                manager.on_request_reset("probe")
                manager.on_request_finished("probe")
            else:
                assert not actions.stores_to_submit
        timed = sorted(measurements[10:])
        print(
            json.dumps(
                {
                    "indexed": indexed,
                    "pending_ops": count,
                    "submitted_tokens": 768 if emit else 0,
                    "samples": samples,
                    "p50_ms": median(timed),
                    "p95_ms": timed[(95 * samples + 99) // 100 - 1],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexed", action="store_true")
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    logging.disable(logging.CRITICAL)
    for count in (0, 18000, 39000):
        run(count, args.indexed, args.samples)
