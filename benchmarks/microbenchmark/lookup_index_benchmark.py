#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CPU microbenchmark for the LOOKUP-side L1 index walk (#6403 item 3).

Replays the index-side CPU work of one LOOKUP at a configurable scale,
with no GPU, server, or RPC:

1. key expansion: ``ipc_key_to_grouped_object_keys`` over the request's
   chunk hashes (one ObjectKey per chunk per kv rank per object group),
2. index walk: the dict operations ``BaseL1Manager.reserve_read`` and the
   prefetch controller's ``_lock_l1_keys`` run per expanded key against a
   fully resident index (index probe, result-dict insert, per-manager
   merge, hit probe),
3. recency: one ``LRUEvictionPolicy.on_keys_touched`` batch over the
   same keys.

Each repetition expands fresh keys and then walks and touches those same
objects, mirroring one request's life: the walk hashes every key first,
the touch reuses the keys the walk already hashed.

Usage:
    python benchmarks/microbenchmark/lookup_index_benchmark.py \
        [--tokens N] [--chunk-size N] [--world-size N] [--object-groups N]
        [--reps N] [--tags N]
"""

# Standard
import argparse
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_grouped_object_keys,
)
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU microbenchmark for the LOOKUP-side L1 index walk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=153_600,
        help="Prompt length in tokens (#6403 reports 150K-token prompts)",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Tokens per chunk")
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="KV ranks per chunk (per-rank layout, e.g. TP8)",
    )
    parser.add_argument(
        "--object-groups",
        type=int,
        default=1,
        help="Object groups per chunk. A hybrid model gets one per attention "
        "window by default (#6041); each is replayed as a full-attention "
        "row set, the worst case for the walk and the touch",
    )
    parser.add_argument("--reps", type=int, default=7, help="Repetitions (median)")
    parser.add_argument(
        "--tags",
        type=int,
        default=0,
        help="Request matching tags per key (lmcache.tag.*); tags make "
        "every key hash carry a dict-snapshot hash",
    )
    args = parser.parse_args()

    num_chunks = args.tokens // args.chunk_size
    chunk_hashes = [i.to_bytes(4, "little") * 8 for i in range(num_chunks)]
    request_configs: dict[str, object] | None = None
    if args.tags:
        request_configs = {f"lmcache.tag.t{i}": f"v{i}" for i in range(args.tags)}
    ipc_key = IPCCacheServerKey(
        model_name="benchmark-model",
        world_size=args.world_size,
        worker_id=None,  # a lookup fans out to every kv rank
        token_bytes=b"\x00" * (args.tokens * 4),
        start=0,
        end=args.tokens,
        request_id="benchmark-request",
        request_configs=request_configs,
    )
    object_group_ids = list(range(args.object_groups))
    attn_desc = AttnWindowDesc(
        num_chunks_in_sw=[-1] * args.object_groups, world_size=args.world_size
    )
    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, args.chunk_size, 8, 128])],
        dtypes=[torch.bfloat16],
    )
    layout_descs = {group_id: layout_desc for group_id in object_group_ids}

    def expand():
        return ipc_key_to_grouped_object_keys(
            ipc_key,
            chunk_hashes,
            object_group_ids,
            layout_descs,
            attn_desc,
            DEFAULT_SHARD_REPLICATION_DESC,
        )

    # A fully resident index and a policy tracking every key: the walk and
    # the touch then cost their worst case (every probe and move happens).
    resident_keys = [key for row in expand() for key in row.keys]
    num_keys = len(resident_keys)
    index = {key: object() for key in resident_keys}
    policy = LRUEvictionPolicy()
    policy.on_keys_created(resident_keys)

    expand_ms: list[float] = []
    walk_ms: list[float] = []
    touch_ms: list[float] = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        rows = expand()
        t1 = time.perf_counter()
        probe_keys = [key for row in rows for key in row.keys]
        lookup = index.get
        t2 = time.perf_counter()
        # reserve_read's walk: one index probe and one result-dict insert
        # per key of every row.
        per_row: list[dict[ObjectKey, tuple[bool, object | None]]] = []
        for row in rows:
            reserved: dict[ObjectKey, tuple[bool, object | None]] = {}
            for key in row.keys:
                entry = lookup(key)
                reserved[key] = (entry is not None, entry)
            per_row.append(reserved)
        # _lock_l1_keys: merge the per-row results, then probe the merge
        # once per key to fill the hit bitmap.
        merged = {
            key: result for row_result in per_row for key, result in row_result.items()
        }
        hits = 0
        for key in probe_keys:
            ok, _entry = merged[key]
            if ok:
                hits += 1
        t3 = time.perf_counter()
        policy.on_keys_touched(probe_keys)
        t4 = time.perf_counter()
        if hits != num_keys:
            raise RuntimeError(f"expected {num_keys} hits, got {hits}")
        expand_ms.append((t1 - t0) * 1e3)
        walk_ms.append((t3 - t2) * 1e3)
        touch_ms.append((t4 - t3) * 1e3)

    expand_med = statistics.median(expand_ms)
    walk_med = statistics.median(walk_ms)
    touch_med = statistics.median(touch_ms)
    total_med = expand_med + walk_med + touch_med
    print(
        f"workload: {args.tokens} tokens, chunk {args.chunk_size} -> "
        f"{num_chunks} chunk hashes, {args.world_size} kv ranks, "
        f"{args.object_groups} object group(s) -> "
        f"{num_keys} keys, median of {args.reps}"
    )
    print(
        f"  key expansion   {expand_med:8.2f} ms"
        f"  ({expand_med * 1e3 / num_keys:5.2f} us/key)"
    )
    print(
        f"  index walk      {walk_med:8.2f} ms"
        f"  ({walk_med * 1e3 / num_keys:5.2f} us/key)"
    )
    print(
        f"  LRU touch       {touch_med:8.2f} ms"
        f"  ({touch_med * 1e3 / num_keys:5.2f} us/key)"
    )
    print(
        f"  TOTAL           {total_med:8.2f} ms"
        f"  ({total_med * 1e3 / num_keys:5.2f} us/key)"
    )


if __name__ == "__main__":
    main()
