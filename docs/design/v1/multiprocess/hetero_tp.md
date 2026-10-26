# Heterogeneous TP lookup (MP mode)

A decode engine at TP=B can serve KV that a prefill engine stored at TP=A.
Two cases are covered: each engine has its own LMCache server joined by P2P
(the native PD data path of RFC #5969, `lmcache_driven` mode), or both
engines share one server. RFC #5969 leaves asymmetric TP open; this design
keeps the store in each rank's native layout and reshards on D. This
document covers phase 1: vLLM to vLLM, attention-only, non-MLA models.

## Problem

1. **Keys do not match.** `ObjectKey.kv_rank` packs `world_size`
   (`distributed/api.py`). A lookup at TP=B expands into B keys with
   `world_size=B`, but P wrote A keys with `world_size=A`. The P2P adapter
   forwards D's exact keys to P, so every lookup misses.
2. **Shapes do not match.** Each rank stores only its own shard: per kernel
   group, a dense `[kv_size, L, T, local_heads * head_elems]`. The GPU copy
   kernel uses one shape for both sides, so the source and destination local
   head counts must be equal.

Pure MLA with PP=1 and DCP=1 already works: `kv_world_size` is 1 for any TP,
so P and D read and write the same key.

## Decision: reshard in the reader's L1

The store path is unchanged. When a lookup misses at its own TP, it looks up
the source TP's keys, pulls those objects unchanged, and copies their head
slices into B-keyed, B-shaped objects in its own L1. Everything after that
(the final lock, retrieve, free accounting, the GPU kernel) sees an ordinary
B-keyed hit.

```
D lookup (B keys)  ── primary stage ──────────────▶ miss (0 chunks)
D lookup (A keys)  ── source stage, P2P/L2/L1 ────▶ hit: A objects in D L1, 1 read lock each
reshard            ── A objects ─▶ B objects in D L1 (head-slice copies)
                      release A read locks
D lookup (B keys)  ── final stage, L1 only ───────▶ hit, num_kv_readers locks
retrieve           ── unchanged
```

Why here and not elsewhere:

- **Not in the GPU copy.** Retrieve, locks and the kernel stay unchanged. The
  network moves exactly P's bytes, once.
- **Not a canonical layout on store** (for example per-head objects). That
  changes `kv_rank` semantics, which orphans every persisted L2 object.
- **Not on P.** P would need to know each reader's TP and serve objects
  shaped for it.

The cost is one CPU copy of the hit bytes on D, plus transient L1 for both
copies. Phase 3 measures it.

## Head mapping

This follows vLLM's `QKVParallelLinear`, with H the model's KV head count and
`replicas = max(1, tp // H)`. Rank `r` holds `max(1, H // tp)` heads, starting
at `(r // replicas) * max(1, H // tp)`. `tp_reshard.plan_head_reshard` walks a
destination rank's heads and takes each one from the lowest source rank that
holds it (replicas are identical).

```
H=8, A=2, B=4: b=3 <- a=1 heads 2..3         (one source per destination)
H=8, A=4, B=2: b=1 <- a=2 heads 0..1, a=3 heads 0..1   (gather)
H=2, A=4, B=1: b=0 <- a=0 head 0, a=2 head 0  (replicated source)
```

H is not derivable from a rank's shape once ranks replicate heads, so vLLM
sends it as the `total_num_kv_heads` layout hint. vLLM sends the hint only
when the reshard is valid:

- **Model:** not MLA, not hybrid, not attention-free (`vllm_layout_hints`).
- **Deployment:** server world size equal to TP and DCP=1
  (`LMCacheMPWorkerAdapter.register_kv_caches`). PP, DCP or several servers
  make server ranks differ from TP ranks.

Without the hint the fallback is off.

## Lookup stages

`LookupModule` holds the fallback as `_HeteroTPState` on the prefetch job.
Each stage is an ordinary prefetch, and the job's `handle` and `row_windows`
are swapped together when a stage is replaced.

1. **Primary stage** (B keys): if it finds no chunk and a candidate remains,
   the next stage is submitted. A partial hit is final.
2. **Source stage** (A keys, layouts from `reshard_layout_desc`,
   `num_kv_readers=1`): a hit is resharded on a background thread, which
   then submits the final stage. A miss moves on to the next candidate.
3. **Final stage** (B keys, `allowed_l2_memory_tags=frozenset()`): its result
   is the lookup's result.

Stages advance inside `query_prefetch_status` / `wait_prefetch_status`. The
reshard runs on `LookupModule`'s own thread pool (4 threads), not on the
request handler's thread, because its copy grows with the hit:
`query_prefetch_status` returns None while it runs, and `wait_prefetch_status`
waits for it, with `timeout` covering every stage. A reshard whose
destination keys another running reshard is writing waits for that one and
writes only the rest; otherwise it would find the keys reserved and the
second lookup of a prefix would miss. Each reshard logs its cost (Qwen3-0.6B,
2048 tokens, measured):

```
Resharded 8 chunks from TP1 to TP2 for request <id>: wrote 224.0 MiB in 38.36 ms (queued 0.20 ms, waited 0.00 ms on 0 other reshards)
```

`free_lookup_locks` consumes without advancing. A source-stage result has its
A read locks released and reports a miss; a running reshard is waited for, so
the release sees the final stage's locks. An exception in any stage is logged
and reported as a miss, so the job always resolves. If a reshard fails
mid-group, the destination objects reserved for that group stay reserved
until the write TTL, because the storage manager cannot abandon a
reservation.

## Configuration

```bash
lmcache server ... --hetero-tp-source-world-sizes 1 4
```

These are the source TP sizes to try, in order. Phase 1 has no discovery:
the operator lists the TP sizes of the other engines.

## Constraints (phase 1)

- Attention-only, non-MLA models without sliding-window object groups.
- LMCache-driven transfer. The engine-driven registration does not send the
  head-count hint, so the fallback is off there.
- The fallback runs only when the primary stage finds no chunk at all.

## Verification

The e2e test is native PD on one node: two servers, a coordinator, NIXL P2P,
`chunk_size=256`, and a 2064-token prompt. Results:

- Qwen3-0.6B (8 KV heads): every ordered pair in {1, 2, 4, 8}.
- Qwen2.5-1.5B (2 KV heads, replicated at TP4): every ordered pair in
  {1, 2, 4}.
- Every pair hit 2048/2064 tokens on D.
- The first-token top-5 logprobs on D matched a cold D run within the
  same-TP noise, and top-1 was the same token.
- Controls with a deliberately broken plan produced garbage text.

## Next phases

| Phase | Content |
| --- | --- |
| 2 | GDN and Mamba2 recurrent groups, and hybrid subpaged attention. This needs a per-kernel-group shard description (segments sharded along one axis), because GDN conv is `[q\|k\|v]_local` and ssm splits by v heads. chunk_size must be a multiple of both engines' block sizes, which can differ by TP in hybrids. |
| 3 | Measure the reshard cost on the PD handoff (logged per reshard); move it to the GPU if it matters. |

## Open questions

1. Source TP discovery: should a peer advertise its layouts (P2P RPC or
   coordinator metadata) instead of using static config?
2. Should a partial hit at B also try the source TPs? A shared system-prompt
   chunk at B currently hides a long prefix at A.
3. After a reshard, keep the A copies in D's L1 for later TP=A readers, or
   drop them?
4. The NIXL load does not compare the remote object size with the local
   buffer (`P2PL2Adapter.submit_load_task`). Fix it separately.
