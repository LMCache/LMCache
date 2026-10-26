# Heterogeneous TP lookup (MP mode)

A decode engine at TP=B can serve KV that a prefill engine stored at TP=A.
Two cases are covered: each engine has its own LMCache server joined by P2P
(the native PD data path of RFC #5969, `lmcache_driven` mode), or both
engines share one server. RFC #5969 leaves asymmetric TP open; this design
keeps the store in each rank's native layout and reshards on D. This
document covers phase 1: vLLM to vLLM, attention-only, non-MLA models.
Later phases are listed under [Next phases](#next-phases).

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
copies. Measured on Qwen3-0.6B, 2048 tokens (224 MiB written): about 12 ms
alone and 15-73 ms under 12 concurrent requests. Each reshard logs its cost
(see [Lookup stages](#lookup-stages)).

## Head mapping

An object does not record which KV heads it holds. With H the model's KV
head count, both TP sizes must divide H, and rank `r` holds the contiguous
heads `[r * H/tp, (r + 1) * H/tp)`. `tp_reshard.plan_head_reshard` walks a
destination rank's heads and takes each one from the source rank that holds
it.

```
H=8, A=2, B=4: b=3 <- a=1 heads 2..3                  (one source per destination)
H=8, A=4, B=2: b=1 <- a=2 heads 0..1, a=3 heads 0..1  (gather)
```

A TP size larger than H is not supported, by choice. The engine then
replicates heads across ranks, and which ranks share a head is the engine's
own rule (vLLM's `QKVParallelLinear` groups adjacent ranks). Contiguous
splitting is the only assumption this design makes about the writer; adding
a replication rule would tie the store format to one engine. Such a lookup
gets no fallback and misses as before.

vLLM sends H as the `total_num_kv_heads` layout hint. The shape alone cannot
tell a rank holding one of H = tp heads from one holding a replicated head,
so the hint is what rules the second case out. vLLM sends the hint only when
the reshard is valid:

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

## After a reshard: only B stays

D keeps only the converted B objects; the A copies it pulled are dropped.
Nothing in the fallback deletes them: the default prefetch policy marks every
object loaded from L2 (P2P included) temporary, so L1 deletes it when the
reshard releases its read lock. A objects that were already in L1, written by
a TP=A engine sharing this server, are not temporary and stay. With
`--l2-prefetch-policy retain` the A copies stay until eviction, as every L2
load does under that policy.

## Configuration

```bash
lmcache server ... --hetero-tp-source-world-sizes 1 4
```

These are the source TP sizes to try, in order. Phase 1 has no discovery:
the operator lists the TP sizes of the other engines.

## Constraints (phase 1)

- Attention-only, non-MLA models without sliding-window object groups.
- Both TP sizes divide the model's KV head count.
- LMCache-driven transfer. The engine-driven registration does not send the
  head-count hint, so the fallback is off there.
- The fallback runs only when the primary stage finds no chunk at all. See
  [Known limitation](#known-limitation-partial-primary-hit).

## Known limitation: partial primary hit

If D already holds the first k chunks at TP=B, the primary stage finds k
chunks, the lookup reports k, and D recomputes the rest, even when the
source TP holds the whole prefix:

```
P store (A keys):  [c0][c1][c2][c3]
D L1 (B keys):     [c0]                 shared system prompt
D lookup:          primary hits c0 -> no fallback -> 1 chunk hit, c1..c3 recomputed
```

A common source of the B chunk is an earlier reshard: request 1 leaves its
B objects in D's L1, and request 2 with the same system prompt hits them.
vLLM's prefix cache on D hides this while it still holds those blocks,
because the lookup then starts after them.

Phase 1 does not handle this; it is the next change. The fix: on a partial
primary hit, look up chunks k..n at the source TP, reshard only those, and
return one combined hit.

## Verification

The e2e test is native PD on one node: two servers, a coordinator, NIXL P2P,
`chunk_size=256`, and a 2064-token prompt. Results:

- Qwen3-0.6B (8 KV heads): every ordered pair in {1, 2, 4, 8}.
- Qwen2.5-1.5B (2 KV heads): TP1 to TP2 and TP2 to TP1.
- Every pair hit 2048/2064 tokens on D.
- The first-token top-5 logprobs on D matched a cold D run within the
  same-TP noise, and top-1 was the same token.
- Controls with a deliberately broken plan produced garbage text.

## Next phases

| Phase | Content |
| --- | --- |
| next | Fall back on a partial primary hit ([Known limitation](#known-limitation-partial-primary-hit)). |
| 2 | GDN and Mamba2 recurrent groups, and hybrid subpaged attention. This needs a per-kernel-group shard description (segments sharded along one axis), because GDN conv is `[q\|k\|v]_local` and ssm splits by v heads. chunk_size must be a multiple of both engines' block sizes, which can differ by TP in hybrids. |
| 3 | Move the reshard copy to the GPU if its logged cost matters on the PD handoff. |

## Open questions

1. Source TP discovery: should a peer advertise its layouts (P2P RPC or
   coordinator metadata) instead of using static config?

## Pre-existing P2P gap: no object size check

This is a P2P issue that exists at equal TP too; this change neither
introduces nor fixes it. `P2PL2Adapter.submit_load_task` hands NIXL the
peer's addresses (which carry the peer object's size) and the local buffers
(sized from the reader's layout) without comparing the two sizes, and the
NIXL channel pairs them page by page without checking either. An object key
does not encode dtype or head count, so a key can match while the shapes
differ, for example P and D with different KV dtypes. When both sizes
round to the same number of pages, D reads wrong bytes and reports a hit.

It matters more here because D derives the source shape (H / A heads per
rank) instead of reading it from its own registration. The fix is a size
comparison in `submit_load_task` that fails the mismatched keys, in its own
PR.
