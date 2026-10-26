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

The store path is unchanged. A lookup at TP=B first counts what it already
hits with its own keys, on every tier. For the chunks after that prefix it
looks up the source TP's keys, pulls those objects unchanged, and copies
their head slices into B-keyed, B-shaped objects in its own L1, read-locked
for the engine. Everything after that (retrieve, free accounting, the GPU
kernel) sees an ordinary B-keyed hit.

```
D lookup (B keys, c0..cn)  ── local stage, L1/L2/P2P ───▶ hit c0..ck-1, num_kv_readers locks
D lookup (A keys, ck..cn)  ── source stage, L1/L2/P2P ──▶ hit ck..cm-1: A objects in D L1, 1 lock each
reshard                    ── A objects ─▶ B objects in D L1 (head-slice copies),
                              admitted with num_kv_readers locks; release A read locks
answer                     ── local hit + resharded hit = c0..cm-1
retrieve                   ── unchanged
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
gets no source stage and misses as before.

vLLM sends H as the `total_num_kv_heads` layout hint. The shape alone cannot
tell a rank holding one of H = tp heads from one holding a replicated head,
so the hint is what rules the second case out. vLLM sends the hint only when
the reshard is valid:

- **Model:** not MLA, not hybrid, not attention-free (`vllm_layout_hints`).
- **Deployment:** server world size equal to TP and DCP=1
  (`LMCacheMPWorkerAdapter.register_kv_caches`). PP, DCP or several servers
  make server ranks differ from TP ranks.

Without the hint the source stage is off.

## Lookup stages

The work splits so that `LookupModule` never deals with heads or TP sizes:

- `tp_reshard.PrefetchStage` is one prefetch: key, chunk hashes, layouts,
  `num_kv_readers` and the allowed L2 tiers.
- `tp_reshard.HeteroTP.plan` decides whether a lookup qualifies (a source TP
  configured and different from the lookup's, head count reported, both TP
  sizes divide it, no sliding window) and returns a per-lookup
  `HeteroTPPlan`.
- `HeteroTPPlan.source_stage(k)` gives the source stage for the chunks from
  `k` on (key, layouts and windows at the source TP); `convert` reshards a
  source hit through `L1Resharder`; `merge` appends the resharded chunks to
  the local stage's result.
- `LookupModule` runs the stages: it submits each as an ordinary prefetch and
  swaps the job's `handle` and `row_windows` together when a stage is
  replaced.

1. **Local stage** (B keys, all chunks, every tier): a full hit is final.
   Otherwise its locks stay held, as part of the answer, and the source
   stage is submitted.
2. **Source stage** (A keys, chunks k..n, layouts from `reshard_layout_desc`,
   `num_kv_readers=1`): a hit is resharded on a background thread. A miss
   releases its locks and the local stage answers.
3. **Reshard**: each B object it writes is admitted with `num_kv_readers`
   read locks in one step (`StorageManager.finish_write_and_reserve_read`),
   so eviction cannot take it before retrieve. A B object another writer
   admitted first is locked as found (`StorageManager.reserve_read`). The
   hit ends at the first chunk with an object neither (out of L1 memory, or
   still being written); locks past it are released. The lookup answers
   with the local result plus these chunks, counted as L2 hits.

The retrieve path needs exactly what a prefetch leaves: every hit object
read-locked `num_kv_readers` times, plus one result with its L1 owners. The
reshard leaves the same, so no further prefetch is needed to lock the new
objects.

Stages advance inside `query_prefetch_status` / `wait_prefetch_status`.
The copy runs in `L1Resharder`, on its own thread pool (4 threads) rather
than the request handler's thread, because it grows with the hit.
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
A read locks released and the local stage answers; a running reshard is
waited for, so the release sees the locks it took. An exception in any
stage, or a failed reshard, is logged and the local stage answers, so the job
always resolves and every lock it took is released. If a reshard fails
mid-group, the destination objects reserved for that group stay reserved
until the write TTL, because the storage manager cannot abandon a
reservation.

## After a reshard: only B stays, in L1

D keeps only the converted B objects; the A copies it pulled are dropped.
The B objects are admitted as prefetched, not as written, so they are not
stored to D's L2: they are derived copies, and the A originals still exist
at the source. After L1 evicts them, a later lookup reshards again.

Nothing in the lookup deletes them: the default prefetch policy marks every
object loaded from L2 (P2P included) temporary, so L1 deletes it when the
reshard releases its read lock. A objects that were already in L1, written by
a TP=A engine sharing this server, are not temporary and stay. With
`--l2-prefetch-policy retain` the A copies stay until eviction, as every L2
load does under that policy.

## Configuration

```bash
lmcache server ... --hetero-tp-source-world-size 1
```

The TP size the other engine stores at. Phase 1 has no discovery: the
operator sets it. A lookup whose own TP equals it skips the source stage.

## Constraints (phase 1)

- Attention-only, non-MLA models without sliding-window object groups.
- Both TP sizes divide the model's KV head count.
- LMCache-driven transfer. The engine-driven registration does not send the
  head-count hint, so the source stage is off there.
- One source TP per server.

## Later: a PD source

The source stage is the single extension point for native PD. Two things
change there, and the stage machine does not:

- **Where A comes from.** From the request (the PD coordinator knows which P
  served it) or from a P2P RPC that asks the peer for its layout, instead of
  the static config value.
- **Which tiers the source stage reads.** PD pulls from P only, so the
  source stage is restricted to the P2P adapter's `REMOTE` tag through its
  `allowed_l2_memory_tags`. The local stage stays on every tier.

**TODO (PD): take the source TP from `kv_transfer_params`, not from config.**
In vLLM PD the proxy already carries P's `kv_transfer_params` (returned by
P's connector in `request_finished`) into D's request. The plan:

1. P's `LMCacheMPConnector.request_finished` adds its KV world size to the
   returned params; today it returns only `num_lmcache_extra_cached_tokens`.
2. The proxy forwards the params to D unchanged, as it already does.
3. D's connector passes the value in the lookup RPC, and `HeteroTP.plan`
   uses it instead of `--hetero-tp-source-world-size`.

The same channel can name the P instance, which xPyD needs: the P2P peer
is a static `peer_mq_server_url` today.

## Verification

The e2e test is native PD on one node: two servers, a coordinator, NIXL P2P,
`chunk_size=256`, and a 2064-token prompt. Results:

- Qwen3-0.6B (8 KV heads): every ordered pair in {1, 2, 4, 8}.
- Qwen2.5-1.5B (2 KV heads): TP1 to TP2 and TP2 to TP1.
- Every pair hit 2048/2064 tokens on D.
- TP1 to TP2 and TP2 to TP1 with D's LMCache already holding the first 2 of
  8 chunks at B (vLLM prefix cache reset): D hit 2048 tokens and resharded
  only the other 6 chunks.
- The first-token top-5 logprobs on D matched a cold D run within the
  same-TP noise, and top-1 was the same token.
- Controls with a deliberately broken plan produced garbage text.

## Next phases

| Phase | Content |
| --- | --- |
| 2 | GDN and Mamba2 recurrent groups, and hybrid subpaged attention. This needs a per-kernel-group shard description (segments sharded along one axis), because GDN conv is `[q\|k\|v]_local` and ssm splits by v heads. chunk_size must be a multiple of both engines' block sizes, which can differ by TP in hybrids. |
| 3 | Move the reshard copy to the GPU if its logged cost matters on the PD handoff. |

## Open questions

1. Source TP discovery: should a peer advertise its layouts (P2P RPC or
   coordinator metadata) instead of using static config? See
   [Later: a PD source](#later-a-pd-source).

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
