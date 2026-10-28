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
D lookup (B keys, c0..cn)  ── same-TP stage, L1/L2/P2P ────▶ hit c0..ck-1, num_kv_readers locks
D lookup (A keys, ck..cn)  ── cross-TP stage, L1/L2/P2P ──▶ hit ck..cm-1: A objects in D L1, 1 lock each
reshard                    ── A objects ─▶ B objects in D L1 (head-slice copies),
                              admitted with num_kv_readers locks; release A read locks
answer                     ── same-TP hit + resharded hit = c0..cm-1
retrieve                   ── unchanged
```

Why here (the rejected placements are under
[Where hetero TP lives](#where-hetero-tp-lives)):

- **Retrieve is unchanged.** Locks and the GPU kernel see ordinary B
  objects. The network moves exactly P's bytes, once.
- **On the CPU, not the GPU.** GPU time is the scarce resource on D; the
  copy stays on D's CPU until its throughput falls below the rate D receives
  cross-TP data (see [Next phases](#next-phases)).

The copy is a strided slice, not a transpose. Per kernel group an object is
`[kv_size, L, T, local_heads * head_elems]` with heads innermost, so a
rank's heads are one contiguous run inside each token row:

```
A=1 -> B=2: dst_b[kv, l, t, :] = src[kv, l, t, b*n*e : (b+1)*n*e]   (slice)
A=2 -> B=1: dst[kv, l, t, :]   = concat_a src_a[kv, l, t, :]         (concat)
```

with `n` heads per destination rank and `e` elements per head. Each run is
small (Qwen3-0.6B: 1 KiB per token and layer at TP2), so P2P cannot read a
rank's share as sub-ranges of P's object; it pulls whole objects and D
slices them. vLLM's NIXL connector avoids the copy because its block layout
puts heads outside tokens, which makes a rank's share one large run.

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
gets no cross-TP stage and misses as before.

vLLM sends H as the `total_num_kv_heads` layout hint. The shape alone cannot
tell a rank holding one of H = tp heads from one holding a replicated head,
so the hint is what rules the second case out. vLLM sends the hint only when
the reshard is valid:

- **Model:** not MLA, not hybrid, not attention-free (`vllm_layout_hints`).
- **Deployment:** server world size equal to TP and DCP=1
  (`LMCacheMPWorkerAdapter.register_kv_caches`). PP, DCP or several servers
  make server ranks differ from TP ranks.

Without the hint the cross-TP stage is off.

## Where hetero TP lives

Source-TP objects can sit in any tier, L1 included:

- **Peer L1 over P2P.** The P2P adapter reads the peer's DRAM L1 (tag
  `DRAM, REMOTE`); in PD that is P's L1, written at A.
- **Shared L1.** Several servers map one region through the memory
  orchestrator; servers whose engines differ in TP write both A and B keys
  into it.
- **Local L1 and L2.** A server or an L2 shared by engines at different TP.

So the cross-TP stage sits in a module above the storage manager and calls
`submit_prefetch_task` over every tier, the same way the P2P controller
serves a peer's lookup (`p2p_controller.py`). The root cause is that keys
match exactly and `kv_rank` encodes the TP; every placement has to map B
keys to A keys and reshape the data somewhere.

| Placement | Why not |
| --- | --- |
| Head-range keys and a cover lookup in L1 (the heads x layers `kv_rank` bitmap in `ComputeKVRank`'s TODO); retrieve reads strided head ranges | No reshard copy and one lookup, but it changes key encoding, the L1 index and the GPU connector, and orphans every persisted L2 object |
| An L2 adapter wrapper that reshards on load, like `SerdeL2AdapterWrapper` | Sees only the adapter it wraps, never L1 |
| An L2 adapter that maps B keys to A keys and looks them up in its own storage manager | No L2 adapter reads local L1 today, and the lookup re-enters the prefetch controller it is serving |
| Reshard on P: P's P2P controller sees B's world size in the keys and serves B objects | Covers P2P only, not local or shared L1. Kept as an option for native PD, where it would remove the source TP setting on D |

Phase 1 below implements the cross-TP stage in the lookup.

## Lookup stages

`LookupModule` never deals with heads or TP sizes. It sends every prefetch
through `hetero_kv_view.HeteroKVView`, which has the storage manager's
prefetch interface (`submit_prefetch_task`, `query_prefetch_status`,
`wait_prefetch_status`), so the lookup code is the same as without hetero TP:

```
LookupModule --spec(B keys)--> HeteroKVView --same-TP spec--> StorageManager
                               (background)  --cross-TP spec->      |
                               merged result <---- L1Resharder <----+
```

The view is generic; the TP knowledge lives in `tp_reshard`:

- `HeteroKVView.plan(spec)` accepts only what it can merge, a locked
  `prefix` prefetch, and asks `HeteroTPPlanner.plan` for the rest: rows that
  fan out to every rank of one world size (a lookup), a source TP configured
  and different from it, head count reported, both TP sizes divide it, no
  sliding window. Anything else, including every other module's prefetches,
  passes through unchanged.
- `HeteroTPPlan.cross_tp_spec(k)` rewrites the spec's rows for chunks from
  `k` on to the source TP's ranks and layouts, keeping the spec's policy and
  tier filters; `convert` reshards a source hit through `L1Resharder`;
  `merge` appends the resharded chunks to the same-TP result.
- A prefetch with a plan runs its stages on a background thread
  (`hetero-kv-lookup`, 32 threads). The view returns the storage manager's
  own handle and answers `query`/`wait` for it from the merged result, once.

1. **Same-TP stage** (B keys, all chunks, every tier): a full hit is final.
   Otherwise its locks stay held, as part of the answer, and the cross-TP
   stage is submitted.
2. **Cross-TP stage** (A keys, chunks k..n, layouts from `reshard_layout_desc`,
   `num_kv_readers=1`): a hit is resharded on a background thread. A miss
   releases its locks and the same-TP stage answers.
3. **Reshard**: each B object it writes is admitted with `num_kv_readers`
   read locks in one step (`StorageManager.finish_write_and_reserve_read`),
   so eviction cannot take it before retrieve. A B object another writer
   admitted first is locked as found (`StorageManager.reserve_read`). The
   hit ends at the first chunk with an object neither (out of L1 memory, or
   still being written); locks past it are released. The lookup answers
   with the same-TP result plus these chunks, counted as L2 hits.

The retrieve path needs exactly what a prefetch leaves: every hit object
read-locked `num_kv_readers` times, plus one result with its L1 owners. The
reshard leaves the same, so no further prefetch is needed to lock the new
objects.

The stages run as one sequential function rather than being advanced by the
engine's polls, so the lookup keeps no stage state:
`query_prefetch_status` returns None until every stage is done, and
`wait_prefetch_status` waits for all of them, with `timeout` covering every
stage. Each
lookup holds a thread while it waits for its prefetches, so the pool size
bounds the cross-TP lookups in flight; a lookup past that queues before its
cross-TP stage. The copy runs in `L1Resharder`, on its own thread pool (4
threads), which bounds concurrent copies. A reshard whose
destination keys another running reshard is writing waits for that one and
writes only the rest; otherwise it would find the keys reserved and the
second lookup of a prefix would miss. Each reshard logs its cost (Qwen3-0.6B,
2048 tokens, measured):

```
Resharded 8 chunks from TP1 to TP2 for request <id>: wrote 224.0 MiB in 38.36 ms (queued 0.20 ms, waited 0.00 ms on 0 other reshards)
```

`free_lookup_locks` on a cross-TP lookup that is still running behaves as for
any running prefetch: it returns at once, and the result keeps its locks for
a later poll or until the read-lock TTL. A finished result is handed over and
released as usual, including the locks the reshard took. An exception in the cross-TP
stage, or a failed reshard, is logged and the same-TP stage answers, so the
future always resolves and every lock it took is released. If a reshard fails
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
operator sets it. A lookup whose own TP equals it skips the cross-TP stage.

## Constraints (phase 1)

- Attention-only, non-MLA models without sliding-window object groups.
- Both TP sizes divide the model's KV head count.
- LMCache-driven transfer. The engine-driven registration does not send the
  head-count hint, so the cross-TP stage is off there.
- One source TP per server.
- Engine KV formats are not a constraint: each engine's GPU connector maps
  its own layout (vLLM and SGLang differ) to the LMCache object format above,
  and the reshard works only on that format. Both sides need the same
  model, dtype, head size and `chunk_size`, or the keys or shapes differ.
- The engine must send the KV head count. vLLM does; the SGLang MP connector
  sends only `kv_list_layout`, so the cross-TP stage is off for SGLang.

## Later: a PD source

The cross-TP stage is the single extension point for native PD. Two things
change there, inside `HeteroKVView`'s TP plan, and `LookupModule` does not:

- **Where A comes from.** From the request (the PD coordinator knows which P
  served it) or from a P2P RPC that asks the peer for its layout, instead of
  the static config value.
- **Which tiers the cross-TP stage reads.** PD pulls from P only, so the
  cross-TP stage is restricted to the P2P adapter's `REMOTE` tag through its
  `allowed_l2_memory_tags`. The same-TP stage stays on every tier.

**TODO (PD): take the source TP from `kv_transfer_params`, not from config.**
In vLLM PD the proxy already carries P's `kv_transfer_params` (returned by
P's connector in `request_finished`) into D's request. The plan:

1. P's `LMCacheMPConnector.request_finished` adds its KV world size to the
   returned params; today it returns only `num_lmcache_extra_cached_tokens`.
2. The proxy forwards the params to D unchanged, as it already does.
3. D's connector passes the value in the lookup RPC, and `HeteroTPPlanner.plan`
   uses it instead of `--hetero-tp-source-world-size`.

The same channel can name the P instance, which xPyD needs: today the P2P
controller builds one adapter per peer the coordinator lists, and a lookup
asks all of them.

## Extensions

`HeteroKVView` is named for heterogeneous engines in general, not for TP:
TP is its first plan. Where the rest would go, none of it built yet:

- **More forms (vertical).** Another difference between the asking engine
  and the stored objects (hybrid models' sliding-window and recurrent groups,
  KV dtype) is another plan next to `HeteroTPPlanner`, with the same
  `cross_tp_spec`/`convert`/`merge` steps. `LookupModule` does not change.
- **GPU conversion (performance).** `convert` is the only step that moves
  bytes. A GPU version would not write B objects into L1; it would return
  the A keys with a head mapping, and the retrieve kernel would copy each B
  rank's heads straight from the A objects into the paged KV. That removes
  the CPU copy and the extra L1 copy, but retrieve then has to know about A
  keys, and each B rank reads whole A objects when A < B. Do it only when the
  reshard log shows the CPU path is the bottleneck.
- **Parallel stages (performance).** The view runs the two prefetches one
  after the other. For PD it can submit both at once; only the view changes.
- **More callers (horizontal).** Other modules (warm prefetch, blend) call
  the storage manager directly. If hetero TP grows beyond PD, so that A and B
  objects live side by side in a shared L2, they switch to the view; the
  view then has to merge `full` and `NO_LOCK` prefetches too.

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
| 3 | If the CPU copy cannot keep up with the cross-TP receive rate, fold the head slicing into the retrieve GPU kernel: it already scatters each object into the engine's paged layout, so reading head ranges from several source objects moves no extra bytes and needs no CPU copy. The cost is a retrieve that reads source-TP objects directly. Gate: measured reshard throughput per thread (about 5.8 GB/s at 224 MiB / 38 ms) and its thread scaling against the PD deployment's cross-TP receive rate. |

## Open questions

1. Source TP discovery for generic hetero TP. The coordinator cannot answer
   it today: registration carries no model or TP; the coordinator only
   tells the P2P controller which peers exist. The candidate is a P2P RPC that returns the
   world sizes the peer has registered for a model. A peer can hold several,
   so the answer is a list.

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
