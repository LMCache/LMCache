# Unfull Chunks

## Summary

LMCache stores KV in fixed-size chunks and, until now, dropped the trailing
partial ("unfull") chunk of every request. For attention-only models that
costs at most one chunk of recompute. For hybrid models it is worse:

- The chunk size must be a multiple of vLLM's unified block size (400 to
  ~1152 tokens for Qwen3.8 / GLM-5.3 / DeepSeek-V4.1), so the dropped tail is
  large.
- A linear-attention (Mamba/GDN) layer can only resume where a state snapshot
  exists, so the recurrent state at the resume point must be stored too.

This feature lets LMCache store and load the unfull chunk at the end of a
request's prompt, so a later request resumes past `kC`. Two kinds of tail
serve the two consumers; both are stored as the same kind of object:

| Tail | Ends at | Stored by | Serves |
|---|---|---|---|
| Exact | `P`, the prompt end | requests that end at the prompt (`max_tokens == 1`, e.g. a disaggregated prefill) | the disaggregated decode, exactly where the prefill stopped |
| Grid | `B = floor((P - 1) / g) * g` on vLLM's hit grid `g` | every request | a later request sharing the prefix (the same prompt again, or the next round of a chat), where vLLM's own prefix cache could resume |

A lookup probes its own end first, so an exact tail wins, then the grid ends
below it. This is the shared groundwork for PD handoff and multi-round reuse;
neither policy lives here.

**Exact tail (PD).** A request that ends at its prompt runs no later forward,
so its block table describes exactly `P`. This is the LMCache counterpart of
NIXL's P/D handoff, where the prefill stops at `P - b` and the decode loads
exactly that and recomputes `b` tokens (`b = 0` for attention-only models,
`1` with Mamba):

- Attention-only: the decode looks up `P` and hits the exact tail; it
  recomputes nothing LMCache could serve.
- Mamba: the decode looks up `P - 1`, the state it can resume from, while the
  prefill's exact tail holds the state at `P`. The two meet once the prefill
  stops at `P - 1`, the NIXL truncation (the P/D change). Until then the
  decode hits the grid tail and recomputes `P - 1 - B < g` tokens more than
  NIXL would.

**Grid tail (reuse).** Why a grid rather than each request's exact end:

- vLLM always recomputes the last prompt token for its logits, so a later
  request with the same prompt resumes at `P - 1`, not `P`; with Mamba the
  connector looks up `P - 1` tokens, and a state at `P` cannot serve it.
- A Mamba state exists only where vLLM snapshots one. On the grid, vLLM hands
  the snapshot block over (boundary-state hand-offs), a copy the next forward
  does not rewrite, so the store is safe while the request keeps decoding.
- A lookup does not know where an earlier request's tail ended. With every
  tail on the grid it probes at most `C / g` candidate ends.

Enabled per connector with `lmcache.mp.save_unfull_chunk: true`
(`kv_connector_extra_config`). Off by default: nothing changes for existing
deployments.

Compressed-attention models (QSA, DSA kpool, DSV4.1 compressor) are not
supported yet and are rejected at startup: resuming mid-chunk also needs the
raw keys of the still-open compression group, which vLLM keeps in a
per-request ring (`CircularBufferSpec`) that LMCache skips (#5042). Storing
that ring is a follow-up.

## Concepts

- **Unfull chunk**: a token range `[kC, B)` with `kC = floor(B / C) * C`
  and `0 < B - kC < C`, where `B` is the exact or the grid end and `C` the
  LMCache chunk size. It is stored as one object per object group with the
  regular chunk layout, plus one block per narrow sliding-window group (see
  below); only the blocks that cover `[kC, B)` are copied.
- **Hit grid `g`**: where vLLM's prefix-cache hits land. For a model with
  align-mode Mamba groups it is vLLM's `hash_block_size` (sub-block hits);
  otherwise the scheduler block, the LMCache connector's
  `_hit_alignment_tokens`.
- **Temporary object**: an L1 object deleted when its last read lock is
  released (`L1ObjectState.is_temporary`), i.e. read once.

## Unfull chunk key

```text
unfull_hash = H(tokens[kC:B], prefix = chunk_hash[k-1])
```

The unfull chunk is hashed like the next full chunk, from the rolling hash
of the full chunks before it. It has fewer than `C` tokens, and every hash
algorithm LMCache supports encodes the token count (blake3 hashes the packed
tokens, `sha256_cbor` and the builtin tuple hash include the tuple length),
so its key differs from every full chunk's: a temporary unfull object never
collides with a persistent full chunk (`reserve_write` would otherwise return
`KEY_NOT_WRITABLE`). A new hash algorithm must keep that property. The key
covers the exact length `B`.

A STORE or RETRIEVE `IPCCacheServerKey` addresses the unfull chunk only when
`include_unfull_chunk` is set and `end` is not chunk-aligned; it is then the
last chunk of the key's range, after any full chunks. Without the flag a
non-aligned `end` keeps its existing meaning (partial chunks are discarded),
which `free_lookup_locks` callers rely on. A LOOKUP key with the flag asks
for the longest unfull chunk on the grid after its hit prefix, up to `end`.

## Data flow

```text
store (B: grid end, or P exactly)           load (lookup end L)
----------------------------------------    -------------------------------------------
STORE [kC, B), include_unfull_chunk          LOOKUP [0, L), include_unfull_chunk,
  blocks: ceil(B/T_g) slices per group         hint lmcache.hint.unfull_chunk_grid = g
  Mamba: the block vLLM handed over            chunk prefetch -> k full chunks hit
  is_temporary from request config             probe e = L (if inside chunk k), then
  (prompt-ending request: also [kC, P),          floor(min(L, kC+C-1)/g)*g, -g, ... > kC
   all blocks from the block table)              (L1 only), stop at first hit;
                                                 read-locked like a chunk
                                               reply (in tokens): e, or k*C
                                             RETRIEVE [s, e), include_unfull_chunk
```

Example, `C = 128`, `g = 16`, `P = 337`: the prompt stores full chunks
`[0, 256)` and the tail `[256, 336)`. The same prompt looks up `L = 336`,
hits both chunks, probes `e = 336` first and hits. A next round that extends
the prompt to 900 tokens hits the same two chunks (round 1 stored no chunk
`[256, 384)`), probes `e = 368, 352, 336` and hits `336`.

A lookup that sets `include_unfull_chunk` is answered in tokens rather than
chunks (`query_prefetch_status`), so the reply can carry `e`. When servers
disagree, the client keeps the full chunks every server hit and frees each
server's locks above them, its unfull chunk included.

A tail store is its own operation `[kC, B)` with no full chunk: vLLM hands the
Mamba snapshot over a step after it computes `B`, usually after the full
chunks were stored. A request can then send two stores in one step; the
worker keeps only its latest store future, which is sound because one
worker's stores run in order on the server (client-affinity pool, one copy
stream), the same ordering that future already relies on across steps. A retrieve of a hit ending mid-chunk carries the unfull
chunk in the same operation as the full chunks: its block IDs follow theirs in
every group. The server handles the full chunks exactly as before, then the
unfull chunk inside the same transaction (same producer-event wait,
all-or-nothing commit, same read-lock release).

The unfull chunk is a single object per object group. Its per-group block
list is trimmed to the last `min(n, keep)` blocks, where `keep` is the
group's per-chunk window in blocks (all blocks for full attention, the last
window for sliding window, one block for Mamba state).

The copy kernel requires exactly `keep` blocks per object
(`num_blocks_per_object * block_size == slots_per_chunk`), so a shorter list
is right-aligned instead of changing the kernel: it is padded at the front
with placeholder ids and the padding is skipped with the kernel's existing
`skip_prefix_n_blocks`, so it is never read or written:

```text
full attention, keep = 16, 3 real blocks:
  block ids : [b0 b0 b0 ... b0 | b0 b1 b2]     (13 placeholders + 3 real)
  skip      : 13
  object    : slots 0..12 untouched, slots 13..15 <- b0 b1 b2
```

Store and load fit the list the same way, so the real block at position `i`
lands in the same object slot on both sides. The unfull object always takes
the staging-kernel path (`transfer_unfull_chunk`), never the direct-copy
path, whose per-chunk planning assumes full chunks. An object group whose
unfull blocks are all null (no real state) stores nothing.

Sliding-window groups need the window ending at `B`, which can start in
chunk `k-1`. The chunk lookup already retains the last window of the hit
prefix (chunk `k-1`), so the full-chunk retrieve restores the earlier part
and the unfull retrieve the rest. Mamba state needs only the block holding
the state at `B`: on store the connector substitutes the block vLLM handed
over; on retrieve it is the last block of the unfull range, where vLLM
expects a resumed request's state.

### Window lead block

A sliding window of `W` tokens narrower than a chunk (`W < C`) keeps
`keep = W / T` blocks per chunk (`T` tokens per block). The token after `B`
attends to `[B - W + 1, B)`; a full chunk ends on a block boundary, so that
range fits in `keep` blocks, but an unfull chunk can end inside a block and
the range then spans `keep + 1` blocks. Example with `T = 4`, `W = 8`,
`keep = 2`, unfull chunk of `n` tokens:

```text
n = 9   window [2, 9)   blocks 0 1 2   -> keep + 1, lead block 0
n = 11  window [4, 11)  blocks 1 2     -> keep
n = 13  window [6, 13)  blocks 1 2 3   -> keep + 1, lead block 1
```

This is the overlap NIXL covers with `cdiv(W, T) + 1` window blocks for P/D;
vLLM's own prefix cache never needs it because its sliding-window hits end
on block boundaries. The block before the last `keep`, the *lead block*, is
kept in the list when needed; both sides derive that from `B mod C`, so it
needs no metadata. Like the full-chunk window, this assumes `W` is a multiple of `T`
(`keep` rounds down).

The unfull object reserves room for it with its own layout
(`get_unfull_chunk_layout_desc`): the full-chunk layout of the object group,
followed by one single-block entry per sliding-window group with `W < C`.
Appending keeps every regular region at its full-chunk offset, and the
full-chunk layout (and everything already stored with it) is unchanged.
The server registers these layouts next to the full-chunk ones
(`LayoutDescRegistry.find_unfull_group_layout_descs`, falling back to the
full-chunk layouts) for the lookup's L1 probe.

The copy kernel is unchanged: the lead block is a second launch with one
block and `lmcache_chunk_size` equal to one block's slots, into the appended
region; the regular launch is as before. Such an object does not fit the
shared staging buffer, so it stages through a buffer allocated on the
transfer stream (stream-ordered reuse makes freeing it after the async copy
safe). On the decode side the lead block is the first block vLLM keeps
(`get_num_skipped_tokens = B - W + 1`), so it is allocated, not null. With
the grid, a sliding-window model's `g` is the scheduler block, a whole number
of sliding-window blocks, so its tails end on a block boundary and need no
lead block; it matters for models whose hit grid is finer than their window
blocks.

## Temporary objects

`request_configs["lmcache.temporary_unfull_chunk"] = True` stores the
request's unfull chunk with `is_temporary`. L1 then deletes it when the last
read lock of a lookup is released (a retrieve on every worker, or
`free_lookup_locks` for an unused hit), and never offloads it to L2. An
unread temporary object is evicted like any other.

## Lookup, locks and sessions

The unfull probe is a prefetch whose spec allows no L2 adapter
(`allowed_l2_memory_tags=frozenset()`). It reads L1 synchronously and takes
`num_kv_readers` read locks per key like the chunk prefetch. The session records the probed keys and their L1 owners
separately from the chunk prefetch result:

- each worker's unfull retrieve releases one lock per key;
- `FREE_LOOKUP_LOCKS` with `include_unfull_chunk` whose `end` is the hit
  `e` (recorded on the session) releases the remaining locks once (vLLM
  already holds the tokens, or the servers disagree on hits);
- anything else is left to the read-lock TTL, like a chunk lock.

## Connector behavior (vLLM MP connector)

- **Grid**: with align-mode Mamba groups, `g` is vLLM's `hash_block_size`
  (`resolve_kv_cache_block_sizes`); otherwise the scheduler block. Sent to the
  server as `lmcache.hint.unfull_chunk_grid` with every lookup.
- **Store, align-mode Mamba**: `SchedulerOutput.kv_connector_block_state`
  carries vLLM's boundary-state hand-offs, `(group, block, boundary)`. vLLM
  hands over only boundaries it can resume from: retained block-end states,
  and with sub-block hits the copy-on-write copy of the prompt-tail state
  (`floor((P - 1) / h) * h`). Each boundary inside the prompt's last chunk
  (`kC < B <= P - 1`) handed over for every Mamba group is stored as a tail,
  with the handed-over block in place of the group's last block. Choosing
  `B` from the hand-offs follows whatever vLLM resolved (sub-block hits or
  not) without re-deriving its rules.
- **Store, other models**: the tail ends at `floor((P - 1) / g) * g`, stored
  once the step computing it is scheduled; its blocks come from the block
  table.
- **Store, requests that end at the prompt**: also the exact tail `[kC, P)`
  once the prompt is computed, with every block (Mamba's running block
  included) from the block table. Skipped when `P` is chunk-aligned, where
  the full chunk covers it.
- **Hand-off timing**: hand-offs are read for the requests scheduled in the
  step; one offered for a request not scheduled then is dropped, and that
  boundary gets no tail.
- **Pinning**: a handed-over block is not in the request's block table, and
  vLLM frees it once the request moves on, so the connector pins it
  (`BlockPool.touch`) when it queues the store and unpins it
  (`free_blocks`) when the request appears in `finished_sending`, after its
  workers finished every store. One block per Mamba group per request.
- **Store invariant**: the tail's blocks must not be written after its store
  is submitted. Attention rows below `B` are never rewritten, and the Mamba
  snapshot is a copy, so the tail is stored while the request keeps
  decoding. A tail is stored once (`tracker.unfull_chunk_ends`), and not when
  the request's own lookup already hit it.
- **Lookup**: sends `end = L` (the lookup token count, after the Mamba
  reserve-last-token rule) with `include_unfull_chunk`, so a hit can end
  mid-chunk.
- **Retrieve**: a hit that ends mid-chunk is retrieved as one operation
  `[start, e)` with `include_unfull_chunk`.
- **Overlap with vLLM's prefix cache**: if vLLM already holds tokens past
  `kC`, loading the unfull chunk would overwrite blocks other requests read,
  so the connector releases the unfull chunk's locks and keeps the hit at
  `kC`.
- **Startup checks**: the option is rejected for models with a
  `CircularBufferSpec` ring (see Summary), with speculative decoding (draft
  rows sit past `P` in relocated Mamba blocks), with
  `lmcache.mp.lazy_offload`, when the worker resolves to an engine-driven
  transfer mode, and for align-mode Mamba on a vLLM without boundary-state
  hand-offs.

## Limits

- Served from L1 only (a persistent unfull object offloaded to L2 is not
  looked up there); LMCache-driven transfer mode only (the CUDA default).
- A prompt shorter than one chunk has no full chunk to look up, so its
  unfull chunk is never probed.
- A lookup probes up to `C / g - 1` ends, each an L1 prefetch; the count is
  in `MP_LOOKUP_PREFETCH_END` (`unfull_chunk_probes`). A non-locking L1
  presence check would make misses cheaper.
- Align-mode Mamba backends that export the prompt-tail state inside a
  prefill chunk (internal prefill checkpoints) hand over no tail state, so
  only block-end boundaries are stored for them.
- Sliding-window blocks below the window of the request's current position
  can be freed while a tail store is in flight, the same exposure as the
  full-chunk sliding-window store.
- A failed unfull-chunk retrieve, or a lookup abandoned for an APC-shrink
  re-lookup (`apc_covered_lookup.md`), leaves the unfull chunk's read locks
  (and a temporary object) to the read-lock TTL: those releases resolve full
  chunks only.
- A model with a sliding window narrower than a chunk stores no unfull
  chunk when an L1 is GDS-backed: GDS copies only through its registered
  staging buffer, and the lead-block layout does not fit it. Supporting it
  means a registered unfull staging region per platform.
- Only prompt tails are stored. A next round hits the previous round's
  prompt tail, not its generated tokens.
- vLLM must accept a non-block-aligned external hit (current vLLM does). DSV4.1
  with bounded-replay SWA still replays its window after a load because
  LMCache reports only the prefix-cacheable groups as loaded.
