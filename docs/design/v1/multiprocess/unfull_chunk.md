# Unfull Chunks

## Summary

LMCache stores KV cache in chunks of `C` tokens. Before this feature it threw
away the last, partial chunk of every prompt (the *unfull chunk*). A later
request with the same prefix then recomputed up to `C - 1` tokens.

That cost is small for attention-only models but large for hybrid models:

- The chunk must be a multiple of vLLM's block size, which is 400 to ~1152
  tokens for Qwen3.8, GLM-5.3 and DeepSeek-V4.1. So the thrown-away part is
  large.
- A Mamba/GDN layer can only resume where its state was saved. Without the
  state at the end of the cached part, the cached KV is not usable.

With this feature, LMCache also stores the unfull chunk at the end of a
prompt, and a lookup can hit part way into a chunk.

```text
prompt, P tokens   |  chunk 0  |  chunk 1  |  chunk 2  | tail |
                   0           C           2C          3C     B   P
before:            [------ stored, hit ---------------]  recomputed
after:             [------ stored, hit ----------------------]  recomputed
                                                         ^ unfull chunk [3C, B)
```

The feature is off by default. With it off, nothing changes.

## How to turn it on

| What | Where | Example |
|---|---|---|
| Turn the feature on | vLLM `kv_connector_extra_config` | `"lmcache.mp.save_unfull_chunk": true` |
| Make vLLM's hit grid finer than the chunk (Mamba hybrids) | `vllm serve` flag | `--prefix-match-unit 32` |
| Make one request's unfull chunk read-once (optional) | the request's `kv_transfer_params` | `"lmcache.temporary_unfull_chunk": true` |

Why `--prefix-match-unit`: LMCache stores the tail at a point on vLLM's hit
grid `g` (see [Which tails are stored](#which-tails-are-stored)). For a Mamba
hybrid, vLLM sets `g` to its block size by default, and the chunk is a
multiple of that. With a one-block chunk (Qwen3.8-27B on vLLM 0.33: block 832,
chunk 832) every grid point is a chunk boundary, so no tail is ever stored.
With `--prefix-match-unit 32` on Qwen3.8-27B and a 4,903-token prompt, a
repeated prompt hits 4,896 tokens instead of 4,160.

Requirements, checked at startup:

- LMCache-driven transfer mode (the CUDA default).
- No speculative decoding, no `lmcache.mp.lazy_offload`.
- For Mamba hybrids, a vLLM whose `SchedulerOutput` has
  `kv_connector_block_state`.

## Parameters: what is sent and where it goes

```text
 user / deployment                       vLLM scheduler process              LMCache server
 -----------------                       ----------------------              --------------
 extra_config                            LMCacheMPConnector
   lmcache.mp.save_unfull_chunk ──────>    reads once at startup
 vllm serve --prefix-match-unit ──────>    grid g = vLLM hash_block_size
 request kv_transfer_params                tracker.request_configs
   lmcache.temporary_unfull_chunk ────>    (copied like any lmcache.* key)
 vLLM (Mamba only)                         │
   kv_connector_block_state ──────────>    picks tail boundaries + state blocks
                                           │
                                           ├─ LOOKUP key ──────────────────────> LookupModule
                                           │    end = lookup length L (unaligned)    probes tails, locks one
                                           │    request_configs:                     replies in tokens
                                           │      lmcache.hint.unfull_chunk_grid = g
                                           │
                                           └─ LoadStoreOp ─> vLLM worker process
                                                end = B (unaligned)   │
                                                                      ├─ STORE / RETRIEVE key ─> LMCacheDrivenTransfer
                                                                      │    end = B (unaligned)       copies the unfull object
                                                                      │    request_configs:          ─> StorageManager
                                                                      │      temporary flag              .reserve_write(is_temporary)
```

No new field travels on the wire. A tail is addressed by an `end` that is not
a multiple of `C`: the key's range is then `[start, kC)` in full chunks plus
the unfull chunk `[kC, end)`.

| Parameter | Set by | Sent as | Read by |
|---|---|---|---|
| `lmcache.mp.save_unfull_chunk` | deployment | vLLM `kv_connector_extra_config` | `LMCacheMPConnector` (scheduler and worker) at startup. Turns on everything below. |
| `--prefix-match-unit` | deployment | `vllm serve` flag | vLLM. It sets `hash_block_size`, which the connector uses as the grid `g`. |
| `lmcache.temporary_unfull_chunk` | client, per request | `kv_transfer_params` → `request_configs` on the STORE key | `LMCacheDrivenTransfer`, which passes `is_temporary=True` to `StorageManager.reserve_write`. |
| `lmcache.hint.unfull_chunk_grid` | connector | `request_configs` on every LOOKUP key | `LookupModule`, to know which tail lengths to probe. Without it the server probes only the lookup length. |
| unaligned `end` | connector (LOOKUP, FREE_LOOKUP_LOCKS), worker adapter (STORE, RETRIEVE) | the existing `IPCCacheServerKey.end` | `LookupModule` (probe and release the tail), `LMCacheDrivenTransfer` (copy the tail). |
| `kv_connector_block_state.boundary_state_offloads` | vLLM | `SchedulerOutput` | Connector, Mamba hybrids only: which boundaries have a saved state and in which block. |

A connector with the feature off sends an aligned LOOKUP `end` and no grid
hint, so the server never hands it a tail it cannot load. The other transfer
paths (engine-driven, qstore, server transfer) still floor an unaligned `end`.

## Interface changes

Wire / IPC:

| Interface | Change |
|---|---|
| `IPCCacheServerKey` | No new field. New method `has_unfull_chunk(chunk_size)`: `end % chunk_size != 0`. |
| QUERY_PREFETCH_STATUS / WAIT_PREFETCH_STATUS reply | **Always a token count** now, for every lookup (was a chunk count). It is a multiple of `C` unless the hit ends with a tail. Every MP client is updated (vLLM `LookupState`, SGLang, TensorRT-LLM, ATOM, SDK, `lmcache bench server`); server and clients must be upgraded together. |
| STORE / RETRIEVE on the LMCache-driven path | An unaligned `end` now also stores or loads the tail `[kC, end)`; its block IDs follow the full chunks' in every group. vLLM, SGLang, TensorRT-LLM, the SDK and the bench send aligned ends unless they mean a tail; ATOM's ranges come from ATOM. |
| FREE_LOOKUP_LOCKS | An unaligned `end` also frees the tail ending exactly there, if this request's lookup locked it. |
| REGISTER_KV_CACHE layout hints | New key `per_request_ring_groups`: `[{"engine_group_id", "layer_indices"}]`, sent when `save_unfull_chunk` is on. The ring layers stay in no KV group. |
| STORE / RETRIEVE block IDs | A transfer ending with a tail appends the request's ring block per ring group after the KV groups' lists. |

Server:

| Interface | Change |
|---|---|
| `MPCacheServerContext` | New `resolve_unfull_chunk_obj_keys(key, object_group_ids)`: the tail's object key per group. `resolve_obj_keys` is unchanged (it floors `end`). |
| `Session` | New `get_unfull_chunk_hash(boundary)`. No tail lookup state: the tail's L1 owners join the chunks' owners through the existing `record_prefetch_result`. |
| `LookupModule.query_prefetch_status` / `wait_prefetch_status` / `free_lookup_locks` | Token reply and tail release as above. |
| `LayoutDescRegistry.register` | New argument `group_unfull_chunk_layout_descs`. New `find_group_unfull_chunk_layout_descs()`, which falls back to the full-chunk layouts. |
| `StorageManager` | `reserve_write` gets `is_temporary: bool = False`. |
| `group_view.slice_block_ids_per_group` | New keyword `include_unfull_chunk`: `end` may be unaligned, and the slice keeps the block holding the last token (ceil instead of floor). A scratch group then keeps all its blocks (a ring's one block). |
| `KVLayerGroupsManager` | New argument and property `per_request_ring_groups` (`PerRequestRingGroup`: engine group, layer tensors, `page_bytes`). |
| `BaseCacheContext` (CPU, CUDA, NPU, MUSA) | New `get_temp_staging_buffer(num_bytes)`: the start of the transfer staging buffer as one flat region (GDS-registered where the buffer is), or `None` if the buffer is smaller. Tail objects with a lead block stage there. |
| `object_group_transfer` | New functions: `has_unfull_chunk_lead_block`, `get_unfull_chunk_layout_desc`, `get_unfull_chunk_ring_groups`, `trim_and_stage_unfull_chunk_block_ids`, `get_unfull_chunk_staging_buffer`, `transfer_unfull_chunk`, `split_unfull_chunk_block_ids`. The CUDA copy kernel is **unchanged**. |
| `MP_LOOKUP_PREFETCH_END` event | New fields `unfull_chunk_boundary`, `unfull_chunk_probes` (tail lengths checked). |

vLLM integration:

| Interface | Change |
|---|---|
| `LMCacheMPSchedulerAdapter.maybe_submit_lookup_request` | New argument `include_unfull_chunk`: send the unaligned lookup length instead of flooring it. The only adapter change. |
| `LMCacheMPRequestTracker` | New fields `unfull_chunk_boundaries` (tails already stored or hit) and `ends_at_prompt` (`max_tokens == 1`). |
| `LMCacheMPRequestMetadata` | New `GetUnfullChunkStoreMetadata(...)`: builds the tail store ops. |
| `kv_cache_groups` | New `is_per_request_ring_spec`, `is_align_mode_mamba_spec`, `get_per_request_ring_groups`. |

## How it works

Terms:

- `C`: LMCache chunk size. `P`: prompt length. `T`: tokens per block.
- **Boundary `B`**: a token count. "The tail at `B`" covers the first `B`
  tokens. vLLM uses the same word (`boundary_tokens`).
- **Unfull chunk**: tokens `[kC, B)` with `kC = floor(B / C) * C` and
  `0 < B - kC < C`.
- **Hit grid `g`**: the token counts where vLLM's prefix cache can resume.
  For Mamba hybrids it is vLLM's `hash_block_size`; otherwise the scheduler
  block. A tail exists only when `g < C`.

### Which tails are stored

| Tail | Boundary | Stored by | Used by |
|---|---|---|---|
| Grid | `B = floor((P - 1) / g) * g` | every request | a later request with the same prefix: the same prompt again, or the next round of a chat |
| Exact | `B = P` | requests that end at the prompt (`max_tokens == 1`, e.g. a disaggregated prefill) | the disaggregated decode, which continues exactly where the prefill stopped |

Why the grid tail stops below `P`:

- vLLM always recomputes the last prompt token to get its logits, so a later
  request resumes at `P - 1` at most.
- A Mamba state exists only where vLLM saved one, and vLLM saves states on the
  grid. It hands the saved block to the connector as a *boundary-state block*:
  a copy that later steps do not overwrite, so storing it while the request
  keeps decoding is safe.
- The lookup does not know where an earlier request's tail ended. Keeping all
  tails on the grid means it has at most `C / g` lengths to try.

For PD with Mamba, the decode looks up `P - 1` but the exact tail holds the
state at `P`, so they do not match yet. The decode hits the grid tail instead
and recomputes `P - 1 - B < g` extra tokens. They will match once the prefill
stops at `P - 1`, like NIXL does (a separate P/D change).

### Key

```text
unfull_hash = H(tokens[kC:B], prefix = chunk_hash[k-1])
```

The tail is hashed like the next full chunk, chained from the hash of the
chunks before it. All supported hash functions include the token count, so a
tail never has the same key as a full chunk. This matters: a temporary tail
with a full chunk's key would make `reserve_write` fail with
`KEY_NOT_WRITABLE`. A new hash function must keep this property.

### Store

```text
connector (scheduler)                 worker                     server
STORE [kC, B)                         blocks: ceil(B / T)        one object per object group,
  Mamba: boundary-state block           per group                temporary if requested
  as that group's last block
```

- **Mamba hybrids**: vLLM lists the boundary-state blocks it offers this step
  in `kv_connector_block_state`. Each offered boundary inside the prompt's last
  chunk (`kC < B <= P - 1`) that every Mamba group has is stored as a tail.
  The connector uses vLLM's choice and does not re-derive it.
- **Other models**: `B = floor((P - 1) / g) * g`, stored once the step that
  computes it is scheduled. Blocks come from the block table.
- **Requests that end at the prompt**: also store the exact tail `[kC, P)`
  once the prompt is computed, all blocks from the block table. Skipped when
  `P` is a multiple of `C`.
- A tail is a store of its own, with no full chunk in it. vLLM hands over the
  Mamba state one step after computing `B`, usually after the full chunks
  were stored. So one request can send two stores in one step. The worker
  keeps only the latest store future, which is safe because one worker's
  stores run in order on the server.
- **Pinning**: a boundary-state block is not in the request's block table and
  vLLM would free it once the request moves on. The connector pins it
  (`BlockPool.touch`) when it queues the store and unpins it when the request
  shows up in `finished_sending`. One block per Mamba group per request.
- Each tail is stored once (`tracker.unfull_chunk_boundaries`), and not at
  all if the request's own lookup already hit it.
- Blocks below `B` must not change after the store is sent. Attention rows
  below `B` are never rewritten, and the Mamba block is a copy, so this
  holds.

### Lookup

```text
LOOKUP [0, L), L unaligned, hint unfull_chunk_grid = g
  stage 1, full chunks (main lookup): one prefetch      -> k full chunks hit
  stage 2, tail (when stage 1 finishes), only inside chunk k:
       candidate lengths, longest first:
         L itself (if inside chunk k)
         floor(min(L, kC + C - 1) / g) * g, then - g, ... while > kC
       one L1-only prefetch per length; stop at the first one that locks
  reply in tokens: the tail length e, or k * C if no tail was locked
```

Example, `C = 128`, `g = 16`, `P = 337`: the prompt stores chunks `[0, 256)`
and the tail `[256, 336)`. The same prompt looks up `L = 336`, hits both
chunks, tries `336` first and hits. A next round with 900 tokens hits the same
two chunks (round 1 stored no chunk `[256, 384)`), tries `368, 352, 336` and
hits `336`.

- Stage 2 searches only the chunk after the full-chunk hit. With 256 tokens
  of chunks and a 32-token tail stored, a 1000-token lookup tries tail lengths
  between 256 and the next chunk boundary, never near 1000.
- Each candidate is tried with its own prefetch that never goes to L2
  (`allowed_l2_memory_tags` is empty), so it finishes right away. It locks
  only when every object group and every KV rank hit, and takes
  `num_kv_readers` read locks per key, like a chunk; a partial hit releases
  what it locked. Shorter candidates are never tried once a longer one locks,
  so a read-once tail below the hit is not deleted by the lookup.
- The tail's L1 owners are recorded with the chunks' owners. FREE_LOOKUP_LOCKS
  frees a tail only if all its keys are among them, so a key ending anywhere
  else releases nothing extra.
- **Several servers**: if they report different hits, the client keeps only
  the full chunks every server hit and frees each server's locks above that,
  its tail included.
- **Overlap with vLLM's prefix cache**: vLLM may already hold `n` tokens
  with `kC < n < B`. The lookup starts at `kC` (vLLM's hit floored to a
  chunk, #5383), so it still locks the tail, like any chunk that straddles
  vLLM's hit. The retrieve then skips `[kC, n)` (see Retrieve), and the hit
  stays `B`.

### Retrieve

A hit that ends mid-chunk is loaded as one RETRIEVE `[start, e)`, `e`
unaligned. The tail's block IDs follow the full chunks' in every
group. The server copies the full chunks as before, then the tail, in the same
transaction: same wait on the producer event, all-or-nothing commit, same
lock release.

- Sliding-window groups need the window that ends at `B`, which can start in
  chunk `k - 1`. The chunk retrieve already restores that earlier part; the
  tail restores the rest.
- Mamba groups need only the block with the state at `B`. On retrieve it is
  the last block of the tail range, where vLLM expects a resumed request's
  state.
- `skip_first_n_tokens` (vLLM's hit minus `start`) protects blocks vLLM
  already shares with other requests. It counts from `start`, so it reaches
  the tail only when the range has no full chunk (`start = kC`). The tail
  then does not write its blocks below `n`: they are added to the
  kernel's `skip_prefix_n_blocks`, and a lead block below `n` is not copied.
  `n` is aligned to every group's block, so the skip is whole blocks. The
  Mamba block (state at `B > n`) is always written.

### Object layout

A tail is one object per object group with the **same layout as a full
chunk**, so no new kernel is needed. It just has fewer real blocks.

The copy kernel expects exactly `keep` blocks per object (`keep` = all blocks
of a chunk for full attention, the window's blocks for sliding window, one
block for Mamba). The block list is cut to its last `min(n, keep)` blocks and
then padded at the front with placeholder IDs. The kernel's existing
`skip_prefix_n_blocks` argument skips the padding, so it is never read or
written:

```text
full attention, keep = 16, 3 real blocks:
  block ids : [b0 b0 b0 ... b0 | b0 b1 b2]     (13 placeholders + 3 real)
  skip      : 13
  object    : slots 0..12 untouched, slots 13..15 <- b0 b1 b2
```

Store and load pad the same way, so block `i` lands in the same slot both
times. Tails always use the staging copy path (`transfer_unfull_chunk`), never
the direct-copy path, which assumes full chunks. An object group whose tail
blocks are all null stores nothing.

#### Window lead block

This case is only for sliding-window groups with window `W < C`.

Such a group keeps `keep = W / T` blocks per chunk. The next token after `B`
looks at tokens `[B - W + 1, B)`. A full chunk ends on a block edge, so that
range fits in `keep` blocks. A tail can end inside a block, and then the range
touches `keep + 1` blocks. Example with `T = 4`, `W = 8`, `keep = 2`, a tail
of `n` tokens:

```text
n = 9   window [2, 9)   blocks 0 1 2   -> keep + 1, lead block 0
n = 11  window [4, 11)  blocks 1 2     -> keep
n = 13  window [6, 13)  blocks 1 2 3   -> keep + 1, lead block 1
```

The extra first block is the *lead block*. Both sides work out whether it is
needed from `B mod C`, so no extra metadata is sent. (NIXL covers the same
overlap for P/D with `cdiv(W, T) + 1` window blocks. This assumes `W` is a
multiple of `T`.)

- Layout: the tail object uses `get_unfull_chunk_layout_desc`: the full-chunk
  layout, plus one single-block region per such group, **appended at the
  end**. All regular regions keep their offsets, and full-chunk objects are
  unchanged. The server registers these layouts next to the full-chunk ones
  for the lookup's L1 prefetch.
- Copy: the regular copy runs as before; the lead block is one more launch of
  the same kernel with one block. The object is larger than one
  (batch, object-group) slot of the staging buffer, so it stages at the
  start of that buffer (`get_unfull_chunk_staging_buffer`, through
  `BaseCacheContext.get_temp_staging_buffer`), spanning the following slots.
  Transfers on the stream run in order, so nothing else uses those slots
  meanwhile. The buffer is registered for GDS, so a GDS-backed L1 can hold
  the object like any other. A tail that does not fit (`None`) is not stored,
  so a retrieve never meets one. With 4 batch slots this cannot happen: each
  lead region is one block, no larger than the regular region it follows.
- On the decode side the lead block is the first block vLLM keeps
  (`get_num_skipped_tokens = B - W + 1`), so it is allocated, not null.
- A sliding-window model's grid is its scheduler block, which is a whole
  number of window blocks, so its grid tails end on block edges and never
  need a lead block. The lead block matters for models whose grid is finer
  than their window blocks, and for exact tails.

#### Per-request ring

Compressed-attention models (DeepSeek-V4.1 compressor, GLM-5-Next kpool
indexer, Step-5 CSA, Qwen4-Exp QSA) merge every `r` tokens into one cached
row. The raw rows of the group still open live in a per-request ring
(`CircularBufferSpec`): one block per request, row `pos` at slot
`pos % capacity`, never prefix-cached. A tail must carry it when the open
group is not empty at `B`:

```text
r = 2, exact tail B = P = 301
  compressed rows  c0 .. c149   tokens [0, 300)    in the tail's KV blocks
  ring             row 300      token 300 (open)   only in the ring block
```

- **Which tails need it**: the exact tail of a P/D prefill. A chunk or grid
  boundary closes every group (vLLM resumes its own prefix-cache hits there
  without the ring), so the rows copied with a grid tail are never read.
  Copying the ring with every tail keeps one layout per object group.
- **Layout**: object group 0's tail object appends one
  `(1, layers, 1, page_bytes)` byte region per ring group, after the lead
  regions, so the object stages at the start of the staging buffer.
- **Copy**: the request's ring block of every layer is copied element by
  element through that layer's block view (a ring block may be padded
  inside), on the transfer stream. Retrieve writes the decode request's own
  ring block, which vLLM claimed before the load. Rows are addressed by
  `pos % capacity`, so the same model config on both sides puts every row
  in its slot. The decode recomputes token `P - 1` and rewrites its row with
  the same value.
- **Block IDs**: the connector slices a scratch group's blocks only for a
  range ending with a tail; the worker appends the groups named in
  `per_request_ring_groups` after the KV groups' lists; the server splits
  them off before the tail's block fitting.
- **Which models read it today** (vLLM 0.33): DeepSeek-V4.1 with
  `--no-swa-bounded-replay`. Under SWA bounded replay (its default) a
  full-prompt hit is floored to a chunk, so the exact tail is dropped and
  only grid tails serve hits. GLM-5-Next and Qwen4-Exp are Mamba hybrids:
  the decode looks up `P - 1`, so their exact tail is not hit until the
  prefill stops at `P - 1` (see [Which tails are stored](#which-tails-are-stored)).
  vLLM lets Step-5 use only the Mooncake or NIXL connector.
- **Prefill-context parallelism**: only DeepSeek-V4.1 runs PCP with a ring
  (vLLM rejects PCP for the other three ring backends). Its compressor
  gathers each request's newest open row on every PCP rank, so every rank's
  ring is equal and each rank's tail carries a full copy. Not yet run end to
  end.
- Validated on a 4-layer DeepSeek-V4.1-Flash (dummy weights,
  `--kv-cache-dtype fp8_ds_mla --no-swa-bounded-replay`, 10,003-token
  prompt): the decode hit 10,002 tokens, and its ring block held the
  prefill's bytes after the retrieve (stale before).

### Temporary tails and locks

- `"lmcache.temporary_unfull_chunk": true` stores the request's tail as
  temporary. L1 deletes it when its last read lock is released and never
  copies it to L2. If nobody reads it, it is evicted like any other object.
- Tail read locks are released by:
  - each worker's retrieve, one lock per key;
  - FREE_LOOKUP_LOCKS whose `end` equals the tail length (no retrieve is
    needed because vLLM holds the whole hit, or the servers disagreed);
  - otherwise the read-lock TTL, like a chunk lock.

## Limits

- Tails are served from L1 only. A persistent tail copied to L2 is not looked
  up there, and tails are not shared over P2P L1. P2P lookup of tails is a
  follow-up for LMCache-native PD.
- A prompt shorter than one chunk has no full chunk to look up, so its tail is
  never tried.
- A lookup tries up to `C / g` tail lengths, one L1-only prefetch (and one
  hash) each. The count is in `MP_LOOKUP_PREFETCH_END` (`unfull_chunk_probes`).
- Requests with `max_tokens == 1` always store an exact tail. Without PD
  nobody reads it, so it only costs L1 space.
- Mamba backends that save the prompt-tail state inside a prefill chunk offer
  no boundary-state block for it, so only block-end boundaries are stored.
- Sliding-window blocks below the window can be freed while a tail store is in
  flight, the same risk as for full-chunk sliding-window stores.
- A failed tail retrieve, or a lookup dropped for an APC-shrink re-lookup
  (`apc_covered_lookup.md`), leaves the tail's locks (and a temporary object)
  to the read-lock TTL.
- Only prompt tails are stored. A next round hits the previous round's prompt
  tail, not its generated tokens.
- vLLM must accept an external hit that is not block-aligned (current vLLM
  does). DSV4.1 with bounded-replay SWA still replays its window after a load,
  because LMCache reports only the prefix-cacheable groups as loaded.
