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

This feature lets LMCache store and load the unfull chunk `[kC, P)` of a
request, so a later request (a decode instance in PD disaggregation, or the
next round of a chat) resumes at `P` instead of `kC`. It is the shared
groundwork for PD handoff and multi-round reuse; neither policy lives here.

Enabled per connector with `lmcache.mp.unfull_chunk: true`
(`kv_connector_extra_config`). Off by default: nothing changes for existing
deployments.

Compressed-attention models (QSA, DSA kpool, DSV4.1 compressor) are not
supported yet and are rejected at startup: resuming mid-chunk also needs the
raw keys of the still-open compression group, which vLLM keeps in a
per-request ring (`CircularBufferSpec`) that LMCache skips (#5042). Storing
that ring is a follow-up.

## Concepts

- **Unfull chunk**: the token range `[kC, P)` with `kC = floor(P / C) * C`
  and `0 < P - kC < C`, where `C` is the LMCache chunk size. It is stored as
  one object per object group with the regular chunk layout; only the blocks
  that cover `[kC, P)` are copied.
- **Temporary object**: an L1 object deleted when its last read lock is
  released (`L1ObjectState.is_temporary`), i.e. read once.

## Unfull chunk key

```text
unfull_hash = H(tokens[kC:P], prefix = chunk_hash[k-1])
```

The unfull chunk is hashed like the next full chunk, from the rolling hash
of the full chunks before it. It has fewer than `C` tokens, and every hash
algorithm LMCache supports encodes the token count (blake3 hashes the packed
tokens, `sha256_cbor` and the builtin tuple hash include the tuple length),
so its key differs from every full chunk's: a temporary unfull object never
collides with a persistent full chunk (`reserve_write` would otherwise return
`KEY_NOT_WRITABLE`). A new hash algorithm must keep that property. The key
covers the exact length `P`.

An `IPCCacheServerKey` addresses the unfull chunk only when
`include_unfull_chunk` is set and `end` is not chunk-aligned; it is then the
last chunk of the key's range, after any full chunks. Without the flag a
non-aligned `end` keeps its existing meaning (partial chunks are discarded),
which `free_lookup_locks` callers rely on.

## Data flow

```text
store (P = prompt end)                     load (lookup end L)
------------------------------------       -----------------------------------------
STORE [s, P), include_unfull_chunk         LOOKUP [0, L), include_unfull_chunk
  full chunks [s, kC) as today               chunk prefetch -> k full chunks hit
  + unfull chunk [kC, P):                    if k == floor(L/C): L1-only probe of the
    blocks: ceil(P/T_g) slices per group       unfull key, read-locked like a chunk
    is_temporary from request config         reply k (+1 if the unfull chunk hit)
                                           client: hit = min(reply * C, L)
                                           RETRIEVE [s, L), include_unfull_chunk
```

A request sends at most one store and one retrieve per step (the worker
adapter tracks futures per request), so the unfull chunk travels in the same
operation as the request's last full chunks: its block IDs follow theirs in
every group. The server handles the full chunks exactly as before, then the
unfull chunk inside the same transaction (same producer-event wait,
all-or-nothing commit, same read-lock release on retrieve).

The unfull chunk is a single object per object group. Its per-group block
list is trimmed to the last `min(n, keep)` blocks, where `keep` is the
group's per-chunk window in blocks (all blocks for full attention, the last
window for sliding window, one block for Mamba state). Store
and load trim the same way, so block `i` maps to object offset `i` on both
sides. The transfer kernel copies exactly the blocks it is given
(`num_blocks_per_object = total_blocks / num_objects`), so a short list
needs no kernel change; the unfull object always takes the staging-kernel
path (`transfer_unfull_chunk`), never the direct-copy path. An object group
whose unfull blocks are all null (no real state) stores nothing.

Sliding-window groups need the window ending at `P`, which can start in
chunk `k-1`. The chunk lookup already retains the last window of the hit
prefix (chunk `k-1`), so the full-chunk retrieve restores the earlier part
and the unfull retrieve the rest. Mamba state needs only the block holding
the state at `P`, which is the last block of the unfull range.

## Temporary objects

`request_configs["lmcache.temporary_unfull_chunk"] = True` stores the
request's unfull chunk with `is_temporary`. L1 then deletes it when the last
read lock of a lookup is released (a retrieve on every worker, or
`free_lookup_locks` for an unused hit), and never offloads it to L2. An
unread temporary object is evicted like any other.

## Lookup, locks and sessions

The unfull probe uses `submit_prefetch_task(..., skip_l2=True)`, which reads
L1 synchronously and takes `num_kv_readers` read locks per key like the chunk
prefetch. The session records the probed keys and their L1 owners
separately from the chunk prefetch result:

- each worker's unfull retrieve releases one lock per key;
- `FREE_LOOKUP_LOCKS` with `include_unfull_chunk` and the lookup's own
  unaligned `end` releases the remaining locks once (vLLM already holds the
  tokens, or the servers disagree on hits);
- anything else is left to the read-lock TTL, like a chunk lock.

## Connector behavior (vLLM MP connector)

- **Store**: the step whose forward computes the prompt's last token `P`
  (`computed == P`, so the Mamba state describes exactly `P`)
  extends its store to `[s, P)` when `P % C != 0` and every full chunk up to
  `kC` is stored by then. At most once per request; a request whose lookup
  already found the unfull chunk does not store it again.
  `num_stored_tokens` stays chunk-aligned.
- **Store invariant**: the unfull chunk's blocks must not be written after
  its store is submitted. Unlike a full chunk, they are live: the next decode
  step rewrites the Mamba running-state block in place (and the last
  attention block gains rows), while the store copies asynchronously on the
  server stream (nothing orders the next forward after it). The connector
  therefore stores the unfull chunk only for a request that ends at the
  prompt (`max_tokens == 1`, e.g. a disaggregated prefill, or a pooling
  request); the request's blocks are held until the store completes.
- **Lookup**: sends `end = L` (the lookup token count, after the Mamba
  reserve-last-token rule) with `include_unfull_chunk`, so a hit can end
  mid-chunk.
- **Retrieve**: a hit that ends mid-chunk is retrieved as one operation
  `[start, L)` with `include_unfull_chunk`.
- **Overlap with vLLM's prefix cache**: if vLLM already holds tokens past
  `kC`, loading the unfull chunk would overwrite blocks other requests read,
  so the connector releases the unfull chunk's locks and keeps the hit at
  `kC`.
- **Startup checks**: the option is rejected for models with a
  `CircularBufferSpec` ring (see Summary), with speculative decoding (draft
  rows sit past `P` in relocated Mamba blocks), with
  `lmcache.mp.lazy_offload`, and when the worker resolves to an
  engine-driven transfer mode.

## Limits

- Served from L1 only (a persistent unfull object offloaded to L2 is not
  looked up there); LMCache-driven transfer mode only (the CUDA default).
- A prompt shorter than one chunk has no full chunk to look up, so its
  unfull chunk is never probed.
- A failed unfull-chunk retrieve, or a lookup abandoned for an APC-shrink
  re-lookup (`apc_covered_lookup.md`), leaves the unfull chunk's read locks
  (and a temporary object) to the read-lock TTL: those releases resolve full
  chunks only.
- Only the prompt-end tail is stored. Finding the tail of an earlier round
  (multi-round chat) needs a separate lookup policy; resuming a decode
  instance at the exact prompt end (PD) additionally needs the prefill side
  to stop where the decoder resumes.
- vLLM must accept a non-block-aligned external hit (current vLLM does). DSV4.1
  with bounded-replay SWA still replays its window after a load because
  LMCache reports only the prefix-cacheable groups as loaded.
