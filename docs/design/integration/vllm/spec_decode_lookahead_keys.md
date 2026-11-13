# Lookahead chunk keys for EAGLE / MTP

Status: opt-in and flag-gated lookahead keys implemented; finish-time save
(§4.2) not implemented. Default for the flag pending GPU measurement (§8).
Tracks: vLLM #61589 (core lookahead block hashes), #64096 (connector opt-in).

## 1. Problem

EAGLE, EAGLE3 and MTP drafters shift their input by one token: the draft KV at
position `i` is computed from target hidden state `i` and token `i + 1`.
So the draft KV at the **last position of a chunk** depends on a token that is
not in the chunk's prefix key.

In general the drafter reads `k` tokens past a position, and vLLM reports `k`
as `num_hash_lookahead_tokens`: `k = 1` for EAGLE, EAGLE3 and single-module
MTP; `k` = number of modules for chained multi-module MTP. LMCache never picks
`k` itself, it uses vLLM's value. The examples below use `k = 1`.

```
chunk size C = 4, k = 1 lookahead token
tokens:        A B C D | E F G H | ...
target KV(D):  f(A..D)                 -> keyed correctly by H(ABCD)
draft  KV(D):  g(A..D, E)              -> depends on E, not in H(ABCD)
```

Request 1 stores `ABCD|E...`. Request 2 sends `ABCD|X...` and hits `H(ABCD)`:
it loads draft KV(D) computed with `E` instead of `X`. Target output is
unaffected (target KV is correct); draft acceptance drops at every chunk
boundary that diverges there.

Where things stand today:

| Component | Behavior |
|---|---|
| vLLM local APC, no connector | Lookahead hashes, no drop (#61589). |
| vLLM local APC, connector not opted in | Legacy hashes + drops last matched block on every hit. |
| vLLM `OffloadingConnector` (legacy mode) | Drops the trailing chunk of EAGLE groups on store and lookup (`_eagle_proof_margins`). |
| LMCache MP connector | Not opted in, **no drop**, keys chunks on prompt tokens only. Stores draft layers too (`lmcache/integration/vllm/utils.py:375`). |

So enabling LMCache with EAGLE/MTP today both (a) shortens vLLM's own APC hits
by one block and (b) can feed stale draft KV at chunk boundaries.

## 2. Goals / non-goals

The two problems are independent and are fixed separately:

- (a) Always: opt into `supports_lookahead_block_hashes`. This only changes
  vLLM's own block hashes and removes its one-block trim on every local hit;
  LMCache keys are untouched.
- (b) Optional, flag `lmcache.mp.lookahead_chunk_keys` (bool, default False):
  an LMCache hit for chunk `i` also requires the `k` tokens after the chunk to
  match. Hits stop at `floor((N - k) / C) * C` for prompt length `N`.

Why (b) is a flag: it is likely a small win and has a real cost (§4.3). Stale
draft KV affects one position per chunk in the drafter's layer(s) only, and
lookahead keys lose a chunk of hits whenever a request diverges exactly at a
chunk boundary. The default is decided by measurement (§8).

Non-goals
- Unfull-chunk + spec decode (still rejected by `save_unfull_chunk`); revisit
  after this lands (§7).
- Splitting draft vs target layers into separate objects (§6, option C).

## 3. vLLM interface

From #64096:
- `KVConnectorBase_V1.supports_lookahead_block_hashes(kv_transfer_config) -> bool`
  (classmethod). Default `False`.
- `vllm_config.num_hash_lookahead_tokens: int` — resolved after configs are
  final; `0` when spec decode is off or any connector didn't opt in.
- MultiConnector opts in only if every child does.

vLLM's `request.block_hashes` become lookahead-aware, but LMCache does **not**
use them for its keys (it hashes `token_bytes` server-side in
`lmcache/v1/multiprocess/session.py`). The opt-in only affects vLLM's local APC
and is a promise that our hits are lookahead-safe.

## 4. Design

### 4.1 Key derivation (server)

Keep the rolling prefix chain unchanged; derive the **object key** of chunk `i`
from the chain hash plus the next `k` tokens:

```
chain_i   = H(tokens[iC:(i+1)C], chain_{i-1})          # unchanged
key_i     = chain_i                                     if k == 0
          = H(tokens[(i+1)C:(i+1)C+k], H(<empty>, chain_i))   if k > 0
```

The `H(<empty>, ·)` step is domain separation: without it,
`H(next_k, chain_i)` equals the unfull-chunk hash of boundary `(i+1)C + k`
(`H(partial_tokens, chain_i)`), so a spec instance and a `save_unfull_chunk`
instance sharing storage would write different data under one key. No full or
unfull chunk is empty, so `H(<empty>, chain_i)` is never a chain value.
Implemented as `TokenHasher.hash_lookahead`.

Why chain-then-derive instead of putting lookahead tokens into the chain:
`chain_i` stays identical for spec and non-spec instances, so the unfull-chunk
hash, KV events and future layer-split work (§6 C) keep one prefix identity.
Only the stored object key changes.

Changes:
- `IPCCacheServerKey`: add `lookahead_tokens: int = 0` (part of cache identity;
  msgspec default keeps old clients decodable).
- `Session.set_tokens(packed, lookahead_tokens)`; `Session.get_hashes`: when
  `k > 0`, return `key_i`; require `end + k <= held` (raise like today's
  `end > held` check); implicit end stops at the last chunk with its
  lookahead tokens present.
- `TokenHasher.compute_packed_chunk_hashes(..., lookahead_tokens=k)` for the
  lookup path, which hashes without a session.
- CacheBlend modules keep plain keys (out of scope).
- `ipc_key_to_object_keys`: unchanged beyond consuming the new hashes.

### 4.2 Connector (scheduler side)

- `LMCacheMPConnector.supports_lookahead_block_hashes` returns `True`
  regardless of the flag.
- Key lookahead = `vllm_config.num_hash_lookahead_tokens` if
  `lmcache.mp.lookahead_chunk_keys` else `0`; passed in every
  `IPCCacheServerKey` (lookup, store, retrieve). Flag off = today's keys.
- **Lookup / hit length:** the lookup key carries the whole prompt, and the
  server only keys chunk `i` if `(i+1)C + k <= len(tokens)`, so hits stop at
  `floor((N - k) / C) * C` with no connector-side cap.
- **Store:** in `GetStoreMetadata`, bound
  `min_available_tokens <= len(all_token_ids) - k`. A chunk whose lookahead
  token is not yet known is not storable this step. Applied when the flag is
  on, **and when lazy offload is on even with the flag off**: lazy offload
  snapshots each vLLM block's `block_hash` when a store is offered and drops
  the store if it later changes, and with the opt-in vLLM hashes a block only
  once its lookahead tokens exist. Without the bound, a chunk ending exactly
  at the computed frontier (common at chunked-prefill boundaries) would see
  `None` then a real hash and be dropped.
- **Finish-time save (not implemented yet):** prompt of `N` tokens with `N % C == 0` and
  `max_tokens = 1`: the last chunk becomes storable only after the sampled
  token exists, but the request finishes in that step. In `request_finished`,
  if `num_stored_tokens < floor((len(all_token_ids) - k) / C) * C`, emit a final
  store op and hold the blocks (return `True` from `request_finished` /
  delayed free) until the store completes. Same pattern as Mooncake Store's
  `register_finished_lookahead_save()` in #64096.
  Respect `save_decode_cache=False`: the chunk covers prompt tokens only; the
  lookahead token is a generated token but is only used in the key.

### 4.3 Config / compatibility

- Flag off (default for now) = today's keys bit-for-bit.
- **Main cost of the flag: sharing.** With lookahead keys, spec-decode and
  non-spec instances (or instances with different `k`) stop sharing the same
  target-model KV across P2P, PD and shared L2 backends, and existing caches
  go cold on upgrade. Confirm whether `ObjectKey` already separates them by
  layer layout (draft layers stored or not); if it does, the flag costs no
  extra sharing.
- KV events: nothing in LMCache depends on vLLM's block-hash values. The
  connector forwards LMCache's own CPU-tier events (LMCache chunk hashes) to
  vLLM's event stream, and lazy offload compares vLLM block hashes only to
  itself (see the store bound above). With the opt-in, vLLM's own GPU-block
  events carry lookahead hashes, same as vLLM without a connector.
- P2P / PD: both sides must agree on `k`; a mismatch only misses (keys
  differ), it never loads wrong data. Failing fast on mismatch is a follow-up.

### 4.4 Known limitations

- Token-based cache-control APIs (`cache_control/key_resolver.py`) hash plain
  keys, so they do not address lookahead-keyed objects.
- CacheBlend modules keep plain keys and stay exposed to stale draft KV.
- Existing spec-decode caches go cold on upgrade (new key format).

## 5. Hit length vs today

| Prompt N, C = 256, k = 1 | Today (unsafe) | Legacy drop (safe) | This design |
|---|---|---|---|
| 1000 | 768 | 512 | 768 |
| 1024 | 768 (`N-1` cap) | 512 | 768 |
| 1025 | 1024 | 768 | 1024 |

Same length as today, but safe; one chunk more than a drop-based fix.

## 6. Alternatives

A. **Drop last LMCache chunk under spec decode** (mirror OffloadingConnector
   legacy). Tiny change, no key format change. Loses a whole 256-token chunk
   per hit, much worse than vLLM's 16-token block drop. Use only as stopgap.

B. **This design** (opt-in always; lookahead object keys behind a flag).

C. **Split draft layers into their own objects**, keyed with lookahead; target
   layers keyed plain. Target KV stays shareable with non-spec instances.
   The gain is small: vLLM has one hit length per request, and recomputing a
   draft position needs the target hidden state at that position, which only
   a target recompute produces. So on a draft-key miss the request still
   recomputes from the boundary; splitting saves at most "boundary − 1 token"
   and needs partial-block hits. Needs per-group objects
   (`ObjectKey.object_group_id`, LMCache #3608).

## 7. Interaction with unfull chunk

`save_unfull_chunk` rejects spec decode because draft rows sit in relocated
Mamba blocks. Lookahead keys solve key safety for the unfull chunk
(`H(lookahead, unfull_hash)`), but not the Mamba relocation issue. Keep the
rejection; revisit separately.

## 8. Test plan

Unit (CPU; implemented in `tests/v1/multiprocess/test_session.py` and
`tests/v1/test_mp_tracker_store_bound.py`):
- Same chunk + different next token → different key; same → same.
- A chunk without its lookahead token is not keyed; explicit `end + k > held`
  raises.
- Lookahead keys never alias plain or unfull-chunk keys.
- Session (store path) keys equal `compute_packed_chunk_hashes` (lookup path)
  keys for `k ∈ {0, 1, 3}` under blake3 and sha256_cbor.
- `GetStoreMetadata` with `k = 1` holds back a chunk until its next token exists.
- Connector (`tests/v1/test_mp_connector_lookahead_keys.py`): flag off sends
  `lookahead_tokens = 0` even when vLLM reports `k > 0`; flag on sends `k`;
  the vLLM opt-in holds either way; a non-boolean flag is rejected.
- Pending with the finish-time save follow-up: save emitted for `N % C == 0`, `max_tokens = 1`.

E2E (GPU), per model (EAGLE3 Llama-3.1-8B; an MTP model, incl. `k > 1` if
available), configs: (a) vLLM only, (b) LMCache flag off, (c) LMCache flag on,
(d) LMCache before the opt-in:
- Boundary stress: shared chunk-aligned prefixes diverging at the next token,
  plus a mid-chunk control. Spec acceptance, TTFT, TPOT, LMCache hit tokens.
- Realistic: long-doc QA or multi-round chat.
- Decision rule: if (b) acceptance is measurably below (a)/(c), default the
  flag on; otherwise keep it off for sharing.

## 9. Rollout

1. This PR: connector opt-in; server key derivation and
   `IPCCacheServerKey.lookahead_tokens` behind `lmcache.mp.lookahead_chunk_keys`;
   store bound; unit tests.
2. GPU measurement (§8 E2E) before marking ready.
3. Follow-up: finish-time save; fail fast on `k` mismatch across peers.
4. Follow-up: option C after #3608.

Minimum vLLM version: first release containing #64096. Older vLLM has no
`supports_lookahead_block_hashes`; `num_hash_lookahead_tokens` absent → treat
as `k = 0`.
