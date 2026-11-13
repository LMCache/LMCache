# Lookahead chunk keys for EAGLE / MTP

Status: implemented except the finish-time save (§4.2) and option C (§6).
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

Goals
- Remove (a): opt into `supports_lookahead_block_hashes` so vLLM drops its
  legacy APC trim.
- Remove (b): an LMCache hit for chunk `i` must imply the `k` tokens after the
  chunk match too.
- Keep hits as long as possible: `floor((N - k) / C) * C` for prompt length `N`.

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

- `LMCacheMPConnector.supports_lookahead_block_hashes` returns `True`.
- Read `k = vllm_config.num_hash_lookahead_tokens` once at init; pass in every
  `IPCCacheServerKey` (lookup, store, retrieve).
- **Lookup / hit length:** the lookup key carries the whole prompt, and the
  server only keys chunk `i` if `(i+1)C + k <= len(tokens)`, so hits stop at
  `floor((N - k) / C) * C` with no connector-side cap.
- **Store:** in `GetStoreMetadata`, bound
  `min_available_tokens <= len(all_token_ids) - k`. A chunk whose lookahead
  token is not yet known is not storable this step.
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

- Enabled automatically when `num_hash_lookahead_tokens > 0`; no new LMCache
  flag. Off = today's keys bit-for-bit.
- Spec and non-spec instances no longer share chunk objects. Acceptable: their
  draft layers differ anyway (non-spec instances don't store draft layers, so
  layer count already splits them). Confirm `ObjectKey` already encodes layer
  layout; if not, this is a pre-existing sharing bug.
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

B. **This design** (lookahead object keys). Recommended.

C. **Split draft layers into their own objects**, keyed with lookahead; target
   layers keyed plain. Target KV stays shareable with non-spec instances, and a
   draft-key miss still loads target KV (draft recomputes). Needs per-group
   objects (`ObjectKey.object_group_id`, LMCache #3608). Follow-up once #3608
   lands.

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
- Pending with the finish-time save follow-up: save emitted for `N % C == 0`, `max_tokens = 1`.

E2E (GPU):
- Correctness: two prompts sharing `ABCD`, diverging at the next token; second
  request's draft acceptance equals cold run.
- Perf: EAGLE3 Llama-3.1-8B and MTP Qwen3.6 hybrid; long-doc QA; report TTFT,
  hit tokens, acceptance rate for {no LMCache, LMCache today, LMCache + this}.
- First step: measure acceptance drop of "LMCache today" to size the problem.

## 9. Rollout

1. This PR: server key derivation, `IPCCacheServerKey.lookahead_tokens`,
   connector opt-in, store bound, unit tests. Inactive at `k = 0`.
2. GPU measurement (§8 E2E) before marking ready.
3. Follow-up: finish-time save; fail fast on `k` mismatch across peers.
4. Follow-up: option C after #3608.

Minimum vLLM version: first release containing #64096. Older vLLM has no
`supports_lookahead_block_hashes`; `num_hash_lookahead_tokens` absent → treat
as `k = 0`.
