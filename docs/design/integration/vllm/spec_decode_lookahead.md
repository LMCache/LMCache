# vLLM lookahead block hashes for EAGLE / MTP

Status: connector opt-in implemented. LMCache lookahead chunk keys were
prototyped, measured (§5) and rejected.
Tracks: vLLM #61589 (core lookahead block hashes), #64096 (connector opt-in).

## 1. Problem

EAGLE, EAGLE3 and MTP drafters shift their input by one token: the draft KV at
position `i` is computed from target hidden state `i` and token `i + 1`. So the
draft KV at the **last position of a block** depends on a token that is not in
the block's prefix hash.

In general the drafter reads `k` tokens past a position; vLLM reports `k` as
`num_hash_lookahead_tokens` (`k = 1` for EAGLE, EAGLE3 and single-module MTP;
the module count for chained multi-module MTP).

```
block size 4, k = 1
tokens:        A B C D | E F G H | ...
target KV(D):  f(A..D)                 -> keyed correctly by H(ABCD)
draft  KV(D):  g(A..D, E)              -> depends on E, not in H(ABCD)
```

vLLM #61589 folds the next `k` tokens into its block hashes. When a KV
connector is attached, vLLM uses them only if the connector opts in; otherwise
it keeps the legacy fix of **dropping the last matched block on every local
prefix-cache hit**.

## 2. Design

`LMCacheMPConnector.supports_lookahead_block_hashes()` returns True
unconditionally. This changes only vLLM's own block hashes and removes vLLM's
one-block drop on local hits. LMCache chunk keys stay token-only, so spec and
non-spec instances keep sharing target KV (P2P, PD, shared L2).

Consequence for lazy offload: it snapshots each block's vLLM hash when a store
is offered and drops the store if the hash later changes or is missing. With
lookahead hashes vLLM hashes a block only once its next `k` tokens exist, so
with `lmcache.mp.lazy_offload` the connector passes `lookahead_tokens = k` to
`GetStoreMetadata`, holding a chunk back until those tokens exist. Without lazy
offload the store bound is unchanged.

KV events: nothing else in LMCache depends on vLLM block-hash values. The
connector forwards LMCache's own CPU-tier events (LMCache chunk hashes); vLLM's
GPU-block events carry lookahead hashes, same as vLLM without a connector.

Remaining exposure: an LMCache hit whose request diverges exactly at a chunk
boundary loads draft KV computed with a different next token at one position.
Target output is unaffected; draft acceptance can drop for that request.
§5 measures this as negligible.

## 3. Rejected: lookahead chunk keys

Fold vLLM's `k` lookahead tokens into LMCache chunk keys
(`key_i = H(next_k, H(<empty>, chain_i))`), so a hit also implies matching
draft KV. Prototyped behind a flag and measured (§5). Rejected because:

- Acceptance gain is within run-to-run noise.
- It loses one chunk of hits (256 tokens) on every request that diverges at a
  chunk boundary, and holds back the last chunk until its next token exists.
- Spec and non-spec instances (or different `k`) stop sharing KV, and existing
  caches go cold.

Other alternatives considered:

- **Drop the last LMCache chunk under spec decode** (like vLLM
  `OffloadingConnector` legacy mode): loses a whole chunk per hit for the same
  negligible effect.
- **Split draft layers into their own lookahead-keyed objects**: vLLM has one
  hit length per request and recomputing a draft position needs the target
  hidden state there, so on a draft-key miss the request still recomputes from
  the boundary. Saves at most "boundary − 1 token"; needs per-group objects.

## 4. Tests

- `tests/v1/test_mp_connector_lookahead_opt_in.py`: the connector opts in.
- `tests/v1/test_mp_tracker_store_bound.py`: `GetStoreMetadata` with `k = 1`
  holds back a chunk until its next token exists.

## 5. Measurements (H200, vLLM c410543d9, LMCache chunk 256, 192 docs/config)

Per doc: store prefix + suffix A in LMCache, clear vLLM's local prefix cache,
then send prefix + suffix B. "Boundary" = A and B diverge at token 1024 (right
after a chunk); "control" = diverge at 1100 (mid-chunk). Spec-decode
acceptance rate; "keys on" is the rejected prototype:

| Model | Case | vLLM only | LMCache (this design) | keys on |
|---|---|---|---|---|
| Llama-3.1-8B + EAGLE3 (3 spec tokens) | boundary | 0.3096 | 0.3095 | 0.3108 |
| | control | 0.3166 | 0.3133 | 0.3133 |
| MiMo-7B-RL MTP (1 spec token) | boundary | 0.7461 | 0.7316 | 0.7382 |
| | control | 0.7450 | 0.7464 | 0.7464 |
| Qwen3.6-35B-A3B MTP (2 spec tokens) | boundary | 0.7583 | no LMCache hits | no LMCache hits |
| | control | 0.7520 | 0.7487 | 0.7493 |

- Differences between LMCache columns are within noise (repeat runs of one
  config varied by up to ~0.03). TTFT and TPOT do not move.
- Keys on cut LMCache hits on boundary requests by a chunk (MiMo 49152 →
  36864 hit tokens).
- Only ~1/C of real requests diverge exactly at a chunk boundary.
- Not measured: multi-module MTP (`k > 1`).

Minimum vLLM version: first release containing #64096. Older vLLM has no
`supports_lookahead_block_hashes` and no `num_hash_lookahead_tokens`
(treated as `k = 0`).
