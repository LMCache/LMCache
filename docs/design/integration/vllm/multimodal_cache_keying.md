# Multimodal Cache Keying

Status: implemented (in-process connector and all MP connector variants)

## Problem

vLLM emits identical placeholder token IDs for every multimodal item, so raw
token IDs cannot distinguish two different images: with token-based chunk
hashing alone, the same text with different images would produce identical
cache keys and silently share KV entries (cross-image contamination).

vLLM's own prefix cache solves this by appending `(mm_hash, offset)` extra
keys to each block hash. LMCache's key derivation is token-based and flows
through interfaces that carry only token IDs (lookup RPC, MP connector
metadata), so LMCache instead substitutes the placeholder token IDs with
values derived from the multimodal identifier before hashing.

## Contract

`lmcache/integration/vllm/utils.py`:

- `mm_hash_to_token_values(identifier, length)` derives a deterministic
  sequence of `length` values, each in `[0, 2**31)`, from the full
  identifier (SHA-256 based, counter-mode expansion). Guarantees:
  - **Full entropy**: every position gets an independent 31-bit value, so a
    chunk overlapping `k` placeholder tokens carries `31*k` bits of item
    identity (effectively bounded only by the 64-bit chunk-hash width for
    `k >= 3`).
  - **Position dependence**: the value encodes the offset within the item,
    mirroring vLLM's `(mm_hash, offset)` extra-key semantics.
  - **Prefix stability**: `values(x, m) == values(x, n)[:m]` for `m <= n`,
    so partial prompts (save-path truncation, chunk boundaries) hash
    consistently with full prompts.
- `apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)`
  overwrites each placeholder span in-place with that sequence, truncating
  at the tensor length.

Values are capped at 31 bits so they stay positive in a signed int32, the
narrowest integer representation token IDs may pass through downstream.

## History

The original implementation (`hex_hash_to_int16`) collapsed the identifier
to 16 bits and filled the whole span with that single value. By the birthday
bound, ~300 distinct same-shape images gave ~50% probability of two images
sharing all their cache keys — a silent false hit serving the wrong image's
KV (issue #3301). `mm_hash_to_token_values` replaces it; on upgrade,
previously cached multimodal entries miss (safe) rather than collide.

## Alternative considered

Passing the full `mm_hash` through `TokenDatabase._hash_tokens(extra_keys=...)`
(the channel reserved for this) is the vLLM-aligned design, but requires
extending every token-carrying interface (lookup ZMQ protocol, MP metadata,
SDK) to carry per-chunk extra keys. The substitution approach achieves
equivalent collision resistance with no interface changes and fixes all
existing substitution call sites at once. `extra_keys` remains the right
channel for request-scoped metadata that must NOT be baked into tokens
(e.g. LoRA IDs).

## Coverage

Substitution is applied on:

- the in-process connector (`vllm_v1_adapter.py`: save, load, lookup),
- the main MP connector (`lmcache_mp_metadata.py`; every key-carrying call
  in `lmcache_mp_connector.py` — lookup, store, retrieve, lock management,
  eager prefetch — goes through the tracker's `get_token_ids()`), and
- the version-pinned MP connector copies (`lmcache_mp_connector_0180.py`,
  `lmcache_mp_connector_0201.py`), which embed the same tracker-level
  substitution so vendored deployments on those vLLM versions are covered.

NOT yet handled (multimodal requests on these paths can still cross-hit and
need either substitution or an MM bypass guard):

- SGLang and TRT-LLM integrations
- token-addressed SDK/CLI paths

## Verification

- Unit: `lmcache/integration/vllm/tests/test_mm_hash_utils.py` (properties +
  16-bit collision regression), `tests/v1/test_mp_connector_mm_keys.py`.
- Acceptance: `tests/e2e_mm/` (real-engine matrix: cross-image isolation,
  collision pressure, chunk-boundary phases, mixed traffic, multi-image,
  video modality, preemption recompute; T3 `mp_connector` scenario reruns
  the T0/T1 core against a real MP cache server through the main MP
  connector, including a per-path detector negative control).
