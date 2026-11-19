# Encryption serde (`aesgcm`)

AES-GCM encryption of KV cache bytes **at rest in the L2 tier**, keyed per
`cache_salt`. Implemented as a serde (`aesgcm.py`, with key providers in
`key_provider.py`), so it plugs into any L2 adapter through the existing
`SerdeL2AdapterWrapper` with no adapter or controller changes.

## Threat model / scope

Protects L2 bytes against a party who can read the remote storage (S3 bucket,
disk, RESP). **In scope:** the bytes persisted to L2. **Out of scope:** L1 (host
RAM) and L0 (GPU) hold plaintext, so a party with access to a running MP server
process is not defended against — a different threat model. This is
at-rest confidentiality for the durable tier, not end-to-end.

Each frame is also authenticated together with its object key: a modified
frame, or a frame found under a key other than the one it was stored under,
fails to load instead of being served as that key's KV (see
[Binding a frame to its object key](#binding-a-frame-to-its-object-key)).

## Why the serde layer

The serde runs on both legs of the L1↔L2 round trip — `serialize` on store,
`deserialize` on load — so a reversible (symmetric) transform is exactly the
right shape. Encryption sits beside the quantizer serdes (`fp8`, `turboquant`);
it is the first serde that needs per-key context (the `cache_salt`), which
PR that added `ObjectKey` to the serde interface made available.

## Method: AES-256/128-GCM

AEAD — confidentiality **and** integrity in one pass. Chosen over alternatives
because on server AES-NI hardware it is the fastest *authenticated* cipher
(~4–8 GB/s/core for AES-128), and it runs off the inference hot path (L1↔L2,
not the L0↔L1 CUDA-IPC transfer), so its cost hides behind S3 latency.

Default `aes_bits=128`: AES-128 is already computationally unbreakable, and KV
cache is short-lived + regenerable so the harvest-now-decrypt-later hedge behind
256 barely applies. `256` is available as a config knob for compliance mandates.

Compression is **not** an alternative (it doesn't make data secret) and is a
poor fit for KV (high-entropy → barely shrinks); quantization is the size lever,
composed before encryption.

### Wire frame (per chunk)

```
[1B version][12B IV][ciphertext || 16B GCM tag]
```

- **version** — format byte, so the scheme can evolve. The current version is
  `2`; version `1` frames are rejected (see below).
- **IV** — a fresh random 96-bit nonce per chunk, stored in the clear (not
  secret; must be unique — a repeated (key, IV) breaks GCM).
- **ciphertext ‖ tag** — AES-GCM output; ciphertext is the plaintext length (no
  padding), tag is the integrity check.

Fixed overhead is 29 bytes/chunk, so `estimate_serialized_size` is exact
(plaintext + 29), not an upper bound.

A tag mismatch (tampering, wrong key, or a frame stored under another object
key) raises `InvalidTag`, which the wrapper turns into a load failure → treated
as a cache miss (re-fetch / recompute), never silent corruption.

### Binding a frame to its object key

The AES key depends only on `cache_salt`, so every object of one salt shares
it. To tie a frame to the object it was computed for, each frame also passes
the full object identity as AES-GCM associated data on both encrypt and
decrypt. A frame then decrypts only under its own key, so a storage-side
mix-up, such as an off-by-one chunk mapping, is a cache miss rather than KV for
other tokens. Associated data is authenticated
but not stored, so the frame size is unchanged. It is encoded unambiguously:

```
"lmcache-l2-aesgcm-aad-v2"
u32 len ‖ chunk_hash
u32 len ‖ model_name (UTF-8)
i64 kv_rank ‖ u64 object_group_id
u32 len ‖ cache_salt (UTF-8)
u32 tag count ‖ (u32 len ‖ name ‖ u32 len ‖ value) per tag, sorted by name
```

Integers are big-endian. Request-scoped `extra_attributes` are not part of the
identity, so they are excluded.

Version `1` frames have no associated data and fail with the
malformed/unknown-version `ValueError`, so objects stored by earlier versions
are recomputed once after an upgrade.

### Deserialize length derivation

The load temp buffer is sized to `estimate_serialized_size` and may be
allocated larger than the stored frame (allocator alignment). `deserialize`
therefore derives the exact ciphertext length from **`dst`** (the KV target,
whose byte size equals the original plaintext), not from the padded `src` —
mirroring how `fp8` reads its length from `dst`.

## Keys: `cache_salt` is the selector, not the key

`cache_salt` (public; it is cleartext in the L2 object name) selects *which*
key; secret material comes from a swappable `KeyProvider`:

- **`HkdfKeyProvider`** (default, shipped) — derives a per-tenant key via
  `HKDF-SHA256(master_key, info=cache_salt)` from one master key read from
  `master_key_path` (e.g. a mounted K8s `Secret`), cached per salt. Any holder
  of the master can derive every tenant's key → **"fleet vs. outside"** trust:
  it protects L2 bytes from outsiders, not tenants from each other.
- **`KeyringKeyProvider`** (deferred) — per-tenant provisioned keys (KMS /
  per-tenant mounts) for true cross-tenant isolation, at real
  distribution/rotation cost. Not implemented; the factory rejects
  `key_provider="keyring"`.

The empty `cache_salt` (anonymous traffic) is a valid tenant bucket; HKDF
accepts an empty `info` and derives a stable key for it.

## Config (`SerdeConfig.kwargs`)

| Key | Default | Meaning |
|---|---|---|
| `key_provider` | `hkdf` | Key source; only `hkdf` implemented |
| `master_key_path` | — | Master-key file (required for `hkdf`) |
| `aes_bits` | `128` | `128` or `256` |
| `max_workers` | `1` | Serde thread-pool size |

## Deferred (future work, not this serde)

Content encryption does not hide L2 object **metadata**: the `cache_salt`
(tenant identity) and a content-derived `chunk_hash` remain in the object name,
so an observer of the bucket can see which tenant stored what and detect
cross-tenant content overlap without decrypting. Closing this is a separate
"metadata-hardening" step (salt the chunk hash; pseudonymize the salt at
ingress). Per-tenant key isolation (`KeyringKeyProvider` + tenant→node
placement) is likewise deferred.
