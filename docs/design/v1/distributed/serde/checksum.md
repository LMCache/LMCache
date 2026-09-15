# Checksum serde (`checksum`)

Lightweight corruption detection for L2 payloads: a version byte, an 8-byte
chunk-hash fingerprint, and an `xxh3_64` hash of the payload, verified on
load. Implemented as a serde (`checksum.py`), so it plugs into any L2
adapter through the existing `SerdeL2AdapterWrapper` with no adapter or
controller changes — same mechanism as `fp8`, `aesgcm`, `asym_k16_v8`.

## Threat model / scope

Detects **unintentional** corruption of L2 bytes: bit rot, truncated/partial
writes, and a payload for one chunk landing in another chunk's slot (race,
misplacement, accidental copy). The motivating case is a file-backed adapter
treating an existing file as valid; `FSL2Adapter`/`fs_native` only check byte
count on load. The serde is generic and can wrap any L2 adapter.

**Out of scope: a determined adversary.** `xxh3_64` is unkeyed — anyone who
can rewrite a file can also recompute a matching hash, so this serde
provides no confidentiality and no protection against deliberate tampering
by a party with write access to the L2 storage. If that is the actual threat
model, use `aesgcm` instead: its GCM tag is a keyed MAC, so a party without
the key cannot forge a valid tag no matter what bytes they write.

## Relationship to `aesgcm`

If `aesgcm` is already enabled on an adapter, this serde is redundant — the
GCM tag already gives corruption detection (and more: confidentiality and
tamper resistance) as a side effect of encryption. `checksum` exists for
deployments that want corruption detection **without** `aesgcm`'s key
management (a `master_key_path` to provision and rotate) or its on-disk
opacity (payloads become unreadable ciphertext, which complicates ad-hoc
disk inspection/debugging). See the Performance section for the focused
in-memory comparison with `aesgcm`.

## Format and validation

xxh3_64 is the shipped non-cryptographic hash for fast large-payload
processing. The committed benchmark intentionally compares only copy,
xxh3_64, and `aesgcm`; the CRC32 selection experiment is outside this script.

### Wire frame (per chunk)

```
[1B version][8B chunk_fingerprint][8B xxh3_64][payload]
```

- **version** — format byte, so the scheme can evolve (e.g. a different
  hash algorithm) without breaking stored blobs.
- **chunk_fingerprint** — the first 8 bytes of `key.chunk_hash` (zero-padded
  if shorter). It fingerprints the chunk rather than binding the full
  `ObjectKey`; no re-hashing or re-encoding is needed. This compact guard
  catches a payload landing in another chunk's slot, while `xxh3_64` checks
  the payload bytes.
- **xxh3_64** — `xxhash.xxh3_64(payload).digest()`, an 8-byte hash.

Fixed overhead is 17 bytes/chunk, so `estimate_serialized_size` is exact
(plaintext + 17), not an upper bound.

A version, fingerprint, or hash mismatch raises `ValueError`. On load, the
payload length comes from `dst` because the source buffer may be padded. A
failed deserialize becomes a clean miss rather than returning corrupted data.

### Why not a magic byte

This serde is opt-in and has no separate magic field. Existing headerless
objects fail frame validation and follow the normal clean-miss path, so
enabling it on a populated adapter should be treated like enabling `aesgcm`:
the old cache misses and repopulates.

## Config (`SerdeConfig.kwargs`)

| Key | Default | Meaning |
|---|---|---|
| `max_workers` | `1` | Serde thread-pool size |

No key material, no per-tenant state — this serde is stateless.

## Performance

Run the benchmark from
[`examples/serde/checksum/README.md`](../../../../examples/serde/checksum/README.md).
It measures the real `Serializer`/`Deserializer` path in memory. Results are
hardware-dependent.

- `copy` is a measured byte-copy baseline, not a zero-cost no-serde path.
- The `checksum` and `aesgcm` rows provide a relative transform-cost
  comparison at representative payload sizes.
- The output is intended for reproducible relative comparisons, not fixed
  performance guarantees.