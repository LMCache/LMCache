# Portable compressed-record format

## Scope

This document defines the version-1 record implemented by `lmcache/v1/distributed/compress_adapters/format.py`, its Python reference codec, and the standalone vendor compatibility probe. It is the portable-format foundation for [GPU decompression RFC #4498](https://github.com/LMCache/LMCache/issues/4498), coordinated with [hardware compression RFC #4149](https://github.com/LMCache/LMCache/issues/4149). No production serde, L2, or retrieve path uses this format yet.

A record identifies its codec, framing, and post-decompression transform independently of the encoder or decoder library. Version 1 supports Deflate with raw or Gzip framing and no post-decompression transform. Device identities, pointers, streams, library versions, and execution engines are runtime concerns and are not stored in the record.

The format describes opaque bytes. Their logical KV layout and expected decoded size come from the caller; compression chunks do not imply model-layer boundaries. This permits the same record structure to describe different KV layouts without fixing a transfer schedule.

## Wire layout

All integers are unsigned and little-endian. A complete record has this shape:

```text
44-byte fixed header
  -> 24-byte descriptor for each independent compression chunk
  -> zero padding to the next 16-byte boundary
  -> compressed chunk 0
  -> zero padding to the next 16-byte boundary
  -> compressed chunk 1 ... final chunk
```

The fixed header is:

| Byte offset | Bytes | Field | Version-1 meaning |
|---|---|---|---|
| 0 | 4 | magic | ASCII `LMCR` |
| 4 | 1 | version | `1` |
| 5 | 1 | codec | `1`: Deflate |
| 6 | 1 | framing | `1`: raw, `2`: Gzip |
| 7 | 1 | post-decompression transform | `0`: none |
| 8 | 4 | header size | `44 + 24 * chunk_count` |
| 12 | 8 | record size | Exact complete record length, including alignment gaps |
| 20 | 8 | uncompressed size | Sum of descriptor output sizes |
| 28 | 4 | chunk count | Number of descriptors |
| 32 | 4 | header CRC-32/IEEE | Checksum of the complete header and descriptor table, with this field zeroed |
| 36 | 4 | compressed-payload CRC-32/IEEE | Checksum of all bytes from `header_size` through `record_size`, including alignment gaps |
| 40 | 4 | reserved flags | Must be zero |

Each descriptor is:

| Descriptor offset | Bytes | Field |
|---|---|---|
| 0 | 8 | Absolute payload offset from the start of the record |
| 8 | 4 | Compressed stream size |
| 12 | 4 | Expected uncompressed size |
| 16 | 4 | CRC-32/IEEE of the uncompressed chunk |
| 20 | 4 | Reserved flags, which must be zero |

The header is limited to 1 MiB, allowing at most 43,688 descriptors. Nonempty chunks have positive compressed and uncompressed sizes. An empty input is a 44-byte record with no descriptors, no payload, and a zero compressed-payload checksum.

## Alignment and stream boundaries

Each payload starts at the smallest 16-byte-aligned offset after the header or previous payload. Alignment gaps contain zero bytes; extra gaps are invalid. The final stream ends exactly at `record_size`, without trailing record padding. Every non-final uncompressed chunk size is divisible by 16, so concatenated output ranges remain aligned when the output base is aligned. A runtime backend must still check the actual allocation addresses against its native requirements.

Each descriptor covers exactly one complete Deflate or Gzip stream. Independent streams allow a future device backend to submit chunks as a batch without repacking the stored bytes. Structural validation cannot prove stream termination: the reference decoder additionally rejects truncated streams, trailing bytes, and concatenated streams within one descriptor.

## Validation and ownership

`parse_record_header()` accepts a contiguous byte buffer containing at least the complete header. It checks version, enum identifiers, reserved fields, bounds, descriptor layout, and the header checksum. It does not validate payload bytes or require the complete record, so its result is useful for size discovery but is not sufficient for native decompression.

`validate_complete_record()` requires immutable `bytes` containing exactly one record. It validates the header, exact record length, payload checksum, and zero alignment gaps, then returns an immutable `ValidatedCompressedRecord` retaining the same bytes without copying. Mutable buffers and memoryviews are rejected because another alias could change them after validation.

CRC-32 detects accidental corruption; it does not authenticate input or establish that a codec stream is safe for a native decoder. A future GPU backend must accept records only from trusted writers or a separately authenticated or memory-safe validation boundary, and must validate decoded output before exposing KV data.

## Reference codec

`encode_reference_record()` uses Python's zlib module to create independent streams and assemble their descriptors, alignment gaps, and checksums. The caller explicitly selects framing and a positive chunk size divisible by 16; this module does not choose production defaults.

`decode_reference_record()` requires the expected uncompressed size from the caller's logical layout and rejects a mismatching record before decompression. It bounds decoded chunk output to the advertised size plus one byte for overflow detection, supplies compressed input in bounded windows, and verifies stream termination, exact output sizes, and output CRCs.

```python
from lmcache.v1.distributed.compress_adapters import (
    CompressionCodec,
    CompressionFraming,
    StoredCompressionFormat,
    validate_complete_record,
)
from lmcache.v1.distributed.compress_adapters.reference import (
    decode_reference_record,
    encode_reference_record,
)

original = b"example KV bytes" * 4
record = encode_reference_record(
    original,
    stored_format=StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    ),
    chunk_size=32,
)
validated = validate_complete_record(record)
assert validated.record_bytes is record
assert decode_reference_record(
    record, expected_uncompressed_size=len(original)
) == original
```

The reference codec is a correctness oracle and fixture producer. It is not registered as a production serde or used as an implicit CPU fallback. Deflate output is not canonical across encoder versions; the frozen fixtures constrain reader compatibility, not every encoder's exact output bytes.

## Compatibility evidence and testing

The [vendor probe](../../../../../tests/v1/distributed/compress_adapters/vendor_compatibility/README.md) copies each frozen record unchanged to a device and passes record-relative chunk offsets to nvCOMP or hipCOMP. The source branch records successful raw-Deflate and Gzip decoding with nvCOMP 5.3.0.16 on an RTX 4060 using CUDA 12.9.86. An AMD hardware result remains pending. These single-chunk fixtures establish framing and address-alignment compatibility, not production throughput, concurrency, or asynchronous ownership guarantees.

Run the format and reference-codec tests from the repository root:

```bash
python -m pytest -q tests/v1/distributed/compress_adapters/test_format.py tests/v1/distributed/compress_adapters/test_reference.py
```

These tests exercise byte records and do not require a GPU. The repository's shared test setup requires the common `lmcache_native` extension. Coverage includes frozen vectors, malformed metadata, resource bounds, immutable ownership, padding, corruption, stream termination, output-size limits, and both framings. Native probe instructions and dependency pins are in its README.

## Versioning and remaining work

Version 1 requires an exact header size and rejects unknown identifiers or flags. Changes to the deployed wire layout require a new record version. While this draft remains undeployed, revisions must update the frozen vectors and vendor fixtures together.

Shared device/request/completion contracts, vendor backends, store-side serde selection, deferred L1 representation, variable-size L2 loads, and retrieve integration belong to subsequent changes. Production framing and chunk size remain open until compatibility and workload measurements support a choice.
