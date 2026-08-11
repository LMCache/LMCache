# Raw Block L2 Adapter Design

This document describes the built-in `raw_block` L2 adapter for LMCache MP
mode. It covers the adapter shape, the shared raw-block core, and the recovery
model.

## Overview

`raw_block` is a persistent MP L2 adapter backed by a raw block device or a
dedicated file. It is designed to keep the MP request flow unchanged while
reusing the existing raw-block on-device metadata format and the low-level Rust
raw-device I/O path.

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
           RawBlockCore
      (index, locks, slots, checkpoints)
                |
                v
         lmcache_rust_raw_block_io
      (pwrite_from_buffer / pread_into)
                |
                v
         raw block device / file
```

## Goals

- Support LMCache MP mode using raw block storage as an L2 cache.
- Reuse the same durable metadata and checkpoint model as the existing
  non-MP raw-block backend.
- Reuse the existing Rust raw-device I/O layer.
- Preserve restart recovery semantics.
- Keep the MP controller flow unchanged: store, lookup-and-lock, load, unlock.

## FDP Placement Base

The MP `raw_block` adapter supports NVMe Flexible Data Placement (FDP) status
discovery when `io_engine="io_uring"` and `use_uring_cmd=true`. At startup, it
queries FDP reclaim unit handle status from the device and reports the
discovered mapping. Startup fails if the query fails or the device reports no
placement identifiers.

FDP plumbing is split by layer: `RawBlockL2Adapter` discovers and registers
non-zero placement identifiers, while `RawBlockCore` enforces that explicit
identifier 0 is never used. `fdp_placement_ids` is the KV data placement pool:
explicit data identifiers are rejected if they overlap with
`meta_checkpoint_placement_id` or are not reported by the device. If
`fdp_placement_ids` is omitted, the adapter registers all device-reported
non-zero identifiers except the metadata checkpoint identifier.

The adapter maps KV data writes onto FDP placement identifiers with a
cache-salt prefix policy. It derives a case-insensitive bucket from the part of
`ObjectKey.cache_salt` before `:` only when the separator is present, assigns
buckets to placement identifiers in first-seen order, and reuses the same
identifier for later writes in that bucket. Values without `:` or with an empty
prefix omit the directive; `rag:` is a valid opt-in to the `rag` bucket. The
mapping is availability-first: if the number of buckets exceeds the number of
registered data placement identifiers, extra buckets fall back to no directive
and the adapter emits one warning. Status reporting tracks a fallback count and
a bounded bucket sample rather than retaining all fallback bucket names. Empty
`cache_salt` values also omit the directive. The mapping is process-local and
may change after restart; read correctness is unaffected because `cache_salt` is
part of the object key rather than the read path's FDP directive. Metadata
checkpoint writes can use an explicit configured placement identifier while
defaulting to no directive when unset. User-facing FDP configuration rules live in
`docs/source/mp/l2_storage/raw_block.rst`; low-level NVMe command encoding
details live in `rust/raw_block/README.md`.

When `pid_affinity` slot reuse is enabled, `RawBlockCore` tracks each free
slot's latest placement identifier in memory and prefers a matching slot during
reuse. If no matching slot is available, it falls back to another free slot or
allocates a new slot.

Slot affinity is not checkpointed because FDP placement assignments are
process-local. After recovery, free slots have no recorded affinity until they
are reused.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/storage_backend/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and blocking I/O:

- raw device open/close
- in-memory key index
- free-slot tracking
- lock refcounts used by MP lookup/load/unlock
- metadata checkpointing and recovery
- direct reads and writes through the Rust binding

This avoids maintaining separate raw-block implementations for MP and non-MP
mode.

## Adapter Contract

`RawBlockL2Adapter` implements `L2AdapterInterface` directly. It exposes:

- three distinct eventfds: store, lookup, load
- non-blocking task submission APIs
- worker-thread execution for blocking raw-device operations
- result maps keyed by adapter-local task id
- listener notifications for stored, accessed, and deleted keys

The adapter uses caller-provided `MemoryObj` buffers for load operations. It
does not allocate destination buffers on the load path.

## Locking Model

LMCache MP already uses L1 locks for CPU-memory object lifetime. `raw_block`
adds a separate L2-side lock refcount so a looked-up key cannot be deleted
between `lookup_and_lock` and `load`.

Rules:

- `exists_many(..., lock=True)` increments the refcount for hits
- `unlock_many(keys)` decrements and floors at zero
- `delete(keys)` skips locked entries

## Persistence and Recovery

`RawBlockCore` keeps the existing metadata checkpoint model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index

The on-device format is intentionally unchanged by the MP adapter work.

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed slots come from global L2 eviction or explicit `delete()` calls.

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "slot_bytes": 1048576,
  "capacity_bytes": 0,
  "use_odirect": true,
  "block_align": 4096,
  "header_bytes": 4096,
  "meta_total_bytes": 268435456,
  "meta_magic": "LMCIDX01",
  "meta_version": 1,
  "meta_checkpoint_interval_sec": 60,
  "meta_enable_periodic": true,
  "load_checkpoint_on_init": true,
  "meta_verify_on_load": true,
  "num_store_workers": 2,
  "num_lookup_workers": 1,
  "num_load_workers": 4
}
```

For FDP configuration examples, see `docs/source/mp/l2_storage/raw_block.rst`.

Important validation rules:

- `block_align` must be a power of two
- `slot_bytes`, `header_bytes`, and `meta_total_bytes` must be aligned to
  `block_align`
- with `use_uring_cmd=true`, `block_align` must be a multiple of the NVMe
  namespace LBA size
- `slot_bytes >= header_bytes + 1`
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`
- with `use_odirect=true`, raw-block I/O rejects offsets and total I/O lengths
  that are not aligned to `block_align`; misaligned write buffers use an
  aligned bounce buffer

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/storage_backend/raw_block/core.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
