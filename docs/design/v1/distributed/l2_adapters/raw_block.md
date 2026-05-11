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

## TODO

- FDP / placement-hint support.
- A raw NVMe command path.

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

`RawBlockCore` checkpoints its in-memory state into a fixed metadata region at
the head of the device. The model is:

- metadata region reserved on the same device, sized by `meta_total_bytes`
- periodic checkpointing from a background thread, plus on-`close` flush
- optional checkpoint load on startup (`load_checkpoint_on_init`)
- optional per-slot header verification on load (`meta_verify_on_load`)
- recovery rebuilds the in-memory index from the latest durable base plus any
  durable delta records that follow it

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed slots come from global L2 eviction or explicit `delete()` calls.

### On-device checkpoint format

The metadata region is split into two equal-size mirrors. Within each mirror::

    [ header (one block) ][ payload (round_up to block_align) ][ delta tail ]

The delta tail starts implicitly at the block-aligned end of the base payload
and runs to the mirror's end. There is no separate delta region.

- Base header (`_META_HEADER_STRUCT`, 32 B): magic, version, monotonic seq,
  payload length, and CRC32 over the payload. The two mirrors are read on
  startup and the higher-seq valid one wins, giving a torn-write-safe ping-pong
  flip.
- Delta record header (`_DELTA_RECORD_HEADER_STRUCT`, 52 B): magic `LMCD`,
  record version, parent `base_seq`, parent `base_crc`, monotonic
  `delta_seq`, `prev_record_crc`, payload length and CRC32, op count, flags,
  total blocks, reserved. Each record is block-aligned and binds itself to its
  base via both `base_seq` *and* `base_crc`, so a stale tail surviving a mirror
  flip is rejected even when the seq number happens to collide.
  `prev_record_crc` chains records together and is anchored at `base_crc` for
  the first record, catching tail bytes that look valid in isolation.
- Compaction is a mirror flip: write a fresh full base into the inactive
  mirror, zero the new mirror's tail head so it is unambiguously empty, then
  commit the new header. A torn compaction leaves the previous active mirror
  untouched as the durable winner.
- Compaction triggers: tail full, oversized record, more than
  `meta_full_checkpoint_max_deltas` deltas accumulated, or more than
  `meta_full_checkpoint_interval_sec` since the last full base.
- Backwards compatibility: a disk written by older code that has no delta
  records reads naturally as "zero deltas applied" -- the tail bytes do not
  carry the `LMCD` magic, replay stops on the first record, and the loaded
  state matches the base alone.

### Replay

On open, the active mirror is selected by valid header with the higher
`base_seq`. Its base payload is decoded into the in-memory index, then the
delta tail is walked record-by-record. Replay stops at the first record that
fails any check (bad magic, mismatched `base_seq` or `base_crc`, broken hash
chain, non-monotonic `delta_seq`, oversized payload, CRC mismatch, or read
error); everything earlier is durable and applied, everything after is treated
as torn or stale.

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
  "meta_full_checkpoint_interval_sec": 600,
  "meta_full_checkpoint_max_deltas": 1024,
  "load_checkpoint_on_init": true,
  "meta_verify_on_load": true,
  "num_store_workers": 2,
  "num_lookup_workers": 1,
  "num_load_workers": 4
}
```

Important validation rules:

- `slot_bytes`, `header_bytes`, and `meta_total_bytes` must be aligned to
  `block_align`
- `slot_bytes >= header_bytes + 1`
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`
- `meta_full_checkpoint_max_deltas` bounds worst-case replay cost; raising it
  reduces compaction frequency at the cost of more delta records to apply on
  recovery
- `meta_full_checkpoint_interval_sec` bounds worst-case staleness of the base
  payload independent of mutation rate

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/storage_backend/raw_block/core.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
