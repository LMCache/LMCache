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
      (index, locks, allocation, checkpoints)
                |
                v
          RawBlockIO engine
         (posix / io_uring)
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

When `pid_affinity` reuse is enabled, freed space keeps the placement
identifier it was last written with, in memory. Among free extents of exactly
the requested length, the allocator prefers one with the requested identifier;
otherwise it takes the normal best fit. Coalescing keeps an identifier only when
both neighbours share it. Space above the highest offset ever allocated has
never been written, so taking it counts neither as an affinity hit nor as a
fallback.

Affinity is not checkpointed because FDP placement assignments are
process-local. After recovery, free space has no recorded affinity until it is
reused.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/distributed/l2_adapters/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and orchestrates I/O through a pluggable
`RawBlockIO` engine:

- raw device open/close
- in-memory key index
- free-space (extent) allocation
- lock refcounts used by MP lookup/load/unlock
- metadata checkpointing and recovery

The low-level "direct reads and writes through the Rust binding" responsibility
lives in the `RawBlockIO` engine so storage and transfer behavior stays behind a
single interface. `PosixBlockIO` implements synchronous `pread`/`pwrite` for
`io_engine="posix"`; `IoUringBlockIO` implements batched io_uring reads/writes for
`io_engine="io_uring"`, including optional NVMe raw-command (`use_uring_cmd`)
passthrough. `RawBlockCore` holds one `RawBlockIO` instance and leases buffers
through its `lease_buffer` / `lease_buffers` context managers, so callers never
touch the engine's buffer primitives directly. This keeps the durable-state logic
in one place while letting the I/O strategy be swapped without touching the core.

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

## Worker Failure and Task Completion

Recoverable io_uring submission errors (`EAGAIN`, `EINTR`, and `EBUSY`) retry
after completion progress or exponential backoff from 1 to 100 ms. Permanent
errors, or 30 seconds without submission or completion progress, stop new
native I/O admission. `RawBlockCore.report_status()` then reports
`is_healthy=False` and the terminal reason in `worker_error`. An ordinary
per-request I/O error does not mark the worker as failed. On a confirmed NVMe
unrecovered payload read, the core quarantines the object's space until
close, invalidates its owner and keeps successful sibling keys in the load bitmap.
The prefetch controller combines available tiers and applies the request’s
fetching policy. A full-attention prefix ends at the first missing chunk.

An open adapter reports a known worker failure through its task results, not
through a synchronous exception from `submit_*`. It allocates a task ID and
passes an already-failed future to the existing completion callback without
queuing work in the thread pool:

- lookup and load complete with all-zero bitmaps of the submitted key count;
- store completes with `L2StoreResult(False, 0)` without writing data;
- each completion decrements its in-flight counter, publishes the result, and
  signals the corresponding eventfd. Results are consumed once.

This lets the controllers release failed-load buffers and lookup/store locks
through their normal completion paths. Existing L1 and healthy-adapter hits
remain usable; missing data can be recomputed by the serving engine. Unlocks
remain available after worker failure. Invalid arguments and submissions after
adapter close still raise. The legacy non-MP backend skips new stores after a
known worker failure without retaining objects or raising to its caller.

I/O already accepted by the kernel keeps its buffers until terminal completion
or cancellation completion; a worker error alone must not complete those
tasks. Close and destructor cleanup release the GIL while draining and joining
the worker. Cleanup can still wait indefinitely for uncancellable I/O: the
submission retry budget is not a shutdown timeout.

## Locking Model

LMCache MP already uses L1 locks for CPU-memory object lifetime. `raw_block`
adds a separate L2-side lock refcount so a looked-up key cannot be deleted
between `lookup_and_lock` and `load`.

Rules:

- `exists_many(..., lock=True)` increments the refcount for hits
- `unlock_many(keys)` decrements and floors at zero
- `delete(keys)` skips locked entries

## Space Allocation

### Why not fixed slots

Fixed slots needed the object size at server start. In MP mode the byte size
of an object is only known after engines register, and one server holds many
sizes at once:

- per model: layers, KV heads, head size and dtype;
- per chunk size (MP multi-chunk-size, #6765 onward);
- per object group (`--separate-object-groups`, the default for hybrid models
  since #6041): full-attention, sliding-window, MLA and state groups differ;
- unfull chunks of sliding-window groups narrower than a chunk carry one
  extra lead block (`get_unfull_chunk_layout_desc`), so they are larger than
  full chunks;
- serde changes the length (AES-GCM adds a tag; quantizing codecs shrink it);
- TP changes the per-rank size.

A slot sized for the largest object wastes the rest of the slot on every
smaller one; a smaller slot rejects the larger objects outright.

### Structure

`ExtentAllocator` (`raw_block/extent_allocator.py`) keeps maximal free extents
indexed by start and by end (O(1) coalescing on free) and by length (a sorted
list of distinct lengths for best fit); Ceph BlueStore's AVL allocator keeps
the same two orders in trees. The unit is `block_align`. An object needs
`header_bytes + round_up(payload_len, block_align)` bytes, at one offset:

- Best fit: the head of the smallest free extent that fits; the remainder
  stays free. With same-size churn this is an exact fit, the same reuse
  pattern as slots. Among equal candidates the most recently freed is taken.
- A store fails when no single free extent fits, even if the free bytes add
  up; eviction then frees more (see Rejected alternatives for how often).
- Free: immediate coalescing with free neighbours. Freeing a range that starts
  or ends where free space does raises, which catches most double frees.
- Cost under `_lock`: dictionary operations plus, when a length appears or
  disappears, one insert into or delete from the sorted length list (memmove,
  O(k) for k distinct free lengths).

The object layout is the slot layout: header, then payload, as two write
requests. Zero-copy direct views, in-place tail zeroing, bounce buffers,
io_uring fixed buffers and io_uring_cmd splitting are unchanged.

The first `block_align` bytes of the data region hold a format marker (see
Upgrades and downgrades); objects are allocated after it.

### Checkpoint format

Format version 2 drops `slot_bytes` and `next_slot`; an entry keeps `offset`
and `size`, and its length follows from `size`. Recovery rebuilds free space
as everything not held by a valid entry; entries outside the data region,
off the block grid, or overlapping another entry are dropped (overlap means
corruption, so both go).

### Upgrades and downgrades

Version 1 (fixed-slot) checkpoints are imported: each object keeps only the
space it needs, and the slot tail is free. The entry in slot 0 is dropped,
since that block becomes the format marker. The checkpoint is then rewritten
as version 2.

A fixed-slot binary rejects version 2, starts empty and writes its first
object at slot 0, which overwrites the marker. A version 2 checkpoint found
without the marker therefore means such a run may have reused its space, and
it is not loaded. This covers a downgraded run that stopped before writing a
checkpoint of its own, unless it stopped after later slots reached the device
but before slot 0 did (one io_uring batch has no order, and posix runs several
store workers); the user docs ask to
clear the metadata region before a downgrade for that reason. An unreadable
marker next to a version 2 checkpoint stops startup rather than guess. The
marker is written before an imported checkpoint is rewritten, so a crash in
between imports it again.

### Rejected alternatives

A trace simulation (`alloc_sim.py`, attached to #6954 with `pinned_sim.py`
and the commands; 16 GiB, LRU eviction controller, working set 3x capacity,
200k operations) at the default watermark 0.8 and eviction ratio 0.2 compares
the allocators on equal terms: every policy treats the device as full after a
failed store, so the next pass evicts (`--full-on-failure`). Raw-block itself
does not do this; see "What ships" below. Hit rate / failed stores:

| Workload | Fixed slot | Exact-size slabs | 1.25x classes, segments returned | 1.25x classes, no return | This allocator |
|---|---|---|---|---|---|
| One size | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % |
| Hybrid (gpt-oss-like groups) | 0.295 / 0 % | 0.362 / 0 % | 0.346 / 0 % | 0.346 / 0 % | 0.362 / 0 % |
| Several models and chunk sizes | 0.227 / 0 % | 0.374 / 2.2 % | 0.366 / 1.1 % | 0.365 / 0 % | 0.373 / 0 % |
| Model mix changes halfway | 0.158 / 0 % | 0.290 / 1.4 % | 0.280 / 1.2 % | 0.221 / 24 % | 0.289 / 0 % |
| 99 % small objects | 0.010 / 0 % | 0.346 / 0 % | 0.340 / 0 % | 0.340 / 0 % | 0.346 / 0 % |
| 1 byte and 64 MiB | 0.226 / 0 % | 0.384 / 8.6 % | 0.384 / 8.6 % | 0.390 / 0 % | 0.390 / 0 % |
| Log-uniform 4 KiB to 64 MiB | 0.069 / 0 % | 0.044 / 48 % | 0.259 / 0.6 % | 0.350 / 0 % | 0.379 / 0 % |
| Two sizes 12 % apart (serde) | 0.356 / 0 % | 0.372 / 0 % | 0.356 / 0 % | 0.356 / 0 % | 0.372 / 0 % |

Slabs carve 256 MiB segments per exact size and return a segment once it is
empty. The 1.25x classes start at 64 KiB and top out at the largest object
(the full-slot class); "segments returned" carves them like the slabs,
"no return" from one shared region with fallback to a larger class.
Power-of-two classes do worse than 1.25x classes for most workloads (several
models at the default watermark: 0.286 vs 0.366) but better for log-uniform
sizes when segments are returned (0.298 vs 0.259). Fixed slots evict two to ten times
the bytes per store (several models: 58.6 MB vs 29.0 MB; log-uniform: 67.0 MB
vs 6.5 MB).

Random instead of LRU eviction order changes these by a few points.

**What ships.** Eviction still starts only at the watermark: a store that
finds no range long enough fails, and nothing is evicted for it. The same
simulation without `--full-on-failure` (hit rate / failed stores):

| Workload | Fixed slot (before) | This allocator |
|---|---|---|
| Hybrid groups | 0.295 / 0 % | 0.362 / 0 % |
| Several models and chunk sizes | 0.227 / 0 % | 0.373 / 0 % |
| Model mix changes halfway | 0.158 / 0 % | 0.289 / 0.1 % |
| Log-uniform 4 KiB to 64 MiB | 0.069 / 0 % | 0.379 / 0 % |
| Two sizes 12 % apart (serde) | 0.356 / 0 % | 0.372 / 0 % |


At the default watermark the free space left keeps failures at 0.1 % or
less. Evicting for failed stores below the watermark (an adapter signal to
the eviction controller) changes nothing there, and around objects that
cannot be evicted it drains the cache (see Known limitations); it is left
out.

- Size classes (the proposal in issue #3392): no splitting, so each object
  wastes up to the gap to its class; power-of-two classes waste about a
  quarter of KV sizes, which are not powers of two. Finer classes come close
  at a fixed mix, but space carved for a class only serves that class: without
  returning segments it calcifies when the mix changes (memcached's slab
  rebalancing problem), and with it, a segment comes back only once every
  object in it is evicted, so spread sizes still lose (log-uniform 0.259 vs
  0.379).
- Exact-size slabs that return empty segments: an LRU rarely empties a whole
  segment, and with many distinct sizes each one holds segments of its own.
- Buddy allocation: the same power-of-two rounding.
- Multi-extent objects (SPDK blobs, BlueStore `PExtentVector`): splitting an
  object across up to 8 free ranges when no single one fits split almost no
  object at the default watermark (0.03 % of stores when the
  model mix changes halfway, none otherwise). It costs more requests per object
  and a header that must also pin the layout. Around locked objects (Known
  limitations) it stored the large objects, but they were rarely hit and
  evicted small ones (small-object hit rate 0.145 to 0.123, with the
  failed-store eviction rule).
- Deletes that write nothing: a deleted object a checkpoint copy may list is
  kept as a hazard, and the store that reuses its space covers its first byte,
  holds the rest of it until its write ends and writes that byte no later.
  It removes the header write per delete (0.4 % of a 1 MiB object's bytes,
  0.04 % of 10 MB), but io_uring then has to submit the payload after the
  header whenever a store reuses a listed object, which with LRU eviction is
  most stores: a put took 3.1 µs more at 4 KiB and +2.6 % at 1 MiB on a
  ramdisk, and more on an ext4 file on NVMe. Holding the rest of the deleted
  object also makes stores of the same batch or of other threads fail or wait
  while that write is in flight. Linking the payload to the header in the
  Rust engine (`IOSQE_IO_LINK`) would remove the wait, at the cost of a
  change to the engine.
- Region or log-structured allocation (CacheLib BlockCache): reclaims whole
  regions, which conflicts with the per-key LRU eviction controller and needs
  reinsertion of live objects.

## Known limitations

- **Unevictable objects split free space for good.** Delete skips locked keys,
  and raw-block locks have no TTL, so leaked locks pin their objects until a
  restart. With fixed slots each one pins a slot; here, locked objects spread
  closer together than a large object leave no range long enough for it,
  whatever is evicted. In a simulation with the real allocator
  (`pinned_sim.py`, attached to #6954: 1 GiB, 1 MiB and 16 MiB objects,
  one 1 MiB object locked every 8 MiB after a small-object warm-up, 10 % of the
  device), 66 % of the 16 MiB stores fail (1.5 % without locked objects);
  usage stays at 0.74 and the small objects keep their hit rate. Evicting on
  every failing pass instead would let 33 % fail but drop usage to 0.55 and
  the small objects' hit rate from 0.153 to 0.107. Locks at random places, or
  held for a while and released, had little effect. A rising
  `alloc_failure_count` with a non-zero `locked_key_count` while idle shows
  the condition; the fix is the leak (for example #6791).
- **High watermarks leave too little free space.** Above about 0.85 the free
  space left is often split into ranges shorter than the larger objects, and
  their stores fail until usage reaches the watermark again (in the
  simulation at 0.9, 60 % of stores when the model mix changes halfway;
  fixed slots fail often there too, 28 % in that case and 21 % with several
  models). A rising `alloc_failure_count` shows the condition.
- **A long delete round blocks the controller.** `adapter.delete()` runs
  under the controller's `_states_lock`, which `report_status()` also takes,
  so `/status` and the other L2 adapters' eviction wait for the round (as
  before). With the header write per
  deleted object, a round of 100k objects takes about 0.6 s with io_uring on a
  ramdisk and several seconds with posix on NVMe.

## Persistence and Recovery

`RawBlockCore` keeps the existing metadata checkpoint model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index
- POSIX recovery validates per-object headers with an internal pool of 8 reader
  threads. Both regular `io_uring` and `io_uring_cmd` batch header reads up to
  `iouring_queue_depth`.

The data layout (object header, then payload) is unchanged; the checkpoint
format changed with extent allocation (see Space Allocation).

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed space comes from global L2 eviction or explicit `delete()` calls.

### Reuse after a delete

Recovery trusts the newest valid checkpoint and verification rereads only
each object's header. With variable sizes, space a deleted object held can be
reused by an object that starts inside it, so the old header can survive
while its payload is overwritten; recovery would then map the old key to new
bytes. A delete therefore overwrites the header first and frees the space
only after that write returns:

```text
delete(K) ──► header zeroed (one header_bytes write) ──► free
                         │
                    write failed: retire the range (as quarantine)
```

- The write happens outside `_lock`, batched per `delete_many` call, with the
  object's placement id; the posix engine writes one request after another,
  so a batch is spread over the 8 recovery threads (2048 deletes of 64 KiB
  objects on an ext4 file on NVMe with O_DIRECT: 224 µs per key serial, 36 µs
  spread; io_uring 4.4 µs; with 8 keys per call on a ramdisk, 9.7 to
  23.7 µs and 5.8 to 11.5 µs for 64 KiB to 16 MiB objects, against 0.6 to
  1.4 µs for the fixed-slot delete). Space is freed as the header writes
  end: per object with posix, per io_uring batch otherwise.
- Stores pay for the allocator, the generation and the in-flight record under
  the lock: on a ramdisk with O_DIRECT, a put took 3.5 % more at 64 KiB with
  io_uring and 10 % with posix, 0.7 % and 1.1 % at 1 MiB, 0.2 % and 0.6 % at
  4 MiB; loads moved within ±4 %. Without I/O the bookkeeping is 4.1
  instead of 3.1 µs per put, about 0.6 µs of it in the allocator's dict
  updates.
- A failed batch is retried one request at a time; only ranges that fail
  again are retired.
- Each write puts a random 64-bit generation in the header (bytes 24-32) and
  the checkpoint entry (`gen`); recovery compares it with the key and size.
  A key re-stored where it was listed (best fit puts it back there) leaves a
  header of another generation, so a crash between its header and payload
  writes drops the entry instead of serving the old range's new bytes. For
  the same reason the space of a failed or cancelled store is free at once.
  An entry imported from fixed slots takes the generation bytes its header
  has, which earlier versions left unset; the import reads the headers even
  with `meta_verify_on_load` off, or a later verified run would drop them.
- An entry dropped by header validation at startup is freed at once when its
  header did not match; when the header could not be read it is retired,
  since the header on the device may still be valid.
- The write happens with `meta_verify_on_load` off too: that run's recovery
  reads no headers, but a later run with verification may load the same
  checkpoint.
- Sequence numbers continue after the highest valid header on the device, so
  a checkpoint written later supersedes the older copies. A run that loads no
  checkpoint (skipped, rejected, or written with another `meta_magic` or
  `meta_version`, which it cannot see) zeroes the header block of every
  metadata container before any store, since its stores can reuse space an
  old checkpoint lists without overwriting those headers; with a rejected
  checkpoint it also zeroes the containers of that checkpoint's own
  `meta_total_bytes`. A metadata read that fails stops startup instead.
- With `meta_verify_on_load=false` there is no header check, so reused space
  can return wrong bytes after a crash, as with fixed slots.
- Nothing is flushed, so the ordering holds for process crashes, not for power
  loss on drives with volatile write caches (issue #6568).

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
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
- `header_bytes` and `meta_total_bytes` must be aligned to `block_align`;
  `slot_bytes` is accepted for old configs but ignored
- with `use_uring_cmd=true`, `block_align` must be a multiple of the NVMe
  namespace LBA size
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true` or `use_uring_cmd=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`
- with `use_odirect=true`, raw-block I/O rejects offsets and total I/O lengths
  that are not aligned to `block_align`; misaligned write buffers use an
  aligned bounce buffer
- with `io_engine="io_uring"` and `enable_zero_copy=true`, a payload that is
  not a multiple of `block_align` (for example after AES-GCM serde) is
  transferred straight from its L1 allocation when that allocation covers the
  aligned length, which the L1 alignment rule above guarantees for L1 objects.
  Writes zero the padding after the payload in place; objects without such
  padding keep the bounce buffer

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/distributed/l2_adapters/raw_block/core.py`
- I/O engine: `lmcache/v1/distributed/l2_adapters/raw_block/io_engine.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
