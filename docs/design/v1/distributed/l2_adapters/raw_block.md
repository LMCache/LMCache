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

## Rust I/O status counters

`RawBlockCore.report_status()["rust_io"]` exposes device-lifetime cumulative
counters from Rust. The adapter forwards them under `core.rust_io`, and the MP
info API returns them under `storage_manager.l2_adapters`. No reset API or
Rust-side metrics exporter is provided. A missing extension snapshot method,
snapshot failure, or closed core produces `rust_io: null`, not fabricated zeros.
The low-level device can still snapshot its counters after close.

The counting unit is a **backend I/O attempt**: one actual POSIX syscall or one
accepted io_uring SQE, including each retry. It is neither a Rust request nor a
hardware command. Byte counters are explicitly named `*_submitted_bytes`:
they count requested lengths, not achieved device throughput or transferred
payload bytes. Request lifecycle gauges have `*_requests` names and must not
be combined with attempt completion counters.

| Fields | Meaning |
| --- | --- |
| `read_attempts`, `write_attempts` | POSIX syscall attempts or accepted io_uring SQEs. |
| `read_submitted_bytes`, `write_submitted_bytes` | Requested bytes for each attempt, including padding and retries. |
| `completed_attempts` | Positive POSIX/ordinary CQE results, successful zero-length SQEs, or zero NVMe command status. |
| `failed_attempts` | Errors, zero-progress nonempty ordinary I/O, or submitted attempts abandoned during shutdown; not exclusively device errors. |
| `outstanding_requests` | Existing Rust outstanding-request count; includes io_uring requests queued or being prepared by the worker. POSIX counts active syscall attempts. |
| `peak_outstanding_requests` | Lifetime maximum of that outstanding count. |
| `queued_requests`, `peak_queued_requests` | Current and peak software request queue length; excludes SQEs already built and pending submission. Zero for POSIX. |
| `queue_full_events` | Worker capacity-limited batches/retries and `EAGAIN` submission results; excludes `EINTR`. |
| `bounce_attempts`, `bounce_submitted_bytes` | Submitted attempts and requested bytes using a Rust bounce buffer. |
| `fixed_buffer_attempts`, `fixed_buffer_submitted_bytes` | Submitted attempts and requested bytes using registered fixed buffers. |

For example, an 8 KiB read that returns 4 KiB and then submits the remaining
4 KiB counts as two reads and 12 KiB of requested bytes. A positive short
completion counts as a successful attempt even if a later retry fails.
Pre-submission validation/build failures and queued requests cancelled before
submission do not increment attempt or failure counters. NVMe administrative
ioctls and Python-side buffer copies are outside these metrics. Checkpoint and
recovery reads/writes through the device are included.

The outstanding gauge intentionally retains the existing shutdown/batch
accounting semantics rather than introducing a second physical-in-flight
counter. Queued requests are a subset, not an additional quantity to add.
Cross-field snapshots are best-effort relaxed atomic loads, not transactional.
Synchronous POSIX attempts are published after the syscall returns; the
outstanding gauge still covers time spent inside the syscall.
Once all submitted attempts have terminated:

```text
read_attempts + write_attempts == completed_attempts + failed_attempts
outstanding_requests == 0
queued_requests == 0
```

The first equality does not hold during live I/O and is not expressed in terms
of `outstanding_requests`: one request may generate multiple attempts. An
initial SQE and all its short-I/O retries retain a single lifecycle request.
The existing `in_flight_count` continues to drive request/batch shutdown and
cleanup; metric naming does not turn it into a kernel-concurrency counter.

### Submission and termination rules

| Event | Attempt and byte counters | Request lifecycle |
| --- | --- | --- |
| Validation or SQE-build failure before submit | No attempt, bytes, or failed attempt | Release any already-acquired request tracking. |
| Request enters the software queue | No submitted attempt yet | Outstanding includes queued work; queued gauge increases. |
| Partial submit accepts N SQEs | Count only those N attempts and their requested lengths | Unaccepted SQEs stay pending, without duplicating request tracking. |
| Submit returns zero, EAGAIN, or EINTR without acceptance | No attempt or failed completion; retry pending SQEs | Outstanding remains unchanged. |
| Fatal submit error for unaccepted SQEs | No submitted attempt or failed attempt | Terminate unsubmitted requests through cleanup. |
| Positive short ordinary completion | Complete that attempt successfully | Request remains outstanding until remaining bytes finish or fail. |
| Short-I/O retry accepted | New attempt and remaining requested bytes | Reuse the same outstanding request. |
| Negative CQE/syscall result, or nonzero NVMe status | One failed submitted attempt | Finish request unless existing request logic retries. |
| Zero result for a nonempty ordinary transfer | One failed attempt, no endless retry | Finish request. A zero-length successful SQE is successful instead. |
| Shutdown cancels queued/unaccepted work | No failed attempt | Release request/queue tracking. |
| Shutdown cancels an accepted attempt | Count its actual terminal CQE, including cancellation errors | Retain buffers and tracking until the CQE is reaped. |

An unrecoverable io_uring submit error stops admission under the same queue
mutex used to enqueue requests. Subsequent submission calls fail immediately.
If failure races with an already-started batch, its rejected entries receive
completion errors without incrementing outstanding or submitted-attempt counts;
previously admitted entries remain tracked until worker cleanup completes.
The caller must still consume that batch with `wait_iouring()` before releasing
its buffers. This preserves buffer ownership without waiting for a stopped worker
to process newly queued work. This admission guard does not batch queue publication.

Shutdown fails queued and unaccepted submissions without publishing them again.
For accepted requests it attempts synchronous ring cancellation, then drains their
terminal CQEs before releasing buffers or notifying completion. If synchronous
cancellation is unsupported or fails, it waits for the accepted requests to
complete naturally. Reaping enters the ring with zero submissions, so unaccepted
SQEs cannot be accidentally submitted during cleanup. There is no arbitrary
one-second cutoff for buffer lifetime: a permanently stalled device can delay
shutdown indefinitely rather than allow the kernel to access released memory.

Buffer counters follow the same accepted-attempt boundary, including retries;
they do not count allocations, registration calls, or preparatory copies.

The worker retains unaccepted SQEs in submission order across partial submits
and retryable errors; it does not rebuild and duplicate them in the software
queue. Short I/O is a new attempt. Zero-progress ordinary I/O terminates rather
than retrying indefinitely. Fatal submit errors stop the worker and use the
shutdown protocol above; submit errors are not counted as device completions.
Before sleeping with outstanding requests, the worker enters the ring to
flush overflowed CQEs; otherwise small rings can stall despite completed I/O.

Counter updates add no locks or per-I/O allocations. Snapshot dictionaries are
allocated only by status collection. Buffer metrics establish the Rust buffer
path, not end-to-end zero-copy. Info API fields do not automatically register
OTel instruments. Exporters must remain in Python and must not label metrics
with device paths, keys, batch IDs, PIDs, errno, or error strings.

### Write-local aggregation

Following the write-local/read-aggregate principle described in Apache brpc's
`docs/cn/bvar.md`, observations are separated from synchronization:

- Each device preallocates 32 producer/POSIX shards and one io_uring worker
  shard, each aligned to 128 bytes. This bounds memory at roughly 4.5 KiB per
  device instead of allocating counters whenever a new thread uses a device.
- A thread-local integer selects a producer shard; it holds no device pointer
  or allocation. Colliding threads use relaxed atomic additions, preserving
  counts even with more than 32 writers. This is bounded striping, not a claim
  that all POSIX updates are contention-free.
- The sole io_uring worker holds a non-Clone, non-Sync recorder. Its attempt,
  completion and buffer counts accumulate in private cells and publish with
  relaxed atomic stores, replacing per-I/O atomic read-modify-write operations.
  A repeated worker-recorder claim safely falls back to a producer shard.
- All increments are published at the original event boundaries, before the
  corresponding completion notification. There is no deferred flush threshold
  or thread-exit merge that could omit completed requests from a snapshot.
- Peaks are per-shard maxima, aggregated with max rather than sum. Writers
  first load the local peak and issue an atomic max only if it may increase.
  Queue length remains an exact shared gauge updated under the existing queue
  lock, isolated from the attempt-counter cache lines. Lifecycle atomics and
  their shutdown/batch semantics are unchanged.
- Snapshot collection sums cumulative counters over 33 shards and takes the
  maximum of peaks. This deliberately increases read-side work. Shards remain
  owned by the device after producer threads exit and disappear on device drop.

Shared published fields remain atomic: ordinary Rust cells must never be read
by the polling thread. Snapshots are still best-effort across fields. The
optimization changes neither `rust_io` names nor their counting units. Performance
claims require remeasurement with concurrent writers and snapshot polling; prior
results in `REPORT.md` describe the earlier shared-counter implementation.

Before production rollout, compare release builds with and without counters
on an explicitly designated test device, using repeated, interleaved runs at
representative request sizes, queue depths, and concurrency. Include fixed and
bounce paths and periodic status collection. The RFC target is less than 2%
throughput regression; temporary-file functional tests do not establish it.

## Adapter goals

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
