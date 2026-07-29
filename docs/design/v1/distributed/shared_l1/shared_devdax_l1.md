# Shared Device-DAX L1 pool

Status: experimental M0 with coordinator-subprocess integration planned

This document describes the in-tree implementation under
`lmcache/v1/distributed/shared_l1/` and the narrow integration boundary for
hosting its allocation state in the existing MP coordinator.

The full design discussion and upstream issue draft are intentionally kept
outside the LMCache source tree.

## Purpose

Two MP servers may map different Device-DAX paths that refer to the same
physical CXL memory. They must not create independent allocators over those
bytes: process-local locks cannot prevent both hosts from allocating the same
offset.

The design uses:

- one coordinator-owned child process as the shared-pool allocation and
  lifetime authority;
- one host-local mapping in each MP server for GPU transfers;
- address-independent object handles;
- Device-DAX for payload bytes only.

There is no additional shared-L1 service, pod, Kubernetes Service, or public
backend class.

## Current source layout

```text
lmcache/v1/distributed/shared_l1/
    __init__.py
    pool.py

tests/v1/distributed/shared_l1/
    test_pool.py

tools/
    shared_l1_pool_smoke.py
```

`pool.py` currently provides:

- `InMemorySharedL1Pool`: lock-protected allocation and object-lifetime state;
- `SharedMemoryRegion`: a host-local mmap plus offset-backed views;
- `SharedRegionContract`: region identity, capacity, alignment, layout ID, and
  generation epoch;
- `SharedObjectHandle`: `{region_id, offset, length, generation}`;
- opaque write and read reservation tokens.

M0 is an executable reference implementation. It is not yet connected to the
MP coordinator or the existing `L1Manager`.

## Coordinator process model

The existing MP coordinator owns the shared-pool feature and supervises one
child process:

```text
+-------------------------------------------------------------+
| existing MP coordinator pod                                 |
|                                                             |
| parent process                                              |
| - existing HTTP API and CoordinatorContext                  |
| - registry, management, and eventual KeyDirectory           |
| - /shared_l1/* routes                                       |
|             |                                               |
|             | bounded local request/reply IPC               |
|             v                                               |
| shared-L1 pool child                                        |
| - free extents                                              |
| - key -> object record                                      |
| - generations                                               |
| - write reservations and read leases                        |
+-------------------------------------------------------------+
          ^                                      ^
          | existing coordinator endpoint        |
          |                                      |
     MP server A                            MP server B
     local DAX mmap                         local DAX mmap
     GPU transfers                          GPU transfers
          \                                      /
           +--------- same physical bytes ------+
```

MP servers call only the existing coordinator endpoint. They never connect
directly to the child. The parent forwards metadata-only requests over local
IPC and reports child readiness through the existing coordinator health
surface.

For the experimental integration:

- one coordinator replica uses `Recreate`, not rolling-update, semantics;
- one HTTP worker spawns exactly one pool child;
- the child starts before shared-L1 readiness becomes true;
- a child exit makes shared-L1 fail closed;
- a replacement child cannot start allocating from offset zero without the
  reset and client-fencing procedure.

The exact bounded IPC mechanism is an implementation choice. It must support
request correlation, backpressure, child-death detection, and deterministic
shutdown without changing the newly defined `/shared_l1/*` coordinator API.

## Ownership

### Coordinator parent

The parent owns:

- the public `/shared_l1/*` API;
- child lifecycle and readiness;
- request validation and bounded forwarding;
- export of child metrics;
- publication of shared-pool events to the eventual `KeyDirectory`.

The `KeyDirectory` is a routing and inspection hint. It never grants access to
bytes or decides that an extent is safe to reuse.

### Pool child

The child is the only process allowed to mutate:

- allocation/free state;
- object state;
- extent generation;
- write reservation state;
- active read leases;
- reclamation state.

Its key index is strong state for one shared region. It is not a second
fleet-wide routing directory.

### MP server

Each MP server owns:

- its host-local Device-DAX path, file descriptor, mmap, and CUDA registration;
- conversion of a handle offset into a bounds-checked local view;
- D2H/H2D scheduling and completion;
- platform publish/acquire visibility operations;
- local cleanup after coordinator operations finish.

No virtual address, CUDA pointer, payload byte, or RDMA descriptor enters the
coordinator protocol.

## Region and handle contract

M0 exposes `SharedRegionContract` with `capacity`, `alignment`, `layout_id`,
and `generation_epoch`. The integrated M1 wire contract expands and renames
those fields as follows:

```text
RegionContract {
    region_id
    region_epoch
    capacity_bytes
    alignment_bytes
    layout_profile_fingerprint
    visibility_mode
}
```

Host-local paths may differ. `region_id` is an operator-provisioned identity
for the same physical bytes; it is not inferred from a path, inode, or virtual
address.

Every object location is:

```text
SharedObjectHandle {
    region_id
    offset
    length
    generation
}
```

`offset` is relative to each local mapping. A reused extent receives a new
generation. A client validates region epoch, generation, bounds, alignment,
and the registered layout before constructing a view.

The first integrated version uses one fixed model/layout profile per region.
The profile contains the model/config digest, chunk size, world size, and an
ordered object-group map of `EngineKVFormat`, shapes, dtypes, and memory
format. A remote reader selects the registered layout by
`ObjectKey.object_group_id`.

## Object lifecycle

The target shared-pool lifecycle is:

```text
FREE -> WRITING -> VALID -> EVICTING -> FREE
```

- `reserve_write` selects an aligned free extent and returns an opaque token.
- Only the token owner may publish or abort the write.
- `finish_write` follows D2H completion and the platform publish operation.
- `reserve_read` returns a handle only for `VALID`.
- A read lease remains active until every dependent H2D operation completes.
- Eviction first enters `EVICTING`, which rejects new readers.
- Reclamation waits for all leases and fenced clients before returning the
  extent to `FREE`.
- Reuse assigns a new generation.

M0 implements exclusive writes, `VALID`-only reads, read pins, and pin-aware
metadata deletion. Its allocator is monotonic: it deliberately does not reuse
aborted or deleted extents. Free-list reuse and `EVICTING` are M1 work.

Lease TTL detects abandoned work but does not prove a remote GPU DMA stopped.
An uncleanly expired operation remains quarantined until its client is fenced.

## Coordinator operations

The coordinator API keeps the reserve/finish/abort semantics and batches the
existing `L1Manager` key list. The integrated operation set includes:

```text
register_client
reserve_write
finish_write
abort_write
reserve_read
finish_read
abort_read
finish_write_and_reserve_read
request_evict
renew
describe_region
usage
close_client
```

One request covers a batch of `ObjectKey` values. One lease covers one
group-specific KV chunk. Internal layer or tensor copies do not create more
coordinator calls; layerwise DMA may pipeline under that lease.

Duplicate writers produce one winner. A committed duplicate returns
`EXISTS_VALID` without allocating a second resident copy. That result does not
authorize a read; the caller still obtains a read lease.

## `L1Manager` integration

Native shared Device-DAX uses a thin path in the existing `L1Manager`, not a
new sibling manager or placement-backend class:

```text
L1Manager
  -> shared_l1.client: batched calls to existing coordinator endpoint
  -> shared_l1.region: handle offset to local TensorMemoryObj view
```

The path bypasses the process-local `DevDaxMemoryAllocator`. When shared mode
is disabled, existing local DRAM, private Device-DAX, and GDS behavior remains
unchanged.

`free()` of a local view releases only process references. Physical reclaim is
completed by the pool child. Delayed reclaim therefore needs a completion
notification from the child through the parent before LMCache publishes
`L1_KEYS_EVICTED`.

Store success must also wait for the strong `finish_write` result. A failed or
timed-out pool commit cannot increment stored count or publish a directory
`STORE` event.

## Events and inspection

The child creates ordered transition notifications after strong state changes.
The parent publishes them through the coordinator event path.

Shared placement identity is `(region_id, region_epoch)`, not the MP server
that first wrote the object. Writer restart must not remove a still-valid
shared placement.

Inspection and metrics are read-only. At minimum they report:

- bytes and objects by state;
- free bytes, largest free extent, fragmentation, and OOM;
- active/expired/quarantined leases;
- duplicate writers suppressed;
- eviction wait and reclaim completion;
- stale epoch, generation, token, and incarnation errors;
- operation latency by batch size and result;
- direct-pinned versus pageable-staging transfer bytes.

## Restart behavior

M0 stores metadata in memory. SQLite is not placed on the reserve/read/release
hot path.

If the coordinator parent or child restarts:

1. shared-L1 enters `RESET_REQUIRED`;
2. a new region epoch fences old handles;
3. the prior parent/child and every old MP client are stopped or fenced;
4. mappings and GPU operations are quiesced;
5. the pool is logically reset;
6. clients register the new contract;
7. only then may the new child enter `ACTIVE`.

The eventual `KeyDirectory` cannot reconstruct allocator state.

## Transfer classification

A CPU mmap view is not a no-copy GPU transfer. Report one of:

- `direct-pinned`: CUDA host registration succeeded and H2D/D2H directly uses
  the registered DAX mapping;
- `pageable-staging`: an intermediate staging path was used;
- `cpu-mapped-only`: only CPU direct access was qualified;
- `unsupported`: visibility or transfer safety could not be established.

`MAP_SHARED`, `mmap.flush()`, and `msync()` do not prove cross-host visibility.
Two-host enablement requires a qualified publish/acquire mechanism.

## Verification

The in-tree gates are:

- unit tests for region contracts, reservations, stale handles, and read pins;
- two processes mapping one regular file at independent virtual addresses;
- duplicate-writer and concurrent non-overlapping-allocation tests;
- mutation and restore verification in `shared_l1_pool_smoke.py`;
- import and CLI checks using the renamed `pool.py` module.

Before enabling real two-host Device-DAX:

- verify both paths refer to the same reserved physical region;
- test bidirectional visibility and checksum correctness;
- record CUDA registration and fallback classification;
- inject parent, child, writer, and reader process failures;
- test concurrency 1, 2, 4, 8, and 16;
- complete a 30-minute mixed allocate/read/evict run with no overlap,
  corruption, stale-generation acceptance, premature reclaim, or leaked clean
  leases.

## Current limits

- No MP-coordinator subprocess wiring yet.
- No `ObjectKey` or registered `MemoryLayoutDesc` in M0; keys are strings.
- No batched/idempotent coordinator wire contract.
- No TTL sweeper, free-list reuse, or automatic failed-client fencing.
- No `TensorMemoryObj` integration.
- No real two-host visibility or GPU-direct qualification.
- P2P, GDS, hybrid DRAM+DAX, and runtime shared-pool hotplug are unsupported.

Related architecture:

- [MP coordinator control-plane RFC](https://github.com/LMCache/LMCache/issues/4226)
- [Fleet-wide eventual key directory](https://github.com/LMCache/LMCache/pull/4275)
- [L1 backend extension discussion](https://github.com/LMCache/LMCache/issues/3654)
- [Runtime private Device-DAX arenas](https://github.com/LMCache/LMCache/pull/3972)
- [Maru shared CXL L1](https://github.com/LMCache/LMCache/pull/4052)
