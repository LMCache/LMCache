# Device-DAX L1 Memory Manager Design

This document describes the Device-DAX-backed L1 tier and its runtime
reconfiguration: adding and removing Device-DAX devices by path while the
process keeps running.

## Goals

- Back L1 with one or more Device-DAX devices, optionally fronted by a DRAM
  tier (hybrid DRAM + Device-DAX).
- Add and remove Device-DAX capacity at runtime by device path, without a
  restart.
- Keep the L1 allocate / free / usage / descriptor interface unchanged for
  callers (`L1Manager`).
- Never move or invalidate a live allocation. L1 hands out raw device pointers
  that requests read and write directly, so an arena with live allocations must
  stay mapped.

## Components

`lmcache/v1/distributed/memory_manager/devdax_l1_memory_manager.py` defines
`DevDaxL1MemoryManager`, the L1 tier object. It is thin: it stores its config,
delegates `allocate` / `free` / `get_memory_usage` / `get_l1_memory_desc` to the
pooled allocator, and exposes the runtime-reconfigure surface
(`add_device`, `remove_device`, `get_arena_statuses`). It also owns the
reconfiguration error boundary, keeping HTTP concerns out of the allocator.

`lmcache/v1/memory_allocators/devdax_memory_allocator.py` defines
`DevDaxMemoryAllocator`, which owns the arena pool, the `host_mem_lock`, the
per-arena `TensorMemoryAllocator` instances, the mmap lifecycle, and
best-effort CUDA host-memory pinning. `_DevDaxArena` is the per-arena record.
`DevDaxArenaState`, `DevDaxRemoveMode`, and `DevDaxArenaStatus` are the public
reconfigure types.

`lmcache/v1/distributed/l1_manager.py` selects `DevDaxL1MemoryManager` when
`memory_config.devdax_path` is set and GDS L1 is not configured. It exposes
Device-DAX status, add, and remove as narrow delegation methods.
`lmcache/v1/distributed/storage_manager.py` provides the HTTP-facing delegates
and publishes capacity after add and drain transitions.

The runtime-reconfigure HTTP surface lives in
`lmcache/v1/multiprocess/http_apis/l1_reconfigure_api.py` as backend-first,
tier-scoped routes: `GET /reconfigure/dax/l1/status` and
`POST /reconfigure/dax/l1/{add,remove}`.
Handlers stay thin -- request-shape validation (422 schema / 400 size values)
happens at the HTTP layer, domain status-code decisions (404 lookup miss, 409
state conflict or non-Device-DAX L1) come from the manager via
`L1ReconfigureError`, the HTTP resolver answers 503 before engine or storage
manager initialization, and arena mechanics stay in the allocator. The HTTP
layer fixes `dax` as the backend and `l1` as the tier segment, keeping it
disjoint from the parametric L2 family
(`/reconfigure/{backend}/l2/*`).
Additional L1 backends are not exposed by this API and require corresponding
routing and delegation support. A URL that omits the tier segment, such as
`/reconfigure/dax/status`, returns `404`. See
[../l2_adapters/dax.md](../l2_adapters/dax.md) for the L2 counterpart.

## Arena Pool

The pool is an optional DRAM local allocator plus an ordered list of Device-DAX
arenas. Each arena owns its own file descriptor, `mmap` (`MAP_SHARED`), flat
`torch.uint8` view of the mapped bytes, and a `TensorMemoryAllocator` with an
arena-local address space, plus a best-effort CUDA host-memory pin and a
lifecycle state.

Allocation order:

1. DRAM local allocator first, if present (hybrid mode).
2. Then each `active` arena in pool order as overflow.

Single-object allocation pre-checks each arena's free capacity and skips arenas
without room instead of probing them with a failed allocation attempt; a failed
probe logs a warning inside the arena allocator, which would emit one line per
full arena on every allocation once earlier arenas fill up.

Batched allocation fills greedily across active arenas and is all-or-nothing: if
the arenas cannot collectively satisfy the request, the partial allocation is
rolled back and the call fails, matching the single-allocator contract.

Free routing: every Device-DAX allocation carries the `DevDaxMemoryAllocator` as
its parent. The owning arena is located by pointer range
(`base_ptr <= data_ptr < base_ptr + size`), because each arena has an
arena-local address space and a freed object must return to the exact address
manager it came from. Batched frees are grouped by owning arena. `base_ptr` is
captured when the arena is mapped, so routing stays correct even for an arena
whose deferred unmap has already dropped its buffer references. After freeing,
the pool re-attempts the reap of every `draining` arena, so an unmap deferred
by lingering views completes on a later free.

## Primary Arena

The primary arena backs `get_l1_memory_desc()` and the `buffer` property and can
never be removed.

- Pure Device-DAX mode (no DRAM tier): the initial arena is primary.
- Hybrid mode (DRAM tier present): DRAM is the primary L1 region, so every
  Device-DAX arena, including the initial one, is removable overflow.

`is_primary` is therefore true only for the initial arena when there is no DRAM
local allocator.

## Runtime Reconfigure

The manager and allocator expose the same return values. The manager translates
request-validation and pre-transition state failures into
`L1ReconfigureError` for the HTTP layer:

- `add_device(device_path, size_in_bytes) -> DevDaxArenaStatus`
- `remove_device(device_path, mode=DevDaxRemoveMode.DRAIN) -> DevDaxArenaStatus`
- `arena_statuses()` on the allocator / `get_arena_statuses()` on the manager,
  returning `list[DevDaxArenaStatus]` in pool order.

Add:

1. Canonicalize the path and reject if the allocator is closed or the current
   node identity matches an already-mapped arena.
2. Map the device -- the mapping attempt itself validates the request
   (non-empty path, positive size, Device-DAX type, and the device's advertised
   sysfs alignment) and acquires the resources: a single `open(O_RDWR)` whose
   fd backs the identity and capacity checks and the `mmap(MAP_SHARED, RW)`.
   The opened identity is checked again before the arena is built; then build a
   `TensorMemoryAllocator` and best-effort pin it.
3. Append the arena as `active` and non-primary. It is immediately available as
   overflow. Existing allocations are untouched.
4. The `StorageManager` entry point publishes the current whole capacity
   topology.

If any setup step fails (mapping, allocator construction, or pin registration),
the freshly opened fd and mmap are released before the error propagates.

Remove (drain):

1. Reject if no arena is mapped at the path, or the arena is primary.
2. Mark the arena `draining`; it is excluded from new allocations.
3. If it has no live allocations, unmap it immediately (`removed`). Otherwise it
   stays `draining`, and the `free` that releases its last allocation unmaps it
   automatically (auto-reap). If the unmap is blocked by lingering external
   views into the mapping (e.g. freed tensors awaiting garbage collection), the
   arena stays `draining` and later frees retry the reap.
4. After drain begins, the `StorageManager` entry point publishes the
   post-transition capacity topology even if synchronization or cleanup later
   fails; a later remove retries cleanup while the path remains mapped.

State machine: `active -> draining -> removed`. `removed` is a report-only
terminal value; a removed arena has already left the pool, so it is never
observed by an in-pool lookup.

Modes that relocate live objects (migrate, evict) are intentionally not
supported here; see Current Limits.

## How to Reconfigure at Runtime

Configure the initial Device-DAX device when the server starts, either with the
MP server CLI flag `--l1-devdax-path /dev/dax0.0` (the mapped size follows the
L1 size settings) or programmatically. The Device-DAX namespace must already
exist and expose the requested capacity; namespace provisioning remains the
operator's responsibility.

```python
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager

l1 = L1Manager(
    L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=32 << 30,
            use_lazy=False,
            shm_name="",
            devdax_path="/dev/dax0.0",
        )
    )
)
```

Reconfiguration is available over HTTP (`/reconfigure/dax/l1/*`).
`StorageManager` is the system entry point that publishes capacity changes;
the `L1Manager` methods below are lower-level delegation methods:

```python
# Grow: map an already-provisioned device and add it to the pool. It serves
# overflow allocations immediately.
status = l1.add_devdax_device("/dev/dax1.0", 32 << 30)

# Inspect per-device usage: used/free bytes, live allocations, state.
for arena in l1.get_devdax_arena_statuses():
    print(arena.device_path, arena.state, arena.used_bytes, arena.free_bytes)

# Shrink: remove a device (currently drain mode only). REMOVED means it was
# empty and is already unmapped; DRAINING means cached entries still live on it.
status = l1.remove_devdax_device("/dev/dax1.0")
```

A `DRAINING` device accepts no new allocations, keeps serving reads for the KV
entries already on it, and unmaps automatically once the last of them is freed
(deleted or evicted). Poll `get_devdax_arena_statuses()` until the path
disappears from the list; `active_allocations` on the draining entry shows how
many allocations still gate the unmap. Calling `remove_devdax_device` again
while the path is still draining is safe and returns the current status; once
the arena has been unmapped the path is no longer known and a repeat raises the
404-mapped lookup error.

`add_device` rejects devices that are already mapped under another spelling,
and `remove_device` rejects the primary arena (the initial device in pure
Device-DAX mode). Lookups accept another path to the same device and statuses
report the canonical path recorded when the arena was mapped. At the allocator
level these raise `ValueError`; the manager translates them into
`L1ReconfigureError` (409, or 404 when the path is not mapped at all). The
Device-DAX path must already exist, be readable and writable, and expose enough
capacity. Runtime reconfigure does not provision or resize DAX namespaces (see
Current Limits).

## Thread Safety

`host_mem_lock` (non-reentrant) serializes every pool mutation and every
per-arena allocate, free, add, remove, and reap. The DRAM local allocator has
its own synchronization and is used outside this lock. Rollback of a failed
batched allocation and arena reaping both run while the lock is held; the
internal helpers assume the lock is held and never re-acquire it.

## mmap and CUDA Host Memory

Each arena maps its device with `mmap(MAP_SHARED, PROT_READ | PROT_WRITE)` and
exposes the bytes as a flat `torch.uint8` tensor via a ctypes array
(`from_buffer`). On unmap every reference into the mmap (the allocator buffer,
the arena buffer, and the ctypes array) is released before `mmap.close()`,
because CPython refuses to close a buffer that still has exported pointers.
`mmap` dups the underlying file descriptor, so unmap releases both the opened fd
and the mmap's dup.

## Device Identity

The allocator identifies what it opened rather than comparing path strings.
It records `st_rdev` for a character device and `(st_dev, st_ino)` for a
regular file used as test backing; other file types are rejected. Device-DAX
nodes that expose the same `major:minor` therefore cannot be mapped twice.
Paths are canonicalized for stable status responses, while add, remove, and
status lookup use the device identity.

Device-DAX character devices normally report `st_size == 0`, so their type,
capacity, and alignment are verified through sysfs by device number rather
than by path basename. `/sys/dev/char/<major>:<minor>` is tried first; if it is
not visible, `/sys/bus/dax/devices` is searched for a matching `dev` attribute.
The entry must be on the `dax` subsystem and expose readable, positive `size`
and `align` values. Verification fails closed when sysfs does not expose enough
information. Regular mmap test files use `fstat` and never consult sysfs.

CUDA host-memory registration (pinning) is per-arena and best-effort; a pin
failure is logged and the arena falls back to pageable host copies.

## Transfer-Channel Compatibility

P2P does not support Device-DAX L1 because
`l1_exposes_single_memory_region()` returns `False`. NIXL-based adapters and
Mooncake over RDMA require a single L1 memory region, so they cannot be used
with hybrid or multi-arena Device-DAX L1. Other adapters allow arenas to be
added and removed normally.

## Capacity

`get_memory_usage()` returns used and total bytes computed under
`host_mem_lock`. Used bytes sum the live allocations of the DRAM local
allocator and every arena (active and draining). Total bytes sum the DRAM
allocator and only the *active* arenas: a draining arena accepts no new
allocations, so its free space is not usable headroom and its capacity is
excluded. A draining arena still holding live bytes therefore pushes used
above total (ratio > 1), which is intentional -- it keeps the eviction
watermark tracking real pressure on the active pool instead of being diluted
by capacity that is being removed.

`L1Manager.get_capacity_bytes_by_backend()` is also used by `report_status()`
and `StorageManager` capacity snapshots. CPU, GDS, and the DRAM half of a hybrid
tier retain their boot-configured values; the Device-DAX entry is the sum of
active arena sizes. Capacity-changing calls through `StorageManager` publish
`SM_CAPACITY_CHANGED` with the whole topology. Delivery is asynchronous and
best-effort, so operation success confirms the local topology change rather
than coordinator receipt.

## Verification

`tests/v1/distributed/test_devdax_l1_allocator.py` unit-tests the pool:
add/remove lifecycle, drain gating, per-arena usage, deferred unmap while
external views are alive, mapping release on setup failure, and the
`StorageManager` reconfiguration delegates.
`tests/v1/distributed/test_memory_capacity.py` verifies capacity reporting and
whole-topology `SM_CAPACITY_CHANGED` publication after successful add/remove
and after a drain transition whose cleanup fails.
`tests/v1/distributed/test_devdax_l1_reconfigure_integration.py` (opt-in via
`RUN_DEVDAX_L1_INTEGRATION=1`) drives real mmap-backed devices end to end,
at the memory-manager level, through the `L1Manager` KV-cache path, and through
the `/reconfigure/dax/l1/*` HTTP lifecycle. KV entries land on a runtime-added
device, stay readable while it drains, and the device is unmapped only after
the last cached entry is deleted. It accepts real `/dev/dax` devices via
`LMCACHE_TEST_DEVDAX_L1_PATHS`.

## Current Limits

- Only drain-based removal. Migrate and evict (relocating live objects to
  another arena or to DRAM) are deferred: L1 hands out raw device pointers that
  live requests read and write, so relocation requires hooking L1 eviction.
- The primary arena in pure Device-DAX mode cannot be removed at runtime.
- Existing arenas cannot be resized; the pool grows and shrinks by whole arenas
  (`add_device` / `remove_device`).
- Runtime reconfigure maps and unmaps already-provisioned Device-DAX devices;
  it does not perform kernel-level CXL/DAX namespace reconfiguration.
