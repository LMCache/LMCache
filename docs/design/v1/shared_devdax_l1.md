# Shared Device-DAX L1

Status: experimental. Scope: TP1, one fixed pool, no reclamation.

## Problem and scope

Private L1 allocators cannot safely allocate into the same physical memory.
Each could hand out an offset already in use by another MP server. This design
gives one Memory Coordinator ownership of keys and offsets. Each MP maps the
same physical pool through its host-local Device-DAX device.

The existing MP Coordinator remains the fleet directory and management service.
Its event-driven view is not allocation authority. It is optional for shared
reads; every read consults the Memory Coordinator directly.

```text
vLLM A --CUDA IPC--> MP A ----metadata----> Memory Coordinator
                      |                   (keys, offsets, write tokens)
                      |                         ^
                shared Device-DAX               |
                      |                         |
vLLM B <--CUDA IPC-- MP B ----metadata-----------+

KV bytes: GPU A -> shared pool -> GPU B
```

This is not Dynamo routing, NIXL payload transfer, or an HA memory service.
There is no change to the default private-L1 path unless shared L1 is enabled.

## Fixed contract

The coordinator publishes `region_id`, `capacity_bytes`, `alignment_bytes`,
`layout_id`, and a fresh `region_epoch`. MPs must match the first four values
and keep the initial epoch. An epoch change permanently fences the client.
Clients must not silently reconnect to a different pool incarnation.

The operator supplies the physical-region identity and layout fingerprint.
They are assertions, not automatic hardware discovery. The layout fingerprint
must identify compatible model weights, tensor layout, dtype, TP, and chunk size.
All offsets are relative to this pool. A local address is:

```text
local mmap base + handle.offset
```

The host-local mapping offset is used only to map the correct physical window.
Virtual addresses never cross hosts. A handle carries region, offset, payload
length, and generation. MPs check the key, tensor layout, alignment, and bounds
before exposing a view. The consumer also checks its registered engine layout.

## Write and read protocol

```text
ABSENT --reserve--> WRITING --D2H complete, publish, finish--> VALID
                       |
                       +--abort--> ABSENT (allocated bytes stay consumed)
```

1. Reserve an entire batch. One writer wins each absent key. Existing or
   in-flight keys return no grant. If the absent keys do not all fit, nothing
   is allocated.
2. The producer copies GPU KV into its CUDA-registered DAX view. It waits for
   D2H completion, then publishes each exact range through the visibility ABI.
3. Only then does it finish the batch with the granted write tokens. The
   coordinator validates the entire batch before making it readable.
4. A reader looks up committed keys, validates the returned descriptors, acquires
   visibility for each range, and copies those bytes to its GPU.
5. Local read references remain held until GPU completion. They prevent local
   unmapping; they are not distributed read pins.

Canonical object keys include the chunk hash, model, KV rank, object group, and
cache salt. The coordinator stores immutable, validated keys directly in its
index. A pending write has a token; a committed object has no write token.

## Why there is no free operation

The allocator only advances. A committed extent is never overwritten or reused.
Aborted writes discard metadata but still consume space. This avoids distributed
read pins, lease expiry, and stale readers accessing reused bytes in this first
implementation. It also means a crashed writer can strand a key and consume space.

Pool pressure requires a coordinated reset, not eviction. `used_bytes` is the
allocation high-water mark, including alignment gaps and abandoned extents.
`object_count` includes both pending and committed records. Neither value is a
full key listing. Shared capacity events prevent each mounting MP from reporting
the same pool as private capacity.

This version supports one shared pool per deployment. Its capacity accounting
does not distinguish multiple independent shared Device-DAX pools.

## Visibility is a platform contract

`MAP_SHARED`, matching device labels, and `msync()` are not cross-host coherence
proof. The operator must qualify a library for the actual fabric. Its ABI is:

```c
#include <stddef.h>
#include <stdint.h>

int lmcache_shared_l1_visibility_v1(
    const char *mode, uint32_t operation, int device_fd,
    void *mapped_address, uint64_t device_offset,
    size_t length, uint64_t generation);
```

The mode is `software_fenced`. Operation `1` publishes completed writes; operation
`2` acquires visibility before reading. The call is synchronous and covers the
exact payload range. It returns zero on success and a nonzero status on failure
(negative errno values give readable errors). The FD is open, the address is
local, and the device offset includes the host-local mapping offset.

The current ABI uses 64-byte visibility granularity. Allocation alignment must
be compatible with it. The library must round a partial final cache line safely
within the isolated allocation. Missing libraries, symbols, registration, or
visibility success are fatal; there is no pageable or no-op fallback. Publication
establishes visibility, not durability or recovery.

## Failures and shutdown

POST transport failures and unexpected server errors can leave an ambiguous
write outcome. The client fences itself instead of retrying an allocation or
assuming that commit failed. A stale-epoch response also fences it. Reserved
bytes are never reused automatically, including during cleanup failures.

An epoch cannot revoke a GPU's old mapping. Therefore, startup atomically creates
a persistent marker with exclusive creation and fsyncs it and its parent. Every
replacement coordinator must use the same marker on storage with reliable
exclusive-create semantics. Even a clean shutdown leaves it in place. It contains
no recoverable metadata and is not a distributed leader-election protocol.

For shutdown, stop admitting requests and drain all model work first. Keep
worker reaping disabled. Drain GPU contexts and completion callbacks before
unregistering host memory and unmapping DAX. Closing with exported views or a
failed CUDA unregistration is refused. Automatic worker-failure recovery and
shutdown racing with admitted handlers are not qualified.

For reset, stop every model worker and MP, stop the coordinator, and verify that
no old mapping or GPU access remains. Only then may the operator remove the
specific startup marker and start a new empty pool. Removing it while workers
are alive defeats the safety barrier. A new epoch does not make that safe.

## Supported combinations and validation

Use `lmcache_driven`, TP1, a fixed Device-DAX mapping, and `noop` eviction.
Reject L2 adapters, hybrid DRAM/DAX, GDS L1, MP P2P, CacheBlend, QStore,
engine-driven transfer, and trace replay. The feature does not add automatic
discovery, authentication between tenants, persistence, HA, or cache reclamation.
Bearer HTTP requires a trusted network or an authenticated TLS proxy.

Tests cover atomic reservation, duplicate writers, tokens and epochs, startup
markers, wire validation, exact visibility ranges, shared accounting, GPU
completion order, and safe teardown. CPU file-backed tests do not qualify a CXL
fabric. Hardware qualification must prove cold-versus-warm output equality and
external-cache hits on every consuming worker, not just successful HTTP replies.

See the [setup and operations guide](../../source/mp/shared_l1.rst).
