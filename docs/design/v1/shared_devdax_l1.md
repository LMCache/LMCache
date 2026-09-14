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

## Measured comparison with Maru

The 2026-09-15 (KST) qualification compared LMCache MP at `9ffd4b14` with Maru
at `626e2e6`, using benchmark-only compatibility fixes described below. Later
code cleanup and documentation commits were not rebenchmarked.

Both arms used the same image, full-attention Qwen2.5-7B-Instruct-1M checkpoint,
bf16, vLLM model runner V2, and six TP1 RTX PRO 6000 Blackwell 96 GB GPUs.
Each GPU had a 16 GiB KV budget, with GPU prefix caching disabled. Prefills ran
on host 196 and decodes on host 197: two prefills/four decodes for 2P4D, and four
prefills/two decodes for 4P2D.

The workload used four repeated, nested document prefixes, 256-token chunks,
8,192/32,768 input tokens, and 128 output tokens. Each concurrency level (1, 4,
16) had 16 measured requests after a separate warmup. Every cell visited all
eight P/D pairs. The table shows concurrency 16 only. Throughput is total output
tokens divided by cell wall time. TTFT is median time to first nonempty output.

| Layout | Input | MP output tok/s | Maru + fixes output tok/s | MP gain | MP TTFT ms | Maru TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| 2P4D | 8K | 815.66 | 732.88 | +11.3% | 753.32 | 1139.11 |
| 2P4D | 32K | 400.99 | 359.40 | +11.6% | 2633.66 | 3527.49 |
| 4P2D | 8K | 776.28 | 692.93 | +12.0% | 674.70 | 1135.36 |
| 4P2D | 32K | 364.59 | 312.26 | +16.8% | 2531.81 | 3800.55 |

Gain is `(MP / Maru - 1) * 100`, calculated before rounding. Across all 24 cells,
384 measured requests completed without errors. Every response had 128 output
tokens; output hashes matched across all eight prompt groups. Every decoder's
measured external-cache hit ratio exceeded 99.9%. No worker or service restarted.

Both arms had 24 GiB capacity. MP used one shared pool; Maru reserved six fixed
4 GiB pools with expansion disabled. The workload occupied 7 GiB; Maru also kept
56 MiB from a separate correctness probe. Neither arm was tested under pressure.

**This is not a stock-Maru result.** Stock Maru failed correctness on this stack.
The benchmark corrected its packed host/GPU staging layout and added the same
qualified CXL publish/acquire fences used by MP. These hooks remain outside the
feature patch. MP also waited for store-complete telemetry; Maru's in-process
path did not emit that event, so its proxy did not wait for it.

These short, closed-loop warm-cache runs show an observed throughput advantage,
not statistical significance or steady-state capacity. They test shared-CXL PD
through a common external proxy, not Dynamo routing or NIXL transfer. Warmup,
failed stock-Maru runs, and packed-layout-only failures are excluded. Raw evidence
is the `mp-*-measure*.jsonl` and `maru-*-fenced-measure*.jsonl` qualification
artifacts; the benchmark harness and logs are not bundled in this minimal patch.

See the [setup and operations guide](../../source/mp/shared_l1.rst).
