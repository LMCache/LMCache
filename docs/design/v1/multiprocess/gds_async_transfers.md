# Concurrent GDS async transfers

The opt-in LMCache-driven CUDA path overlaps existing gather/scatter kernels
and cuFile asynchronous I/O across up to four independently owned staging slots.
It retains `cuFileReadAsync`/`cuFileWriteAsync`, the backend interface, KV object
layout, and the store requirement of one object per gather kernel.
`io_concurrency=0` retains the original single-stream path.

## Bounds and ownership

`GDSContext.io_scope()` serializes optimized object-group scheduling
for the slab. Only one GDS slab is supported per process. This keeps admission
bounded across cache contexts without adding CPU worker threads: the scheduler
holds at most `io_concurrency` objects and `max_inflight_bytes` logical bytes.
Different cache contexts take turns using that budget; their object groups do
not overlap. This is a deliberate limit of this first optimization.

The CUDA cache context reuses its four staging slices and adds one persistent,
separately registered cuFile stream per active slot. Buffer registration is not
repeated per transfer. Registration regions end at slot boundaries, so concurrent
slots never share a native registration. For example, four 18 MiB slots use eight
regions of 16 MiB and 2 MiB, retaining the same 72 MiB allocation. Actual staging
remains four times the sum of all
object-group sizes and must fit the configured byte budget per cache context.
The cache-context lease prevents teardown from releasing engine mappings or
staging while the pipeline is active.

## Ordering and completion

An input event is recorded after producer dependencies and block-ID staging on
the original stream. Each slot stream waits on this recorded event. Its next
generation cannot be recorded until the previous pipeline has drained.

```
retrieve: read_async -> completed read -> exact byte check -> scatter -> complete
store:    gather -> write_async -> completed write -> exact byte check
```

The scheduler fills available slots and drains the oldest slot before reusing
it. A read is checked before scatter; all accepted work completes before the
normal engine IPC event is recorded and exported. These scoped waits run in the
existing blocking affinity handler, leaving the MP transport event loop free.
No CUDA host callback invokes CUDA APIs, and no per-transfer device-wide
synchronization is added.

Original object indices still determine block-ID slicing, sliding-window skips,
and protected prefixes. Missing host block IDs or aliased retrieve destinations
use the ordered fallback. Every object resolves its slab by L1 owner.

## Failure and teardown

Both paths validate deferred results: each registered-region segment must
complete its exact requested byte count. Negative and short results fail before
successful publication. A submission failure drains work already accepted; a
slot remains owned through scatter completion even when another slot fails.
Existing store/retrieve handlers retain their failure and owner-release behavior.

Context close waits for the pipeline, drains and deregisters slot streams, and
then releases registered buffers and engine mappings. Slab close also waits for
its scheduling scope. Stream flags remain `0x7`, lengths and offsets are
unchanged, and completion adds no `fsync` requirement. Native direct DMA depends
on the runtime and storage configuration; async concurrency does not establish
that cuFile avoids its compatibility path.
