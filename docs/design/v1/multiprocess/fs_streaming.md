# Bounded FS streaming for MP retrieval

## Motivation and scope

When the DRAM L1 is full, prefetching a cold FS prefix through L1 competes
with resident objects. An opt-in FS streaming reader uses a bounded share of
that same DRAM budget to move KV objects to the device without admitting the
FS-read objects to L1. The remaining L1 uses its configured eviction policy.
This path applies when the sole active L2 adapter is an unwrapped FS adapter.
When multiple L2 adapters are active, lookup uses normal prefetch across them.
Reserving streaming capacity reduces resident L1 capacity in either case.

## Lookup and ownership

For every requested object, the mixed planner first reserves available L1
objects and looks up missing keys on FS. It folds the combined hit bitmap
across KV ranks and attention windows to find the longest usable prefix.
L1 locks beyond that prefix are released. If the retained prefix contains no
FS object, all temporary locks are released and the existing prefetch path
runs. On a lookup timeout the locks are released and the lookup reports a
miss. A read-locked unfull L1 chunk can extend the prefix as described in
[unfull_chunk.md](unfull_chunk.md).

The session records the streaming range and the owners of its L1 locks.
Retrieve reads resident objects from their owners and FS objects through the
shared streaming window pool. It slices the staged block IDs for each run of
L1 objects or FS windows, so the device sees the same object order as the
lookup prefix. Tokens after the hit are computed by the inference engine.

## Buffer lifecycle

The FS reader owns a pinned allocator and an admission-controlled window
pool. Each window holds at most one object; batches are limited by the pool's
capacity. A completed filesystem read leaves its window reserved through the
GPU transfer. A reaper polls the device completion fence and releases the
window for reuse without synchronizing the transfer stream on the caller.
If an I/O times out while still in flight, a reaper retains the reservation
until that I/O completes. Reader shutdown waits for outstanding I/O and active
windows up to its deadline and does not free memory still in use.

L1 object locks are independent of window reservations. Retrieve submits their
release after the device work, including locks for a group whose transfer fails.
If an FS object disappears after lookup, retrieve reports failure so the
connector's configured failure policy can handle the missing KV.

## Configuration and tradeoffs

`streaming_buffer_size_bytes` is disabled by default. When enabled it must be
4096-byte aligned, smaller than the selected DRAM L1 budget, and large enough
for each object layout. The storage manager's resident L1 capacity is reduced
by the configured pool size before either allocator is constructed. The
allocators are distinct; their combined configured capacity is the original
L1 budget. Objects larger than a window use normal prefetch. The pool does not
pin resident hot objects or suppress ordinary L1 evictions and stores.

Validation covers mixed L1/FS lookup, partial prefixes, lock and timeout
lifecycle, unfull L1 tails, bounded window reuse, and GPU byte equality for
concurrent readers. User-facing configuration is documented in
[the FS adapter reference](../../../source/mp/l2_storage/fs.rst).
