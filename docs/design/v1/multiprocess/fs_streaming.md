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

The existing prefetch controller locks available L1 objects, looks up missing
objects on FS, and plans the longest usable prefix across ranks and attention
windows. For streaming requests it returns this plan without reserving L1 write
buffers or loading FS objects. The lookup module records the planned FS ranges
and retained L1 owners. A read-locked unfull L1 chunk can extend the prefix as
described in [unfull_chunk.md](unfull_chunk.md).

The session records the streaming range and the owners of its L1 locks.
Retrieve reads resident objects from their owners and FS objects through the
shared streaming window pool. It slices the staged block IDs for each run of
L1 objects or FS windows, so the device sees the same object order as the
lookup prefix. Tokens after the hit are computed by the inference engine.

## Buffer lifecycle

The DRAM L1 manager permanently reserves a portion of its existing arena for
the FS reader. The reader suballocates admission-controlled windows within that
reservation. Each window holds at most one object; batches are limited by the
pool's capacity. A completed filesystem read leaves its window reserved through the
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
for each object layout. The reservation is allocated from the storage
manager's existing DRAM L1 arena at startup. Resident objects cannot use these bytes, and the reported
hot-L1 capacity excludes them. Objects larger than a window use normal
prefetch. The pool does not pin resident hot objects or suppress ordinary L1
evictions and stores.

User-facing configuration is documented in
[the FS adapter reference](../../../source/mp/l2_storage/fs.rst).
