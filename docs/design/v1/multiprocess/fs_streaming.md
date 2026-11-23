# Bounded FS streaming for MP retrieval

## Motivation and scope

When the DRAM L1 is full, prefetching a cold FS prefix through L1 competes
with resident objects. Opt-in reserved FS streaming windows use a bounded
share of that DRAM budget and serve FS reads outside resident L1. Resident L1
uses its configured eviction policy. With streaming enabled, eligible FS hits
use the window path at any level of L1 occupancy.

The default MP engine in `lmcache_driven` mode can serve deferred FS hits.
The `engine_driven` and `auto` modes and the blend engine use ordinary
prefetch and keep the full resident L1 budget. With streaming enabled, lookup
uses the window path when the sole active L2 adapter is an unwrapped `fs`
adapter and each object layout fits. Other lookups use ordinary prefetch
while the reservation stays in place.

## Lookup and ownership

The existing prefetch controller locks available L1 objects, looks up missing
objects on FS, and plans the longest usable prefix across ranks and attention
windows. For streaming requests the result reports planned FS hits. The
controller leaves their data on FS and retains read locks on L1 hits. A
read-locked unfull L1 chunk can extend the prefix as described in
[unfull_chunk.md](unfull_chunk.md).

The session records the mixed streaming range and the owners of its L1 locks.
Retrieve reads resident objects from their owners and FS objects through the
reserved window pool. It slices the staged block IDs for each run of
L1 objects or FS windows, so the device sees the same object order as the
lookup prefix. Tokens after the hit are computed by the inference engine.
When servers report unequal hits, the scheduler waits for the excess-tail
releases to complete before exposing their common hit. When the scheduler
replaces a lookup, it sends the abandoned lookup's range for cleanup. Its end
can be the shorter common hit across servers. The server releases that
lookup's remaining L1 locks, including any retained unfull L1 tail. A partial
FREE removes only the released locks from the record. When vLLM trims the
current hit for bounded SWA replay, it releases the omitted tail as a current
lookup range.

## Buffer lifecycle

The DRAM L1 manager permanently reserves a portion of its existing arena for
the FS reader. The reader suballocates one window per object, up to the pool's
capacity. The existing stream completion dispatcher returns windows after the
device copy. Shutdown drains device streams before stopping this dispatcher
so the reader can return the idle reservation to L1.

Lookup uses the standard prefetch completion API. The first window acquisition
and the FS batch load each have a 30-second timeout. The FS adapter publishes a
failure result when an asynchronous task raises, so both ordinary prefetch
and streaming can settle the task. A reaper retains windows whose reads are
still in flight until the adapter reports completion.

L1 object locks are independent of window reservations. Retrieve submits their
release after the device work, including locks for a group whose transfer fails.
If an FS object disappears after lookup, retrieve reports failure so the
connector's configured failure policy can handle the missing KV.

## Configuration and tradeoffs

`streaming_buffer_size_bytes` defaults to zero. A positive reservation must be
4096-byte aligned. Its aligned allocation must leave room for at least one
resident L1 allocation. The reservation comes from
the existing DRAM L1 arena and is excluded from the reported resident L1
capacity. Any request with an object larger than the reservation uses ordinary
prefetch. Resident L1 continues to admit stores and run its eviction policy.

User-facing configuration is documented in
[the FS adapter reference](../../../source/mp/l2_storage/fs.rst).
