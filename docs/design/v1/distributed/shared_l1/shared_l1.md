# Shared L1

`RegionMedium` owns one mapping, its GPU registration and tensor views.
DRAM uses a regular file on tmpfs/hugetlbfs; DEVDAX requires a character device
with readable sysfs capacity. The caller supplies the path, backend, capacity
and alignment. Mapping uses the same helper as the private Device-DAX allocator.

Views validate offset, alignment, extent length and payload size. They have no
parent allocator: dropping a view cannot free shared bytes. GPU registration
must succeed; a failure closes acquired resources. After all copies drain,
`close()` unregisters and unmaps; outstanding views defer unmapping.

For noncoherent memory, `Visibility("software_fenced")` writes back and
invalidates cache lines using CLFLUSHOPT + SFENCE, or CLFLUSH + MFENCE.
The writer must publish after D2H and before exposing metadata. The reader must
acquire before H2D. Native flushes release the GIL and a batch uses up to eight
threads. `coherent` performs no cache maintenance.

```text
D2H completion -> publish fence -> metadata commit
read grant -> acquire fence -> H2D -> release read
```

The operator establishes that peer paths map the same physical region.
Allocation, write publication and read leases belong to the memory orchestrator;
these primitives do not own them.

Tests cover view aliasing, extent bounds, backing types, failed registration
cleanup, and native/cache-visibility operations. The private Device-DAX allocator
continues using the same mapping behavior after the helper extraction.

## Manager lifecycle

`SharedL1Manager` implements L1ManagerInterface using SharedL1Config and the
orchestrator's canonical ObjectKey, layouts and tokens. DRAM/DEVDAX configuration
subclasses select the medium. The binding uses the same L1 interface as private
pools.

ReserveWrite returns private write tokens/views. After D2H completes, FinishWrite
publishes visibility and commits metadata. Failed publication aborts its tokens.
ReserveRead acquires independent leases and fences before H2D; overlapping reads
retain separate counts. Failed acquire releases only that call's leases. Malformed
extents, payloads and layouts release known tokens without disturbing existing
reads. CloseClient retires this incarnation after copies have drained.

Startup verifies capacity, page alignment and the epoch across mapping and
registration. Epoch fencing disables the binding. Unavailable metadata makes
reads miss and writes return OUT_OF_MEMORY; failed-close leases remain until
re-registration and writes also expire under the orchestrator TTL. Extents are
never reclaimed, and shared objects cannot be evicted or forcibly deleted.

Status and common L1 gauges report region-wide allocation/capacity and local
staging payload bytes. RPCs and visibility operations run outside metadata locks.
Device-DAX hot-plug is a separate capability; the shared binding does not expose
it, and the storage-manager API rejects unsupported reconfiguration.

Shared placement events are excluded from the per-server fleet directory, since
several bindings refer to the same region-owned copy.

## Serving integration

The `shared` section of `--l1-manager` selects SharedL1Manager. One shared region
per MP process is supported, with explicit client identity, `noop` eviction and
`lmcache_driven` transfer. Shared GDS, L2 affinity, hot-plug, CacheBlend, P2P and
runtime plugins are rejected. P/D proxy/readiness changes are a separate follow-up.

Writes follow configured L1 order and overflow privately when the region is full.
Local hits avoid shared lookups. Shared peer layout mismatches release every
acquired read before the lookup fails. Store failures abort through each memory
object's owner on the copy stream, after partial transfers drain.

Existing key/rank/group encoding and grouped prefix rules support equal
TP1/2/4/8 peers, including hybrid windows. A missing rank truncates the prefix;
overlapping reads retain independent leases. Grouped-lookup tests exercise
late shards, peer layouts and read cleanup for each TP size.

User configuration, compatibility requirements, status/metrics and reset
instructions are in `docs/source/mp/shared_l1.rst`. Shared allocation/capacity
are region-wide and must not be summed across attached servers.
