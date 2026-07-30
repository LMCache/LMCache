# Shared Device-DAX L1 M0

Status: experimental, non-reclaiming functional addon

This implementation proves that SGLang and vLLM MP servers can use one physical
Device-DAX/CXL KV pool without running independent allocators over the same
bytes. It is the monotonic M0 described by
[issue #4307](https://github.com/LMCache/LMCache/issues/4307), not the complete
production design.

## Scope

The supported profile is intentionally fixed:

- one existing MP coordinator with one metadata child process;
- one shared region and layout ID;
- two or more MP servers with host-local mappings of the same physical bytes;
- TP=1 and LMCache-driven transfers;
- immutable KV objects and monotonic allocation;
- CUDA registration and a platform-qualified visibility operation.

P2P, L2, GDS, hybrid DRAM/DAX, eviction, reuse, hotplug, coordinator HA, and
automatic crash recovery are rejected or deferred. The manager transport uses
authenticated Python `BaseManager` RPC and assumes a trusted test cluster.

## Ownership and data path

Two Device-DAX paths may expose the same CXL bytes while using different path
names and virtual addresses. Only the coordinator child allocates offsets:

```text
MP coordinator pod
  FastAPI parent
    `-- starts/stops one metadata child
          `-- allocation cursor, key state, reservations
                       ^
                       | authenticated metadata-only RPC
             +---------+---------+
             |                   |
        MP server A         MP server B
        local DAX mmap      local DAX mmap
        GPU D2H/H2D         GPU D2H/H2D
             \                   /
              +-- same CXL bytes +
```

Serving-engine pods do not map Device-DAX. Payload bytes, virtual addresses,
CUDA pointers, and GPU descriptors never pass through the coordinator.

## Region and object contracts

Every MP server validates this immutable startup contract:

```text
SharedRegionContract {
    region_id, capacity, alignment, layout_id, generation_epoch
}
```

`region_id` is operator-provisioned; LMCache cannot infer physical identity
from a device path. `generation_epoch` changes with each metadata child and
fences a coordinated reset.

Each object has an address-independent handle:

```text
SharedObjectHandle {
    region_id, offset, length, generation
}
```

The coordinator also retains the existing `MemoryLayoutDesc` written with the
object. A remote reader receives that layout with its reservation, so the
public list-oriented `L1Manager.reserve_read(keys, extra_count)` API is
unchanged.

## M0 state and batching

M0 implements:

```text
absent -> WRITING -> VALID
             |
             +-> aborted metadata; extent remains consumed
```

- one writer wins a duplicate-key race;
- only its opaque token can finish or abort the write;
- readers see only `VALID` objects;
- read tokens remain active until H2D completion or cancellation;
- committed objects are immutable;
- abort never rewinds the allocation cursor;
- allocation, reserve, finish, and release are serialized by the child.

Each existing `L1Manager` key list produces one coordinator operation:

```text
reserve_writes(keys)
finish_writes(reservations)
abort_writes(reservations)
reserve_reads(keys)
finish_reads(reservations)
abort_reads(reservations)
```

Write reservation and commit validate the whole batch before changing state.
Read reservation is partial: hits receive tokens while misses return `None`.
Internal per-layer DMA work does not create more metadata calls.

## Transfer ordering

Write publication is:

```text
reserve_writes
  -> enqueue all D2H copies into mapped tensors
  -> synchronize the transfer event
  -> publish every exact DAX range
  -> atomically finish_writes
  -> report STORE success
```

Any transfer, synchronization, visibility, or commit failure aborts remaining
`WRITING` reservations. The normal private L1 path retains its asynchronous
host callback.

Read consumption is:

```text
reserve_reads
  -> validate contract, bounds, generation, and stored layout
  -> acquire each exact DAX range
  -> expose TensorMemoryObj views
  -> enqueue H2D
  -> finish_reads after H2D completion
```

Shared reads reject `extra_count > 0`. Shutdown drains every registered GPU
transfer stream before releasing tokens, unregistering, and unmapping.

## Visibility and CUDA registration

The operator supplies an absolute library path exporting:

```c
int lmcache_shared_l1_visibility_v1(
    const char *mode,
    uint32_t operation,
    int device_fd,
    void *mapped_address,
    uint64_t device_offset,
    size_t length,
    uint64_t generation);
```

Mode is `software_fenced`; operation 1 publishes and operation 2 acquires.
Return zero on success or a negative errno. `MAP_SHARED`, `msync`, or equal
physical media alone are not accepted as cross-host visibility proof.

The entire mapping is registered with CUDA at startup. Registration failure is
fatal; M0 has no pageable-staging fallback. A successful path is
`direct-pinned`, but DAX-to-GPU consumption still performs H2D DMA.

## Configuration

The coordinator uses:

```text
LMCACHE_MP_COORDINATOR_SHARED_L1_{HOST,PORT,AUTHKEY_FILE,REGION_ID,
                                  CAPACITY_BYTES,ALIGNMENT_BYTES,LAYOUT_ID}
```

Each MP server supplies:

```text
--l1-devdax-path PATH
--l1-size-gb SIZE --no-l1-use-lazy --shm-name ""
--shared-l1-coordinator HOST:PORT
--shared-l1-authkey-file ABSOLUTE_PATH
--shared-l1-region-id ID --shared-l1-layout-id ID
--shared-l1-mapping-offset BYTES
--shared-l1-visibility-library-path ABSOLUTE_PATH
--supported-transfer-mode lmcache_driven
--eviction-policy noop
```

Authentication key bytes are read from mounted files, not serialized config.

## Verification and deferred work

Tests cover batched state transitions, duplicate writers, non-overlapping
concurrent allocation, stale tokens/handles, exact visibility ranges, CUDA
registration failure, coordinator authentication/lifecycle, `L1Manager`
integration, and STORE failure cleanup. The two-host Kubernetes qualification
additionally proves an SGLang-produced object can be loaded by vLLM through the
same physical region.

Free-list reuse and `EVICTING`, TTL renewal, client incarnation/idempotency,
restart fencing, metrics, production transport, and HA remain follow-ups.
Restart requires quiescing all MP servers, resetting the pool, and validating
the new epoch. The eventual fleet directory in
[issue #4226](https://github.com/LMCache/LMCache/issues/4226) cannot authorize
shared reads or reconstruct allocator state.
