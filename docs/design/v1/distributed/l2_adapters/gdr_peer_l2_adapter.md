# GPU-Direct Borrowing Peer Adapter (GDR)

`lmcache/v1/distributed/l2_adapters/gdr_peer_l2_adapter.py`,
`lmcache/v1/distributed/l2_adapters/lending_peer_l2_adapter.py`,
`lmcache/v1/gpu_connector/gdr_context.py`

A copying P2P adapter ([`p2p_l2_adapter.md`](p2p_l2_adapter.md)) pulls a
peer's KV object into this host's DRAM L1 at lookup time and later copies it
DRAM-to-GPU at retrieve time: two hops and L1 capacity for every borrowed
chunk. The GDR adapter removes both. It **borrows** the object instead (the
owner keeps its read lock while this host uses it) and moves the bytes once, at
retrieve time, from the owner's DRAM straight into GPU memory over the
transfer channel (RDMA with GPUDirect). Nothing is staged in this host's L1.

```
--p2p-load-mode l1 (copy)                 --p2p-load-mode gdr (borrow)

lookup: owner DRAM ──RDMA──► local DRAM L1   lookup: owner read-locks; shadow L1
                              (reserve_write)         indexes a remote handle (0 bytes)
retrieve: local DRAM ──PCIe──► GPU            retrieve: owner DRAM ──RDMA──► GPU landing
unlock:   after the lookup-time copy                   slot ──kernel──► paged KV
                                              unlock:   after the last local reader
```

Enable with `--p2p-load-mode gdr --p2p-transfer-engine nixl --l1-shadow` on top
of the usual P2P flags (see the P2P user guide). `--l1-shadow` (or
`"shadow": true` in the DRAM `--l1-manager`) pairs the DRAM L1 with the shadow
L1 that indexes the borrowed objects.

## What is borrowed

`RemoteBorrowedMemoryObj` (`memory_management.py`) is a
`BorrowedMemoryObj` without local bytes: `tensor` is None, `data_ptr`
raises, and its metadata carries the owner-relative offset and size. Its
lease is the owner's read reservation, bounded locally by the owner's read
TTL minus a safety margin (`min(1 s, TTL / 10)`); `is_valid()` is False once
the lease ended, `release()` ends it. `read_into(region_id, offset)` starts
the transfer-channel read of the whole object into a registered local device
region and returns a `PendingRead` to poll.

The owner reports its read-lock TTL with every lookup hit
(`TransferChannelAddress.read_ttl_seconds`); a hit without one cannot be lent
and is released at once.

## Class layout

```
PeerL2AdapterBase             lookup/unlock RPCs, event fds, no-op stores, close
├── P2PL2Adapter              copy over the transfer channel into the L1
└── LendingPeerL2AdapterBase  task-scoped reservations, local TTL bound, batched
    │                         unlock worker, finalizers, drain on close
    └── GdrPeerL2Adapter      remote handles read into a device region at retrieve
```

`LendingPeerL2AdapterBase` owns everything a lender needs and asks a subclass
two things: `_accept_address(address)` (can this hit be lent at all) and
`_make_borrowed_object(address, metadata, lease_is_valid, on_release)` (what
the borrowed object is). It never copy-loads: `submit_load_task` raises, and
the prefetch controller never plans a load against an adapter bound to a
shadow L1.

Releases only enqueue work: a worker thread sends one unlock RPC per batch of
queued releases (`submit_unlock(keys, wait=True)`), so a slow owner costs one
timeout per batch and no finalizer ever takes a lock or blocks. A borrowed
object dropped without `release()` is released by its `weakref.finalize`. A
reservation past its local expiry is not unlocked: the owner's own TTL reclaims
it, and a late key-only unlock could otherwise release a newer reservation of
the same key.

## Retrieve path

The retrieve handler reads the shadow L1 like any L1 (`unsafe_read`), so the
object list it hands to `transfer_kv_per_object_group` may contain remote
handles. That function routes such an object group to `run_borrowed_transfer`
(`multiprocess/object_group_transfer.py`):

```
for each batch of ring.slots_per_bank chunks:
    ring.acquire(bank)               host waits: kernel that last read the bank finished
    remote handle  → handle.read_into(slot.region_id, slot.offset)   RDMA → GPU slot
    local object   → lmcache_memcpy_async_h2d(obj, slot)             ordinary staging
    wait until every read of the batch finished (a failure raises RuntimeError)
    multi_layer_block_kv_transfer(slots, paged blocks)               scatter kernel
    ring.release(bank, stream)       records a completion event on the stream
    bank ^= 1
```

Transfer-channel reads are not stream-ordered, so the host must know a bank's
slots are free before submitting reads into them, and must know the reads
finished before queuing the kernel. The ring's two banks let the reads of
batch *i+1* overlap the kernel of batch *i*. The handler's thread blocks while
a batch's reads are in flight; the GPU stream is never blocked.

`GdrLandingRing` (`gpu_connector/gdr_context.py`) is created by
`GPUCacheContext` when a `DeviceRegionRegistrar` is installed, with
`2 × max_batch_size` slots of the largest object size, rounded up to the
transfer channel's alignment, and registered as one device region. The P2P
controller installs `TransferChannelRegionRegistrar` under
`--p2p-load-mode gdr`, so the GPU side never names the transport. Without it
no ring exists and no GPU memory is spent. The ring is freed when the cache
context closes.

The direct copy path (`cudaMemcpyBatchAsync`) and the native staged plan both
require host bytes; `direct_transfer_supported` rejects object groups holding
remote handles, and the dispatch runs before either.

## Transfer channel additions

- `TransferChannelAddress.read_ttl_seconds`: the owner's read-lock TTL (0 when
  not reported). Wire-compatible: a new field with a default on the ZMQ
  (msgspec) and gRPC (`p2p_service.proto`) encodings.
- `TransferChannelContext.register_device_region(ptr, size, device_index)` /
  `unregister_device_region(id)`: extra local read destinations in device
  memory, never shared with peers. Offsets follow the L1 alignment contract.
  Only `nixl` implements it (registers the buffer as `VRAM`); the base class
  reports `supports_device_regions()` as False and raises
  `NotImplementedError`, and the controller refuses GDR mode at startup with
  an engine that reports no support.
- `TransferChannelClient.submit_read(..., local_region=L1_REGION)`: which local
  region the destination addresses belong to.

## Lease timing

```
lookup sent ─── owner read lock (TTL) ──────────────────────────────┤ owner may free
      │◄── lease valid: started_at + TTL − margin ──►│ margin │
      admit → ... → retrieve: read_into checks validity, then reads
```

The margin is how long a read that passed the validity check has to finish
before the owner may reuse the bytes: 1 s, or a tenth of the TTL for short
TTLs. A retrieve of hundreds of chunks reads them in batches within one
handler call, so the check-to-read gap is the retrieve itself; with the
default 300 s TTL and GPU-rate reads that is far inside the margin. An expired
handle is never read (`read_into` returns a failed `PendingRead`) and a
retrieve that hits one fails, exactly like a missing object.

## Failure handling

| Failure | Behavior |
|---|---|
| Owner dropped the object or the read errors | `PendingRead.succeeded()` is False; the retrieve raises `RuntimeError` before any kernel of that batch runs. |
| Read exceeds `load_timeout_s` | Same as a failed read. |
| Lease expired before retrieve | `unsafe_read` does not return the handle (unreadable), so the key is a miss; a handle already held is refused by `read_into`. |
| Owner unreachable for unlocks | One RPC timeout per batch, logged; reservations follow the owner's TTL. |
| Peer removed with live handles | `close()` raises, `delete_l2_adapter` keeps the adapter for a retry; the prefetch controller drains live borrows first (`get_active_borrow_count`). |

## Trade-off

Borrowing puts the network read on the retrieve path, where the copying mode
pays only a DRAM-to-GPU copy; in exchange it uses no L1 capacity or DRAM
bandwidth and moves each byte once. With two banks the reads of one batch
overlap the scatter kernel of the previous one, so a retrieve is bound by the
slower of the fabric and the kernel. Measure before choosing: on a single H200
host, NIXL loopback reads pinned DRAM into GPU memory at about 46 GB/s.

## Tests

- `tests/v1/distributed/l2_adapters/test_gdr_peer_l2_adapter.py`: lending,
  reads into a region, TTL bound, finalizers, draining (mocked transport).
- `tests/v1/multiprocess/test_borrowed_transfer.py`: the retrieve ordering
  contract with a fake ring and fake reads.
- `tests/v1/gpu_connector/test_gdr_context.py`: ring layout, alignment and
  bank hand-off.
- `tests/v1/multiprocess/test_p2p_controller.py`: GDR startup requirements,
  registrar lifetime, adapter type, served TTL.
- `tests/v1/distributed/transfer_channel/test_nixl_impl.py`: a read into a
  registered CUDA region (GPU + NIXL).
- `tests/v1/distributed/l2_adapters/test_gdr_peer_l2_adapter_integration.py`:
  owner and borrower in one process over real NIXL, bytes verified in GPU
  memory (GPU + NIXL).
- `tests/v1/platform/devices/cuda/test_gdr_landing_ring.py`: the GPU cache
  context creates and frees the ring (GPU).

Not covered: a retrieve through the full `LMCacheDrivenTransferModule` with
a vLLM worker, and TTFT measurements against the copying mode.
