# GPU-Direct Borrowing Peer Adapter (GDR)

`lmcache/v1/distributed/l2_adapters/gdr_peer_l2_adapter.py`,
`BorrowingPeerL2AdapterBase` in `lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`,
`run_device_read_transfer` in `lmcache/v1/multiprocess/object_group_transfer.py`

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

Enable with `--p2p-load-mode gdr --p2p-transfer-engine nixl` on top of the
usual P2P flags (see the P2P user guide). The borrowed objects are indexed by
a DRAM shadow L1 ([`../l1_manager/l1-manager-types.md`](../l1_manager/l1-manager-types.md),
"Shadow L1"); the server adds one (`shadow.dram`) in `gdr` mode when the L1
configuration has none.

## What is borrowed

A `BorrowedMemoryObj` (`distributed/memory_manager/shadow_l1_memory_manager.py`)
the shadow L1 staged as a placeholder, which this adapter's load binds to a
remote handle: no local bytes (`tensor` is None, `data_ptr` raises), the
owner-relative offset in its metadata, and a `read_into(region_id, offset)`
that starts the transfer-channel read of the whole object into a registered
local device region and returns a `PendingRead` to poll. Its lease is the
owner's read reservation, bounded locally by the owner's read TTL minus a
safety margin (`min(1 s, TTL / 10)`); `is_valid()` is False once the lease
ended, `release()` ends it. `same_bytes` compares the owner's address, so a
resident handle adopts a newer lease of the same bytes, and `local_device()`
is None, so the shadow synchronizes every GPU before it closes.

The owner reports its read-lock TTL with every lookup hit
(`TransferChannelAddress.read_ttl_seconds`); a hit without one cannot be borrowed
and is released at once.

## Class layout

```
PeerL2AdapterBase               lookup/unlock RPCs, event fds, no-op stores, close
├── P2PL2Adapter                copy over the transfer channel into the L1
└── BorrowingPeerL2AdapterBase    borrow views into a shadow L1: per-key reservations,
    │                           local TTL bound, batched unlock worker, lease records,
    │                           drain on close
    └── GdrPeerL2Adapter        binds placeholders to remote handles read into a device region at retrieve
```

`BorrowingPeerL2AdapterBase` (#6207's interface, made concrete here) implements
the borrowing contract and asks a subclass two things: `_accept_address(address)`
(can this hit be borrowed at all) and `_bind_borrowed_object(obj, address, lease)`
(how the placeholder is bound to the bytes). Every lookup hit adds one
reservation for its key; `submit_load_task(keys, objects)` binds one
reservation per key into the shadow L1's placeholder and `submit_unlock`
returns one, so a key two requests looked up has two reservations and each
request's unlock or load consumes its own. `query_load_result` has the bound
bitmap at once: nothing is copied.

Releases only enqueue work: a worker thread sends one unlock RPC per batch of
queued releases (`submit_unlock(keys, wait=True)`), so a slow owner costs one
timeout per batch and ending a lease only flags its record and enqueues the unlock, so a
reader's `finish_read` never blocks on the owner. The shadow
L1 frees every placeholder it staged, so every lease ends. A
reservation past its local expiry is not unlocked: the owner's own TTL reclaims
it, and a late key-only unlock could otherwise release a newer reservation of
the same key.

## Retrieve path

The retrieve handler reads the shadow L1 like any L1 (`unsafe_read`), so the
object list it hands to `transfer_kv_per_object_group` may contain remote
handles. That function routes such an object group to `run_device_read_transfer`
(`multiprocess/object_group_transfer.py`):

```
wait for the stream: an earlier transfer may still use the staging slots
for each batch of max_batch_size chunks:
    wait for the kernel that last read this bank (none the first time round)
    slot = cache_context.get_landing_slot(bank, chunk, object_group)
    remote handle  → handle.read_into(slot.region_id, slot.offset)   RDMA → GPU slot
    local object   → lmcache_memcpy_async_h2d(obj, slot.buffer)      ordinary staging
    wait until every read of the batch finished (a failure raises RuntimeError)
    multi_layer_block_kv_transfer(slots, paged blocks)               scatter kernel
    record a completion event for the bank on the stream
    bank = (bank + 1) % landing_banks
```

Transfer-channel reads are not stream-ordered, so the host must know a bank's
slots are free before submitting reads into them, and must know the reads
finished before queuing the kernel. Two banks let the reads of batch *i+1*
overlap the kernel of batch *i*. The handler's thread blocks while a batch's
reads are in flight; the GPU stream is never blocked.

The banks are the cache context's own staging buffer. When a
`DeviceRegionTarget` is installed, `GPUCacheContext` builds the buffer with a
second bank of `max_batch_size` batch slots, starts every object-group slot on
the transfer channel's alignment, and registers the whole buffer as one
device region with `register_device_region` (next to its GDS cuFile
registration); `landing_banks` and `get_landing_slot(bank, chunk, object_group)`
expose the slots, bank 0 being the stream-ordered copy paths' own slots. The
P2P controller installs the target (the transfer channel context and its L1
alignment, `set_device_region_target`) under `--p2p-load-mode gdr`, so the GPU
side never names a transport or a load mode. Without it the buffer keeps its
single packed bank and no extra GPU memory is spent. The region is
unregistered when the cache context closes.

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
| Peer removed with live handles | `close()` raises while a handle holds a lease the owner's TTL has not ended; `delete_l2_adapter` keeps the adapter for a retry, and the P2P controller retries on every later cycle. A lease past the owner's TTL is drained: no read of it can be in flight. |

## Trade-off

Borrowing puts the network read on the retrieve path, where the copying mode
pays only a DRAM-to-GPU copy; in exchange it uses no L1 capacity or DRAM
bandwidth and moves each byte once. With two banks the reads of one batch
overlap the scatter kernel of the previous one, so a retrieve is bound by the
slower of the fabric and the kernel. Measure before choosing: on a single H200
host, NIXL loopback reads pinned DRAM into GPU memory at about 46 GB/s.

## Tests

- `tests/v1/distributed/l2_adapters/test_gdr_peer_l2_adapter.py`: borrowing,
  reads into a region, TTL bound, lease adoption, draining (mocked transport).
- `tests/v1/multiprocess/test_device_read_transfer.py`: the retrieve ordering
  contract with fake slots, events and reads.
- `tests/v1/platform/devices/cuda/test_gpu_cache_context.py`: the staging
  buffer's banks and slot alignment.
- `tests/v1/multiprocess/test_p2p_controller.py`: GDR startup requirement on
  the engine, device-region target lifetime, adapter type, served TTL.
- `tests/v1/distributed/transfer_channel/test_nixl_impl.py`: a read into a
  registered CUDA region (GPU + NIXL).
- `tests/v1/distributed/l2_adapters/test_gdr_peer_l2_adapter_integration.py`:
  owner and borrower in one process over real NIXL, bytes verified in GPU
  memory (GPU + NIXL).
- `tests/v1/platform/devices/cuda/test_gdr_staging_region.py`: the GPU cache
  context registers and unregisters its staging buffer (GPU).

Not covered: a retrieve through the full `LMCacheDrivenTransferModule` with
a vLLM worker, and TTFT measurements against the copying mode.
