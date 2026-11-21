# GPU-Direct Borrowing Peer Adapter (GDR)

`lmcache/v1/distributed/l2_adapters/gdr_peer_l2_adapter.py`,
`BorrowingPeerL2AdapterBase` in `lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`,
`NicDmaEngine` in `lmcache/v1/multiprocess/object_group_transfer/nic_dma.py`

The copying P2P adapter ([`p2p_l2_adapter.md`](p2p_l2_adapter.md)) pulls a
peer's object into the local DRAM L1 at lookup, then copies it to the GPU at
retrieve. The GDR adapter **borrows** it instead: the owner keeps its read
lock, and the bytes move once, at retrieve, from the owner's DRAM straight
into GPU memory. Nothing is staged in the local L1.

```
--p2p-load-mode l1 (copy)                  --p2p-load-mode gdr (borrow)
lookup:   owner DRAM ─RDMA─► local L1      lookup:   owner read-locks; shadow L1 indexes a handle
retrieve: local L1 ─PCIe─► GPU             retrieve: owner DRAM ─RDMA─► GPU slot ─kernel─► paged KV
unlock:   after the copy                   unlock:   after the last local reader
```

Enable with `--p2p-load-mode gdr --p2p-transfer-engine nixl`. Borrowed objects
are indexed by a DRAM shadow L1
([`../l1_manager/l1-manager-types.md`](../l1_manager/l1-manager-types.md));
the server adds one (`shadow.dram`) when none is configured. `gdr` excludes
`--enable-p2p-write`.

## Classes

`GdrPeerL2Adapter` extends `BorrowingPeerL2AdapterBase`; the class tree is in
[`p2p_l2_adapter.md`](p2p_l2_adapter.md).

- Each lookup hit adds one reservation for its key.
- `submit_load_task` binds the shadow L1's `BorrowedMemoryObj` placeholder to
  one reservation, which becomes the object's lease. Nothing is copied.
- `submit_unlock` returns the reservations that were not borrowed.
- Ending a lease fires the unlock RPC without waiting for the answer, so
  `finish_read` never blocks on the owner.
- `close` refuses while a lease is live; the P2P controller retries the
  removal every discovery cycle.

## Lease

A lease lasts until `lookup start + read TTL - margin`, with
`margin = min(1 s, TTL / 10)`, and must have `load_timeout_s` left for a read
to start. After that the object is invalid, `read_into` refuses, and the key
is a miss. An expired reservation is not unlocked; the owner's TTL reclaims it.

The TTL is not sent over the wire: **every node must use the same
`--l1-read-ttl-seconds`**, and the borrower uses its own. A peer with a shorter
TTL would let the borrower read bytes the owner already reused. Nothing
checks this today (known limitation); sending the owner's TTL with the lookup
reply is a possible follow-up.

## Retrieve

`transfer_kv_per_object_group` splits a group into runs by transfer engine
(see [`lmcache_driven_transfer_copy_paths.md`](../../multiprocess/lmcache_driven_transfer_copy_paths.md)).
Peer-owned objects select the NIC DMA engine, under the same staged loop as
the other engines:

```
for each batch:                                       (run_staged_loop)
    NicDmaEngine.stage(objects, staging slots):
        synchronize the stream
        obj.read_into(region_id, slot offset)         RDMA → GPU slot
        wait for every read (a failure raises RuntimeError)
    block kernel: slots → paged KV
```

The reads are outside the stream, so the engine waits for the stream before
reading and for every read before returning. No read is abandoned: it could
land in a slot the next batch reuses. The unfull chunk uses the same engine.

The slots are `GPUCacheContext`'s staging buffer. Under `gdr` the P2P
controller calls `enable_staging_buffers(context, alignment)`; the cache
context then aligns every slot (`staging_alignment`) and registers the buffer
(`register_staging_buffer`), which the engine looks up by address
(`find_staging_region`). Reads and kernels of consecutive batches do
not overlap; double buffering is a follow-up (#6399).

## Failures

| Failure | Behavior |
|---|---|
| Read fails | The retrieve raises `RuntimeError` before the batch's kernel. |
| Read exceeds `load_timeout_s` | It is cancelled (`TransferChannelClient.cancel`) and fails once the transport confirms it stopped. Until then the retrieve waits: a read still running could land in a reused slot. |
| Read TTL not above `--p2p-load-timeout` + 1 s | The server refuses to start in `gdr` mode. |
| Lease expired | The key is a miss; a held object refuses `read_into`. |
| Owner unreachable for unlocks | The unlock is lost; the owner's TTL reclaims the locks. |
| Peer removed with live leases or a read in flight | `close()` raises; the controller retries each cycle. |

## Known limitations

- A cancelled read relies on nixl's contract that a successful release of an
  active transfer means no more bytes land. The tests only cancel a transfer
  that already finished.
- At shutdown, if an adapter could not be removed (leases or reads still
  live), the P2P controller leaves the transfer channel open instead of
  closing it under a read; the process exit reclaims it.
- A read that completes between a poll and its cancel is reported failed.
