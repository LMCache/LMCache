# GPU-Direct Borrowing Peer Adapter (GDR)

`lmcache/v1/distributed/l2_adapters/gdr_peer_l2_adapter.py`,
`BorrowingPeerL2AdapterBase` in `lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`

The copying P2P adapter ([`p2p_l2_adapter.md`](p2p_l2_adapter.md)) pulls a
peer's object into the local DRAM L1 at lookup, then copies it to the GPU at
retrieve. The GDR adapter **borrows** it instead: the owner keeps its read
lock, and the bytes move once, at retrieve, from the owner's DRAM straight
into GPU memory. Nothing is staged in the local L1.

```
copying adapter (p2p)                      borrowing adapter (p2p_gdr)
lookup:   owner DRAM ─RDMA─► local L1      lookup:   owner read-locks; shadow L1 indexes a handle
retrieve: local L1 ─PCIe─► GPU             retrieve: owner DRAM ─RDMA─► GPU region
unlock:   after the copy                   unlock:   after the last local reader
```

**Status: the adapter is not selectable yet.** This document covers the
adapter (L2 adapter type `p2p_gdr`) and its read interface. Nothing in the
server creates one: the load mode that selects it, the shadow L1 the server
adds for it, and the retrieve path that reads borrowed objects into GPU
memory land in follow-up changes. Borrowed objects are indexed by a DRAM
shadow L1
([`../l1_manager/l1-manager-types.md`](../l1_manager/l1-manager-types.md)),
which the adapter wants through its memory tags `((DRAM, SHADOW),)`.

## Classes

`GdrPeerL2Adapter` extends `BorrowingPeerL2AdapterBase`; the class tree is in
[`p2p_l2_adapter.md`](p2p_l2_adapter.md).

- Each lookup hit adds one reservation for its key.
- `submit_load_task` binds the shadow L1's `BorrowedMemoryObj` placeholder to
  one reservation, which becomes the object's lease. Nothing is copied.
- `submit_unlock` returns the reservations that were not borrowed.
- Ending a lease fires the unlock RPC without waiting for the answer, so
  `finish_read` never blocks on the owner.
- `close` refuses while a lease is live or a read is in flight; the caller
  retries once they ended.

## Lease

A lease lasts until `lookup start + read TTL - margin`, with
`margin = min(1 s, TTL / 10)`, and must have `load_timeout_s` left for a read
to start. After that the object is invalid, a read of it is refused, and the key
is a miss. An expired reservation is not unlocked; the owner's TTL reclaims it.

The TTL is not sent over the wire: **every node must use the same
`--l1-read-ttl-seconds`**, and the borrower uses its own. A peer with a shorter
TTL would let the borrower read bytes the owner already reused. Nothing
checks this today (known limitation); sending the owner's TTL with the lookup
reply is a possible follow-up.

## Reading a borrowed object

The adapter binds each `BorrowedMemoryObj` to its lease, to the owner's
address of the object, and to the adapter's batch read:

```python
obj.source                           # the adapter: objects with one source can share a read
obj.read_many(region_id, targets)    # targets: [(obj, offset in the region), ...]
```

`read_many` reads every object in `targets` into a memory region the caller
registered with the transfer channel, as **one** transfer (one NIXL submit),
and returns a `PendingRead` to poll. Per-object submits gave the same
bandwidth but 10x the polling CPU, so a caller groups the objects of one
source into one call.

- A read is refused (`succeeded()` is False at once, nothing is submitted)
  when any object's lease has less than `load_timeout_s` left.
- A read that runs longer than `load_timeout_s` is cancelled
  (`TransferChannelClient.cancel`) and reported failed once the transport
  confirms it stopped, also if it finished before the cancel took effect: the
  owner's lock may lapse before late bytes are used.
- The adapter counts reads in flight; `close` refuses until they ended, so
  the peer's client is never removed under a read.

`GdrPeerL2AdapterConfig.check_timeouts(read_ttl_s, load_timeout_s)` raises
`ValueError` unless the read TTL is more than 1 s above the load timeout:
otherwise no lease could ever start a read.

## Failures

| Failure | Behavior |
|---|---|
| Read fails | The `PendingRead` finishes with `succeeded()` False; the caller treats the objects as not loaded. |
| Read exceeds `load_timeout_s` | It is cancelled and fails once the transport confirms it stopped. A cancel the transport never confirms keeps the read unfinished (known gap). |
| Lease expired | The key is a miss; a read of a held object is refused. |
| Owner unreachable for unlocks | The unlock is lost; the owner's TTL reclaims the locks. |
| Adapter closed with live leases or a read in flight | `close()` raises; the caller retries. |

## Known limitations

- A cancelled read relies on nixl's contract that a successful release of an
  active transfer means no more bytes land. The tests only cancel a transfer
  that already finished.
- A read that completes between a poll and its cancel is reported failed.
