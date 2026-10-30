# P2P L2 Adapter

`lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`

An L2 adapter that treats a **single peer cache server** as a read-only L2
tier. Instead of a storage backend, it looks up objects that are resident in
the peer's L1 and reads them from there. How the bytes are read depends on
the peer's L1:

- **DRAM L1:** `P2PL2Adapter` pulls them over the transfer channel (RDMA).
- **Device-DAX L1 on a CXL pool that both hosts map:** the bytes are already
  addressable from this host, so a `PeerL2AdapterBase` subclass can read them
  in place without a network copy.

One adapter instance is created per connected peer; the peer discovery
and the dynamic add/remove that wire these into the storage manager are owned
by separate PRs (the P2P controller and the runtime `add_l2_adapter`
interface).

See the Confluence design *LMCache MP P2P design → P2P L2 adapter design* for
the full system context.

## What it talks to

| Dependency | Used for |
|---|---|
| `RequestClient` → peer's P2P controller | lookup-and-lock, query, and unlock RPCs over ZMQ or gRPC |
| `TransferChannelContext` / `TransferChannelClient` | translating local L1 addresses + reading the peer's L1 |
| `PeriodicEventNotifier` | pulsing the lookup / load event fds (see below) |

## Class layout

```
PeerL2AdapterBase             lookup/unlock RPCs, event fds, no-op stores, close
├── P2PL2Adapter              import the peer region; RDMA load over the transfer channel
└── BorrowingPeerL2AdapterBase  borrow the peer's bytes: per-key reservations under the owner's
    │                         lock, a load that binds the shadow L1's placeholders, batched unlocks
    └── GdrPeerL2Adapter      remote handles (see [gdr_peer_l2_adapter.md](gdr_peer_l2_adapter.md))
```

The base holds the RPC control plane so another way of reading a peer can
reuse it. Subclasses implement three protected hooks:

| Hook | When | On failure |
|---|---|---|
| `_register_peer_region()` | once, inside the base constructor | everything acquired before it is released |
| `_unregister_peer_region()` | first step of `close()` | adapter stays open, `close()` can be retried |
| `_process_lookup_result(keys, addresses)` | per completed lookup; returns the hit bitmap | |

Because the region hook runs inside the base constructor, a subclass sets the
state its hook needs before `super().__init__()` and acquires nothing after it.

`_process_lookup_result` is a hook rather than inline code in
`query_lookup_and_lock_result` because the read strategy decides what a hit
means, while the polling, timeout and failure handling around it stay shared:

| Strategy | What `_process_lookup_result` does with a hit |
|---|---|
| Copy (`P2PL2Adapter`) | records the address; a later load copies the bytes into local L1 |
| Borrow (`BorrowingPeerL2AdapterBase`, e.g. GPU-direct or CXL in place) | keeps the reservation; the load binds it into the shadow L1's placeholder; no copy, no local capacity |

`BorrowingPeerL2AdapterBase` goes through the prefetch controller's ordinary
load pipeline: the controller stages a `BorrowedMemoryObj` placeholder per
planned key in the shadow L1 and submits a load with them; `submit_load_task`
binds each placeholder to the key's reservation (its lease) through the
`_bind_borrowed_object` hook, and `_accept_address` says whether a hit is
usable at all. `_process_lookup_result(task_id, keys, addresses, started_at)`
receives the time just before the lookup RPC was sent, a lower bound on when
the owner's read locks started; `submit_unlock(keys, wait=True)` waits for the
owner's acknowledgment, which borrowers use so peer removal drains their
unlocks. A reservation bound into an object belongs to the object and returns
at its release; `submit_unlock` returns only the others. `close` refuses while
an object holds a live lease and drains one the owner's TTL has ended
(`get_local_borrow_count` reports the live ones). The adapter wants a
`SHADOW` L1 first: the
configured shadow L1 of the segment it borrows. See
`../l1_manager/l1-manager-types.md`, "Shadow L1".

## Lifecycle

```
lookup-and-lock  → query (addresses)  → load (RDMA read)  → unlock
```

1. **lookup-and-lock** — send `P2P_LOOKUP_AND_LOCK([keys, group_layout_descs])` to the
   peer; it read-locks every L1-resident key (sparse; gaps allowed) and
   returns a task id.
2. **query** — poll `P2P_QUERY_LOOKUP_RESULTS(task_id)`; once ready the peer
   returns one `TransferChannelAddress` per key (invalid offset for keys it did
   not lock). The adapter stashes the valid addresses keyed by `ObjectKey` and
   reports a found/not-found `Bitmap` to the prefetch controller.
3. **load** — for the found keys, translate the local destination objects'
   `(shm_offset, shm_byte_length)` into transfer-channel addresses, pair them
   with the stashed remote addresses, and issue `submit_read`. `query_load_result`
   maps the read's `succeeded_mask` to a `Bitmap`.
4. **unlock** — `P2P_UNLOCK_OBJECTS([keys])` releases the peer's read locks and
   drops the stashed addresses. Fire-and-forget: the result future is not awaited.

`group_layout_descs` (one `MemoryLayoutDesc` per object group) is the advisory
hint added to the L2 lookup interface; the P2P adapter forwards the real
descriptors verbatim so the peer can size objects.

## Why the periodic notifier

Neither the lookup RPC nor the transfer-channel read exposes a completion fd:
the request response resolves asynchronously in the selected transport, and
the RDMA read completes inside the transfer engine. So the adapter registers
its lookup and load event fds with the `PeriodicEventNotifier` singleton, which
pulses them every few milliseconds. Each pulse drives the prefetch controller
to re-poll `query_lookup_and_lock_result` / `query_load_result`, which in turn
check the in-flight RPC / transfer state. The store fd is not pulsed by the
notifier; it is signaled directly on submit (see *No store* below).

Both RPCs are non-blocking handlers, so the adapter uses **submit-then-wait**:
`submit_lookup_and_lock_task` submits the lookup and waits (hard-coded ~3 s) for
the task id; each `query_lookup_and_lock_result` pulse submits
`P2P_QUERY_LOOKUP_RESULTS` and waits — `None` means not ready (re-queried on the
next pulse), addresses mean ready.

## Timeouts

P2P is error-prone, so both queries are bounded by a per-task deadline
(`lookup_timeout_s` / `load_timeout_s`). A lookup past its deadline returns an
all-zero `Bitmap` (treated as a miss); a load past its deadline returns an
all-zero `Bitmap` (treated as a failure). The prefetch controller then trims
those keys as if the peer never had them.

On the owner, `P2P_LOOKUP_AND_LOCK` drops finished lookups older than
`--p2p-lookup-timeout` that the requester never queried, scanning at most once
per timeout. Their read locks are
left to expire by TTL; a late query gets `None` and counts as a miss.

## No store / no eviction

Writing to a peer's L1 is intentionally unsupported (it would require remote
allocation and existence/status checks that can corrupt both nodes on failure).
But the store controller still tracks every task it submits — and the L1 read
locks it reserved for that task — until the result is popped. So rather than
drop store tasks, `submit_store_task` records a 0-byte success and signals the
store fd immediately, and `pop_completed_store_tasks` returns it; the controller
then finalizes its bookkeeping and releases the read locks instead of leaking
them. No data is moved and no bytes are accounted.

The adapter is constructed with `max_capacity_bytes=0`, so
`supports_global_eviction` is `False`, `delete` is the inherited no-op, and
`get_usage` reports the (empty) base counters.

## Configuration

The adapter is built through the standard factory (`create_l2_adapter`), so the
storage manager owns its lifecycle. The config carries the peer's two URLs:

```json
{
  "type": "p2p",
  "peer_mq_server_url": "tcp://peer-host:5555",
  "peer_transfer_channel_server_url": "peer-host:7600",
  "lookup_timeout_s": 10.0,
  "load_timeout_s": 10.0
}
```

## close()

`close()` calls `_unregister_peer_region()` first; for `P2PL2Adapter` that
removes the transfer-channel client. If it succeeds, `close()` marks the
adapter closed, unregisters the lookup and load fds from the periodic
notifier, and closes the request client and the three event fds. Later calls
are no-ops.

If `_unregister_peer_region()` raises, the adapter stays open with its
request client and fds, and `close()` can be called again.
`StorageManager.delete_l2_adapter` propagates the error and keeps the adapter
registered but inactive: it is excluded from `l2_adapters()`, status,
touches and the capacity topology, but still counts as the owner of any
device range it maps. Calling `delete_l2_adapter` again skips the drain and
retries the close.

The P2P controller retries a failed removal on every later cycle (a
borrower refuses to close while local readers hold its objects);
`StorageManager.close()` closes every registered adapter again at shutdown;
if any of them fails, it leaves the L1 managers open, because that adapter
may still access L1 buffers.
Region access shared per peer, such as a transfer-channel client keyed by the
peer's URL, must not be re-acquired for that peer while a close-failed
adapter still holds it.
