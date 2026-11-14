# P2P L2 Adapter

`lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`

An L2 adapter that treats a **single peer cache server** as an L2 tier.
Instead of a storage backend, it looks up objects that are resident in the
peer's L1 and reads them from there. It is read-only by default; with
`write_enabled` it also stores objects into the peer's L1 (see *Store*). How
the bytes are read depends on the peer's L1:

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
| `RequestClient` → peer's P2P controller | lookup-and-lock, query, unlock, reserve-write and finish-write RPCs over ZMQ or gRPC |
| `TransferChannelContext` / `TransferChannelClient` | translating local L1 addresses + reading and writing the peer's L1 |
| `PeriodicEventNotifier` | pulsing the lookup / load event fds, and the store fd when writes are enabled (see below) |

## Class layout

```
PeerL2AdapterBase   lookup/unlock RPCs, event fds, no-op stores, close
└── P2PL2Adapter    import the peer region; RDMA load and, when enabled,
                    RDMA store over the transfer channel
```

The base holds the RPC control plane so another way of reading a peer can
reuse it. Subclasses implement three protected hooks and may override a
fourth:

| Hook | When | On failure |
|---|---|---|
| `_register_peer_region()` | once, inside the base constructor | everything acquired before it is released |
| `_unregister_peer_region()` | first step of `close()` | adapter stays open, `close()` can be retried |
| `_process_lookup_result(keys, addresses)` | per completed lookup; returns the hit bitmap | |
| `_stores_need_polling()` (optional, default `False`) | once, inside the base constructor; `True` adds the store fd to the pulsed fds | |

Because the region hook runs inside the base constructor, a subclass sets the
state its hook needs before `super().__init__()` and acquires nothing after it.

`_process_lookup_result` is a hook rather than inline code in
`query_lookup_and_lock_result` because the read strategy decides what a hit
means, while the polling, timeout and failure handling around it stay shared:

| Strategy | What `_process_lookup_result` does with a hit |
|---|---|
| Copy (`P2PL2Adapter`) | records the address; a later load copies the bytes into local L1 |
| Lend (a follow-up subclass, e.g. GPU-direct or CXL in place) | keeps the reservation and hands L1 a borrowed view; no copy, no local capacity |

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
check the in-flight RPC / transfer state. With writes disabled the store fd is
not pulsed; the no-op store signals it directly on submit. With writes enabled
the store fd is pulsed too, so the store controller keeps calling
`pop_completed_store_tasks` to notice finished transfers and expired stores
(see *Store* below).

Both RPCs are non-blocking handlers, so the adapter uses **submit-then-wait**:
`submit_lookup_and_lock_task` submits the lookup and waits (hard-coded ~3 s) for
the task id; each `query_lookup_and_lock_result` pulse submits
`P2P_QUERY_LOOKUP_RESULTS` and waits — `None` means not ready (re-queried on the
next pulse), addresses mean ready.

## Timeouts

A lookup past `lookup_timeout_s` returns an all-zero `Bitmap` (a miss).
A load observed complete past `load_timeout_s` logs a warning and returns the
transfer channel's actual success bitmap. Already-transferred keys are retained
instead of being fetched again. Until the transfer is terminal,
`query_load_result` returns `None` and continues polling the read. In the normal
load-result path, the prefetch controller therefore does not release destination
buffers or send the peer unlock RPC while the transfer is active.
Polling through completion also lets the transfer channel release its handle.

The load deadline does not cancel I/O or bound resource lifetime. A permanently
stalled transfer remains pending; releasing its buffers safely requires
confirmed transport termination. Independent TTL and shutdown policies are
unchanged. Reporting failure early would let a late read write into a buffer
reassigned to another key.

On the owner, `P2P_LOOKUP_AND_LOCK` drops finished lookups older than
`--p2p-lookup-timeout` that the requester never queried, scanning at most once
per timeout. Their read locks are
left to expire by TTL; a late query gets `None` and counts as a miss.

## Store

### Writes disabled (default)

The store controller tracks every task it submits — and the L1 read locks it
reserved for that task — until the result is popped. So rather than drop
store tasks, `submit_store_task` records a 0-byte success and signals the
store fd immediately, and `pop_completed_store_tasks` returns it; the
controller then finalizes its bookkeeping and releases the read locks instead
of leaking them. No data is moved and no bytes are accounted.

### Writes enabled (`write_enabled`)

A store pushes the objects into the peer's L1, which makes the peer an
extension of this server's DRAM (for example a headless peer with no
inference engine). It takes two RPCs around one RDMA write:

```
requester (P2PL2Adapter)                      peer (P2PController + L1)
  submit_store_task
    P2P_RESERVE_WRITE(keys, layout)  ───────►  reserve one buffer per key
                                     ◄───────  one address per key
  transfer-channel submit_write      ═══════►  bytes land in the buffers
    P2P_FINISH_WRITE(keys, written)  ───────►  commit written, drop the rest
                                     ◄───────  one committed flag per key
  pop_completed_store_tasks → result
```

- `P2P_RESERVE_WRITE` returns exactly one address per key, in request order.
  A valid address is a buffer of the layout's size, invisible to readers
  until committed. An invalid address means nothing was reserved for that
  key: the peer's L1 is out of memory, the key is already resident there, or
  another peer write of the key is in progress.
- `P2P_FINISH_WRITE` takes the keys that got a valid address with one
  `succeeded` flag each. `True` commits the buffer, `False` discards it. It
  returns one `committed` flag per key, in request order: `True` only when the
  key was committed and is therefore resident on the peer, so always `False`
  for a key passed with `False`, and `False` for a key whose reservation was
  gone.

**Nothing waits.** The store controller's loop is one thread shared by every
L2 adapter, so a slow or dead peer must not stall stores to other tiers.
`submit_store_task` only sends the reserve request and
`pop_completed_store_tasks` only queries state. What moves a store from one
step to the next:

```
RESERVING ──► WRITING ──► COMMITTING ──► success
    │            │             │
    └────────────┴─────────────┴──────► failure
```

| Step done | Noticed by | What happens |
|---|---|---|
| Reservation answered | the request's done-callback (`MessagingFuture.add_done_callback`), on the request client's receive thread | the transfer is submitted right there |
| Transfer finished | `pop_completed_store_tasks` on a store-fd pulse (`query_status`; the transfer channel has no completion callback) | the commit request is sent |
| Commit answered | the request's done-callback | it signals the store fd, so the store loop pops the result at once |
| Deadline passed | `pop_completed_store_tasks` on a pulse | the store fails |

Because the reservation's callback runs on the receive thread while the store
loop polls, the store bookkeeping is guarded by one lock. A callback for a
store that already failed (deadline, close) does nothing.

Why callbacks and not polling or waiting, measured across two nodes with
about 1 GB per store task over ZMQ (median per-task latency):

| Object size × count | Poll every step on the 5 ms pulse | Wait for the reservation in `submit_store_task` | Callbacks |
|---|---|---|---|
| 64 MiB × 16 | 20.1 ms | 15.1 ms | 10.0 ms |
| 10 MiB × 100 | 19.7 ms | 14.6 ms | 14.5 ms |
| 1 MiB × 1000 | 59.3 ms | 65.5 ms | 55.0 ms |

Waiting in `submit_store_task` blocks the store loop for the RPC round trip,
which grows with the task (about 0.6 ms for 16 keys, 2.2 ms for 100, 18 ms
for 1000) and lasts the full wait timeout when the peer is down. With
callbacks `submit_store_task` returns in under 0.1 ms in every case; the
receive thread is busy for 0.25 ms (16 keys) to 2.4 ms (1000 keys) while it
submits the transfer.

**Success means every key of the task is resident on the peer.** A store
policy may delete the local copies after a successful store, so a task where
the peer reserved, received or committed only some keys is a failure
(`L2StoreResult(False, 0)`), and the local copies stay. The peer still keeps
the keys that were written and committed.

| Failure | Requester | Peer |
|---|---|---|
| Peer L1 full | invalid addresses, task fails | nothing reserved; counted in `l1.allocation.failed` with `during="p2p_store"` |
| Key already resident or being written on the peer | invalid address, task fails | untouched |
| Reserve request lost, rejected, or peer too old to know it | task fails on the request failure or at the deadline | whatever it reserved expires by the L1 write TTL |
| RDMA write fails for some keys | sends the mask, task fails | commits the written keys, frees the rest |
| Transfer not finished at the deadline | tells the peer to discard, task fails | frees the buffers |
| Requester dies mid-store | — | reservation expires by the L1 write TTL; eviction or the next writer of the key takes the buffer |
| Reservation gone before the commit | `committed=False`, task fails | nothing admitted |

One deadline, `store_timeout_s`, covers the whole task from submit. It must
stay below the peer's L1 write TTL (`--l1-write-ttl-seconds`, default 600 s),
or the peer may hand a buffer to another writer while this one still writes.

The peer keeps no state for a write: reserve, commit and discard are single
L1 calls under the writer tag `"p2p"`, and an unfinished reservation is
reclaimed by the L1 write TTL like any other abandoned write. A committed
object is an ordinary L1 object on the peer, so the peer's own store
controller offers it to the peer's L2 adapters.

Enable writes on only one side of a pair. With both sides writing, an object
one side stored and then dropped locally can be stored back by the other.

Known limitation: the transfer channel has no cancel, so a write still in
flight at the deadline can land after the peer freed the buffer. The load
path has the same exposure for the local buffer.

### No eviction

The adapter is constructed with `max_capacity_bytes=0`, so
`supports_global_eviction` is `False`, `delete` is the inherited no-op, and
`get_usage` reports the (empty) base counters. Stores are not accounted
either: the adapter never learns when the peer evicts an object.

## Configuration

The adapter is built through the standard factory (`create_l2_adapter`), so the
storage manager owns its lifecycle. The config carries the peer's two URLs:

```json
{
  "type": "p2p",
  "peer_mq_server_url": "tcp://peer-host:5555",
  "peer_transfer_channel_server_url": "peer-host:7600",
  "lookup_timeout_s": 10.0,
  "load_timeout_s": 10.0,
  "write_enabled": false,
  "store_timeout_s": 10.0
}
```

`write_enabled` and `store_timeout_s` are optional and default to the values
shown. Adapters the P2P controller creates for discovered peers use the
defaults, so they do not write.

## close()

`close()` calls `_unregister_peer_region()` first; for `P2PL2Adapter` that
removes the transfer-channel client. If it succeeds, `close()` marks the
adapter closed, unregisters the pulsed fds from the periodic notifier, and
closes the request client and the three event fds. Later calls are no-ops.
Stores still in flight are dropped; their reservations on the peer expire by
its L1 write TTL.

If `_unregister_peer_region()` raises, the adapter stays open with its
request client and fds, and `close()` can be called again.
`StorageManager.delete_l2_adapter` propagates the error and keeps the adapter
registered but inactive: it is excluded from `l2_adapters()`, status,
touches and the capacity topology, but still counts as the owner of any
device range it maps. Calling `delete_l2_adapter` again skips the drain and
retries the close.

The P2P controller logs a failed removal and does not retry;
`StorageManager.close()` closes every registered adapter again at shutdown;
if any of them fails, it leaves the L1 managers open, because that adapter
may still access L1 buffers.
Region access shared per peer, such as a transfer-channel client keyed by the
peer's URL, must not be re-acquired for that peer while a close-failed
adapter still holds it.
