# Maru CXL Shared L1 Design

This document describes the `MaruL1Manager` L1 backend for LMCache MP mode: how it
stores the L1 tier in a cross-node CXL shared pool, the components involved,
the control-plane RPC flows, and the state-machine invariants it maintains.

## Overview

Maru replaces MP mode's L1 tier storage — CPU DRAM → a cross-node **CXL shared
pool** reached through the external [maru](https://github.com/xcena-dev/maru)
runtime. Multiple LMCache instances on separate nodes `mmap` the same physical
pool for zero-copy reads. A **single** `MaruServer` — one per shared CXL pool — is
the directory: it maps each key to its `(region, offset)` and holds the cross-node
`pin_count`, and every node sharing the pool connects to it.

Control stays with LMCache and only the storage medium changes: the stock L1↔L2
tiering (write-through, discard-evict, promote-on-miss), the controllers, the L2
adapters, and eviction all run unchanged on top. `MaruL1Manager` executes those
decisions against the MaruServer directory and the CXL allocator; MaruServer itself
is passive (a directory + region broker with no policy).

```
StoreController / PrefetchController / L1EvictionController
                    │  (L1ManagerInterface only)
                    ▼
              MaruL1Manager
  RPC control surface + _pending_read/_pending_write staging + TTL sweeper
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  MaruMemoryAllocator      MaruHandler (RPC)
  (CxlMemoryAdapter)             │
        │                       ▼
        │                  MaruServer  (passive directory)
        │            key → (region, offset), kv_ref_count,
        │            cross-node pin_count
        ▼                       │
   CXL shared pool  ◄───────────┘
   (mmap, zero-copy views)
```

## Goals

- Store the MP L1 tier in a shared CXL pool so multiple instances hit the same KV
  cache with zero-copy reads.
- Keep all tiering control (controllers, L2 cascade, eviction) with the stock
  stack; swap only the L1 medium.
- Keep the L1 control-state-machine core (`l1_manager.py`, `l1_memory_manager.py`,
  `internal_api.py`) and all controller/L2 logic byte-identical to `dev`.
- Isolate every Maru concern behind one class and one Protocol, so the stack does
  not know which L1 backend it runs on.

## Key Design Choice

Maru is integrated as a **sibling** L1 manager — `MaruL1Manager` — beside the
existing `L1Manager`, rather than by modifying it.

Maru is a **shared** tier: LMCache instances on separate nodes read and write the
same CXL pool, so two parts of the L1 control state can no longer be local:

- **Membership** — whether a key is resident is a property of the shared pool, not
  of any one node. A node that never wrote key K must still find it if a peer did.
- **Protection (pins)** — a node must not evict a page another node is still
  reading, which needs a reference count visible across nodes.

Both have to be authoritative in the shared directory (MaruServer) and reached over
RPC; a local dictionary and local locks cannot express them. So Maru needs its own
manager reimplementing the L1 control surface against MaruServer, while
`MaruMemoryAllocator` supplies the CXL medium. Crucially this moves only *where the
L1 state lives*, not the *decisions* — when to evict, pin, write through, or promote
all stay with the stock controllers, which see only the `L1ManagerInterface` and
never learn which manager backs it.

The sibling form is a deliberate first step. Because Maru is external storage, it
keeps LMCache's core control logic untouched — `L1Manager`, the controllers, and the
tiering stack are all unchanged, and `MaruL1Manager` just satisfies the same
interface alongside them. The trade-off is that it reimplements the entire
L1-manager surface; once the shared-L1 approach is validated and accepted upstream, a
cleaner follow-up is to lift just the state-authority difference into a small
`L1StateBackend` interface inside the existing manager, so a shared tier like Maru
becomes a compact backend instead of a full sibling.

## Components

### `MaruL1Manager` (`maru_l1_manager.py`)

The sibling L1 manager. Reimplements the `L1ManagerInterface` control surface over
MaruServer RPCs and owns:

- `_pending_write: dict[ObjectKey, _PendingWrite]` — pages reserved between
  `reserve_write` and `finish_write` (the CXL page, `is_temporary`, a write-TTL
  deadline).
- `_pending_read: dict[ObjectKey, _PendingRead]` — staged reads between
  `reserve_read` and the last `finish_read` (the zero-copy `MemoryObj`, a
  `refcount`, the real server-pin count `pinned`, `is_temporary`, a read-TTL
  deadline).
- The `MaruMemoryAllocator` (as `self._allocator`), the registered listeners, and a
  daemon TTL sweeper thread.

Membership and pins live remotely (MaruServer); these local dicts hold only
in-flight/staged state. A single non-reentrant lock serializes every public method.

### `L1ManagerInterface` (`l1_protocol.py`)

A structural `typing.Protocol` (`@runtime_checkable`). Both `L1Manager` and
`MaruL1Manager` satisfy it without inheritance; each method's docstring is the
behavioral contract (listener firing, PINNED retry, `extra_count` balance).
`StorageManager` selects the implementation in one line:

```python
self._l1_manager: L1ManagerInterface = (
    MaruL1Manager(cfg) if maru_config else L1Manager(cfg)
)
```

The stock controllers call only this surface (no private access), so widening their
`l1_manager` parameter type from concrete `L1Manager` to `L1ManagerInterface`
(runtime-identical) lets the whole stack run over either implementation.

### `MaruMemoryAllocator` (`memory_manager/maru_memory_allocator.py`)

The CXL medium allocator, alongside the other media allocators — a thin wrapper over
the external `maru_lmcache.CxlMemoryAdapter`. Two-phase init: the pool is typed by
the KV layout, so the `MaruHandler` and `CxlMemoryAdapter` are built on the first
`init_layout` (once at startup, before any store/retrieve).
`allocate`/`batched_allocate` hand out CXL pages and return `None` on OOM. Pages are
reclaimed through LMCache's eviction (executed as the manager's `delete`), not the
allocator, so `free` is a no-op.

## Operation Flow

All three flows run on the stock controllers; `MaruL1Manager` only implements the
`L1ManagerInterface` calls they make.

### Store (write-through)

```
StoreController
  ├─ reserve_write(keys)   allocate a CXL page, stage in _pending_write
  ├─ (D2H copy into the page)
  ├─ finish_write(keys)    create_store_handle + batch_store (register in directory)
  │                        → fire on_l1_keys_write_finished
  └─ StoreController wakes, unsafe_read()s the zero-copy MemoryObj, copies it to L2
```

A key another instance already registered is dup-skipped by the server and its page
auto-freed.

### Retrieve (hit, and miss → promote)

```
StorageManager.lookup
  ├─ reserve_read(keys)    batch_pin + batch_retrieve + get_by_location (zero-copy)
  │                        stage in _pending_read → fire on_l1_keys_reserved_read
  ├─ unsafe_read(keys)     return the staged MemoryObj (no new pin), H2D copy to GPU
  └─ finish_read(keys)     batch_unpin the pins; drop the entry at refcount 0

miss → PrefetchController promote:
  reserve_write → (load L2 bytes into the page) → finish_write_and_reserve_read
     temporary (default):  private page, no batch_store/pin, served once then freed
     retained (hot-cache): batch_store, then re-resolve the authoritative page
```

### Eviction (watermark → delete → PINNED retry)

```
L1EvictionController (each cycle)
  ├─ get_memory_usage()    (used, total) over the CXL pool
  ├─ if over watermark: the LRU policy picks victims, filtered by is_key_evictable
  └─ delete(victims)       directory delete; MaruServer refuses PINNED keys (another
                           node is reading). Only truly-deleted keys fire
                           on_l1_keys_deleted_by_manager; PINNED keys stay in the
                           policy and retry next cycle.
```

Discard-on-evict is safe because write-through already placed the data in L2.

## State-Machine Invariants

1. **A key is in at most one of `_pending_write` / `_pending_read`**, mirroring
   stock's per-key write/read lock exclusion. `reserve_write` and
   `finish_write_and_reserve_read` reject a key already staged the other way, and
   `reserve_read` excludes mid-write keys from its pin/stage step — a peer may have
   registered the same key, so the pin would otherwise succeed and double-stage,
   stranding the in-flight write.

2. **`pinned` tracks real server pins separately from `refcount`**
   (`0 <= pinned <= refcount`); release paths unpin `pinned`, not `refcount`. A pure
   temporary stage has `pinned == 0`; a temporary that absorbs an overlapping
   `reserve_read`'s pins records them so they are released, not leaked. `finish_read`
   never releases more than it holds.

3. **temporary vs retained promote.** The default prefetch policy marks every
   promote temporary — private staging, never registered, discarded after one read
   (the shared pool is populated only by store write-through). Only a hot-cache
   policy marks a promote retained and registers it in the directory.

4. **PINNED cross-node retry.** MaruServer refuses to delete a key with
   `pin_count > 0`; `delete` fires the deleted-listener only for keys it actually
   removed, so a refused key stays in the eviction policy and retries. This relies on
   the stock LRU contract that a key leaves the policy only via `on_keys_removed`.

## Orphan Reclaim and Crash Recovery

A reserve→finish flow that stalls leaves an orphan the pending dicts cannot
distinguish from an in-progress one, so time is the only abandonment signal (reusing
the stock `write_ttl`/`read_ttl` values). A daemon sweeper reclaims both under the
lock: an expired `_pending_write` page returns to the owner's free-list
(`abort_alloc`); an expired `_pending_read` unpins its remaining refcount. A late
`finish_read`/`unsafe_read` then sees `KEY_NOT_EXIST` and recomputes — the same path
as a stock TTL expiry.

A full client crash (the MP-server process dies, so the local sweeper cannot run) is
out of scope for `MaruL1Manager` and relies on maru-side mechanisms: read pins are
released when the maru side drops a disconnected client's pin set, and an in-flight
write page is covered by region owner-release. These are premises of this design,
not guarantees provided here.

## Configuration

Maru L1 is enabled via CLI flags that build a `MaruL1Config` (mutually exclusive
with the pinned-DRAM / DevDax / GDS tiers):

- `--maru-server-url` — MaruServer endpoint (`maru://host:port`, rewritten to
  `tcp://`).
- `--maru-pool-size-gb` — CXL pool size to request (> 0). A pool is a single region
  that must fit within one CXL device's free space (a region cannot span devices);
  with `auto_expand` the pool then grows across devices by adding regions.
- `--maru-instance-id` — stable client id for ownership tracking (auto-generated if
  unset).
- `--maru-auto-expand` / `--no-maru-auto-expand` — whether the owned pool auto-expands
  into free CXL when it fills (default on).

`MaruL1Config` fields:

| Field | Default | Purpose |
|---|---|---|
| `server_url` | (required) | MaruServer endpoint. |
| `pool_size_bytes` | (required) | CXL pool size requested from MaruServer. |
| `instance_id` | auto | Stable client id for ownership tracking. |
| `timeout_ms` | 5000 | RPC timeout. |
| `use_async_rpc` | True | Use the async RPC path. |
| `max_inflight` | 64 | Max concurrent in-flight RPCs. |
| `eager_map` | True | Eagerly mmap peer regions for cross-node zero-copy reads. |
| `auto_expand` | True | Grow the owned pool into free CXL when full; off hard-caps it at `pool_size_bytes`. |

`register_kv_layout` binds the KV layout to the allocator (pool bring-up) on the
first `register_kv_cache`; it is wired only on the `lmcache_driven` transfer path.

### Startup guards

`validate_storage_manager_config` rejects unsupported combinations at startup rather
than failing deep in the request path: maru + gds/devdax (mutual exclusion), maru +
`skip_l1` store policy, maru + registered/RDMA-type L2 (only copy-type L2 is
allowlisted, since `get_l1_memory_desc()` is `None`), maru + p2p, and maru +
engine_driven/auto transfer mode (maru requires
`--supported-transfer-mode lmcache_driven`).

## Capacity and Eviction

`get_memory_usage()` returns `(used, total)` over the CXL pool: `used` is the
owned-pool allocation and `total` is the owned pool plus the CXL device free
(`cxl_pool.free_size` from `get_stats`), so the eviction watermark tracks
whole-device fill rather than only the owned pool — the pool auto-expands into free
CXL before evicting. With `auto_expand` off, `total` is the owned pool alone (which
is hard-capped at `pool_size_bytes`), so eviction engages before it is exhausted.

## Current Limits

- **Copy-type L2 adapters only.** Registered/RDMA-type adapters and p2p are rejected
  at startup; registered-L2 support needs an allocator region accessor + per-region
  descriptor (follow-up).
- **Single model / single object group.** The pool is fixed to the first layout; a
  different or hybrid (`num_object_groups > 1`) layout fails fast. Multiple instances
  of the same model are fine.
- **LRU is a per-instance local view** — fed only by this instance's listener
  events, so other nodes' access recency is not seen. A known approximation for a
  shared cache.
- **Cross-owner reclaim.** A per-key `delete` returns the page only to the owner's
  local free-list; evicting a key another instance owns removes the directory entry
  but not that owner's page. Region-level return is owner-only, outside this critical
  path.

## References

- Implementation: `lmcache/v1/distributed/maru_l1_manager.py`
- Protocol: `lmcache/v1/distributed/l1_protocol.py`
- Allocator: `lmcache/v1/distributed/memory_manager/maru_memory_allocator.py`
- Config: `lmcache/v1/distributed/config.py` (`MaruL1Config`)
- User docs: `docs/source/mp/configuration.rst` (Maru CXL Shared L1 section)
- External runtime: [maru](https://github.com/xcena-dev/maru)
