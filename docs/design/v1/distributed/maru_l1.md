# Maru CXL Shared L1 (`MaruL1Manager`)

Design doc for the MP-mode L1 backend that stores the L1 tier in a
cross-instance **CXL shared pool** instead of local CPU DRAM. Covers the code in
`lmcache/v1/distributed/maru_l1_manager.py`, `l1_protocol.py`, and
`memory_manager/maru_memory_allocator.py`.

This is a review-oriented summary: it states the **contracts and invariants** the
implementation must satisfy so the code can be checked against them. It is not an
exhaustive walkthrough.

## 0. One-line summary

**Maru replaces MP mode's L1 tier storage (CPU DRAM → a cross-instance shared CXL
pool); everything above L1 — L1↔L2 tiering (write-through / discard-evict /
promote-on-miss), the controllers, the L2 adapters, eviction — runs unchanged on
top.** LMCache never uses L1 in isolation, so Maru is not an "L1-only" cache
either.

The whole integration reduces to one decision:

> Isolate all Maru logic in a single class, `MaruL1Manager`, and run the existing
> stack unchanged on top of it through the `L1ManagerInterface` Protocol seam.

## 1. Model — what changes and what does not

### 1.1 Maru = one cross-instance shared L1 tier

- **Shared medium**: a physical pool reached over a CXL switch. Multiple LMCache
  instances `mmap` the same physical memory → **zero-copy**, no P2P transfer.
- **Shared index**: `MaruServer` holds the `key → (region, offset)` directory, a
  per-region `kv_ref_count`, and a cross-node `pin_count`.
- **Access**: a retrieve pins + looks up the key on `MaruServer`, then
  materializes a `MemoryObj` pointing at the resident CXL page (no copy).

### 1.2 Control (LMCache) / Directory (MaruServer) split

- **LMCache decides**: store / evict / promote / pin are decided by the existing
  LMCache controllers.
- **MaruServer is passive**: it executes and records those decisions as a shared
  directory + region broker. It holds no policy.
- LMCache's L1 manager therefore needs Maru RPC hooks for store/retrieve/pin/
  delete, and `MaruL1Manager` is the home of those hooks.

### 1.3 Unchanged (byte-identical to `dev`)

- `L1Manager`, `L1MemoryManager`, `internal_api.py` — zero Maru code.
- `StoreController` / `PrefetchController` / `L1EvictionController` / the L2
  adapters / `L2EvictionController` — logic unchanged; only the `l1_manager`
  parameter type is widened to the Protocol (§3).
- The data plane (`mmap` zero-copy, `metadata.address` bit-pack, `gpu_ops`).

### 1.4 The three layers of MP L1 — reuse / reimplement / replace

The recurring confusion is *which layer* "we keep the control logic" refers to.
MP L1 has three layers and the Maru seam cuts each differently:

| Layer | Components | Stock local state | Role | Maru seam |
|---|---|---|---|---|
| **1 — Controllers / tiering** | `StoreController`, `PrefetchController`, `L1EvictionController`, `L2EvictionController`, L2 adapters, eviction `policy` (LRU) | policy's `OrderedDict` order (fed by listeners) | **decide** write-through / promote / eviction; act only on the L1Manager public surface | **reused** (only the `l1_manager` param type is widened; behavior-neutral) |
| **2 — L1Manager control state machine** | `L1Manager` | `_objects: dict[ObjectKey, L1ObjectState]` with per-key `write_lock`/`read_lock` (TTLLock) | authority for **this instance's** membership + lock state | **reimplemented** as sibling `MaruL1Manager` (MaruServer RPC instead of a local dict) |
| **3 — Memory manager / allocator** | `L1MemoryManager` + `MemoryAllocator` (DRAM / DevDax / GDS) | pinned DRAM/devdax/GDS buffer pool | the medium the KV bytes live in | **replaced** by `MaruMemoryAllocator` (CXL, zero-copy) |

"Control logic = stock" is a statement about **layer 1**, and it holds. But layer
2 cannot be reused as-is, which is why Maru is a **sibling** (layer-2
reimplementation), not merely an allocator swap (layer-3 replacement).

### 1.5 Why an allocator swap (layer-3 only) is not enough

Keeping stock layer 2 (`L1Manager._objects`) and swapping only the allocator
breaks, because `_objects` is a **per-instance local index**:

1. **Membership**: if instance B stores key `K`, this instance's local `_objects`
   never learns about `K`, so `reserve_read(K)` misses — even though `K` is
   physically present in the shared pool. Cross-instance read collapses.
2. **Pin**: stock `read_lock`/`write_lock` are **local** (a C++ atomic TTLLock),
   invisible to other instances. Maru must prevent node A from evicting a page
   node B is reading, which requires a **cross-node shared refcount** (MaruServer
   `pin_count`). A local TTLLock cannot express that.

GDS/DevDax work as allocator swaps because they are **private per-instance**
tiers, so `_objects` is complete and accurate. Maru is a **shared** tier, so the
authority for membership and pins must be **remote** (MaruServer), and layer 2 is
reimplemented over RPC. (The allocator interface has only
`allocate`/`free`/`get_memory_usage`/…; it has no "does K exist?" or "pin K"
control operation, so a layer-3 swap cannot express Maru's needs.)

> "Same `MemoryObj`" means the *buffer type* is shared (so the data plane and
> allocator machinery are reusable); it does **not** mean the membership/pin
> control state may stay local. Those are independent.

### 1.6 Stock local structures → Maru replacements

Layer 2's authority moves from local to remote (MaruServer); only in-flight state
stays local:

| Stock (layer 2, local) | Role | Maru replacement | Where |
|---|---|---|---|
| `_objects` dict | membership authority | MaruServer directory `key→region/offset` + `kv_ref_count` | **remote (shared)** |
| `L1ObjectState.memory_obj` | buffer handle | CXL page `MemoryObj` (`get_by_location`, zero-copy view) | local view |
| `write_lock`/`read_lock` (TTLLock) | local eviction/write protection | cross-node pin via MaruServer `pin_count` (delete refuses with PINNED) | **remote (shared)** |
| (no in-flight-write tracking) | reserve_write → finish_write window | `_pending_write` (mem_obj, `is_temporary`, deadline) + write-TTL | local |
| `L1ObjectState.is_temporary` | promote lifetime | `_PendingRead.is_temporary` (propagated from `_pending_write` by promote) | local |
| (no staged-read tracking) | reserve_read → finish_read window | `_pending_read` (`_PendingRead` refcount + `pinned` + `is_temporary`) + read-TTL | local |
| `L1MemoryManager` (DRAM pool) | medium | `MaruMemoryAllocator` (CXL pool) | local |

**Everything Maru keeps local is in-flight/staged side-channel state; all
authority (membership, pins) is remote.** Layer 1 and the data plane are
untouched. TTL reclaim of the side-channel reuses the stock `write_ttl`/`read_ttl`
config values, but sweeps a monotonic deadline on the pending entries instead of a
TTLLock.

## 2. The seam — sibling + structural Protocol

`StorageManager` picks the L1 manager in one line at construction:

```python
self._l1_manager: L1ManagerInterface = (
    MaruL1Manager(cfg) if maru_config else L1Manager(cfg)
)
```

Both implementations satisfy `L1ManagerInterface`, so the stack above does not
know which one it has.

**Three new files:**

- `l1_protocol.py` — the structural `typing.Protocol` `L1ManagerInterface`
  (`@runtime_checkable`). `L1Manager` and `MaruL1Manager` both satisfy it
  **without inheritance**. Each method's docstring states its behavioral contract
  (when listeners fire, the PINNED retry, `extra_count` balance).
- `maru_l1_manager.py` — `MaruL1Manager`. Implements the whole RPC control
  surface and owns `MaruMemoryAllocator` directly as `self._allocator`. No
  separate dispatch layer.
- `memory_manager/maru_memory_allocator.py` — `MaruMemoryAllocator` (same
  directory as the devdax/gds allocators). Thin wrapper over the external
  `maru_lmcache.CxlMemoryAdapter`; connects the handler and creates the adapter on
  the first `init_layout`. `free` is a no-op (CXL page lifecycle is MaruServer's).

The stock controllers call only the `L1ManagerInterface` surface (no private
access — verified), so widening each controller's `l1_manager` parameter type from
concrete `L1Manager` to `L1ManagerInterface` (runtime-identical) lets the whole
stack run over `MaruL1Manager`. The only remaining Maru-specific concrete
references are the L1-selection branch in `StorageManager` and the maru-only gate
in its `register_kv_layout` wrapper — both legitimately Maru-only.

## 3. What `MaruL1Manager` implements

The `L1ManagerInterface` methods, reimplemented over RPC. "Fire" means invoking
the registered listeners' `on_l1_keys_*` callbacks (the eviction LRU and the store
controller consume these).

| Method | Implemented behavior |
|---|---|
| `reserve_read(keys, extra_count)` | `batch_pin` + `batch_retrieve` + `get_by_location` (zero-copy) → stage in `_pending_read`. Takes `1 + extra_count` server pins per key (MLA + TP>1). A key mid-write on this instance is **excluded** (stays `KEY_NOT_READABLE`, see §4 invariant 1). Fires `on_l1_keys_reserved_read`. |
| `unsafe_read` / `finish_read` | Return the staged object / release `1 + extra_count` holds. Each release drops `refcount` and unpins `min(released, pinned)` server pins (§4 invariant 2). At `refcount == 0` a **temporary** entry frees its private page (`on_l1_keys_deleted_by_manager`); a directory read just drops the entry. |
| `reserve_write` / `finish_write` | Allocate a CXL page + stage in `_pending_write` (`is_temporary` + write-TTL); fire `on_l1_keys_reserved_write`. / `create_store_handle` + `batch_store` (directory register); fire `on_l1_keys_write_finished`. In `mode="new"`, a key already staged locally **or** already in the directory returns `KEY_NOT_WRITABLE` (cross-instance dedup). |
| `finish_write_and_reserve_read` (promote) | Branches on the staged `is_temporary` (§4 invariant 3). **temporary** (the default prefetch policy): no `batch_store`, no pin — the loaded local page moves straight to `_pending_read`; freed at `finish_read == 0`. **retained**: `batch_store`, then re-resolve the authoritative directory page with pins (a dup-skip auto-freed our page). Both fire only `on_l1_keys_finish_write_and_reserve_read` — **never** `on_l1_keys_write_finished` (that would make the store controller re-store the promoted key to L2). |
| `delete(keys)` | Directory delete. A local read-hold → `KEY_IS_LOCKED`; a cross-node pinned key → MaruServer refuses with **PINNED**. Only actually-deleted keys fire `on_l1_keys_deleted_by_manager` (§4 invariant 4). |
| `is_key_evictable(key)` | False if locally pinned (`_pending_read`); lock-free. Cross-node pins are re-checked authoritatively by `delete`. |
| `touch_keys` / `register_listener` | `register_listener` only stores the listener; each method fires its own event. `touch_keys` fires `on_l1_keys_accessed`. |
| `get_memory_usage()` | `handler.get_stats` → `(used, total)`. `used` = owned-pool allocated; `total` = owned pool + CXL device free (`cxl_pool.free_size`), so the eviction watermark tracks whole-device fill, not just the owned pool. The last known free is cached to survive a transient `get_stats` RPC failure. Before init, returns `(0, configured pool size)` to avoid a 0-division in the controller watermark. |
| `get_l1_memory_desc()` | `None` — the shared pool has no single registrable region (§5 L2 limit). |
| `register_kv_layout(...)` | Maru-only: binds the KV layout to `MaruMemoryAllocator` (pool bring-up). `num_object_groups > 1` is rejected by the `StorageManager.register_kv_layout` wrapper (§6). |
| `clear` / `memcheck` / `report_status` / `close` | Delegated under the lock (§4). `clear` empties the local side-channel and releases staged pins but does **not** touch shared data (a shared tier must not be wiped by one node). `close` stops the TTL sweeper. |

Eviction and LRU stay with the stock stack: `L1EvictionController` decides, and
recency lives in the eviction **policy** (fed by the listener events).
`MaruL1Manager` keeps no local object/LRU dict.

## 4. Invariants (check the implementation against these)

These are the contracts a reviewer should verify; the two correctness bugs found
in review were both violations of invariants 1 and 2.

1. **A key is in at most one of `_pending_write` / `_pending_read`.** This mirrors
   stock `L1Manager`'s per-key `write_lock`/`read_lock` mutual exclusion (a
   mid-write key is not readable). Enforced in both directions: `reserve_write`
   and `finish_write_and_reserve_read` reject a key already staged the other way,
   and `reserve_read` **excludes** mid-write keys from the pin/stage step (a peer
   may have registered the same key, so the pin would otherwise succeed and double
   stage — stranding the in-flight write).

2. **Pin accounting: `pinned` tracks real server pins separately from
   `refcount`** (`0 <= pinned <= refcount`). Release paths unpin `pinned`, not
   `refcount`. A pure temporary stage has `pinned == 0` (a private local page, no
   server pin); a temporary that absorbs an overlapping `reserve_read`'s pins
   records them in `pinned` so they are released, not leaked. `N` reserves take
   `N` pins and `N` finishes release `N`; `finish_read` never releases more than
   it holds (over-release would corrupt the server `pin_count`).

3. **temporary vs retained promote.** The default prefetch policy marks every
   promote **temporary**: the page is private staging, never registered in the
   directory, and discarded after one read (the shared pool is populated only by
   store write-through). Only a hot-cache policy marks a promote **retained**,
   which registers it in the directory.

4. **PINNED cross-node retry (no controller change).** MaruServer refuses to
   delete any key with `pin_count > 0` (PINNED). `MaruL1Manager.delete` fires
   `on_l1_keys_deleted_by_manager` only for keys it **actually** deleted, so a
   PINNED-refused key stays in the eviction policy and is retried next cycle. This
   relies on the stock LRU contract that a key is removed from the policy only via
   the `on_keys_removed` event, never popped by `get_eviction_actions`.

### Thread safety

A single non-reentrant `threading.Lock` (`_maru_l1_synchronized`) serializes every
public method that touches the side-channel or allocator. The decorator is typed
with `ParamSpec`/`Concatenate` so wrapped signatures survive the Protocol
conformance check.

### TTL sweeper + crash-recovery premises

An entry whose reserve→finish flow is interrupted (an *orphan*) cannot be
distinguished from an in-progress one by inspecting the pending dicts, so **time
(TTL) is the only abandonment signal** (reusing the stock `write_ttl`/`read_ttl`
values). A daemon sweeper thread (scanning under the lock) reclaims both:

- `_pending_write`: on write-TTL expiry, return the page to the owner's local
  free-list (`abort_alloc`) + pop. An unregistered page is invisible to the
  server, so client-side reclaim is the only option.
- `_pending_read`: on read-TTL expiry, `batch_unpin × remaining refcount` + pop.

A late `finish_read`/`unsafe_read` after expiry sees `KEY_NOT_EXIST` → retrieve
returns False → recompute (the same failure path as a stock TTL expiry). Double
unpin is impossible under the serializing lock.

**Premises for a full client crash** (the MP-server process dies, so the local
sweeper cannot run):

- read pins are assumed to be released by the **maru side** tracking each client's
  pin set and dropping them on disconnect;
- an in-flight write page is assumed to be covered by **region owner-release**.

These two mechanisms live outside `MaruL1Manager` (on the Maru/MaruServer side)
and are premises of this design, not guarantees provided by this PR.

## 5. Scope and known limits

- **Copy-type L2 adapters only.** `get_l1_memory_desc()` returns `None`, so
  copy-type adapters (aerospike, dax, fs, fs_native, hfbucket, mock, resp, s3,
  plugin, raw_block, mooncake-TCP) work fully, while registered/RDMA-type adapters
  (nixl_store, nixl_store_dynamic, mooncake-RDMA) and p2p are **rejected at
  startup** (§6 guards; an allowlist, so a new registered adapter fails safe).
  Registered-L2 support needs an allocator region accessor + per-region descriptor
  and is follow-up work.
- **Single model / single object group.** `MaruMemoryAllocator` fixes the pool to
  a single layout on the first `init_layout`, so a different model (different
  shapes/dtypes/fmt/chunk) or a hybrid model (`num_object_groups > 1`) fails fast
  with a `ValueError`. Multiple instances of the *same* model are fine. Mixed
  layouts are follow-up work.
- **Single-device pool bound.** A pool is one contiguous allocation within one DAX
  device, so `pool_size_bytes` cannot exceed a single CXL device's capacity.
- **LRU is a per-instance local view.** The policy is fed only by this instance's
  listener events, so it does not see other nodes' access recency. A known
  approximation for a shared cache.
- **Cross-owner reclaim.** A per-key `delete` returns the page only to the
  *owner's* local free-list. Evicting a key another instance owns removes the
  directory entry but does not reclaim that owner's page. Region-level return to
  the shared pool is an owner-only concern outside this PR's critical path. (This
  and the expand-vs-evict watermark interaction are the two multi-instance
  consistency issues tracked for a separate team review.)

## 6. File changes (this PR)

**New**

- `l1_protocol.py` — `L1ManagerInterface` + the `L1OperationResult` alias.
- `maru_l1_manager.py` — `MaruL1Manager` (~1050 lines).
- `memory_manager/maru_memory_allocator.py` — `MaruMemoryAllocator`.
- Tests under `tests/v1/distributed/` (fake-maru harness + manager unit +
  conformance + config guard + control integration).

**Modified (all additive / behavior-neutral)**

- `store_controller.py` / `prefetch_controller.py` / `eviction_controller.py` —
  `l1_manager` parameter type widened to `L1ManagerInterface`.
- `storage_manager.py` — the L1-selection branch (`MaruL1Manager` when
  `maru_config` is set) + the maru-only `register_kv_layout` wrapper.
- `config.py` — the `maru_config` field, CLI, and validation (§ startup guards).
- `lmcache_driven_transfer.py` — the `register_kv_layout` engine hook (a no-op for
  the default backend), wired in `register_kv_cache` and wrapped in try/except so a
  failure closes the cache context.

**Unchanged (byte-identical to `dev`)**

- `l1_manager.py`, `l1_memory_manager.py`, `internal_api.py`, all controller
  logic, all L2 adapters.

### Startup guards (`validate_storage_manager_config`)

Unsupported combinations are rejected at startup: maru + gds/devdax (mutual
exclusion), maru + `skip_l1` store policy, maru + registered/RDMA-type L2, maru +
p2p, and maru + engine_driven/auto transfer mode (maru requires
`--supported-transfer-mode lmcache_driven`). Without these guards the failures are
silent or surface as an `AttributeError` deep in the request path.

## 7. Status

Implemented on branch `maru-mp-l1`: the sibling `MaruL1Manager` + `l1_protocol.py`
+ `MaruMemoryAllocator`, with `l1_manager.py` / `l1_memory_manager.py` /
`internal_api.py` kept byte-identical to `dev`. The full Maru test suite passes on
GPU (`CUDA_VISIBLE_DEVICES=1`).

A later phase may extract an `L1StateBackend` seam so Maru becomes a small backend
rather than a full sibling; that is separate follow-up work after this sibling
merges.
