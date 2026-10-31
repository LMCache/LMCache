# L1 manager types

`lmcache/v1/distributed/l1_manager/` holds one object-lifecycle state machine
(`BaseL1Manager`, package-private) and one subclass per memory medium. Every
subclass implements `L1ManagerInterface`
([../l1_manager_interface.md](../l1_manager_interface.md)); the write path
they share is in [staging-and-write-tags.md](staging-and-write-tags.md).

| File | Class | Medium | `backend_type` | Memory tags |
|---|---|---|---|---|
| `base.py` | `BaseL1Manager` | (abstract) | abstract | abstract |
| `dram.py` | `DramL1Manager` | pinned host DRAM (`L1MemoryManager`) | `DRAM` | `(DRAM, LOCAL)` |
| `devdax.py` | `DevDaxL1Manager` | Device-DAX arenas (`DevDaxL1MemoryManager`) | `DEVDAX` | `(CXL, LOCAL)` |
| `gds.py` | `GdsL1Manager` | NVMe slab via GPUDirect Storage (`GDSL1MemoryManager`) | `GDS` | `(GDS, LOCAL)` |
| `shadow.py` | `ShadowL1Manager` | none: borrowed views of a peer's bytes (`ShadowL1MemoryManager`) | the paired L1's | the paired L1's segment, `REMOTE` |
| `factory.py` | `create_l1_manager`, `create_l1_managers` | picks the subclass from `L1ManagerConfig` | | |

## One medium per manager

A manager has exactly one medium: `backend_type` (an `L1BackendType`:
`DRAM`, `DEVDAX` or `GDS`) names it, `get_capacity_bytes()` is one number,
and every object the manager reports in an `L1ObjectMeta` carries that
`backend_type`. The former hybrid DRAM + Device-DAX tier (a DRAM pool
spilling into a DAX arena inside one manager) is gone: configure a `DRAM` and
a `DEVDAX` manager and let the write policy place keys. A `dax` L2 adapter
on the Device-DAX L1's device range is a configuration error
(`validate_storage_manager_config`), not an overflow hint. `StorageManager`
sums managers of one medium into one capacity compartment
(`l1_capacity_bytes_by_backend` in `/status`, `SM_CAPACITY_CHANGED`), so the
coordinator's `(tier, backend)` key is unchanged.

## Why a base class plus subclasses

Before this split one `L1Manager` chose its memory manager with an
`if gds / elif devdax / else` chain and then re-derived the medium in
`get_memory_tags`, a per-medium capacity dict, `owns_range`,
`memory_region_count`, the gauge label and a `_require_devdax_memory_manager`
guard, while `L1ManagerInterface` carried the Device-DAX hot-plug calls for
every medium. Each medium now says once what it is:

- `BaseL1Manager.__init__(config, memory_manager)` takes the allocator facade
  (`L1MemoryManagerProtocol`) from the subclass and owns keys, locks, staging,
  listeners, event-bus publishing, `report_status`, `memcheck` and the three
  OTel gauges. `get_capacity_bytes()` returns the boot-configured size
  (`get_configured_capacity_bytes(config)`); `get_l1_memory_desc()` delegates
  to the memory manager. The gauge class state is addressed as
  `BaseL1Manager.<attr>` so one registration per process observes every
  subclass instance.
- Subclasses implement the `backend_type` property and `get_memory_tags()`,
  build their memory manager, and add what only they can do:
  - `DevDaxL1Manager` owns the hot-plug surface (`add_devdax_device`,
    `remove_devdax_device`, `get_devdax_arena_statuses`,
    `get_devdax_arena_status`, `owns_range`, `memory_region_count`),
    overrides `get_capacity_bytes()` with the live sum of ACTIVE arenas and
    `get_l1_memory_desc()` with the primary-arena warning; see
    [../memory_manager/device-dax-l1.md](../memory_manager/device-dax-l1.md).
  - `GdsL1Manager` initializes the process-wide GDS context for its slab in
    its constructor and closes it in `close()`; its `get_l1_memory_desc()` is
    `None` (slab offsets, no host buffer).

Per manager, `report_status()` reports `"backend"` (the `backend_type` wire
value: `dram`, `devdax`, `gds`) and `memory_configured_bytes`
(`get_capacity_bytes()`); there is no per-backend dict inside one manager.

## Factory

`create_l1_manager(config)` selects GDS when `config.gds_l1_config` is set,
Device-DAX when `memory_config.devdax_path` is set, otherwise DRAM, and
rejects a config that sets both. `create_l1_managers(configs)` builds the
list in placement order, refuses a Device-DAX config whose range an earlier
`DevDaxL1Manager.owns_range` already maps (checked against the live mapping
so device aliases resolve the way the allocator sees them), and closes the
already-created managers on any failure. `StorageManager.__init__` is the
only production caller; tests build single managers with
`create_l1_manager(config)`.

## Write placement

```text
StorageManager.reserve_write(keys, layout)
    |
    | write_policy.select_write_targets(keys, [L1ManagerInterface...])
    v
{ l1_manager_id -> [keys] }        one target per key, no retry elsewhere
    |  validated first: ids registered, every key exactly once
    v
L1[id].reserve_write(keys, ...)    per id; OUT_OF_MEMORY stays the key's result
```

`WritePolicy` (`storage_controllers/write_policy.py`) has the same shape as
`StorePolicy`: an ABC plus a registry (`register_write_policy`,
`create_write_policy`), selected by `StorageManagerConfig.write_policy`
(default `"default"`; there is no CLI flag yet). `DefaultWritePolicy` sends
every key to the first L1 in placement order. The policy sees the live L1
manager instances, so it can place by tags, usage or configuration; its
answer must place every key exactly once, which `reserve_write` checks before
touching any L1. Objects carry their owner id, so completion
(`finish_write(keys, l1_owners)`, owners from `l1_owners_of(objects)`)
never re-runs the policy; see [../storage_manager.md](../storage_manager.md).

## Shadow L1

The read path (lookup, read locks, `finish_read`, eviction, events) only
serves objects an L1 manager indexes, so data an L2 adapter can lend instead
of copying -- bytes a peer owns that this host's GPU can read directly -- needs
an L1 that indexes them without owning them. `ShadowL1Manager` is that L1:

```text
L2 adapter that lends views                 ShadowL1Manager (metadata only)
  take_borrowed_objects(task, keys) ──────►   admit({key: view}, read_locks)
  -> BorrowedTensorMemoryObj per key           resident + read-locked, 0 bytes
       is_valid(): lease still holds            ...ordinary readers...
       release():  end lease  ◄──────────────  free (last finish_read, delete,
                                                eviction, release_all)
```

- **No payload.** `ShadowL1MemoryManager` refuses `allocate`, reports
  `(0, 0)` usage and no `L1MemoryDesc`; `free(obj)` calls `obj.release()`.
  Write placement never targets a shadow L1 (`StorageManager` hands the write
  policy only the other managers).
- **Admission.** `admit` replaces `reserve_write` + `finish_write`: one locked
  pass makes each view resident (temporary) and read-locked. A view of a key
  that is already resident is released and the resident one is read-locked; a
  view whose lease already ended is refused (`KEY_IN_WRONG_STATE`), as is a
  resident view whose lease ended while readers still hold it
  (`KEY_NOT_READABLE`).
- **Expiry.** A lease can end on its own (e.g. a peer's reservation TTL).
  `BaseL1Manager` treats an object whose `is_valid()` is False as unreadable
  in `reserve_read` and `unsafe_read`, the only lease-related rule in the base
  class. A reader that already holds an expired view keeps it until its
  `finish_read`, so a copy in flight is never torn down; `reap_expired()`
  releases expired views no reader holds.
- **Tags.** `ShadowL1Manager.next_to(local)` pairs the shadow with a local L1
  and reports that L1's backend and segment with `REMOTE` residency, so an
  adapter binds to it through `get_wanted_l1_memory_tags()` and P2P serving
  (which resolves a `LOCAL` L1) never re-exports a borrowed view.
- **Who borrows.** The prefetch controller borrows exactly when an L2
  adapter's L1 affinity is a shadow L1: it calls `take_borrowed_objects` and
  `admit` instead of reserving L1 buffers and planning a copy load.

The first lender is the shared-pool CXL peer adapter
([../../cxl-kv-sharing.md](../../cxl-kv-sharing.md)): `create_l1_managers`
appends a shadow L1 after each Device-DAX L1 that owns a CXL slab.

## Addressing one L1

`StorageManager.get_l1_manager_by_tags(tags)` returns the one manager whose
`get_memory_tags()` equals `tags` (`ValueError` for zero or several). The P2P
controller resolves `(DRAM, LOCAL)` this way and registers that manager's
`get_l1_memory_desc()` with the transfer channel; the `/reconfigure/dax/l1/*`
routes resolve `(CXL, LOCAL)` and narrow it to `DevDaxL1Manager` (`409` when
there is no Device-DAX L1 or more than one). See
[../memory_tags.md](../memory_tags.md).

## Tests

`tests/v1/distributed/test_l1_manager_factory.py` (medium per config,
placement order, GDS + DAX rejected, overlapping DAX ranges rejected with the
earlier managers closed), `test_l1_interface.py` (contract against
`DramL1Manager`; the medium-specific names are absent from the interface),
`test_memory_tags.py` (tags per medium), `test_l1_l2_state_metrics.py`
(gauge labels), `test_multi_l1.py` and `test_write_policy.py` (placement,
registry), `test_devdax_l1_allocator.py` (primary-region warning, single-region
adapter checks, hot-plug through the `/reconfigure/dax/l1/*` routes),
`test_shadow_l1_manager.py` (shadow L1 admission, expiry, deletion and
release with fake leases) and `test_prefetch_controller.py::TestBorrowing`
(a lending mock adapter through the prefetch controller).
