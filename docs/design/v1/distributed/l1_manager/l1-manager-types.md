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
adapter checks, hot-plug through the `/reconfigure/dax/l1/*` routes).
