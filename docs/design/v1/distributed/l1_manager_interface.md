# L1ManagerInterface

`L1ManagerInterface` (`lmcache/v1/distributed/internal_api.py`) is the
Protocol of the calls `StorageManager`, the storage controllers and the serde
wrapper make on an L1. It is the *generic* L1 contract: every member makes
sense for every memory medium. Medium-specific operations live on the
concrete classes in `lmcache/v1/distributed/l1_manager/` (see
[l1_manager/l1-manager-types.md](l1_manager/l1-manager-types.md)).

## Shape

- Members are exactly the calls the callers make today: reserve and finish
  for writes and reads, including the combined `finish_write_and_reserve_read`
  and `finish_write_and_delete`; `unsafe_read`, `delete`, `clear`,
  `touch_keys`, `is_key_evictable`, `register_listener`; `get_memory_usage`,
  `get_staging_memory_usage`, `get_capacity_bytes`, `get_l1_memory_desc`,
  `get_memory_tags`, `report_status`, `memcheck`, `close`; and the
  `l1_manager_id`, `config` and `backend_type` (`L1BackendType`) properties.
  An implementation that cannot honour a call answers with the documented
  `L1Error` codes instead of raising, so the callers keep one code path.
- **Not** members: Device-DAX hot-plug (`add_devdax_device`,
  `remove_devdax_device`, `get_devdax_arena_status(es)`), device ownership
  (`owns_range`) and `memory_region_count`. They belong to `DevDaxL1Manager`.
  `StorageManager` reaches them by resolving the `(CXL, LOCAL)` L1 with
  `get_l1_manager_by_tags` and narrowing it with `isinstance`
  (`_devdax_l1()`); `add_l2_adapter` checks `memory_region_count()` only when
  the affinity L1 is a `DevDaxL1Manager`. `test_l1_interface.py` asserts
  these names are not part of the contract.
- `get_capacity_bytes()` is one number per manager: the boot-configured size
  for DRAM and GDS, the live ACTIVE-arena sum for Device-DAX. `StorageManager`
  sums it per `backend_type` for capacity reporting.
- `get_l1_memory_desc()` is typed `L1MemoryDesc | None`: `GdsL1Manager`
  returns `None` (slab offsets, no host buffer). A `DevDaxL1Manager` with
  arenas added at runtime returns the *primary* arena only and logs a
  warning, because an L2 adapter or transfer channel registers exactly one
  buffer.
- `next_l1_manager_id()` and `validate_read_locks()` in `l1_manager/base.py`
  are public so every implementation draws from one id counter (the id is the
  owner tag stamped on memory objects) and clamps read-lock counts the same
  way.
- Implementations are built by `create_l1_manager(config)` /
  `create_l1_managers(configs)` (`l1_manager/factory.py`), which pick the
  subclass from the config; callers never name a subclass.

```text
StorageManager, PrefetchController, StoreController,
L1EvictionController, SerdeL2AdapterWrapper
          |  typed against L1ManagerInterface
          v
     BaseL1Manager            (state machine, listeners, gauges)
      /     |      \
DramL1Manager DevDaxL1Manager GdsL1Manager
                 + hot-plug, owns_range
```

## Addressing a particular L1

Callers never hold a "the L1" reference. `StorageManager.l1_managers()`
returns every manager in placement order and
`StorageManager.get_l1_manager_by_tags(tags)` the one manager whose
`get_memory_tags()` equal `tags` (`ValueError` for zero or several; see
[memory_tags.md](memory_tags.md)). The P2P controller uses the latter to find
the local DRAM pool whose buffer the transfer channel registers; the
Device-DAX reconfigure routes use it to find the `(CXL, LOCAL)` L1.

## Tests

`tests/v1/distributed/test_l1_interface.py` runs the contract against
`DramL1Manager`: write, commit, read and release with listener notifications;
an aborted write is never readable; `finish_write_and_reserve_read` hands back
a locked object; tags are independent writers; the medium-specific names are
absent from the interface.
