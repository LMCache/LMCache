# L1ManagerInterface

`L1ManagerInterface` (`lmcache/v1/distributed/internal_api.py`) is the
Protocol of the calls `StorageManager`, the storage controllers and the serde
wrapper make on an L1. They were typed against the concrete `L1Manager`;
typing them against the Protocol lets another L1 implementation serve them
without changes on the caller side.

## Shape

- Members are exactly the calls the callers make today: reserve and finish
  for writes and reads, including the combined `finish_write_and_reserve_read`
  and `finish_write_and_delete`; `unsafe_read`, `delete`, `clear`,
  `touch_keys`, `is_key_evictable`, `register_listener`; the usage, capacity
  and status getters; the Device-DAX hot-plug calls; `owns_device`,
  `memory_region_count`, `memcheck`, `close`. An implementation that cannot
  honour a call answers with the documented `L1Error` codes instead of
  raising, so the callers keep one code path.
- It is `runtime_checkable`; `L1Manager` satisfies it unchanged.
- `next_l1_manager_id()` and `validate_read_locks()` in `l1_manager.py` are
  public so every implementation draws from one id counter (the id is the
  owner tag stamped on memory objects) and clamps read-lock counts the same
  way.
- `StorageManager.l1_memory_desc` raises `ValueError` when the single L1 has
  no registerable buffer; the interface types `get_l1_memory_desc()` as
  `L1MemoryDesc | None`, which the previous unannotated return left implicit.

```text
StorageManager, PrefetchController, StoreController,
L1EvictionController, SerdeL2AdapterWrapper
          |  typed against L1ManagerInterface
   +------+-----------------------+
   |                              |
L1Manager                 another implementation
```

## Tests

`tests/v1/distributed/test_l1_interface.py` runs the contract against
`L1Manager`: write, commit, read and release with listener notifications; an
aborted write is never readable; `finish_write_and_reserve_read` hands back a
locked object; tags are independent writers.
