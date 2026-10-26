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
- Read acquisition methods optionally populate `read_generations` with the
  acquired generation per key. `finish_read` checks that captured map before
  releasing a lock; absent or stale entries cannot release a newer generation.
  Omitting the map retains ordinary key-only completion.
- `pin_read_buffers` validates the captured generation and holds a resident
  source buffer independently of the read TTL. `finish_read_transfer` removes
  that unique transfer tag and releases its original share at most once.
  Normal `delete`, `clear`, eviction and temporary-object cleanup preserve
  pinned buffers. Forced deletion still requires the caller to quiesce I/O.
- A generation identifies the shared lock lifetime, not an individual reader.
  Before pinning, the caller must exclusively claim the original read share;
  it must not release that share separately through `finish_read`. The dense
  MP path uses Session claims so FREE releases only shares not already given
  to a worker. Completion uses its immutable original owner/generation receipt
  even if the Session has been removed.
- It is `runtime_checkable`; `L1Manager` declares it explicitly
  (`class L1Manager(L1ManagerInterface)`) so the contract is visible at the
  implementation site, not only at the call sites.
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
