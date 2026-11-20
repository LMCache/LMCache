# PrefetchController

Design of `lmcache/v1/distributed/storage_controllers/prefetch_controller.py`:
how a prefetch request is planned across L1 and every L2 adapter, how its
locks are tracked, and what the result means.

## Request model

A prefetch request is a **grid**: one row per key group (an object group on
one kv rank) and one column per chunk. `PrefetchTaskSpec.key_groups` supplies
the rows; every row has the same number of keys. Every lookup result, plan and
lock set is a `Bitmap2D` over that grid, and every per-tier map is a
`MapState` (tier index -> `Bitmap2D`). Both live in `storage_controllers/utils.py`.

The request's lock state is exactly three maps, all on `PrefetchKeyState`:

| Map | Index | Meaning |
|---|---|---|
| `l1_locked_keys` | L1 manager | Cells read-locked in that manager (resident, or admitted from L2). |
| `l2_locked_keys` | L2 adapter | Cells that adapter holds locked for this request. |
| `l1_reserved_keys` | L1 manager | Cells with a staging buffer reserved for an L2 load. |

The maps are consistent after every step of the flow. Abort and shutdown
release exactly what the maps record, so any step can fail and the request
leaves nothing behind.

## Invariants

- **Observation is acquisition.** L1 is probed by read-locking it
  (`reserve_read`); L2 is probed by `lookup_and_lock`. A key required for the
  reported hit is lock-held from the moment it is discovered until the request
  finishes, so a concurrent eviction can shrink what gets discovered but never
  invalidate a reported hit.
- **One planning step, one fold.** The policy plans once on the union of the
  L1 locked map and every L2 locked map. The prefix fold runs once, at finish,
  over the final L1 locked map. L1 is never folded on its own and then handed
  to L2 as a remainder.
- **Every planned cell is locked in the tier it is planned from, and no cell
  is planned twice.** This is the policy contract (`PrefetchPolicy.plan_load`).
  The controller releases every lock outside the plan and then sets the locked
  maps equal to the plan.
- **Admission is per adapter.** Each adapter's load result is admitted into L1
  as it arrives; the request does not wait for the slowest adapter before
  making the fast ones' objects resident and read-locked.
- **An adapter stays attached while any request references it**: an in-flight
  lookup or load task, or an entry in `l2_locked_keys`. Draining adapters are
  never planned or looked up; they detach once no request references them.
- **LRU is decoupled from locking.** Taking and releasing read locks never
  refreshes eviction recency. Finish touches the hit keys explicitly
  (`L1Manager.touch_keys`), so locks held while the request decides do not
  pin recency.

## Interface

```python
PrefetchController(l1_managers, l1_manager_descriptors,
                   l2_adapters, adapter_descriptors, policy, max_in_flight=8)

submit_prefetch_request(spec: PrefetchTaskSpec, skip_l2=False) -> PrefetchRequestId
query_prefetch_result(request_id) -> PrefetchResult | None   # consume-once
wait_prefetch_result(request_id, timeout) -> bool            # does not consume
add_adapter(adapter_id, adapter, descriptor) / request_remove_adapter(adapter_id)
start() / stop()
```

`submit_prefetch_request` is thread-safe. When `skip_l2` is set, or no L2
adapter is attached, the request is served from L1 on the calling thread and
its result is queryable when the call returns. Otherwise it is queued for the
background loop and admitted up to `max_in_flight` at a time.

`PrefetchResult` carries `hit_cells` (one bitmap per key group, in
`key_groups` order), split into the disjoint `l1_hit_cells` (L1 already held
them) and `l2_hit_cells` (loaded by this request). Under `LOCK` every hit cell
is read-locked `num_kv_readers` times for the caller.

## Flow

The background thread polls one submission eventfd, one adapter-control
eventfd, and every adapter's lookup and load eventfd. Requests advance through
two phases, `LOOKUP` while any lookup task is outstanding and `PLAN_AND_LOAD`
afterwards.

1. **Lock pass** (`_lock_l1_keys`). `reserve_read` every key of the request in
   every L1 manager; the hits become `l1_locked_keys`. Locks are taken
   regardless of `lock_mode`.
2. **Lookup** (`_start_lookup_phase`). Submit one `lookup_and_lock` task per
   non-draining adapter, over the full key list. As each result arrives
   (`_poll_lookup_results`) it becomes that adapter's row in `l2_locked_keys`.
3. **Plan** (`_plan_load`). Call `policy.plan_load` on the two maps. Release
   every L1 lock and every L2 lock outside the plan, then set both locked maps
   to the plan.
4. **Reserve** (`_transition_to_load_phase`). For each L2 adapter in the plan,
   `reserve_write` staging buffers in its affinity L1 manager under a
   per-request write tag, with retention decided per key by
   `policy.plan_l1_retention` (every key is retained under `NO_LOCK`).
   Reservation failure is **per cell**: `L2_PREFETCH_FAILED` is published with
   reason `l1_oom` or `l1_contended`, the affected L2 locks are returned, and
   the request is re-planned on what was reserved. Staging buffers the replan
   drops are deleted (`finish_write_and_delete`).
5. **Load** (`_submit_load_tasks`). One load task per adapter in the plan,
   carrying the reserved buffers.
6. **Admit** (`_poll_load_results`), per adapter as its result arrives. Loaded
   keys go through `finish_write_and_reserve_read` and join `l1_locked_keys`
   and the request's `l2_loaded_cells`; failed keys are deleted
   (`finish_write_and_delete`, `L2_PREFETCH_FAILED` reason `not_found`); the
   adapter's L2 locks are returned and its entries leave `l2_locked_keys` and
   `l1_reserved_keys`.
7. **Finish** (`_finish_request`), once no load task is outstanding. Fold the
   merged `l1_locked_keys` (`fold_unfold_grouped`, prefix fetching only; `full`
   keeps every locked cell). Release every L1 lock outside the hit, or every L1
   lock under `NO_LOCK`. Touch the hit keys. Publish the `PrefetchResult` with
   the L1/L2 split taken from `l2_loaded_cells`.

Per-segment view of the locks for one key group under `prefix` fetching and
`LOCK` mode. "L1 hit" is what the lock pass found, "planned" is what the
policy assigned to an L2 adapter, "hit" is the final folded prefix:

| Step | in L1 hit | planned from L2 | outside the plan | past the hit |
|---|---|---|---|---|
| lock pass | L1 read lock | - | L1 read lock if resident | L1 read lock if resident |
| lookup | L1 read lock | L2 lock | L2 lock if found | L2 lock if found |
| plan | L1 read lock | L2 lock | released | released |
| reserve | L1 read lock | L2 lock + staging buffer (`l1_oom` / `l1_contended` cells: L2 lock returned, replanned) | - | - |
| admit | L1 read lock | loaded: L1 read lock, L2 lock returned; failed: buffer deleted | - | - |
| finish | L1 read lock if in hit, else released | L1 read lock if in hit, else released | - | - |

Under `NO_LOCK` the finish row releases every L1 lock; loaded objects stay
resident and evictable. Under `full` fetching there is no fold: every locked
cell is a hit cell.

**Errors.** Any exception while advancing a request aborts it
(`_abort_request`): the three maps are walked to return L2 locks, delete
staging buffers and release L1 read locks, an all-zero result is published,
and the request is retired. `stop()` does the same for every in-flight
request. Outstanding adapter tasks are left to complete on their own.

## Policy

`PrefetchPolicy` (`prefetch_policy.py`) is **pure**: `plan_load` sees the two
locked maps plus the L1 manager and L2 adapter descriptors and returns a
`PrefetchPlan` (`l1_planned_keys`, `l2_planned_keys`); `plan_l1_retention`
returns one flag per key loaded from a given L2 adapter into a given L1
manager. Neither raises; invalid input yields an empty plan.

- `"default"`: serve each needed cell from the lowest-indexed tier that holds
  it (L1 managers before L2 adapters); nothing loaded from L2 is retained.
  Temporary objects are deleted once the retrieve that requested them
  finishes.
- `"retain"`: same plan; everything loaded from L2 stays in L1.

`"prefix"` fetching plans the longest prefix every row can serve under its
window (`fold_unfold_grouped`). `"full"` fetching plans every found cell and
refuses sliding-window rows (empty plan, error log).

Policies self-register with `register_prefetch_policy(name, cls)` and are
selected with `--l2-prefetch-policy`.

## Events

| Event | When |
|---|---|
| `L2_PREFETCH_LOOKUP_SUBMITTED` | Lookup tasks submitted (once per request). |
| `L2_PREFETCH_LOAD_SUBMITTED` | Load tasks submitted (once per request). |
| `L2_LOAD_TASK_SUBMITTED` / `L2_LOAD_TASK_COMPLETED` | Per adapter, per request. |
| `L2_PREFETCH_LOAD_COMPLETED` | Per adapter, as its load is admitted. |
| `L2_PREFETCH_FAILED` | `l1_oom` / `l1_contended` from the reserve step, `not_found` from admission. |

Gauges: `lmcache_mp.num_inflight_l2_loads`,
`lmcache_mp.inflight_load_memory_usage_bytes` (per adapter) and
`lmcache_mp.l2_prefetch_adapters` (by `state`: active or draining).

## Known limitations

- **One L1 manager.** `_get_l2_affinity_manager` returns the only manager;
  `L1ManagerDescriptor` is the hook for describing managers to the policy once
  L1-L2 affinity is configurable.
- **Static admission.** `max_in_flight` is a constructor constant rather than
  a function of L1 memory committed to in-flight loads.
- **Full key list to L2.** Lookups are submitted with every key, not only the
  L1 misses, because the adapter interface carries a flat key list.
- **Write conflicts are misses.** A key that becomes resident between the
  lookup and the reserve step fails reservation (`l1_contended`) and is
  reported as a miss; the engine recomputes and re-stores it.
- **Uniform rows.** Every key group has the same number of keys.

## Contract-anchoring tests

`tests/v1/distributed/test_prefetch_controller.py` drives the public interface
with a real `L1Manager` and `MockL2Adapter`:

- `TestL1AndL2` — an L1 suffix extends an L2 prefix into one hit; resident
  keys are not reloaded; L1 hits outside the prefix are released.
- `TestReservationFailures` — an out-of-memory or contended cell drops its L2
  lock and the request replans (`"prefix"` shortens the hit, `"full"` keeps
  the other cells); an L1 hit survives a racing evictor.
- `TestLockModeAndRetention` — `NO_LOCK` releases every L1 lock at finish and
  leaves loaded objects permanent; `"default"` deletes loaded keys after the
  retrieve, `"retain"` keeps them.
- `TestSlidingWindowRows` — a windowed row keeps and loads only its trailing
  chunks; `"full"` refuses windowed rows.
- `TestRuntimeAdapters`, `TestShutdown` — draining adapters are not planned
  and detach once unreferenced; `stop()` releases what the maps record.

`tests/v1/distributed/test_prefetch_policy.py` tests the policies black-box
against their docstrings.
