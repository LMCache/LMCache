# L2 Store and Prefetch Controller Design

## Architecture

```
              ┌──────────────────┐
              │  StorageManager   │
              └───┬──────────┬───┘
                  │          │
       ┌──────────┘          └──────────┐
       ▼                                ▼
┌──────────────┐              ┌──────────────────┐
│StoreController│             │PrefetchController │
│ (bg thread)   │             │ (bg thread)       │
│ L1 write done │             │ external submit   │
│  → store to L2│             │  → lookup → load  │
└──────┬────────┘             └────────┬──────────┘
       │                               │
       ▼                               ▼
┌─────────────────────────────────────────────┐
│         L2AdapterInterface(s)               │
│  store / lookup_and_lock / load / unlock    │
│  3 distinct eventfds per adapter            │
└─────────────────────────────────────────────┘
```

Both controllers run a background thread using `select.poll()` on eventfds.
They share the same L2 adapter instances (thread-safe by contract).

## Key Invariants

1. **All eventfds are globally unique** across all adapters and operation types.
2. **L2 task IDs are per-adapter.** Use `(adapter_index, task_id)` as composite keys.
3. **Query results are one-shot.** `query_*_result()` returns non-None exactly once.
4. **`submit_unlock` must eventually succeed.** Controllers never retry — adapters handle retries internally.
5. **Prefix-only loading.** Only the contiguous prefix of found keys is loaded from L2.
6. **Listener callbacks run inside L1Manager's lock.** `StoreListener` must be non-blocking.
7. **Atomic write→read transition.** `finish_write_and_reserve_read()` prevents eviction gaps.
8. **Both controllers release all locks on shutdown.**
9. **L2 adapters are thread-safe** for concurrent store and prefetch calls.

## Implementing a New L2 Adapter

Implement `L2AdapterInterface` (see `mock_l2_adapter.py` for reference).
Register a config class in `config.py` and add a factory branch in `__init__.py`.

For native (C++/Rust) backends, see [`csrc/storage_backends/README.md`](../../../../csrc/storage_backends/README.md).

## Implementing a New Policy

Create a single file in `storage_controllers/` — no changes to existing files needed.
The module is auto-discovered via `pkgutil.iter_modules()` at import time.

- **Store policy:** Subclass `StorePolicy`, implement `select_store_targets()` and `select_l1_deletions()`, call `register_store_policy("name", cls)` at module level.
- **Prefetch policy:** Subclass `PrefetchPolicy`, implement `select_load_plan()`, call `register_prefetch_policy("name", cls)` at module level.
