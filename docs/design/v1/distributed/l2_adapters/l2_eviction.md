# L2 Adapter Eviction Design

This document describes the eviction mechanism for L2 adapters: how eviction is
configured, how the eviction controller tracks key lifecycle events, and how
adapters participate in the eviction loop.

## Overview

L2 eviction is **per-adapter** and **opt-in**. Each L2 adapter instance can
independently declare an eviction policy via its JSON config. Adapters without
an `"eviction"` key in their config are excluded from the eviction loop.

A single `L2EvictionController` manages all adapters that have eviction enabled.
Its background thread loops over every adapter each cycle, checking usage and
triggering eviction independently per adapter.

## Architecture

```
StorageManager
  │
  └─ L2EvictionController  (single instance, single thread)
       │
       ├─ L2AdapterEvictionState[0]
       │     adapter ──► L2Adapter[0]   (eviction_config set)
       │     policy  ──► EvictionPolicy
       │     listener ─► L2EvictionPolicy (bridge)
       │
       ├─ (L2Adapter[1] has no eviction config → not tracked)
       │
       └─ L2AdapterEvictionState[1]
             adapter ──► L2Adapter[2]   (eviction_config set)
             policy  ──► EvictionPolicy
             listener ─► L2EvictionPolicy (bridge)
```

Each adapter with eviction enabled gets an `L2AdapterEvictionState` that bundles:

- An **`EvictionPolicy`** instance (e.g., LRU) that tracks key state.
- An **`L2EvictionPolicy`** listener bridge registered on the adapter. The
  bridge translates adapter events into policy `on_keys_*` calls via
  composition (no multi-inheritance).

The controller and each adapter communicate through two channels:

1. **Listener callbacks** — the `L2EvictionPolicy` bridge is registered as an
   `L2AdapterListener` on the adapter. The adapter fires events when keys are
   stored, accessed, or deleted, and the bridge forwards them to the eviction
   policy to keep its key tracking up-to-date.

2. **`delete(keys)`** — when the eviction policy decides to evict, the
   controller calls `adapter.delete(keys)` directly. The adapter removes those
   keys from its storage and fires an `on_l2_keys_deleted` callback, which the
   bridge forwards to the policy so it removes them from its tracking state.

## Configuration

Eviction is configured as an optional `"eviction"` sub-object in each adapter's
JSON spec passed to `--l2-adapter`:

```json
{
  "type": "mock",
  "max_size_gb": 10,
  "mock_bandwidth_gb": 4,
  "eviction": {
    "eviction_policy": "LRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

| Field               | Type    | Default | Description                                                     |
|---------------------|---------|---------|-----------------------------------------------------------------|
| `eviction_policy`   | string  | —       | Policy name: `"LRU"`, `"IsolatedLRU"`, or `"noop"`. Required.  |
| `trigger_watermark` | float   | `0.8`   | Usage fraction [0, 1] above which eviction is triggered.        |
| `eviction_ratio`    | float   | `0.2`   | Fraction of **used** capacity to evict each cycle.              |

If the `"eviction"` key is absent, no `L2AdapterEvictionState` is created for
that adapter instance and it is excluded from the eviction loop.

The eviction config is parsed by `L2AdapterConfigBase._parse_eviction_config()`
and stored as `adapter_config.eviction_config: EvictionConfig | None`.

## Key Components

### `L2AdapterListener` (`internal_api.py`)

Abstract interface for receiving L2 adapter events:

```python
class L2AdapterListener:
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]): ...
    def on_l2_keys_accessed(self, keys: list[ObjectKey]): ...
    def on_l2_keys_deleted(self, keys: list[ObjectKey]): ...
```

### Listener Infrastructure in `L2AdapterInterface` (`l2_adapters/base.py`)

The base class owns the listener list and provides:

- `register_listener(listener)` — adds a listener; called by
  `L2AdapterEvictionState.__init__`.
- `_notify_keys_stored(keys, sizes)` / `_notify_keys_accessed(keys)` /
  `_notify_keys_deleted(keys, sizes)` — protected helpers that update
  base-class byte accounting and fan out to all registered listeners.
  Adapter implementations call these after mutating their storage.

No per-adapter code is needed to support listeners — just call
`super().__init__(max_capacity_bytes=...)` and use the `_notify_*` helpers.

### `L2AdapterEvictionState` (`storage_controllers/eviction_controller.py`)

Bundles the per-adapter eviction state: the adapter reference, its
`EvictionConfig`, an `EvictionPolicy` instance, and an `L2EvictionPolicy`
listener bridge. On construction, it registers the bridge on the adapter:

```python
L2AdapterEvictionState(adapter, eviction_config)
  → creates EvictionPolicy from config
  → creates L2EvictionPolicy(policy)   # listener bridge
  → adapter.register_listener(bridge)  # subscribe to adapter events
```

### `L2EvictionPolicy` (`eviction.py`)

Listener bridge that inherits only `L2AdapterListener` and delegates events
to an `EvictionPolicy` via composition:

| Callback               | Delegates to              |
|------------------------|---------------------------|
| `on_l2_keys_stored`    | `policy.on_keys_created`  |
| `on_l2_keys_accessed`  | `policy.on_keys_touched`  |
| `on_l2_keys_deleted`   | `policy.on_keys_removed`  |

### `L2EvictionController` (`storage_controllers/eviction_controller.py`)

A single controller that manages all adapters with eviction enabled. It owns
one background thread and a list of `L2AdapterEvictionState` objects.

**Eviction loop:**

Every second, the thread iterates over all adapter states. For each adapter,
it calls `adapter.get_usage()` which returns an `AdapterUsage` snapshot. For
aggregate policies, `AdapterUsage.usage_fraction` is compared with that
adapter's `trigger_watermark`; if it exceeds the watermark, the policy's
`get_eviction_actions(eviction_ratio)` is called and the resulting keys are
passed to `adapter.delete()`. For isolated policies, the controller compares
`AdapterUsage.bytes_by_cache_salt` with runtime quotas.

### Eviction Policy (`eviction_policy/`)

The eviction policy is a pure data structure — it tracks keys and decides which
to evict. It has no knowledge of adapters or listeners:

```
EvictionPolicy (abstract)
  ├─ LRUEvictionPolicy   — evicts least-recently-used keys
  └─ NoOpEvictionPolicy  — never evicts
```

Policies are created by `CreateEvictionPolicy(eviction_config)` in
`eviction_policy/factory.py`.

## Adapter Implementation Guide

To support eviction in a new adapter:

1. **Call `super().__init__(max_capacity_bytes=...)`** in the adapter's
   `__init__`. Pass a positive value when the adapter supports aggregate
   capacity eviction. Use `0` for unlimited or unknown capacity.

2. **Fire `_notify_keys_stored(keys, sizes)`** after keys are durably written
   to L2 storage (e.g., after a store task completes). Pass one byte size per
   key; this drives aggregate and per-`cache_salt` usage accounting.

3. **Fire `_notify_keys_deleted(keys, sizes)`** inside `delete()` after keys are
   removed. Only fire for keys that were actually removed or that were already
   missing but still accounted by the adapter. Skip keys not found and never
   accounted, and keys skipped due to pinning.

4. **Fire `_notify_keys_accessed(keys)`** when a lookup or load marks a key as
   recently used (optional — improves LRU accuracy).

5. **Implement `delete(keys)`** to remove keys from storage. Pinned keys (in use
   by an in-flight load) should be skipped; the eviction controller will retry
   them on the next cycle.

6. **Use base-class `get_usage()`** unless the adapter has a more precise
   backend-native accounting source. The base implementation returns
   `AdapterUsage(total_bytes_used, total_capacity_bytes, bytes_by_cache_salt)`
   from `_notify_keys_stored` / `_notify_keys_deleted`.

Adapters that do not support eviction (e.g., a remote adapter with unbounded
capacity) can omit steps 2–6 and rely on the base class no-op defaults.

## Adapter Support Matrix

| Adapter                    | `delete` | `get_usage` | Listener events     |
|----------------------------|----------|-------------|---------------------|
| `MockL2Adapter`            | ✓        | ✓           | stored, deleted     |
| `NixlStoreL2Adapter`       | ✓ (skips pinned) | ✓ (pool-based) | stored, deleted |
| `RawBlockL2Adapter`        | ✓ (skips locked) | ✓ | stored, accessed, deleted |
| `FSL2Adapter`              | ✓ (skips locked, deletes `.data` files) | ✓ (requires `max_capacity_gb` for aggregate eviction) | stored, accessed, deleted |
| `NativeConnectorL2Adapter` | ✓ (via `submit_batch_delete`) | ✓ (client-side, requires `max_capacity_gb`) | stored, deleted |

**Note on `FSL2Adapter`:** Eviction support is opt-in. Configure
`max_capacity_gb > 0` and add an adapter-local `"eviction"` object. The adapter
stores raw tensor bytes without a metadata header; it recovers usage by scanning
direct child `.data` files and deriving `ObjectKey` values from filenames. Legacy
filenames that predate the explicit `object_group_id` field are mapped to
`object_group_id=0`, so existing cache directories continue to hit and can be
governed without renaming files.

**Note on `NativeConnectorL2Adapter`:** Eviction support requires two things:

1. The underlying C++ connector must implement `do_single_delete()` (built-in Redis
   and FS connectors do; third-party plugins may not — in which case `delete()` is a
   no-op).
2. The adapter must be configured with `max_capacity_gb > 0` to enable client-side
   size tracking for `get_usage()`. Without it, `usage_fraction == -1.0` and
   the eviction controller will not trigger aggregate eviction.

Example configuration with eviction enabled:

```json
{
  "type": "resp",
  "host": "localhost",
  "port": 6379,
  "max_capacity_gb": 10,
  "eviction": {
    "eviction_policy": "LRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

## Data Flow: Eviction Cycle

```
[Background thread — every 1s]
  │
  ▼
for each L2AdapterEvictionState:
  │
  ▼
  state.adapter.get_usage()
    → AdapterUsage
    │
    ├─ usage_fraction < watermark → skip this adapter
    │
    └─ usage_fraction ≥ watermark
         │
         ▼
    state.policy.get_eviction_actions(eviction_ratio)
         → list[EvictionAction(keys, destination=DISCARD)]
         │
         ▼
    state.adapter.delete(eviction_action.keys)
         │
         ├─ removes keys from storage
         └─ calls _notify_keys_deleted(deleted_keys, deleted_sizes)
              │
              ▼
         L2EvictionPolicy bridge → policy.on_keys_removed
              → updates internal tracking (e.g., LRU order)
```

## Relationship to L1 Eviction

L1 and L2 eviction share the same policy classes (`LRUEvictionPolicy`,
`NoOpEvictionPolicy`) and the same listener-bridge pattern (composition over
multi-inheritance). They differ in how they are wired:

| Aspect              | L1                                   | L2                                    |
|---------------------|--------------------------------------|---------------------------------------|
| Controller          | `L1EvictionController`               | `L2EvictionController`                |
| Listener bridge     | `L1EvictionPolicy`                   | `L2EvictionPolicy`                    |
| Listener interface  | `L1ManagerListener`                  | `L2AdapterListener`                   |
| Usage source        | `L1Manager.get_memory_usage()`       | `L2AdapterInterface.get_usage()`      |
| Config location     | `StorageManagerConfig.eviction_config` | `L2AdapterConfigBase.eviction_config` |
| Cardinality         | One per `StorageManager`             | One controller for all adapters       |
| Created by          | `StorageManager.__init__`            | `StorageManager.__init__`             |
