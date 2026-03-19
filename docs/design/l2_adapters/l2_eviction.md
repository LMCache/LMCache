# L2 Adapter Eviction Design

This document describes the eviction mechanism for L2 adapters: how eviction is
configured, how the eviction controller tracks key lifecycle events, and how
adapters participate in the eviction loop.

## Overview

L2 eviction is **per-adapter** and **opt-in**. Each L2 adapter instance can
independently declare an eviction policy via its JSON config. Adapters without
an `"eviction"` key in their config have no eviction controller created for them.

When eviction is enabled for an adapter, a dedicated `L2EvictionController`
background thread monitors that adapter's storage utilization and periodically
evicts keys according to the configured policy (e.g., LRU).

## Architecture

```
StorageManager
  │
  ├─ L2Adapter[0]  ──► L2EvictionController[0]  (if eviction_config set)
  │     ▲                  │
  │     │ events           │ delete(keys)
  │     └──────────────────┘
  │
  ├─ L2Adapter[1]  (no eviction config → no controller)
  │
  └─ L2Adapter[2]  ──► L2EvictionController[2]  (if eviction_config set)
        ▲                  │
        │ events           │ delete(keys)
        └──────────────────┘
```

The `L2EvictionController` and the adapter communicate through two channels:

1. **Listener callbacks** — the controller registers itself as an
   `L2AdapterListener` on the adapter. The adapter fires events when keys are
   stored, accessed, or deleted, keeping the eviction policy's key tracking
   up-to-date.

2. **`delete(keys)`** — when the eviction policy decides to evict, the
   controller calls `adapter.delete(keys)` directly. The adapter removes those
   keys from its storage and fires an `on_l2_keys_deleted` callback, which the
   controller forwards to the policy so it removes them from its tracking state.

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
| `eviction_policy`   | string  | —       | Policy name: `"LRU"` or `"noop"`. Required.                    |
| `trigger_watermark` | float   | `0.8`   | Usage fraction [0, 1] above which eviction is triggered.        |
| `eviction_ratio`    | float   | `0.2`   | Fraction of **used** capacity to evict each cycle.              |

If the `"eviction"` key is absent, no `L2EvictionController` is created for
that adapter instance.

The eviction config is parsed by `L2AdapterConfigBase._parse_eviction_config()`
and stored as `adapter_config.eviction_config: EvictionConfig | None`.

## Key Components

### `L2AdapterListener` (`internal_api.py`)

Abstract interface for receiving L2 adapter events:

```python
class L2AdapterListener:
    def on_l2_keys_stored(self, keys: list[ObjectKey]): ...
    def on_l2_keys_accessed(self, keys: list[ObjectKey]): ...
    def on_l2_keys_deleted(self, keys: list[ObjectKey]): ...
```

### Listener Infrastructure in `L2AdapterInterface` (`l2_adapters/base.py`)

The base class owns the listener list and provides:

- `register_listener(listener)` — adds a listener; called by
  `L2EvictionController.__init__`.
- `_notify_keys_stored(keys)` / `_notify_keys_accessed(keys)` /
  `_notify_keys_deleted(keys)` — protected helpers that fan out to all
  registered listeners. Adapter implementations call these after mutating
  their storage.

No per-adapter code is needed to support listeners — just call
`super().__init__()` and use the `_notify_*` helpers.

### `L2EvictionController` (`storage_controllers/eviction_controller.py`)

Extends both `EvictionController` (background thread + policy) and
`L2AdapterListener` (event receiver):

```
L2EvictionController
  ├─ inherits: EvictionController      (stop flag, thread, eviction loop)
  └─ implements: L2AdapterListener     (delegates events → policy)
```

**Initialization:**

```python
L2EvictionController(l2_adapter, eviction_config)
  → super().__init__(eviction_config)   # creates policy + thread
  → l2_adapter.register_listener(self)  # subscribe to adapter events
```

**Listener delegation:**

| Callback               | Delegates to              |
|------------------------|---------------------------|
| `on_l2_keys_stored`    | `policy.on_keys_created`  |
| `on_l2_keys_accessed`  | `policy.on_keys_touched`  |
| `on_l2_keys_deleted`   | `policy.on_keys_removed`  |

**Eviction loop:**

Every second, the thread calls `adapter.get_usage()` which returns
`(current_usage, usage_after_ongoing_eviction)`. If `current_usage` exceeds
`trigger_watermark`, the policy's `get_eviction_actions(eviction_ratio)` is
called and the resulting keys are passed to `adapter.delete()`.

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

1. **Call `super().__init__()`** in the adapter's `__init__`. This initializes
   the listener list from the base class.

2. **Fire `_notify_keys_stored(keys)`** after keys are durably written to L2
   storage (e.g., after a store task completes).

3. **Fire `_notify_keys_deleted(keys)`** inside `delete()` after keys are
   removed. Only fire for keys that were actually removed — skip keys not found
   and keys skipped due to pinning.

4. **Fire `_notify_keys_accessed(keys)`** when a lookup or load marks a key as
   recently used (optional — improves LRU accuracy).

5. **Implement `delete(keys)`** to remove keys from storage. Pinned keys (in use
   by an in-flight load) should be skipped; the eviction controller will retry
   them on the next cycle.

6. **Implement `get_usage() -> (current, projected)`** to return the current
   storage utilization as a fraction in [0, 1]. `projected` may equal `current`
   if in-flight deletions are not tracked.

Adapters that do not support eviction (e.g., a remote adapter with unbounded
capacity) can omit steps 2–6 and rely on the base class no-op defaults.

## Adapter Support Matrix

| Adapter                    | `delete` | `get_usage` | Listener events     |
|----------------------------|----------|-------------|---------------------|
| `MockL2Adapter`            | ✓        | ✓           | stored, deleted     |
| `NixlStoreL2Adapter`       | ✓ (skips pinned) | ✓ (pool-based) | stored, deleted |
| `FSL2Adapter`              | no-op    | `(0, 0)`    | none                |
| `NativeConnectorL2Adapter` | no-op    | `(0, 0)`    | none                |

## Data Flow: Eviction Cycle

```
[Background thread — every 1s]
  │
  ▼
adapter.get_usage()
  → (current_usage, _)
  │
  ├─ current_usage < watermark → sleep, repeat
  │
  └─ current_usage ≥ watermark
       │
       ▼
  policy.get_eviction_actions(eviction_ratio)
       → list[EvictionAction(keys, destination=DISCARD)]
       │
       ▼
  adapter.delete(eviction_action.keys)
       │
       ├─ removes keys from storage
       └─ calls _notify_keys_deleted(deleted_keys)
            │
            ▼
       on_l2_keys_deleted → policy.on_keys_removed
            → updates internal tracking (e.g., LRU order)
```

## Relationship to L1 Eviction

L1 and L2 eviction share the same policy classes (`LRUEvictionPolicy`,
`NoOpEvictionPolicy`) and the same `EvictionController` base class. They differ
in how they are wired:

| Aspect              | L1                                   | L2                                    |
|---------------------|--------------------------------------|---------------------------------------|
| Controller          | `L1EvictionController`               | `L2EvictionController`                |
| Listener interface  | `L1ManagerListener`                  | `L2AdapterListener`                   |
| Usage source        | `L1Manager.get_memory_usage()`       | `L2AdapterInterface.get_usage()`      |
| Config location     | `StorageManagerConfig.eviction_config` | `L2AdapterConfigBase.eviction_config` |
| Cardinality         | One per `StorageManager`             | One per adapter (only if configured)  |
| Created by          | `StorageManager.__init__`            | `StorageManager.__init__` (per adapter) |
