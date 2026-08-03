# Runtime Management Policy

This document describes the node-local Phase 1 implementation of RFC #4360.
It covers only policies whose state can be reused after an update.

## Control Flow

```text
HTTP /config/policies
        |
        v
StorageManager
  - version check
  - full validation
  - controller routing
        |
        +--> StoreController.update_policy()
        +--> PrefetchController.update_policy()
        +--> L1EvictionController.update_tunables()
        `--> L2EvictionController.update_tunables()
```

`StorageManager` owns the monotonic runtime policy version. A request is
validated completely before any controller is mutated. `expected_version`, when
provided, implements optimistic concurrency and returns HTTP 409 on a stale
write.

## Phase 1 Contract

The following changes are hot-updateable:

| Field | Effective on |
| --- | --- |
| `store_policy` | Next store plan |
| `prefetch_policy` | Next prefetch request |
| `l1_eviction.tunables.trigger_watermark` | Next eviction loop tick |
| `l1_eviction.tunables.eviction_ratio` | Next eviction loop tick |
| `l2_eviction[].tunables.trigger_watermark` | Next eviction loop tick |
| `l2_eviction[].tunables.eviction_ratio` | Next eviction loop tick |

Store and prefetch tasks capture the policy instance when they enter their
planning lifecycle. Updating the controller does not recompute an in-flight
plan, and a store task uses its captured policy for the eventual L1 deletion
decision.

Eviction policy classes are stateful and are not replaced at runtime. A class
change returns `state_migration_required`. Startup-only fields such as
`chunk_size`, L1 capacity, serde, and adapter type return `restart_required`.

## HTTP API

`GET /config/policies` returns the current version, current policy names, and
capability metadata. The `registered` lists are the source of truth for valid
store and prefetch selectors.

`POST /config/policies/validate` accepts the same body as `PATCH` and never
mutates the process.

`PATCH /config/policies` applies an update. The body uses stable L2 adapter IDs:

```json
{
  "expected_version": 7,
  "store_policy": "skip_l1",
  "prefetch_policy": "retain",
  "l1_eviction": {
    "tunables": {
      "trigger_watermark": 0.9,
      "eviction_ratio": 0.1
    }
  },
  "l2_eviction": [
    {
      "adapter_id": 0,
      "tunables": {
        "trigger_watermark": 0.85,
        "eviction_ratio": 0.1
      }
    }
  ]
}
```

Successful updates increment the version and return the applied field list.
Invalid policy names return 400, stale versions return 409, and unknown L2
adapter IDs return 404. Validation errors never partially apply an update.

## Future Extensions

Coordinator fan-out can use the same versioned domain API without changing
controller ownership. Stateful eviction policy migration remains a separate
extension and requires an explicit snapshot/import compatibility contract.
