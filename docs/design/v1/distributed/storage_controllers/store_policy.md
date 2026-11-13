# Store Policy

`lmcache/v1/distributed/storage_controllers/store_policy.py`,
`lmcache/v1/distributed/storage_controllers/p2p_store_policy.py`

A store policy answers two questions for the `StoreController` after an
object has been written to L1:

1. `select_store_targets(keys, adapters)`: which L2 adapters get which keys.
2. `select_l1_deletions(keys, adapter)`: after `adapter` stored `keys`
   successfully, which of them leave L1.

The store flow around these two calls is described in
[`l2_adapters/overall.md`](../l2_adapters/overall.md#storecontroller).

## Interface

```python
class StorePolicy(ABC):
    def __init__(self, l1_managers: list[L1ManagerInterface]) -> None: ...
    def select_store_targets(self, keys, adapters) -> dict[int, list[ObjectKey]]: ...
    def select_l1_deletions(self, keys, adapter) -> list[ObjectKey]: ...
```

- **`l1_managers`** is every L1 manager of the storage manager, in
  write-placement order. A policy that needs L1 state (memory usage, memory
  tags, configuration) picks the manager it cares about from this list. The
  list is the same for every policy instance; a policy must not assume it only
  contains the L1 of the store controller that owns it.
- **`adapters`** in `select_store_targets` are the descriptors of the adapters
  attached to the calling store controller. An adapter is attached to the
  controller of the one L1 it has affinity with (`resolve_l1_affinity`), so a
  policy only ever sees adapters that can read that L1's objects.
- **`adapter`** in `select_l1_deletions` is the descriptor of the adapter
  whose store task succeeded. It lets a policy drop the L1 copy after a store
  into one kind of adapter and keep it after a store into another.

The storage manager creates one policy instance per store controller with
`create_store_policy(name, l1_managers)`. Both methods are called only from
that controller's loop thread, so a policy may keep plain (unlocked) state.

## Built-in policies

| Name | Class | Targets | L1 deletions |
|---|---|---|---|
| `default` | `DefaultStorePolicy` | every key to every adapter | none |
| `skip_l1` | `BufferOnlyStorePolicy` | every key to every adapter | every key, after any successful store |
| `p2p` | `P2PStorePolicy` | see below | keys stored into a peer |

Select one with `--l2-store-policy`. A new policy is a module in
`storage_controllers/` that calls `register_store_policy(name, cls)` at
import time; the package imports every module in the directory, so no
existing file changes.

## `P2PStorePolicy`

For a server whose P2P peers extend its DRAM (for example headless peers with
no inference engine). Objects stay in the local DRAM L1 while it has room;
once it is fuller than a threshold, a newly written object is stored into one
peer and its local copy is dropped.

At construction the policy keeps the L1 manager whose memory tags are
DRAM/LOCAL: the L1 registered with the transfer channel, and therefore the
only one whose objects a P2P adapter can send.

`select_store_targets(keys, adapters)`:

| Adapter | Gets the keys? |
|---|---|
| Not a P2P adapter | Always, as under `default` |
| P2P adapter with `write_enabled=False` | Never. (Under `default` it gets a task that the adapter completes as a 0-byte no-op.) |
| P2P adapters with `write_enabled=True` | Exactly **one** of them, and only when `used / total` of the local DRAM L1 is at or above that adapter's `write_l1_usage_threshold` |

The write-enabled peers take turns: a counter advances each time a task is
sent to a peer, so tasks spread evenly without the policy tracking peer
capacity. A task is not retried on another peer when the chosen one rejects
it; the objects then simply stay local.

`select_l1_deletions(keys, adapter)` returns every key when `adapter` is a
write-enabled P2P adapter and nothing otherwise. The store controller defers
the deletion of a key a reader still holds.

What this adds up to:

```
          local DRAM L1 usage
 0 ───────────────┬───────────────────┬──────────── 1
                  │ threshold (0.7)   │ eviction trigger (0.8)
   objects stay   │  new objects go   │  local eviction would
   local          │  to a peer and    │  discard objects
                  │  leave L1         │
```

- Local usage settles around the threshold: every object stored into a peer
  frees its local buffer, which brings usage back under the threshold.
- An object written below the threshold is never moved later. Only newly
  written objects are candidates.
- A store task that fails (peer full, peer unreachable, key already on the
  peer) keeps its objects local, where normal eviction applies.
- The threshold must be below the L1 eviction trigger watermark, otherwise
  eviction discards local objects before any are stored into a peer. The
  server warns at startup when it is not.

Interaction with the prefetch policy: the store policy only decides where a
newly written object goes. When an object is later read back from a peer, the
prefetch policy decides whether the main server keeps that copy. Under the
default prefetch policy it does, so read-backs accumulate in the local L1,
lift it over the eviction watermark, and evict objects that live nowhere
else. In a two-node run with a 4 GiB main L1 this cut the share of prompt
tokens served from cache from 95 % to 63–74 %. `--enable-p2p-write` therefore
also selects the `skip_l1` prefetch policy, which makes read-backs temporary,
when `--l2-prefetch-policy` is `default`.

Where the threshold comes from: it is a field of `P2PL2AdapterConfig`
(`write_l1_usage_threshold`), filled by the P2P controller from
`--p2p-write-l1-usage-threshold` for every adapter it creates. It lives on the
adapter config because that is the only per-adapter data a policy receives.

`--enable-p2p-write` selects this policy automatically when
`--l2-store-policy` is `default`. With another policy chosen explicitly, that
policy decides what goes to peers and the threshold does not apply. The two
selections are independent: each replaces only a policy left at `default`.
