# Memory Tags for L1 Managers and L2 Adapters

> **Scope — read before using these types.**
> `MemorySegment`, `MemoryResidency` and `MemoryTags` exist for exactly two
> purposes:
>
> 1. **L1-L2 affinity**: choosing the L1 manager whose buffers an L2 adapter
>    stores from and loads into.
> 2. **Prefetch target filtering**: restricting a prefetch to part of L1/L2
>    (e.g. P2P serving looks only at local DRAM L1 and local L2).
>
> Do **not** use them for anything else, such as store routing, eviction,
> quotas, metrics labels, capacity reporting, or feature gating, without first
> discussing it with the author (@ApostaC). Any new use must also be added to
> this section.

Source: `lmcache/v1/distributed/api.py` (types),
`lmcache/v1/distributed/storage_controllers/utils.py` (`resolve_l1_affinity`).

## Why

Before memory tags, each `--l2-adapter` JSON carried a hand-written
`affinity_tag` naming an L1, defaulting to `_default`. Users had to know which
L1s an adapter could work with (GDS was rejected by a special case), and a
prefetch could not be restricted to "local" tiers: P2P serving could only turn
L2 off entirely. Memory tags make those properties intrinsic to each L1 manager
and L2 adapter, so the runtime can decide.

## Types

```python
class MemorySegment(str, enum.Enum):
    VRAM = "vram"        # reserved for GDS/GDR L2 adapters; nothing reports it yet
    DRAM = "dram"
    CXL = "cxl"          # Device-DAX
    GDS = "gds"
    STORAGE = "storage"  # any other medium

class MemoryResidency(str, enum.Enum):
    LOCAL = "local"      # bytes reside on this host
    REMOTE = "remote"    # bytes reside elsewhere, or in a domain shared with other hosts
    DUMMY = "dummy"      # reserved: placeholder L1 objects for GDS/GDR L2 adapters

@dataclass(frozen=True)
class MemoryTags:
    segment: MemorySegment
    residency: MemoryResidency
```

Tags are compared by exact equality. `MemoryTags` is frozen, so it is hashable
and can be put in sets.

There is deliberately no free-form `extra_attrs` field. Neither use case reads
one, string matching has no defined semantics (exact? subset?), and an open bag
of strings invites the misuse the scope rule forbids. If a use case needs more,
such as NUMA-aware affinity, add a typed field with a default (e.g.
`numa_node: int | None = None`); that is backward compatible.

## Who declares what

| Declaration | Where | Why there |
|---|---|---|
| L1 tags | `L1ManagerInterface.get_memory_tags()` | L1 managers are built first, so instances exist whenever tags are read. |
| L2 own tags | `L2AdapterInterface.get_memory_tags()` (abstract) | Read after adapters are built; every adapter must declare its locality. |
| L2 wanted L1 tags | `L2AdapterConfigBase.get_wanted_l1_memory_tags()` | Needed **before** the adapter exists (see below). |

`StorageManager` builds things in this order:

```
1. L1 managers                      -> L1 tags available
2. L2 adapters, one at a time:
     wanted tags (config) + L1 tags -> resolve_l1_affinity -> chosen L1
     create_l2_adapter(config, chosen_l1.get_l1_memory_desc())
     SerdeL2AdapterWrapper(..., l1_manager=chosen_l1)   # if serde is set
3. Store / prefetch controllers     -> route with the same resolution
```

The wanted tags cannot be an instance method. Step 2 passes the chosen L1's
memory descriptor to the adapter factory: NIXL, Mooncake over RDMA and
raw_block register that buffer in their constructors, and the serde wrapper
reserves temporary buffers in that L1.

### L1 tags

All L1 managers are `LOCAL`. The segment names the primary pool:

| L1 | Tags |
|---|---|
| `DRAM` | `(DRAM, LOCAL)` |
| `DEVDAX` (pure Device-DAX) | `(CXL, LOCAL)` |
| legacy hybrid DRAM + Device-DAX overflow | `(DRAM, LOCAL)` |
| `GDS` slab | `(GDS, LOCAL)` |

### L2 own tags

`REMOTE` means the bytes are off this host or in a domain shared with other
hosts. Adapters on local media honor the generic `"shared": true` JSON key by
reporting `REMOTE`, because a shared filesystem is visible to every peer.

| Adapter | Tags |
|---|---|
| `fs`, `fs_native`, `raw_block` | `(STORAGE, LOCAL)`; `REMOTE` when shared |
| `dax` | `(CXL, LOCAL)`; `REMOTE` when shared |
| `nixl_store`, `nixl_store_dynamic`, `nixl_native` | `POSIX`, `GDS`, `GDS_MT` backends: `(STORAGE, LOCAL)`; `REMOTE` when shared; any other backend: `(STORAGE, REMOTE)`. GDS backends still load through host L1 buffers, so they are plain storage. |
| `mock` | `(STORAGE, LOCAL)` |
| `p2p` | `(DRAM, REMOTE)` |
| `s3`, `resp`, `valkey`, `mooncake_store`, `aerospike`, `bigtable`, `hfbucket`, `hf3fs`, `sagemaker-hyperpod`, `native_plugin` | `(STORAGE, REMOTE)` |
| `plugin` | declared by the user's adapter class |
| `fault_inject`, serde wrapper | forwarded from the inner adapter |

## L1-L2 affinity

`get_wanted_l1_memory_tags()` returns acceptable L1 tags, most preferred first.
The base default is:

```python
((DRAM, LOCAL), (CXL, LOCAL))
```

`resolve_l1_affinity(adapter_config, l1_managers)` takes the first wanted entry
that any L1 matches and returns the first matching L1 in write-placement
order. If nothing matches it raises `ValueError` naming the adapter type, the
wanted tags and the configured L1 tags, and startup or `add_l2_adapter` fails.
The function is pure and its inputs are fixed after startup, so
`StorageManager` and `PrefetchController` each call it and always agree.

| L1 managers (placement order) | Adapter binds to |
|---|---|
| `DRAM` | the DRAM L1 |
| `DRAM`, `DEVDAX`, `GDS` | the DRAM L1 |
| `DEVDAX`, `DRAM` | the DRAM L1 (DRAM is preferred over placement order) |
| `DRAM a`, `DRAM b` | `a` (first in placement order) |
| `DEVDAX` | the DEVDAX L1 (CXL fallback) |
| `GDS` | error: no L1 has the wanted tags |

Consequences:

- An L2 adapter sees completed stores only from its L1. In a DRAM + DEVDAX
  server, objects that overflow into Device-DAX are not stored to L2.
- With several DRAM L1s, every adapter binds to the first one. Choosing among
  them (e.g. by NUMA node) would need the typed field described above.
- A future GDS/GDR adapter overrides `get_wanted_l1_memory_tags()`, for
  example to `((VRAM, DUMMY),)`, once a dummy L1 exists.

## Prefetch target filtering

`PrefetchTaskSpec` carries two filters, one per tier. They are separate because
an L1 and an L2 can share tags (a `dax` L2 and a CXL L1 are both
`(CXL, LOCAL)`).

```python
PrefetchTaskSpec(
    key_groups=...,
    allowed_l1_memory_tags=frozenset({MemoryTags(DRAM, LOCAL)}),  # hit sources
    allowed_l2_memory_tags=frozenset(),                           # no L2 at all
)
```

| Value | Meaning |
|---|---|
| `None` (default) | every L1 manager / L2 adapter is allowed |
| empty `frozenset()` | none is allowed |
| non-empty set | only managers / adapters whose tags are in the set |

- **L1 filter = hit sources only.** Only allowed L1 managers are read-locked,
  so only their resident objects count as L1 hits.
- **L2 filter = lookup and load sources.** Only allowed, non-draining adapters
  are looked up and planned.
- **Loads still land in the affinity L1.** An allowed L2 adapter loads into its
  affinity L1 even when that L1's tags are not allowed. A key already resident
  there cannot be reserved again, so it is reported as a miss.
- **Synchronous fast path.** A spec with an empty L2 filter, or a controller
  with no L2 adapter attached, is served from L1 on the calling thread. A
  non-empty filter that matches no adapter is served from L1 on the loop
  thread.

P2P serving passes `allowed_l1_memory_tags=frozenset({(DRAM, LOCAL)})` and
`allowed_l2_memory_tags=frozenset()`: a peer lookup only locks what this node
already holds in its DRAM L1, the L1 registered with the transfer channel, and
never recurses into this node's L2 adapters (including its own P2P adapters). Serving peers from local L2 is a
follow-up: it moves P2P lookups onto the asynchronous L2 path, which interacts
with the requester-side lookup timeouts.
