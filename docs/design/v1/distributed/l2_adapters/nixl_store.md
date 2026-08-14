# Nixl Store L2 Adapter Design

## Overview

The Nixl L2 adapter family implements `L2AdapterInterface` using the
[Nixl](https://github.com/ai-infra-org/nixl) library to offload KV-cache
objects from L1 (DRAM/VRAM) to a secondary storage tier via DMA. There are
two variants:

| Adapter | Type name | Storage mode | Persist | Backends |
|---|---|---|---|---|
| `NixlStoreL2Adapter` | `nixl_store` | Static (pre-allocated files) | Not supported | GDS, GDS_MT, POSIX, HF3FS, OBJ, AZURE_BLOB |
| `DynamicNixlStoreL2Adapter` | `nixl_store_dynamic` | Dynamic (per-operation descriptors) | File: adapter-managed (`persist_enabled`, default on); Object: backend-managed retention with presence-based recovery | GDS, GDS_MT, POSIX, HF3FS, OBJ, AZURE_BLOB |

The **static** adapter pre-allocates all storage descriptors at init and
registers them with Nixl as a single prepped descriptor list. The **dynamic**
adapter opens/registers descriptors per operation. File backends use
deterministic file names for persist/recover across restarts and avoid
open-file-descriptor limits. Object backends (`OBJ`, `AZURE_BLOB`) use
deterministic object keys and NIXL presence queries; they do not support
adapter-side deletion or global capacity-based eviction.

---

## Static Adapter: `NixlStoreL2Adapter`

### Key Components

#### `NixlStoreObj`
Metadata record for a single cached object in Nixl storage:
- `page_indices` — list of pre-allocated storage slot indices holding the data.
- `size` — byte size of the stored object.
- `layout` — optional `MemoryLayoutDesc` (shapes/dtypes) for reconstruction.
- `pin_count` — reference count preventing eviction while a load is in flight.

#### `NixlObjPool`
Thread-safe integer index pool representing the fixed set of pre-allocated
storage slots (`pool_size` entries). Slots are allocated before a store and
freed after a failed transfer or when the object is evicted.

#### `NixlStorageAgent`
Thin wrapper around the Nixl agent API. Responsibilities:
- Register the L1 memory buffer with Nixl (`init_mem_handlers`).
- Register storage slots (files or object keys) with Nixl
  (`init_storage_handlers_file` / `init_storage_handlers_object`).
- Produce pre-prepared transfer handles for batched DMA reads/writes
  (`get_mem_to_storage_handle`, `get_storage_to_mem_handle`).
- Drive transfers asynchronously (`post_non_blocking`).

#### `NixlStoreL2Adapter`
The public adapter implementing `L2AdapterInterface`. It owns:
- A background asyncio event loop (in a dedicated daemon thread) that
  executes all DMA coroutines.
- Three Linux event-fds (store / lookup / load) used to signal completion
  to the caller without polling.
- A shared `dict[ObjectKey, NixlStoreObj]` as the in-memory index.
- A single `threading.Lock` protecting all shared state.

---

## Operation Flow

### Store
```
submit_store_task(keys, objects)
  └─ schedules _execute_store_in_the_loop on the asyncio loop
       ├─ for each key/object: allocate storage slots, collect page indices
       ├─ issue single batched DMA write (mem → storage)
       ├─ on success: record key→NixlStoreObj in _memory_objects
       └─ on failure: free allocated slots; mark task failed
  └─ signals store event-fd
```

### Lookup & Lock
```
submit_lookup_and_lock_task(keys)
  └─ schedules _execute_lookup_in_the_loop (sync, via call_soon_threadsafe)
       ├─ for each key present: set bitmap bit, increment pin_count
       └─ records bitmap in _completed_lookup_tasks
  └─ signals lookup event-fd

submit_unlock(keys)
  └─ schedules pin_count decrement for each key (fire-and-forget)
```

### Load
```
submit_load_task(keys, objects)
  └─ schedules _execute_load_in_loop on the asyncio loop
       ├─ for each found key: collect mem/storage page indices, set bitmap bit
       ├─ issue single batched DMA read (storage → mem)
       └─ records bitmap in _completed_load_tasks
  └─ signals load event-fd
```

---

## Threading Model

| Thread | Role |
|---|---|
| Caller thread(s) | Call `submit_*` / `query_*`; never touch storage directly |
| Event-loop thread | Executes all Nixl DMA coroutines; owns `_memory_objects` mutations |
| Shared lock | Protects `_memory_objects`, task result dicts, and task-id counter |

Lookup is synchronous (scheduled via `call_soon_threadsafe`); store and load
are async coroutines (scheduled via `run_coroutine_threadsafe`).

---

## Memory Address → Page Index Mapping

L1 memory is registered with Nixl as a single contiguous buffer split into
fixed-size pages of `align_bytes`. A memory object at address `addr` of size
`sz` maps to page indices:

```
[addr // align_bytes, addr // align_bytes + 1, ..., addr // align_bytes + sz // align_bytes - 1]
```

Both `addr` and `sz` must be multiples of `align_bytes`.

---

## Dynamic Adapter: `DynamicNixlStoreL2Adapter`

**Source:** `l2_adapters/nixl_store_dynamic_l2_adapter.py`

### Motivation

The static adapter pre-allocates all storage files and registers them with
Nixl at init time. This has two limitations:

1. **OS file descriptor limits.** Each storage slot requires an open fd,
   limiting pool size in practice.
2. **No persist/recover.** Files are created with random UUIDs and the
   in-memory index (`_memory_objects`) is lost on shutdown.

The dynamic adapter solves both for file backends by opening/registering files
per operation and using deterministic file names derived from `ObjectKey`.
Object backends dynamically register deterministic object keys derived from
`ObjectKey`.

### Key Differences from Static

| Aspect | Static | Dynamic |
|---|---|---|
| Descriptor lifecycle | All registered at init, released at shutdown | Registered per store/load, released after each transfer |
| Storage naming | Random UUID (`obj_{i}_{uuid}.bin`) or object key | Deterministic from ObjectKey (`{model}_{rank}_{hash}.bin`) |
| Nixl registration | Single prepped dlist for all storage | Per-operation register → transfer → deregister |
| Pool / page indices | `NixlObjPool` manages fixed slots | No pool; `NixlStoreObj.page_indices` unused (`[]`) |
| Capacity control | Pool size (slot count) | File: `max_capacity_gb`; object: unsupported |
| Persist/recover | Not supported | File: supported; object: presence lookup only |
| Batching | One DMA transfer per batch of keys | One DMA transfer per key (each key = separate file) |

### Key Components

#### `DynamicNixlStorageAgent`

Base class for dynamic NIXL storage agents. It owns the NIXL agent, L1 memory
registration, page-index calculation, transfer lifecycle, and shutdown. The
backend-specific subclasses operate on `ObjectKey` values rather than exposing
storage paths to the adapter; this keeps the shared mechanics reusable by both
file and object-storage backends.

#### `FileDynamicNixlStorageAgent`

The file-backed implementation registers L1 memory at initialization and
performs file registration per operation:

- `dynamic_store(mem_indices, key)` — create the key's data file,
  register with Nixl, DMA write, deregister, close fd.
- `dynamic_load(mem_indices, key)` — open the key's existing data file,
  register, DMA read, deregister, close fd.
- `dynamic_delete(key)` — delete the key's data file with `os.unlink()`.

#### `ObjectDynamicNixlStorageAgent`

The object-backed implementation registers deterministic object keys per
transfer. `dynamic_store` and `dynamic_load` use NIXL's `OBJ` memory type;
`get_stored_size` performs a NIXL presence query and returns zero on a hit
because object backends do not expose a backend-neutral size query.

#### `DynamicNixlStoreL2Adapter`

Same `L2AdapterInterface` contract as the static adapter. Differences:

- **Store:** Iterates per key, calling `dynamic_store` for each. File backends
  enforce `max_capacity_gb`; object backends do not enforce aggregate capacity.
- **Delete:** File backends remove data via `dynamic_delete`; object backend
  deletion is a no-op.
- **Capacity:** File backends track `_total_bytes` for the eviction controller.
  Object backends use `max_capacity_bytes=0`, disabling global eviction.
- **Close:** Stops the event loop first. File backends honor
  `persist_enabled`; object backends have no adapter-side cleanup.
- **Lookup:** A lookup miss always falls through to a synchronous
  secondary lookup on disk; see the Persist / Secondary Lookup section
  below.

### Operation Flow

#### Store
```
submit_store_task(keys, objects)
  └─ schedules _execute_store_in_the_loop on the asyncio loop
       ├─ for each key/object:
       │    ├─ check capacity for file backends
       │    ├─ derive deterministic file path or object key from ObjectKey
       │    ├─ register storage with Nixl, DMA write, deregister
       │    └─ record key→NixlStoreObj in _memory_objects, update _total_bytes
       └─ signals store event-fd
```

#### Load
```
submit_load_task(keys, objects)
  └─ schedules _execute_load_in_loop on the asyncio loop
       ├─ for each found key:
       │    ├─ derive file path or object key from ObjectKey
       │    └─ register storage with Nixl, DMA read, deregister
       └─ signals load event-fd
```

Lookup and unlock are identical to the static adapter (in-memory index
lookup + pin count management).

---

## Persist / Secondary Lookup

### Config

`PersistConfig` (`l2_adapters/config.py`) has one boolean flag:

| Field | Default | Purpose |
|---|---|---|
| `persist_enabled` | `True` | If True, data files are kept on disk at shutdown. |

Parsed from the adapter JSON config key `"persist_enabled"` by
`L2AdapterConfigBase._parse_persist_config()`.

Lookup always checks secondary storage on miss — this is not configurable.

Only dynamic file backends use `persist_enabled`; static and dynamic object
backends ignore it.

### How it works

There is no dedicated `persist()` or `recover()` method on the
`L2AdapterInterface`. Persist and recover are implemented implicitly
through two existing hooks:

#### Persist (file retention at shutdown)

In `close()`, after the event loop has stopped:

- If `persist_enabled`, data files are left on disk untouched.
- Otherwise, every file in `_memory_objects` is `os.unlink`'d to avoid
  orphaned storage.

No metadata JSON is written — the deterministic `ObjectKey → filename`
mapping is sufficient to rediscover each file on restart.

#### Secondary Lookup

For file backends, `_execute_lookup_in_the_loop` extends the in-memory index
lookup with a secondary lookup on miss:

1. Compute deterministic file path from the ObjectKey.
2. `os.stat(file_path)` — if the file exists, treat as a hit.
3. Populate `_memory_objects[key]` lazily with `size` from the stat
   result and `layout=None`.
4. Update `_total_bytes`; enforce capacity (skip if it would exceed).

For object backends, secondary lookup derives the deterministic object key and
uses NIXL `query_memory`. A hit creates an entry with `size=0`, because object
stores do not provide a backend-neutral size query.

The `NixlStoreObj.layout` field is left as `None` on secondary lookup. Layout
information is only needed at load time, where the caller supplies it
via the provided `MemoryObj`'s shape/dtype/phy_size.

---

## Configuration

### Static Adapter (`nixl_store`)

```json
{
  "type": "nixl_store",
  "backend": "POSIX",
  "backend_params": {
    "file_path": "/path/to/storage",
    "use_direct_io": "false"
  },
  "pool_size": 100
}
```

### Dynamic Adapter (`nixl_store_dynamic`)

File backend:

```json
{
  "type": "nixl_store_dynamic",
  "backend": "POSIX",
  "backend_params": {
    "file_path": "/path/to/storage",
    "use_direct_io": "false",
    "max_capacity_gb": "10"
  },
  "persist_enabled": true
}
```

OBJ backend:

```json
{
  "type": "nixl_store_dynamic",
  "backend": "OBJ",
  "backend_params": {
    "bucket": "<bucket_name>"
  }
}
```

AZURE_BLOB backend:

```json
{
  "type": "nixl_store_dynamic",
  "backend": "AZURE_BLOB",
  "backend_params": {
    "account_url": "https://<account_name>.blob.core.windows.net",
    "container_name": "<container_name>"
  }
}
```

---
