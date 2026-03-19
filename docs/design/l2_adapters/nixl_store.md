# NixlStoreL2Adapter Design

## Overview

`NixlStoreL2Adapter` implements `L2AdapterInterface` using the
[Nixl](https://github.com/ai-infra-org/nixl) library to offload KV-cache
objects from L1 (DRAM/VRAM) to a secondary storage tier via DMA. Supported
backends include file-based storage (GDS, GDS_MT, POSIX, HF3FS) and
object-based storage (OBJ).

---

## Key Components

### `NixlStoreObj`
Metadata record for a single cached object in Nixl storage:
- `page_indices` — list of pre-allocated storage slot indices holding the data.
- `size` — byte size of the stored object.
- `layout` — optional `MemoryLayoutDesc` (shapes/dtypes) for reconstruction.
- `pin_count` — reference count preventing eviction while a load is in flight.

### `NixlObjPool`
Thread-safe integer index pool representing the fixed set of pre-allocated
storage slots (`pool_size` entries). Slots are allocated before a store and
freed after a failed transfer or when the object is evicted.

### `NixlStorageAgent`
Thin wrapper around the Nixl agent API. Responsibilities:
- Register the L1 memory buffer with Nixl (`init_mem_handlers`).
- Register storage slots (files or object keys) with Nixl
  (`init_storage_handlers_file` / `init_storage_handlers_object`).
- Produce pre-prepared transfer handles for batched DMA reads/writes
  (`get_mem_to_storage_handle`, `get_storage_to_mem_handle`).
- Drive transfers asynchronously (`post_non_blocking`).

### `NixlStoreL2Adapter`
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