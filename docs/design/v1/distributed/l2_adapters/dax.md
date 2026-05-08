# DAX L2 Adapter Design

This document describes the built-in `dax` L2 adapter for LMCache
multiprocess mode and how it shares implementation with the non-MP DAX
storage backend.

## Goals

- Reuse one synchronous DAX core for MP and non-MP DAX storage.
- Keep the MP controller flow unchanged: adapters still use the normal
  submit, event-fd, and query-result contract.
- Keep PR1 volatile-only. Keys are indexed in process memory and are not
  recovered from device bytes after restart.

## Components

`lmcache/v1/storage_backend/dax/core.py` defines `DaxCore[KeyT]`. The core owns
the mapped DAX arena, fixed-size slot allocation, in-memory index, LRU order,
in-flight writes, external lock refcounts, active read borrow counts, close
coordination, and direct `ctypes.memmove` copies.

`lmcache/v1/storage_backend/plugins/dax_backend.py` is the non-MP wrapper. It
keeps existing non-MP behavior such as the local CPU backend requirement, TP=1
validation, optional async put, and the staging-slab batched restore path.

`lmcache/v1/distributed/l2_adapters/dax_l2_adapter.py` is the MP adapter. It
self-registers adapter type `dax`, owns separate event notifiers and worker
pools for store, lookup, and load operations, and uses `DaxCore[ObjectKey]`.

## Slot State

Each committed key points to one fixed-size slot. A slot is reusable only when:

- The key has been removed from the index.
- No external lock is held for the key.
- No store is in flight for the key.
- No active read has borrowed the slot.

Delete operations remove unlocked keys from the index immediately. If a read
has borrowed the slot, the slot is marked pending-free and recycled when the
borrow count reaches zero.

## MP Flow

Store:

1. `StoreController` calls `submit_store_task(keys, objects)`.
2. A store worker copies each object into a DAX slot through `DaxCore.put_many`.
3. The adapter records task-level success as `all(per_key_results)`.
4. The store event fd is signaled and store listeners are notified for the keys
   that were actually accepted by the core.

Lookup and load:

1. `PrefetchController` calls `submit_lookup_and_lock_task(keys)`.
2. The adapter calls `DaxCore.exists_many(keys, lock=True)` and returns a full
   bitmap, including holes.
3. Load workers call `DaxCore.load_many_into(keys, objects)` to copy directly
   from the mapped arena into caller-provided L1 buffers.
4. `submit_unlock(keys)` releases the external lock refcounts acquired by
   lookup.

## Restart Behavior

The adapter stores keys and metadata only in memory. Closing the adapter and
opening a new adapter against the same DAX device starts with an empty index.
Old bytes may remain on the device, but they are unreachable because PR1 does
not define any on-device metadata, scan, checkpoint, or recovery format.

## Capacity And Eviction

Usage is slot-based, not payload-byte-based. `get_usage()` reports occupied
slot capacity because the DAX arena is exhausted by slot count. The eviction
controller calls `delete(keys)`, which skips externally locked keys and
reclaims slots after active read borrows drain.

## Current Limits

- One server-owned mapped DAX path per adapter instance.
- No per-TP partitioning and no multi-device striping.
- No restart recovery.
- Only single-buffer objects are supported.
