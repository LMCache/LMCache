# PD L2 Adapter — Architecture Design

## Overview

The PD L2 Adapter enables cross-node KV Cache transfer between prefill and decode instances in LMCache Multiprocess (MP) mode. It implements `L2AdapterInterface` and plugs into the existing StoreController / PrefetchController eventfd-driven polling loop with zero changes to controller code.

## System Architecture

```
  Prefill Node                                     Decode Node
 ┌──────────────────-───────┐                      ┌──-───────────────────────┐
 │  vLLM Engine             │                      │  vLLM Engine             │
 │       │                  │                      │       ▲                  │
 │       ▼                  │                      │       │                  │
 │  StorageManager          │                      │  StorageManager          │
 │   ├─ StoreController     │                      │   ├─ PrefetchController  │
 │   │   poll(store_efd)    │                      │   │   poll(lookup_efd,   │
 │   │       │              │                      │   │        load_efd)     │
 │   │       ▼              │                      │   │       │              │
 │   │  PdL2Adapter         │                      │   │  PdL2Adapter         │
 │   │  (role=sender)       │                      │   │  (role=receiver)     │
 │   │       │              │                      │   │       ▲              │
 └───┼───────┼──────────────┘                      └───┼───────┼──────────────┘
     │       │                                         │       │
     │       │    ┌──────────────────────────────┐     │       │
     │       └───►│  ZMQ (alloc request/response)│─────┘       │
     │            └──────────────────────────────┘             │
     │            ┌──────────────────────────────┐             │
     │            │  TransferChannel (NIXL/RDMA) │─────────────┘
     │            └──────────────────────────────┘
     │            ┌──────────────────────────────┐
     └───────────►│  ZMQ PUSH → Proxy (notify)   │
                  └──────────────────────────────┘
```

## Data Flow

### Sender Store Path

```
StoreController                    PdL2Adapter (sender)                   Receiver
      │                                  │                                    │
      │  submit_store_task(keys, objs)   │                                    │
      │─────────────────────────────────►│                                    │
      │                                  │  AllocRequest (ZMQ)                │
      │                                  │───────────────────────────────────►│
      │                                  │                    AllocResponse   │
      │                                  │◄───────────────────────────────────│
      │                                  │                                    │
      │                                  │  L1 → staging buffer (copy_)       │
      │                                  │  TransferChannel.batched_write()   │
      │                                  │───────────────────────────────────►│
      │                                  │                                    │
      │                                  │  eventfd_write(store_efd, 1)       │
      │  pop_completed_store_tasks()     │                                    │
      │◄─────────────────────────────────│                                    │
```

### Receiver Load Path

```
PrefetchController                     PdL2Adapter (receiver)
      │                                      │
      │  submit_lookup_and_lock_task(keys)   │
      │─────────────────────────────────────►│
      │                                      │  check staging data dict
      │                                      │  eventfd_write(lookup_efd, 1)
      │  query_lookup_and_lock_result()      │
      │◄─────────────────────────────────────│  return Bitmap
      │                                      │
      │  submit_load_task(keys, l1_objs)     │
      │─────────────────────────────────────►│
      │                                      │  staging → L1 MemoryObj (copy_)
      │                                      │  eventfd_write(load_efd, 1)
      │  query_load_result()                 │
      │◄─────────────────────────────────────│  return Bitmap
      │                                      │
      │  submit_unlock(keys)                 │
      │─────────────────────────────────────►│
```

## Task State Machine

```
submit_xxx_task()
        │
        ▼
  ┌───────────┐      async execution      ┌────────────┐
  │  PENDING  │───────────────────────-──►│  COMPLETED │
  │  (queued) │                           │  (in dict) │
  └───────────┘                           └──────┬─────┘
                                                 │
                                           eventfd_write()
                                                 │
                                                 ▼
                                        query/pop → return once → remove
```

- Thread-safe task ID allocation via `self._lock`
- Separate completion dicts per operation type (store / lookup / load)
- Single-consume semantics on query results

## Key Components

| Component           | Location           | Description                                                     |
|---------------------|--------------------|-----------------------------------------------------------------|
| `PdL2AdapterConfig` | `pd_l2_adapter.py` | Typed config: role, peer host/ports, buffer, channel settings   |
| `PdL2Adapter`       | `pd_l2_adapter.py` | `L2AdapterInterface` impl with 3 eventfds, async task execution |
| `key_mapper`        | `key_mapper.py`    | Lossless `ObjectKey ↔ string` serialization for wire protocol   |
| `ProxyNotif`        | `pd_l2_adapter.py` | ZMQ PUSH notification to proxy on last prefill completion       |

## Thread Model

```
StoreController thread ────► submit_store_task() ───┐
                                                    │  self._lock
PrefetchController thread ─► submit_lookup/load() ──┤
                                                     ▼
                                             Adapter async loop thread
                                               ├─ _execute_store()
                                               ├─ _execute_lookup()
                                               ├─ _execute_load()
                                               └─ _mem_alloc_loop() (receiver)
```

All shared state protected by `threading.Lock`. Eventfds are inherently thread-safe.
