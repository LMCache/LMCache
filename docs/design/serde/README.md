# Serde Integration for Distributed Storage Controllers

## Overview

Serialization/deserialization (serde) support for the L1-L2 data path. When enabled, all data transferred between L1 and L2 passes through a `SerdeProcessor` that compresses/decompresses using temporary byte buffers allocated from L1.

When serde is disabled (`None`), both controllers behave identically to the existing code path.

## Architecture

Two-layer interface:
- **Sync** (`Serializer`/`Deserializer`): users implement pure transform logic
- **Async** (`SerdeProcessor`): eventfd-driven interface consumed by controllers, wraps sync implementations via `AsyncSerdeProcessor`

The `SerdeProcessor` follows the same submit/eventfd/query pattern as L2 adapters, integrating naturally into the poll-based event loops.

**Serde is per-adapter:** each L2 adapter has its own optional `SerdeProcessor`. A single prefetch/store request can mix serde-enabled and serde-disabled adapters within its plan. Each adapter independently decides whether to use temp buffers and async serialize/deserialize.

## Module Layout

```
lmcache/v1/distributed/serde/
  base.py             # Serializer, Deserializer (sync ABCs)
                      # SerdeProcessor (async ABC)
  async_processor.py  # AsyncSerdeProcessor (thread pool wrapper)
  utils.py            # serialized_layout_desc, make_temp_key
```

## Store Controller Data Flow

```mermaid
sequenceDiagram
    participant L1Mgr
    participant StoreListener
    participant Main Loop
    participant L2Mgr
    participant SerdeMgr as SerdeProcessor

    L1Mgr ->> StoreListener: on_write_finished(keys)
    activate StoreListener
    StoreListener ->> Main Loop: submit store "request"
    deactivate StoreListener
    activate Main Loop
    Main Loop ->> Main Loop: determine keys to store in L2
    Main Loop ->> L1Mgr: reserve_read(l1_keys)
    L1Mgr ->> Main Loop: l1_keys

    opt some reserve_read failed
        Main Loop ->> Main Loop: skip failed keys (best-effort)
    end

    Main Loop ->> Main Loop: storage_keys = l1_keys

    rect rgba(255, 200, 100, 0.2)
        alt if serde is enabled (this adapter)
            Main Loop ->> L1Mgr: reserve_write(tmp_keys, byte layout)
            L1Mgr ->> Main Loop: reserved tmp_keys

            opt some reserve_write(tmp) failed
                Main Loop ->> L1Mgr: finish_read(failed l1_keys)
                Main Loop ->> L1Mgr: finish_write(failed tmp_keys)
                Main Loop ->> L1Mgr: delete(failed tmp_keys)
                Main Loop ->> Main Loop: drop failed keys from batch
            end

            Main Loop ->> SerdeMgr: submit_serialize(l1_objs, tmp_objs)
            SerdeMgr ->> Main Loop: serde task id
            deactivate Main Loop
            activate SerdeMgr
            SerdeMgr ->> SerdeMgr: serialize(l1_objs -> tmp_objs)
            SerdeMgr -->> Main Loop: serialize_fd is ready
            deactivate SerdeMgr
            activate Main Loop
            Main Loop ->> SerdeMgr: query_serialize_result(serde task id)
            SerdeMgr ->> Main Loop: success / failure

            alt serialize succeeded
                Main Loop ->> L1Mgr: finish_read(l1_keys)
                Main Loop ->> L1Mgr: finish_write_and_reserve_read(tmp_keys)
                Main Loop ->> Main Loop: storage_keys = tmp_keys
            else serialize failed
                Main Loop ->> L1Mgr: finish_read(l1_keys)
                Main Loop ->> L1Mgr: finish_write(tmp_keys)
                Main Loop ->> L1Mgr: delete(tmp_keys)
                Main Loop ->> Main Loop: abort (no L2 store)
            end
        end
    end

    Main Loop ->> L2Mgr: submit_store_task(storage_keys)
    L2Mgr ->> Main Loop: store task id
    deactivate Main Loop
    activate L2Mgr
    L2Mgr ->> L2Mgr: execute real store operation
    L2Mgr -->> Main Loop: event fd is ready
    deactivate L2Mgr
    activate Main Loop
    Main Loop ->> L2Mgr: pop completed tasks
    L2Mgr ->> Main Loop: completed task id
    Main Loop ->> Main Loop: update internal "request" status and check finished "requests"
    rect rgba(200, 220, 255, 0.2)
        alt no serde (storage_keys = l1_keys)
            Main Loop ->> L1Mgr: finish_read(l1_keys)
        else serde (storage_keys = tmp_keys)
            Main Loop ->> L1Mgr: finish_read(tmp_keys)
            L1Mgr ->> L1Mgr: auto-delete(tmp_keys)
            note right of L1Mgr: is_temporary=True → finish_read<br/>triggers automatic deletion from L1.
        end
    end

    alt L2 store succeeded
        Main Loop ->> Main Loop: determine keys to delete from L1
        Main Loop ->> L1Mgr: delete(l1_keys)
    else L2 store failed
        Main Loop ->> Main Loop: log warning, no L1 deletion
    end

    deactivate Main Loop
```

## Prefetch Controller Data Flow

```mermaid
sequenceDiagram
  participant Main Loop
  participant L1Mgr
  participant L2Mgr
  participant SerdeMgr as SerdeProcessor

  Main Loop ->> Main Loop: execute a new prefetch task with keys
  activate Main Loop 
  Main Loop ->> L2Mgr: submit_lookup_and_lock(keys)
  L2Mgr ->> Main Loop: lookup_and_lock task id
  deactivate Main Loop
  activate L2Mgr
  L2Mgr ->> L2Mgr: execute the lookup and lock
  L2Mgr -->> Main Loop: trigger lookup event fd
  deactivate L2Mgr
  activate Main Loop 
  Main Loop ->> L2Mgr: check lookup and lock results
  L2Mgr ->> Main Loop: lookup and lock results
  Main Loop ->> Main Loop: update task internal states
  note right of Main Loop: ...wait for other L2 mgr lookup
  Main Loop ->> Main Loop: determine real load plan
  Main Loop ->> L1Mgr: reserve_write(keys_in_the_plan, KV layout)
  L1Mgr ->> Main Loop: reserved keys_in_the_plan

  opt some reserve_write(real) failed
    Main Loop ->> Main Loop: drop failed keys, recompute prefix
  end

  Main Loop ->> Main Loop: load_buffers = keys_in_the_plan

  rect rgba(255, 200, 100, 0.2)
    alt if serde is enabled (this adapter)
      Main Loop ->> L1Mgr: reserve_write(tmp_keys, byte layout)
      L1Mgr ->> Main Loop: reserved tmp_keys

      opt some reserve_write(tmp) failed
        Main Loop ->> L1Mgr: finish_write(real keys whose tmp failed)
        Main Loop ->> L1Mgr: delete(real keys whose tmp failed)
        Main Loop ->> Main Loop: drop failed keys, recompute prefix
      end

      Main Loop ->> Main Loop: load_buffers = tmp_keys
    end
  end

  Main Loop ->> Main Loop: update the real load plan
  Main Loop ->> L2Mgr: unlock(keys_not_in_the_plan)
  Main Loop ->> L2Mgr: submit_load_task(keys_in_the_plan, load_buffers)
  L2Mgr ->> Main Loop: load task id
  deactivate Main Loop
  activate L2Mgr
  L2Mgr ->> L2Mgr: execute the load (not release locks)
  L2Mgr -->> Main Loop: update load fd 
  deactivate L2Mgr
  activate Main Loop
  Main Loop ->> L2Mgr: query_load_status(load task id)
  Main Loop ->> L2Mgr: unlock(keys in the plan)

  rect rgba(255, 200, 100, 0.2)
    alt if serde is enabled (this adapter)
      Main Loop ->> SerdeMgr: submit_deserialize(tmp_keys, keys_in_the_plan)
      SerdeMgr ->> Main Loop: serde task id
      deactivate Main Loop
      activate SerdeMgr
      SerdeMgr ->> SerdeMgr: deserialize(tmp_keys -> keys_in_the_plan)
      SerdeMgr -->> Main Loop: deserialize_fd is ready
      deactivate SerdeMgr
      activate Main Loop
      Main Loop ->> SerdeMgr: query_deserialize_result(serde task id)
      SerdeMgr ->> Main Loop: success / failure

      Main Loop ->> L1Mgr: finish_write(adapter_tmp_keys)
      Main Loop ->> L1Mgr: delete(adapter_tmp_keys)
      note right of Main Loop: Temp buffers released regardless of<br/>deserialize success or failure.

      alt deserialize failed
        Main Loop ->> Main Loop: zero adapter's load_result bitmap
        note right of Main Loop: Affected keys treated as "failed"<br/>in the finalize step below.
      end
    end
  end

  Main Loop ->> Main Loop: compute loaded vs failed from result bitmap
  note right of Main Loop: failed = write_reserved but not in loaded bitmap <br> (includes surplus non-prefix keys)
  Main Loop ->> L1Mgr: finish_write_and_reserve_read(loaded_keys) if successful
  Main Loop ->> L1Mgr: finish_write(failed_keys) + delete(failed_keys)

  Main Loop ->> Main Loop: compute prefix hits
  opt loaded keys beyond prefix (gap from partial load failure)
    Main Loop ->> L1Mgr: finish_read(non_prefix_loaded_keys)
  end

  Main Loop ->> Main Loop: update task status
  deactivate Main Loop
```

## Event Loop Integration

Both controllers register serde eventfds in their poll loop alongside L2 adapter fds:

| Controller | Polls | Handler |
|---|---|---|
| Store | `serialize_event_fd` | `_process_serialize_and_submit_l2_store` |
| Prefetch | `deserialize_event_fd` | `_process_deserialize_completions` |

## Buffer Sizing

Temp buffers are allocated at exactly `estimate_serialized_size()` bytes. The serializer is responsible for including any safety margin in its estimate (e.g., the built-in fp8 serializer returns `1.5 * num_elements`).

