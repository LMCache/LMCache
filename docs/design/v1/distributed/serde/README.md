# `lmcache.v1.distributed.serde` — Serialization / Deserialization Package

## Scope

This package holds the **generic serde primitives** used by the
distributed storage stack. It does not know anything about L2 adapters
or controllers — it just defines:

- A pair of **sync interfaces** users implement (`Serializer`,
  `Deserializer`) for the actual byte transform.
- An **async interface** (`SerdeProcessor`) with the same
  `submit → eventfd → query` shape as an L2 adapter, so downstream
  consumers can poll it uniformly.
- A default implementation (`AsyncSerdeProcessor`) that turns any pair
  of sync implementations into the async interface by running them in
  a thread pool and signaling an eventfd on completion.
- A factory / registration mechanism so adapters can reference a serde
  by name (`{"type": "fp8", ...}` in JSON config).
- One built-in serde: **fp8 quantization**.

How the async interface is actually plugged into the L2 path lives in
[`docs/design/v1/distributed/l2_adapters/serde_wrapper.md`][wrapper-doc]
— the wrapper is the sole consumer of `SerdeProcessor`'s event fds.

[wrapper-doc]: ../l2_adapters/serde_wrapper.md

## Module Layout

```
lmcache/v1/distributed/serde/
  base.py             # Serializer, Deserializer (sync ABCs)
                      # SerdeProcessor (async ABC)
                      # SerdeConfig, SerdeTaskId
  async_processor.py  # AsyncSerdeProcessor (thread-pool + eventfd wrapper)
  factory.py          # register_serde_factory / create_serde_processor
  fp8.py              # Fp8QuantizationSerializer / Deserializer
  utils.py            # serialized_layout_desc, make_temp_key
```

## Two-Layer Interface

```
                 ┌────────────────────────────────┐
 user writes →   │  Serializer / Deserializer     │  sync transform
                 │  (pure: src MemoryObj → dst)   │
                 └──────────────┬─────────────────┘
                                │ wrapped by AsyncSerdeProcessor
                                ▼
                 ┌────────────────────────────────┐
 wrapper uses →  │  SerdeProcessor                │  async:
                 │    submit_serialize(...) → id  │   submit / eventfd /
                 │    query_serialize_result(id)  │   query
                 │    get_serialize_event_fd()    │
                 │    (plus deserialize pair)     │
                 └────────────────────────────────┘
```

- **Sync layer** is where the user cares. Pure Python (or torch) code,
  no threads, no fds. Two abstract methods: `serialize(src, dst)` and
  `estimate_serialized_size(layout_desc)`.
- **Async layer** is what the `SerdeL2AdapterWrapper` talks to. It
  owns two eventfds (one for serialize, one for deserialize) that
  must be distinct, and queues completed tasks in a dict the wrapper
  drains.

`AsyncSerdeProcessor` is the default and typically only implementation
of the async layer — most custom serdes only need to provide the sync
classes and register a factory.

## Contracts

### `Serializer.serialize(src, dst) -> int`

- `src` is a `MemoryObj` holding KV data (read-locked by the caller).
- `dst` is a `MemoryObj` byte buffer (write-locked by the caller),
  sized ≥ `estimate_serialized_size(layout_of_src)`.
- Must return the number of bytes actually written to `dst`.
- Must be **deterministic** given the same `src` — the wrapper relies
  on the serialize step being reproducible across retries.

### `Serializer.estimate_serialized_size(layout_desc) -> int`

- Called once per batch to size the temp buffer **before** any work.
- Must be an **upper bound** on the actual serialized output. Include
  any safety margin inside this method (the fp8 serializer returns
  `1.5 × num_elements` for exactly this reason).
- Must only depend on `layout_desc` (shapes + dtypes). The wrapper
  uses the first object's layout to size temps for the whole batch,
  so a data-dependent estimate would break all-or-nothing allocation.

### `Deserializer.deserialize(src, dst) -> None`

- `src` is a byte-buffer MemoryObj filled by L2 load.
- `dst` is a KV-shaped MemoryObj (write-locked), already the correct
  shape and dtype.
- No return value — the caller observes completion via the async
  layer's event fd.

### `SerdeProcessor` (async)

- `submit_serialize(src_objs, dst_objs) → SerdeTaskId` must be
  non-blocking. The actual transform runs asynchronously.
- `query_serialize_result(task_id) → bool | None` is
  **non-idempotent**: it returns a non-None value exactly once per
  task id. `None` means the task is still in flight.
- `get_serialize_event_fd()` returns an eventfd signaled once per
  completed serialize task; distinct from the deserialize fd.
- The deserialize side has the identical shape.
- `close()` must release both event fds and any worker threads.

## `AsyncSerdeProcessor`

Wraps any `(Serializer, Deserializer)` pair. Internal design:

- One `ThreadPoolExecutor(max_workers=N)` runs both serialize and
  deserialize tasks; they're independent, so the pool is shared.
- One task-id counter under a single lock; two `_completed_*` dicts
  (one per direction) protected by the same lock.
- Two `os.eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)` file descriptors —
  one per direction, signaled after the result is written to the
  completion dict.

**Why a pool, not an asyncio executor:** the CPU-bound fp8 / encryption
transforms release the GIL under torch / native calls, so a real
thread pool is useful. `N=1` is a safe default (one in-flight
transform at a time), and the fp8 factory accepts `max_workers` to
bump it.

## Factory / Registration

```python
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor, Deserializer, Serializer,
    register_serde_factory,
)

def _create_my_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    return AsyncSerdeProcessor(MySerializer(...), MyDeserializer(...))

register_serde_factory("mine", _create_my_serde)
```

The factory receives the type-specific kwargs from the JSON config
(everything except `"type"`). The registry is process-global and
rejects duplicate names, matching the pattern already used by
`register_l2_adapter_type`.

The factory is called exactly once per `SerdeL2AdapterWrapper`
construction — each wrapped adapter gets its own `SerdeProcessor`
instance.

## Built-in fp8

`Fp8QuantizationSerializer` / `Fp8QuantizationDeserializer`:

- Cast each element to `torch.float8_e4m3fn` (default) or
  `torch.float8_e5m2`, reinterpret the bytes as `uint8`, and copy into
  the temp buffer.
- Deserialize reinterprets the `uint8` bytes as the chosen fp8 dtype,
  reshapes back to the original KV shape, and casts to the destination
  tensor's dtype.
- `estimate_serialized_size` returns `int(total_elements × 1.5)` — the
  exact fp8 size is `num_elements × 1 byte`, and the 1.5× headroom
  absorbs future format changes or alignment padding.

## Extension Guide

Most custom serdes only need:

1. Two classes implementing `Serializer` / `Deserializer`.
2. A factory function registered at import time.
3. A JSON `serde` sub-dict on the adapter config: `{"type": "mine", ...}`.

Everything else — temp buffer allocation, eventfd plumbing, lock /
lifecycle transitions, all-or-nothing failure handling — is provided
by `AsyncSerdeProcessor` and the wrapper. Users never need to touch
controllers, eventfds, or L1 locks.
