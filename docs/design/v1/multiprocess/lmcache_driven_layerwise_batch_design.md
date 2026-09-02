# Layerwise Batch KV Loading: Design

## 1. Overview

`--layerwise-batch N` enables layer-major H2D KV cache transfer in
LMCache's multiprocess (ZMQ) mode.  Instead of copying all layers for
each chunk (chunk-major), the server copies all chunks for N layers at
a time (layer-major), records one IPC event per batch, and streams
the event handle to the worker immediately so worker can start attention
on those layers while later batches are still transferring.

Key parameters:
- `N = 0` (default): layerwise disabled; all layers transferred at once
  per chunk (chunk-major, original path).
- `N = 1`: per-layer transfer; one event per layer.
- `N > 1`: batch N consecutive layers per H2D + scatter, one event per
  batch, best balance of event overhead vs. pipeline granularity.

For a model with L total layers and `--layerwise-batch N`,
the server produces `ceil(L / N)` batches, each covering up to N
consecutive same-kernel-group layers.  Each batch triggers one IPC
event and one intermediate response frame to the worker.

### 1.1 Supported Scope

Layer-wise loading is supported for the **LMCache-driven transfer mode on
CUDA only**. It is not supported with `--supported-transfer-mode
engine_driven`, which never loads the layer-wise module, nor on non-CUDA
platforms: only the CUDA and ROCm build profiles compile
`csrc/cuda/mp_mem_kernels_layerwise.cu`, and the `layerwise` argument exists
only in the `cuda_ops` bindings.

Neither restriction is enforced at startup today. `--layerwise-batch N` is
accepted and silently ignored under `engine_driven`, and on a non-CUDA build
the server starts and registers normally but the first layer-wise retrieve
fails inside the native call. Widening the scope -- and enforcing it -- is
deferred until the CUDA path is validated upstream.

---

## 2. Architecture

```
+----------------------- Server Process (lmcache_server) ----------------------+
|                                                                              |
|  +----------------------+    +------------------------------------------+    |
|  |  MQ Main-Loop Thread |    |  Affinity Pool Thread                    |    |
|  |  (mq-server-thread)  |    |  (one per ZMQ client, i.e. per vLLM wkr) |    |
|  |                      |    |                                          |    |
|  |  * zmq ROUTER recv   |    |  Runs: retrieve() handler                |    |
|  |  * dispatch to pool  |    |                                          |    |
|  |  * zmq ROUTER send   |    |  (Conditional) CPU: read chunks from L2  |    |
|  |    (per-batch frames)|    |                                          |    |
|  |                      |    |  +-- per batch (N layers) -----------+   |    |
|  |  Drains output_queue |    |  | 1. GPU: H2D memcpy                |   |    |
|  |  into ZMQ socket     |    |  | 2. GPU: scatter kernel            |   |    |
|  |                      |    |  | 3. CPU: record_event(pool[i])     |   |    |
|  |                      |    |  | 4. CPU: response_channel(pool_idx)|   |    |
|  |                      |    |  |      +-> output_queue.put()       |   |    |
|  |                      |    |  +-----------------------------------+   |    |
|  +------+---------------+    +------------------------------------------+    |
|         | ZMQ ipc://                                                         |
+---------+--------------------------------------------------------------------+
          |
          |  partial frames (pool index, int)
          |  or final frame  (completion status)
          |
+---------+-------------------- Worker Process (vLLM) -------------------------+
|         |                                                                    |
|  +------v---------------+    +------------------------------------------+    |
|  |  Client Polling Loop |    |  vLLM Model Runner Thread                |    |
|  |  (singleton thread)  |    |                                          |    |
|  |                      |    |  +-- per layer (0..L-1) -------------+   |    |
|  |  * zmq DEALER recv   |    |  |                                   |   |    |
|  |  * !final -> queue   |    |  | 1. CPU: wait_for_layer(layer_idx) |   |    |
|  |  * final  -> set_rslt|    |  |    1st in batch: drain _partial_q |   |    |
|  |                      |    |  |  + pool.event_at(idx) + wait_event|   |    |
|  |  queue.Queue links:  |    |  |    rest N-1: cached event, no-op  |   |    |
|  |   -> _partial_queue  |    |  |                                   |   |    |
|  |                      |    |  | 2. GPU: attention(layer_idx)      |   |    |
|  |   -> set_result()    |    |  +-----------------------------------+   |    |
|  |                      |    |                                          |    |
|  |                      |    |  Reads from:                             |    |
|  |                      |    |   LayerwiseDeviceMessagingFuture         |    |
|  +-----------------------+    +------------------------------------------+   |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## 3. Pipeline: Streaming ZMQ Partial Frames

With `--layerwise-batch N` and L total layers (ceil(L/N) batches), the per-batch
flow on the server affinity pool thread is:

```
Batch 0 (layers 0 .. N-1):
  +-- H2D memcpy: num_chunks x N x per_layer_bytes        (GPU)
  +-- scatter kernel: interleaved -> per-layer KV blocks  (GPU)
  +-- record_event(layer_events[0], server_stream)        (CPU)
  +-- response_channel(pack(0, N, pool_idx=0))            (CPU)
       +-> output_queue -> MQ main loop -> ZMQ ROUTER -> wire

Batch 1 (layers N .. 2N-1):
  +-- H2D memcpy ...
  +-- scatter kernel ...
  +-- record_event(layer_events[N], server_stream)
  +-- response_channel(pack(N, N, pool_idx=N))

  ... (batches 2 .. ceil(L/N)-2 identical pattern) ...

Batch ceil(L/N)-1 (last N layers):
  +-- H2D + scatter + record_event(pool[last])
  +-- response_channel(pack((ceil(L/N)-1)*N, N, pool_idx))

Handler returns ([], True)
  +-> done-callback -> output_queue -> final frame
```

**Worker-side** (vLLM model runner thread), concurrently:

```
Layer  0: wait_for_layer(0)
            -> _drain_until_layer(0) blocks on partial_queue.get()
            -> receives frame (0, N, pool_idx) -> pool.event_at(idx) -> wait_event
          attention(layer 0)  <-- GPU, overlaps with server batch 1 H2D

Layer  1: wait_for_layer(1)
            -> already in _layer_event_map (same batch as layer 0)
            -> evt is _last_waited_event -> skip (dedup)
          attention(layer 1)

  ... layers 2 .. N-1: same event, all dedup-skipped ...

Layer  N: wait_for_layer(N)
            -> not in map -> drain queue -> receives frame (N, N, pool_idx)
            -> pool.event_at(idx) -> wait_event
          attention(layer N)  <-- GPU, overlaps with server batch 2 H2D

  ... layers N+1 .. L-1: same pattern ...

All layers done -> future.wait() -> drain remaining -> synchronize last event
```

---

## 4. Contiguous Per-Batch H2D Memcpy

### 4.1 Host Memory Layout: kv_interleaved

When `--layerwise-batch > 0`, the store path (D2H) writes CPU
cache chunks in per-layer interleaved layout:

```
Per-chunk (kv_interleaved=False, default):
  chunk = [K_layer0, K_layer1, ..., K_layerN, V_layer0, V_layer1, ..., V_layerN]

Per-layer interleaved (kv_interleaved=True, layerwise mode):
  chunk = [K_layer0, V_layer0, K_layer1, V_layer1, ..., K_layerN, V_layerN]
```

This interleaved layout is carried by `PageBufferShapeDesc.kv_interleaved`
(`lmcache/v1/platform/ops_types.py`).

The flag is a **deployment-wide invariant**, not a per-call argument: it
is latched once in `LMCacheLayerwiseTransferModule._ensure_event_pool()`
on every kernel group's `shape_desc`, guarded by
`layerwise_batch > 0`.  That runs from the
`REGISTER_LAYERWISE_IPC_EVENT_POOL` handler, which the worker issues
immediately after `REGISTER_KV_CACHE` (see 6.1), so it is in place
before any transfer.  Both the store (D2H) and retrieve (H2D) paths
then simply read it.  Configuring it at registration — rather than latching
it on the first store — also keeps cold-start retrieves correct when a process
reads chunks written by a previous run before storing anything itself.

Consequently the per-chunk transfer helpers
(`_run_object_group_transfer_plan`, `transfer_kv_per_object_group`)
take no layout parameter and are byte-identical to the pre-layerwise
implementation.

**The flag is a no-op when `kv_size == 1`.**  Both layouts are produced
by the same expression in `calculate_lmcache_global_offset`, differing only
in where the layer stride sits relative to the K/V stride.

A layout mismatch between the D2H writer and the H2D reader is only observable
when `kv_size == 2`.

### 4.2 Interleaved Enables Contiguous Batch Memcpy

With the interleaved layout, N consecutive layers in one chunk occupy
a contiguous byte range in the CPU buffer:

```
chunk memory (interleaved):
  offset 0:                      [K0, V0]          <-- layer 0
  offset per_layer_bytes:        [K1, V1]          <-- layer 1
  ...
  offset (N-1)*per_layer_bytes:  [K_{N-1}, V_{N-1}]  <-- layer N-1

For batch of layers [i, i+N):
  src = chunk.data_ptr + kg_byte_offset + i * per_layer_bytes
  len = N * per_layer_bytes
  -> single contiguous memcpy per chunk
```

### 4.3 Merged H2D Path

When native ops are available and the staging buffer is large enough,
`transfer_kv_layerwise` uses the N-layer merged path:

1. **StagingCopy per batch N-layer per chunk**: source = `memory_obj.data_ptr +
   src_layer_offset`, size = `n_in_batch * per_layer_bytes`.  One
   contiguous memcpy copies N layers from one chunk in a single call.

2. **Scatter kernel**: `execute_object_group_transfer` with
   `PageBufferShapeDesc(nl=n_in_batch, kv_interleaved=True)`.  The
   GPU kernel reads the interleaved staging buffer and scatters to
   per-layer K/V paged block tensors.

3. **Fallback**: when native ops are absent or the staging buffer
   overflows, a per-layer fallback loop does N separate H2D copies
   (`multi_layer_block_kv_transfer` with single-layer sd).

---

## 5. IPC Event Pool

### 5.1 Motivation

Without pooling, each layerwise retrieve creates L events
(`cudaEventCreate`), exports B handles (`cudaIpcGetEventHandle`), 
and the worker imports B handles (`cudaIpcOpenEventHandle`). 

### 5.2 Design

A fixed pool of `EVENT_POOL_SIZE = 256` interprocess events is
pre-allocated per (context, worker) pair at **registration time**:

```
register_layerwise_ipc_event_pool(instance_id):
  1. pool = _ensure_event_pool(instance_id)     # creates pool on first call:
     a. assert num_total_layers <= EVENT_POOL_SIZE
     b. pool = EventPool(backend, device)        # 256 x cudaEventCreate
     c. pool.handles -> export all 256 events    # 256 x cudaIpcGetEventHandle
  2. return (layerwise_batch, pool.handles)      # REGISTER_LAYERWISE_IPC_EVENT_POOL response
```

The worker sends `REGISTER_LAYERWISE_IPC_EVENT_POOL` right after
`REGISTER_KV_CACHE` and imports all 256 handles **once**:

```
LMCacheLayerwiseTransferContext.register():
  1. super().register(...)                       # REGISTER_KV_CACHE -> None
  2. send_request(REGISTER_LAYERWISE_IPC_EVENT_POOL, [instance_id])
  3. layerwise_batch, pool_handles = future.result(timeout=mq_timeout)
  4. pool = EventPool.import_pool(backend, device, pool_handles)
     # 256 x cudaIpcOpenEventHandle — one-time cost at startup
```

### 5.3 Per-Request Hot Path (Zero Driver Calls)

During retrieve, the server indexes into the pre-allocated pool:

```
layer_events = [pool.event_at(i) for i in range(num_total_layers)]
...
record_event(pool_event[batch_leader], stream)   # cudaEventRecord only
response_channel((struct.pack("<3i", first_layer, count, batch_leader),
                  False, False))                 # int index, not a handle
```

The worker receives a pool index (int) and looks up the pre-imported
event — no `cudaIpcOpenEventHandle` on the forward path:

```
first_layer, count, pool_idx = struct.unpack("<3i", payload)
evt = pool.event_at(pool_idx)
stream.wait_event(evt)
```

### 5.4 Wire Encoding

Every frame is a `tuple[bytes, bool, bool]` -- `(payload, is_final,
succeeded)` -- so intermediate and closing frames share one response class:
- **Intermediate frame:** `payload = struct.pack("<3i", first_layer, count,
  pool_idx)`, `is_final = False`.
- **Closing frame:** `payload = struct.pack(f"<{L}i", *indices)` where L is the
  number of total layers (empty when indices were already reported
  frame by frame), `is_final = True`.

### 5.5 Invariants

- `num_total_layers <= EVENT_POOL_SIZE` — validated at registration; fails
  loudly otherwise.
- Pool is always present when layerwise is enabled (`assert event_pool`
  in retrieve).
- No fallback path exists — if the pool can't be created, registration
  fails.

---

## 6. `--layerwise-batch N` Configuration Flow

```
--layerwise-batch N
    |
    v
MPServerConfig(layerwise_batch=N)
    |
    v
MPCacheServerContext._layerwise_batch = N
    |
    +---> N = 0: server.py loads LMCacheDrivenTransferModule
    |            serves RETRIEVE (per-chunk) only
    |
    +---> N > 0: server.py loads LMCacheLayerwiseTransferModule
                 serves RETRIEVE_LAYERWISE; store writes kv_interleaved
```

The two modes are **mutually exclusive per server node**. A node started
with `N > 0` serves the layer-wise path exclusively and does not serve
per-chunk retrieves.

### 6.1 Pairing the Worker With the Server

The worker's mode is **not** negotiated at registration. It is fixed at
process start by which connector vLLM imports, so the operator must pair
the two sides explicitly:

| MP server | vLLM `--kv-transfer-config` |
|---|---|
| `--layerwise-batch 0` (or omitted) | `"kv_connector": "LMCacheMPConnector"` — no `kv_connector_module_path` |
| `--layerwise-batch N` (N > 0) | `"kv_connector": "LMCacheLayerwiseMPConnector"` **and** `"kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector_layerwise"` |

Layer-wise launch example:

```bash
vllm serve $MODEL \
  --kv-transfer-config '{
    "kv_connector": "LMCacheLayerwiseMPConnector",
    "kv_connector_module_path":
        "lmcache.integration.vllm.lmcache_mp_connector_layerwise",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://localhost",
      "lmcache.mp.port": 5555
    }
  }'
```

Confirm the pairing took effect: a correct layer-wise worker logs

```
Layerwise transfer context registered (batch=N, pool_size=...)
```


Notes: Keeping mode off `REGISTER_KV_CACHE` is deliberate: it leaves the
per-chunk protocol byte-for-byte unchanged when layer-wise is disabled.

---

## 7. Layout Uniformity & Mixed-Mode Considerations

### 7.1 Current Invariant

Layout is fixed per deployment: a server is uniformly per-layer
(`kv_interleaved=True`) or uniformly per-chunk
(`kv_interleaved=False`).  The `store` handler writes D2H in the
layout dictated by `layerwise_loading`, and the `retrieve` handler
reads the same layout.

This works because `layerwise_loading` is a server-level config, not
a per-request flag.  All chunks stored by this server instance use
the same interleaving.

### 7.2 Note: Rolling Upgrades & Shared L2

If servers sharing persistent L2 storage are upgraded from
`--layerwise-batch 0` to `N > 0` (or vice versa), stale chunks with
the old layout may remain.  Reading a chunk with mismatched layout
produces corrupt KV data.

In practice this is a non-issue: mixed deployments offer no benefit,
and old chunks are naturally evicted.  If rolling upgrades are needed,
flushing L2 between mode changes is sufficient.

**Exposure is per kernel group**  `kv_size` is a per-group property
-- one `PageBufferShapeDesc` per group spec -- so a single registration
can mix exposed and inert groups.  if model pairs a `kv_size == 2` main
K/V group with a `kv_size == 1` key-only indexer. Chunks from a `kv_size == 1`
group are always reusable whatever the flag says; only `kv_size == 2` groups
can observe a mismatch. `kv_size == 1` may be about fused K/V, MLA or key-only
side caches.

**Future alternative:** store `kv_interleaved` in each chunk's
`MemoryObjMetadata` at D2H time and have the token database lookup treat
`kv_size == 2 and stored != current` as a cache miss -- the stale entry is
never read, and a subsequent store overwrites it in the correct layout.
This avoids both silent corruption and wasted L2 read I/O, and lets the
system self-heal without a manual flush.

---

## 8. Multi-Frame ZMQ Responses

### 8.1 Protocol Surface

Three additions to the protocol layer; none of them modifies an existing
definition.

- **`HandlerType.STREAMING`** (`protocols/base.py`) -- a fourth handler
  type alongside `SYNC`, `BLOCKING` and `NON_BLOCKING`.  Declaring it on
  a request is what makes `add_handler` route to the streaming path.
- **Two `RequestType` members** (`protocols/base.py`):
  `REGISTER_LAYERWISE_IPC_EVENT_POOL` and `RETRIEVE_LAYERWISE`.
- **`protocols/layerwise.py`** -- a new module holding both definitions,
  so `protocols/engine.py` is untouched and the per-chunk protocol is
  byte-for-byte identical when layer-wise is disabled.

| Request Type | Handler type | Frames / request | Response class |
|---|---|---|---|
| `RETRIEVE` | `BLOCKING` | 1 | `tuple[bytes, bool]` |
| `RETRIEVE_LAYERWISE` | `STREAMING` | ceil(L/N) + 1 | `tuple[bytes, bool, bool]` |
| `REGISTER_LAYERWISE_IPC_EVENT_POOL` | `SYNC` | 1 | `tuple[int, list[bytes]]` |

`RETRIEVE_LAYERWISE` declares exactly the same `payload_classes` as
`RETRIEVE` (`[KeyType, int, list[list[int]], bytes, int]`); only the
response widens, from `(handle, succeeded)` to
`(payload, is_final, succeeded)`.  A dedicated request type keeps the
plain `RETRIEVE` dispatch path completely untouched.

One response class is declared per request type, and
`get_response_class(RETRIEVE_LAYERWISE)` is what
`_call_streaming_handler` encodes *every* frame with.  Intermediate and
closing frames are therefore the same msgspec type on the wire; the
transport never learns that some of them are partial.  `is_final` is the
only field that distinguishes them -- see 8.3.

`REGISTER_LAYERWISE_IPC_EVENT_POOL` is an ordinary `SYNC` request, kept
off `REGISTER_KV_CACHE` so registration keeps its plain `None` response
for every non-layer-wise deployment (see 6.1).

### 8.2 How the Message Queue Stays Neutral

Both file pairs follow the same shape: the layer-wise module subclasses
the default one, and is imported only when `--layerwise-batch > 0`.

```
  mq.py                          mq_streaming.py
  -------------------------      ------------------------------------------
  MessageQueueServer        <--  StreamingMessageQueueServer
    _call_handler()                _call_handler()  -> STREAMING? else super()
    add_handler()                  add_handler()    -> STREAMING? else super()
                                   _call_streaming_handler()           [new]
                                   add_streaming_handler()             [new]

  BlockingRequestHandler    <--  StreamingRequestHandler
    __call__(payloads,             __call__(payloads,
             affinity_key)                  affinity_key,
                                            response_channel)      [+1 kwarg]

  MessageQueueClient
    .submit_request()          ~   submit_streaming_request(client, .., future)
    .process_inbound()             (reused verbatim; not overridden)


  futures.py                     futures_layerwise.py
  -------------------------      ------------------------------------------
  MessagingFuture           <--  LayerwiseRawFuture
    set_result()             |     set_result()  -> buffer + re-arm, or finish
    wait() / result()        |     bind_registry()                     [new]
       ^                     |
       |                     +-- LayerwiseDeviceMessagingFuture
       |                            owns raw_future_: LayerwiseRawFuture
       |                            _layer_event_map, wait_for_layer()
       |
  DeviceMessagingFuture  (sibling: one completion event, per-chunk retrieve)
```

`<--` is subclassing, `~` a sibling helper with no inheritance
relationship.  `server.py` picks the server class at construction:

```python
server_cls: type[MessageQueueServer] = MessageQueueServer
if mp_config.layerwise_batch > 0:
    from lmcache.v1.multiprocess.mq_streaming import StreamingMessageQueueServer
    server_cls = StreamingMessageQueueServer
```

so a deployment that never registers a streaming handler never imports
`mq_streaming.py` at all.  Registration itself is unchanged:
`add_handler_helper` reads `get_handler_type(request_type)` from the
protocol table and passes it to `server.add_handler`, which the
subclass intercepts.

`response_channel` must be **keyword-only** on the handler.
`_inspect_handler_signature` matches a handler's *positional* parameters
against the declared `payload_classes`, so a positional
`response_channel` would fail validation before dispatch.

`mq.py` has no notion of a partial result, and none of its existing
classes were modified.  Two additions carry the whole mechanism:

1. **The future re-arms itself.**  `process_inbound` is unchanged: it
   pops the pending entry and calls `future.set_result(...)`, exactly as
   it always has.  `LayerwiseRawFuture` overrides `set_result` so that a
   non-final frame is buffered *and* the future puts itself back into the
   pending table under the same uid.  It holds that table because
   `submit_streaming_request` handed it over via `bind_registry` before
   the request reached the polling loop.  Both halves of the multi-frame
   contract therefore live in the future; `mq.py` is byte-identical to
   before this feature, and so is `futures.py`.
2. **A handler may answer more than once.**  This lives entirely in
   `mq_streaming.py`, which `mq.py` never imports.
   `StreamingMessageQueueServer` subclasses `MessageQueueServer` and
   intercepts `HandlerType.STREAMING` in `add_handler` / `_call_handler`,
   delegating every other handler type to `super()`.
   `StreamingRequestHandler` subclasses `BlockingRequestHandler`, so
   thread-pool assignment and validation need no changes, and passes the
   handler the same frame-emitting closure the server already uses for the
   final response, as a keyword-only `response_channel` argument.
   `server.py` builds the subclass only when `layerwise_batch > 0`; every
   other deployment runs the unmodified server dispatch.

### 8.3 Frame Formats

All frames are identical in shape, so no marker byte or length probe is
needed:

```
[zmq_identity, request_uid, request_type, msgpack((payload, is_final, succeeded))]
```

**Intermediate** (ceil(L/N) of them, one per batch): `is_final = False`,
`payload = struct.pack("<3i", first_layer, count, pool_index)`.

**Closing** (one, emitted after the handler returns): `is_final = True`,
`payload` empty when the indices were already reported.

### 8.4 Frame Sequence for One Request

A single `RETRIEVE_LAYERWISE` request produces every frame.  The
multiplexing is done by reusing the request UID: the server copies the
same `prefix_frames` for each send, and the client re-arms the pending
entry under that UID until the future reports the exchange is over.

```
client                                  server (affinity pool thread)
  |                                        |
  |-- RETRIEVE_LAYERWISE (uid=7) --------->|
  |   future.bind_registry(pending, 7)     |
  |   pending_futures[7] = future          |
  |                                        |  batch 0: H2D + scatter
  |                                        |           record_event(e0)
  |<-- [7, type, (first=0, cnt=N, idx=0)] -|  response_channel(...)
  |   set_result -> buffers, re-arms [7]   |
  |                                        |  batch 1: H2D + scatter
  |                                        |           record_event(e1)
  |<-- [7, type, (first=N, cnt=N, idx=N)] -|  response_channel(...)
  |   set_result -> buffers, re-arms [7]   |
  |             ...                        |             ...
  |                                        |  handler returns
  |<-- [7, type, (b"", True, succeeded)] --|  done-callback
  |   set_result -> completes, stays out   |
  |                                        |
```

Client dispatch is the stock `MessageQueueClient.process_inbound`,
unchanged: it pops the entry and calls `set_result`.  The re-arm happens
inside `LayerwiseRawFuture.set_result`, in `futures_layerwise.py`:

```python
def set_result(self, result):
    payload, is_final, _ = result
    if not is_final:
        self._partial_queue.put(payload)
        if self._registry is not None:
            # Runs on the polling-loop thread, the only writer of this table.
            self._registry[self._request_uid] = self
        return
    self._partial_queue.put(None)
    super().set_result(result)
```

The entry is re-inserted under the **same** UID until the closing frame
arrives.  There is no second request and no side channel.

Server dispatch, in `StreamingMessageQueueServer._call_streaming_handler`
(`mq_streaming.py`): the same closure serves both frame kinds, so
intermediate and closing frames are indistinguishable on the wire apart
from `is_final`:

```python
def _send_response(response):
    frames_to_send = list(prefix_frames)     # [identity, uid, type]
    if response is not None:
        frames_to_send.append(msgspec_encode(response, cls=response_cls))
    self.output_queue.put(frames_to_send)
    self._output_efd.notify()
```

The ROUTER socket consumes `identity` for routing, so the client's
DEALER receives `[uid, type, *response]`.

**Why the future must be pre-created.**  This is the reason the
layer-wise path submits through `submit_streaming_request` rather than
`MessageQueueClient.submit_request`.  The multi-frame behaviour is baked
into the future's *type*, so it has to be the object handed to the
polling loop:

```python
submit_streaming_request(
    self._mq_client, RequestType.RETRIEVE_LAYERWISE, payloads,
    layerwise_future.raw_future_,
)
```

`submit_request` builds a plain `MessagingFuture` internally, which the
first intermediate frame would complete outright, dropping the rest.  The
helper additionally calls `bind_registry` *before* the `input_queue.put`,
because a reply can land the instant the polling loop sees the request.
Because the behaviour comes from the class rather than from state
attached afterwards, there is no window in which the future is submitted
only half-configured.

The helper duplicates the few lines of request-building in
`submit_request` rather than adding a flag to it, so the per-chunk path
carries nothing about streaming.  The copy is pinned to the original by
`test_streaming_submit_matches_the_base_client`, which compares the two
enqueued `WrappedRequest` objects field by field via
`dataclasses.fields`, so a field added to the base path but missed here
fails the build.

**Frames signal enqueue, not completion.**  Each intermediate frame is
emitted immediately after `record_event`, which follows an *enqueue-only*
native call.  The closing frame therefore means "no more frames", not
"the H2D copies have landed".  Completion is carried by the events:
`wait_for_layer` inserts a stream-ordered `wait_event` (the host does not
block), and `LayerwiseDeviceMessagingFuture.wait()` calls
`synchronize_event` on the last layer's event -- that is the only point
at which all transfers are provably done.
