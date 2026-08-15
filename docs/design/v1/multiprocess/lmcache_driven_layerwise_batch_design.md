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
event and one streaming partial frame to the worker.

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
|  |    (partials + final)|    |                                          |    |
|  |                      |    |  +-- per batch (N layers) -----------+   |    |
|  |  Drains output_queue |    |  | 1. GPU: H2D memcpy                |   |    |
|  |  into ZMQ socket     |    |  | 2. GPU: scatter kernel            |   |    |
|  |                      |    |  | 3. CPU: record_event(pool[i])              |   |    |
|  |                      |    |  | 4. CPU: sink.send_partial(pool_idx)       |   |    |
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
|  |  * partial? -> queue |    |  | 1. CPU: wait_for_layer(layer_idx) |   |    |
|  |  * final? -> set_rslt|    |  |    1st in batch: drain _partial_q |   |    |
|  |                      |    |  |      + pool.event_at(idx) + wait_event|   |    |
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
  +-- sink.send_partial(msgpack(0, N, pool_idx=0))            (CPU)
       +-> output_queue -> MQ main loop -> ZMQ ROUTER -> wire

Batch 1 (layers N .. 2N-1):
  +-- H2D memcpy ...
  +-- scatter kernel ...
  +-- record_event(layer_events[N], server_stream)
  +-- sink.send_partial(msgpack(N, N, pool_idx=N))

  ... (batches 2 .. ceil(L/N)-2 identical pattern) ...

Batch ceil(L/N)-1 (last N layers):
  +-- H2D + scatter + record_event(pool[last])
  +-- sink.send_partial(msgpack((ceil(L/N)-1)*N, N, pool_idx))

Handler returns ([], True)
  +-> done-callback -> output_queue -> final frame
```

**Worker-side** (vLLM model runner thread), concurrently:

```
Layer  0: wait_for_layer(0)
            -> _drain_until_layer(0) blocks on partial_queue.get()
            -> receives partial (0, N, pool_idx) -> pool.event_at(idx) -> wait_event
          attention(layer 0)  <-- GPU, overlaps with server batch 1 H2D

Layer  1: wait_for_layer(1)
            -> already in _layer_event_map (same batch as layer 0)
            -> evt is _last_waited_event -> skip (dedup)
          attention(layer 1)

  ... layers 2 .. N-1: same event, all dedup-skipped ...

Layer  N: wait_for_layer(N)
            -> not in map -> drain queue -> receives partial (N, N, pool_idx)
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

This interleaved layout is set by `PageBufferShapeDesc.kv_interleaved`
(`lmcache/v1/platform/ops_types.py`).

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
register_kv_cache():
  1. assert num_total_layers <= EVENT_POOL_SIZE
  2. pool = EventPool(backend, device)          # 256 x cudaEventCreate
  3. handles = [backend.export_event(e) for e]  # 256 x cudaIpcGetEventHandle
  4. return (layerwise_batch, handles)          # sent once in registration response
```

The worker imports all 256 handles **once** at registration:

```
register() response:
  pool = EventPool.import_pool(backend, device, handles)
  # 256 x cudaIpcOpenEventHandle — one-time cost at startup
```

### 5.3 Per-Request Hot Path (Zero Driver Calls)

During retrieve, the server indexes into the pre-allocated pool:

```
layer_events = [pool.event_at(i) for i in range(num_total_layers)]
...
record_event(pool_event[batch_leader], stream)   # cudaEventRecord only
send_partial(msgpack(first_layer, count, batch_leader))  # int, not bytes
```

The worker receives a pool index (int) and looks up the pre-imported
event — no `cudaIpcOpenEventHandle` on the forward path:

```
pool_idx = decode(partial)  # int
evt = pool.event_at(pool_idx)
stream.wait_event(evt)
```

### 5.4 Wire Encoding

Pool indices are sent as:
- **Streaming (partial frames):** `msgpack(first_layer, count, pool_idx: int)`
- **Non-streaming (final response):** `struct.pack("<Ni", *indices)` encoded
  as `bytes` to stay within `tuple[bytes | list[bytes], bool]` (msgspec
  cannot union two array-like types).

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
MPCacheServerContext
    ._layerwise_batch = N
    ._layerwise_loading = (N > 0)
    |
    +---> Retrieve: transfer_kv_layerwise(batch=N)
    +---> Store:    kv_interleaved = (N > 0)
    |
    +---> Worker registration (ZMQ handshake):
              server sends layerwise_batch=N to vLLM worker
              |
              v
          LMCacheDrivenTransferContext._layerwise_batch = N
              |
              +---> N > 0: submit RETRIEVE_LAYERWISE (streaming)
              +---> N = 0: submit RETRIEVE (per-chunk, legacy)
```

---

## 7. Layout Uniformity & Mixed-Mode Considerations

### 6.1 Current Invariant

Layout is fixed per deployment: a server is uniformly per-layer
(`kv_interleaved=True`) or uniformly per-chunk
(`kv_interleaved=False`).  The `store` handler writes D2H in the
layout dictated by `layerwise_loading`, and the `retrieve` handler
reads the same layout.

This works because `layerwise_loading` is a server-level config, not
a per-request flag.  All chunks stored by this server instance use
the same interleaving.

### 6.2 Note: Rolling Upgrades & Shared L2

If servers sharing persistent L2 storage are upgraded from
`--layerwise-batch 0` to `N > 0` (or vice versa), stale chunks with
the old layout may remain.  Reading a chunk with mismatched layout
produces corrupt KV data.

In practice this is a non-issue: mixed deployments offer no benefit,
and old chunks are naturally evicted.  If rolling upgrades are needed,
flushing L2 between mode changes is sufficient.

---

## 8. Streaming ZMQ Protocol

### 7.1 Request Types

| Request Type | `streaming` | Used By |
|---|---|---|
| `RETRIEVE` | `False` | Per-chunk retrieve (legacy, untouched) |
| `RETRIEVE_LAYERWISE` | `True` | Layerwise retrieve (streaming) |

Both share identical `payload_classes` and `response_class`.  The
only difference is the protocol-level `streaming` flag, which
controls whether `_call_blocking_handler` allocates a `StreamingSink`.

### 7.2 Frame Formats

**Partial frame** (total ceil(L/N), one per batch):
```
[zmq_identity, request_uid, request_type, b'\x00', msgpack(first_layer, count, pool_index)]
```

**Final frame** (only 1 for completion, after handler returns):
```
[zmq_identity, request_uid, request_type, msgpack(([], True/False))]
```

**Discrimination:** partial frames have `len(b_response) >= 2` and
`b_response[0] == b'\x00'`.  Final frames have `len(b_response) == 1`
(or 0 for None response).  No collision possible.
