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
|  |                      |    |  | 3. CPU: record_event              |   |    |
|  |                      |    |  | 4. CPU: export_event              |   |    |
|  |                      |    |  | 5. CPU: sink.send_partial()       |   |    |
|  |                      |    |  |      +-> output_queue.put()       |   |    |
|  |                      |    |  +-----------------------------------+   |    |
|  +------+---------------+    +------------------------------------------+    |
|         | ZMQ ipc://                                                         |
+---------+--------------------------------------------------------------------+
          |
          |  partial frames (event IPC handle)
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
|  |                      |    |  |      + import + wait_event(stream)|   |    |
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
  +-- export_event(layer_events[0]) -> handle bytes       (CPU)
  +-- sink.send_partial(msgpack(0, N, handle))            (CPU)
       +-> output_queue -> MQ main loop -> ZMQ ROUTER -> wire

Batch 1 (layers N .. 2N-1):
  +-- H2D memcpy ...
  +-- scatter kernel ...
  +-- record_event(layer_events[N], server_stream)
  +-- export_event(layer_events[N]) -> handle bytes
  +-- sink.send_partial(msgpack(N, N, handle))

  ... (batches 2 .. ceil(L/N)-2 identical pattern) ...

Batch ceil(L/N)-1 (last N layers):
  +-- H2D + scatter + record_event + export_event
  +-- sink.send_partial(msgpack((ceil(L/N)-1)*N, N, handle))

Handler returns ([], True)
  +-> done-callback -> output_queue -> final frame
```

**Worker-side** (vLLM model runner thread), concurrently:

```
Layer  0: wait_for_layer(0)
            -> _drain_until_layer(0) blocks on partial_queue.get()
            -> receives partial (0, N, handle) -> import_event -> wait_event
          attention(layer 0)  <-- GPU, overlaps with server batch 1 H2D

Layer  1: wait_for_layer(1)
            -> already in _layer_event_map (same batch as layer 0)
            -> evt is _last_waited_event -> skip (dedup)
          attention(layer 1)

  ... layers 2 .. N-1: same event, all dedup-skipped ...

Layer  N: wait_for_layer(N)
            -> not in map -> drain queue -> receives partial (N, N, handle)
            -> import_event -> wait_event
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

## 5. `--layerwise-batch N` Configuration Flow

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

## 6. Layout Uniformity & Mixed-Mode Considerations

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

## 7. Streaming ZMQ Protocol

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
[zmq_identity, request_uid, request_type, b'\x00', msgpack(first_layer, count, handle_bytes)]
```

**Final frame** (only 1 for completion, after handler returns):
```
[zmq_identity, request_uid, request_type, msgpack(([], True/False))]
```

**Discrimination:** partial frames have `len(b_response) >= 2` and
`b_response[0] == b'\x00'`.  Final frames have `len(b_response) == 1`
(or 0 for None response).  No collision possible.
