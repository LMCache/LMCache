# Layerwise Batch KV Loading: Design & Architecture

## 1. Overview

`--layerwise-batch N` enables **layer-major** H2D KV cache transfer in
LMCache's multiprocess (ZMQ) mode.  Instead of copying all layers for
each chunk (chunk-major), the server copies all chunks for N layers at
a time (layer-major), records one IPC event per batch, and **streams**
the event handle to the worker immediately so vLLM can start attention
on those layers while later batches are still transferring.

Key parameters:
- `N = 0` (default): layerwise disabled; all layers transferred at once
  per chunk (chunk-major, original path).
- `N = 1`: per-layer transfer; one event per layer.
- `N > 1`: batch N consecutive layers per H2D + scatter, one event per
  batch, best balance of event overhead vs. pipeline granularity.

Typical deployment: Qwen2.5-32B, 64 layers, GQA (8 KV heads per TP
rank), head_size=128, bf16, TP=2, chunk_size=256, block_size=16.
With `--layerwise-batch 8` -> 8 batches of 8 layers.

---

## 2. Process & Thread Topology

```
+----------------------- Server Process (lmcache_server) ----------------------+
|                                                                              |
|  +----------------------+    +------------------------------------------+   |
|  |  MQ Main-Loop Thread |    |  Affinity Pool (per-identity thread)      |   |
|  |  (mq-server-thread)  |    |                                          |   |
|  |                       |    |  Runs: retrieve() handler                |   |
|  |  * zmq ROUTER recv    |    |  * CPU: read prefetched chunks from L2  |   |
|  |  * dispatch to pool   |    |  * GPU: H2D memcpy (async, server strm) |   |
|  |  * zmq ROUTER send    |    |  * GPU: scatter kernel                  |   |
|  |    (partials + final) |    |  * CPU: record_event + export_event     |   |
|  |                       |    |  * CPU: sink.send_partial()             |   |
|  |  Drains output_queue  |    |        +-> output_queue.put()           |   |
|  |  into ZMQ socket      |    |                                          |   |
|  +------+---------------+    +------------------------------------------+   |
|         | ZMQ ipc://                                                         |
+---------+--------------------------------------------------------------------+
          |
          |  partial frames: [identity, uid, type, \x00, data]
          |  final   frame:  [identity, uid, type, msgspec([], bool)]
          |
+---------+-------------------- Worker Process (vLLM) -------------------------+
|         |                                                                    |
|  +------v---------------+    +------------------------------------------+   |
|  |  Client Polling Loop |    |  vLLM Model Runner Thread                |   |
|  |  (singleton thread)  |    |                                          |   |
|  |                       |    |  for layer_idx in 0..63:                 |   |
|  |  * zmq DEALER recv    |    |    future.wait_for_layer(layer_idx)      |   |
|  |  * partial? -> queue  |    |      -> drain queue until layer_idx      |   |
|  |  * final? -> set_rslt |    |      -> import_event + wait_event(strm)  |   |
|  |                       |    |    attention(layer_idx, kv_cache)  [GPU]  |   |
|  |  Writes to:           |    |                                          |   |
|  |   MessagingFuture     |    |  Reads from:                             |   |
|  |   ._partial_queue     |    |   LayerwiseDeviceMessagingFuture         |   |
|  |   .set_result()       |    |   ._partial_queue (same queue.Queue)     |   |
|  +-----------------------+    +------------------------------------------+   |
|                                                                              |
+------------------------------------------------------------------------------+
```

**Thread-safety boundaries:**
- `output_queue` (server): `queue.Queue`, written by affinity pool
  thread (partials + done-callback for final), drained by MQ main-loop
  thread.
- `_partial_queue` (worker): `queue.Queue`, written by client polling
  loop thread (`on_partial`), read by vLLM model runner thread
  (`_drain_until_layer`).
- `_layer_event_map` (worker): only accessed by vLLM model runner
  thread -- no lock needed.

---

## 3. Pipeline: Streaming ZMQ Partial Frames

With `--layerwise-batch 8` and 64 layers (8 batches), the per-batch
flow on the **server affinity pool thread** is:

```
Batch 0 (layers 0-7):
  +-- H2D memcpy: N_chunks x 8_layers x per_layer_bytes  (async GPU)
  +-- scatter kernel: interleaved -> per-layer KV blocks  (async GPU)
  +-- record_event(layer_events[0], server_stream)        (CPU, ~us)
  +-- export_event(layer_events[0]) -> handle bytes       (CPU, ~us)
  +-- sink.send_partial(msgpack(0, 8, handle))            (CPU, ~us)
       +-> output_queue -> MQ main loop -> ZMQ ROUTER -> wire

Batch 1 (layers 8-15):
  +-- H2D memcpy ...
  +-- scatter kernel ...
  +-- record_event(layer_events[8], server_stream)
  +-- export_event(layer_events[8]) -> handle bytes
  +-- sink.send_partial(msgpack(8, 8, handle))

  ... (batches 2-6 identical pattern) ...

Batch 7 (layers 56-63):
  +-- H2D + scatter + record_event + export_event
  +-- sink.send_partial(msgpack(56, 8, handle))

Handler returns ([], True)
  +-> done-callback -> output_queue -> final frame
```

**Worker-side** (vLLM model runner thread), concurrently:

```
Layer  0: wait_for_layer(0)
            -> _drain_until_layer(0) blocks on partial_queue.get()
            -> receives partial (0, 8, handle) -> import_event -> wait_event
          attention(layer 0)  <-- GPU, overlaps with server batch 1 H2D

Layer  1: wait_for_layer(1)
            -> already in _layer_event_map (same batch as layer 0)
            -> evt is _last_waited_event -> skip (dedup)
          attention(layer 1)

  ... layers 2-7: same event, all dedup-skipped ...

Layer  8: wait_for_layer(8)
            -> not in map -> drain queue -> receives partial (8, 8, handle)
            -> import_event -> wait_event
          attention(layer 8)  <-- GPU, overlaps with server batch 2 H2D

  ... layers 9-63: same pattern ...

All layers done -> future.wait() -> drain remaining -> synchronize last event
```

**Ordering guarantee:** the handler thread queues all partials before
returning.  The done-callback (`_notify_response`) fires only after
return, queuing the final frame.  `output_queue` is FIFO.  ZMQ ipc://
preserves per-connection order.  Partials always arrive before final.

---

## 4. Contiguous Per-Batch H2D Memcpy

### 4.1 CPU Memory Layout: kv_interleaved

When `--layerwise-batch > 0`, the **store** path (D2H) writes CPU
cache chunks in **per-layer interleaved** layout:

```
Per-chunk (kv_interleaved=False, default):
  chunk = [K_layer0, K_layer1, ..., K_layerN, V_layer0, V_layer1, ..., V_layerN]

Per-layer interleaved (kv_interleaved=True, layerwise mode):
  chunk = [K_layer0, V_layer0, K_layer1, V_layer1, ..., K_layerN, V_layerN]
```

This interleaved layout is set by `MemoryLayoutDesc.kv_interleaved`
(`lmcache/v1/platform/ops_types.py`).

### 4.2 Why Interleaved Enables Contiguous Batch Memcpy

With the interleaved layout, N consecutive layers in one chunk occupy
a **contiguous byte range** in the CPU buffer:

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

### 4.3 Current Merged H2D Path

When native ops are available and the staging buffer is large enough,
`transfer_kv_layerwise` uses the **N-layer merged path**:

1. **StagingCopy per chunk**: source = `memory_obj.data_ptr +
   src_layer_offset`, size = `n_in_batch * per_layer_bytes`.  This is
   one contiguous memcpy per chunk (not per layer).

2. **Scatter kernel**: `execute_object_group_transfer` with
   `PageBufferShapeDesc(nl=n_in_batch, kv_interleaved=True)`.  The
   GPU kernel reads the interleaved staging buffer and scatters to
   per-layer K/V paged block tensors.

3. **Fallback**: when native ops are absent or the staging buffer
   overflows, a per-layer fallback loop does N separate H2D copies
   (`multi_layer_block_kv_transfer` with single-layer sd).

### 4.4 Optimization Opportunity: Cross-Chunk Contiguous Memcpy

Currently, each chunk's N-layer region is copied separately (one
`StagingCopy` per chunk per batch).  If multiple chunks were laid out
contiguously in host memory (e.g., within the same
`LazyMemoryAllocator` pin chunk), a single DMA call could copy all
chunks' N-layer data in one shot:

```
Current (per-chunk H2D within a batch):
  chunk0: memcpy(gpu_slot0, cpu_chunk0 + offset, N * per_layer_bytes)
  chunk1: memcpy(gpu_slot1, cpu_chunk1 + offset, N * per_layer_bytes)
  ...

Potential (single memcpy across chunks):
  Requires: cpu_chunk0, cpu_chunk1, ... contiguous in host memory
  memcpy(gpu_base, cpu_base + offset, num_chunks * N * per_layer_bytes)
```

This is a future optimization requiring the storage manager to
allocate chunks from a contiguous arena and the staging logic to
detect contiguous regions.  The current approach (one H2D per chunk
per batch, fused into a `BatchStep` list handed to
`execute_object_group_transfer`) already achieves good throughput by
running all copies on the same CUDA stream with minimal launch
overhead.

---

## 5. vLLM MP Connector Integration

### 5.1 Attention Decorator

Every attention layer's `forward()` is wrapped by
`@maybe_transfer_kv_layer` (`vllm/.../kv_transfer_utils.py`):

```python
@maybe_transfer_kv_layer
def forward(self, ..., layer_name: str, ...):
    ...  # attention computation
```

The decorator:
1. **On entry:** calls `connector.wait_for_layer_load(layer_name)`.
2. **Executes:** the attention kernel (GPU compute on KV blocks).
3. **On exit:** calls `connector.save_kv_layer(layer_name, kv_cache,
   attn_metadata)`.

### 5.2 Worker Adapter: wait_for_layer_load

`VllmMultiProcessAdapter.wait_for_layer_load(layer_name)`
(`lmcache/integration/vllm/vllm_multi_process_adapter.py`):

1. Parses `layer_name` (e.g., `"model.layers.5.self_attn"`) to
   extract `layer_idx = 5`.
2. For each pending retrieve request, checks if the future is a
   `LayerwiseDeviceMessagingFuture`.
3. Calls `future.wait_for_layer(layer_idx)`:
   - **Streaming mode:** drains the `_partial_queue` until layer 5's
     event arrives, calls `import_event`, then `wait_event` on the
     compute stream (stream-ordered, non-blocking on CPU).
   - Event dedup: layers in the same batch share one event object;
     `wait_event` is called only once per unique event via identity
     check (`evt is not self._last_waited_event`).
4. Returns immediately -- the GPU attention kernel is now
   stream-ordered after the H2D event for that layer's batch.

### 5.3 Worker-Side Request Routing

`LMCacheDrivenTransferContext.submit_retrieve()` (`worker_transfer.py`):

```
layerwise=True  -> RequestType.RETRIEVE_LAYERWISE (streaming=True)
                   -> LayerwiseDeviceMessagingFuture(streaming=True)

layerwise=False -> RequestType.RETRIEVE           (streaming=False)
                   -> DeviceMessagingFuture (original per-chunk future)
```

A dedicated `RETRIEVE_LAYERWISE` request type isolates streaming to
the layerwise path.  The plain `RETRIEVE` dispatch is completely
untouched -- no `StreamingSink` allocated, no extra kwargs forwarded.

---

## 6. `--layerwise-batch N` Configuration Flow

```
CLI / ENV                         Config Parse               Server Init
---------                         ------------               -----------
--layerwise-batch 8       ->  MPServerConfig(              MPCacheServerContext(
                                layerwise_batch=8)           layerwise_batch=8)
                                                             ._layerwise_loading = True
                                                             ._layerwise_batch = 8

Transfer Module                   Retrieve Handler           Store Handler
---------------                   ----------------           -------------
self._ctx.layerwise_batch -> 8   if layerwise:              kv_interleaved=
self._ctx.layerwise_loading       transfer_kv_layerwise(      self._ctx.
  -> True                           layerwise_batch=8)          layerwise_loading
                                 else:                        (True -> interleaved
                                   transfer_kv_per_object_      D2H layout)
                                     group()
```

**Property definitions** (`MPCacheServerContext`):
- `layerwise_loading: bool = layerwise_batch > 0` -- gates per-layer
  vs. per-chunk code paths.
- `layerwise_batch: int` -- batch size passed directly to
  `transfer_kv_layerwise`.

**Worker-side detection:**
`LMCacheDrivenTransferContext._layerwise_batch` is communicated from
the server during worker registration.  The worker checks
`self._layerwise_batch > 0` to decide whether `submit_retrieve`
should send `RETRIEVE_LAYERWISE` and return a
`LayerwiseDeviceMessagingFuture`.

---

## 7. Layout Uniformity & Mixed-Mode Considerations

### 7.1 Current Invariant

Layout is **fixed per deployment**: a server is uniformly per-layer
(`kv_interleaved=True`) or uniformly per-chunk
(`kv_interleaved=False`).  The `store` handler writes D2H in the
layout dictated by `layerwise_loading`, and the `retrieve` handler
reads the same layout.

This works because `layerwise_loading` is a server-level config, not
a per-request flag.  All chunks stored by this server instance use
the same interleaving.

### 7.2 TODO: Mixed Per-Layer / Per-Chunk Deployments

If two server instances **share the same persistent L2 storage**
(e.g., a shared Redis or filesystem backend), one running with
`--layerwise-batch 8` and another with `--layerwise-batch 0`, the
stored chunks will have **different memory layouts** but no metadata
to distinguish them.

**Required changes:**

1. **Layout discriminator in metadata:** the `ObjectKey` or
   `MemoryObj.meta` must include a layout tag (e.g.,
   `kv_interleaved: bool`) so the retrieve path can detect layout
   mismatch.

2. **On-the-fly transcoding (optional):** if a per-chunk server reads
   an interleaved chunk (or vice versa), it must either:
   - Reject the chunk (safe, simple -- cache miss penalty only).
   - Transcode the layout in a CPU staging buffer before H2D.

3. **Overhead analysis:**
   - **Metadata overhead:** 1 bit per chunk in the object key or
     metadata header.  Negligible storage and wire cost.
   - **Reject-on-mismatch:** the chunk is treated as a cache miss.
     The request falls through to recompute, same cost as a cold
     miss.  No correctness risk; only a hit-rate reduction
     proportional to the fraction of cross-layout chunks.
   - **On-the-fly transcode:** requires one extra CPU memcpy pass
     to reorder K/V within the chunk.  Cost:
     `num_chunks * chunk_size_bytes` of CPU bandwidth.  For
     Qwen2.5-32B with 256-token chunks and bf16, a single chunk
     is ~4 MB -> transcoding 16 chunks is ~64 MB of CPU memcpy,
     adding ~1-2 ms on modern CPUs.  This is dominated by the H2D
     transfer time and therefore acceptable if needed.
   - **Recommended approach:** add the 1-bit layout tag to metadata
     and reject on mismatch (cache miss).  Transcoding is not worth
     the complexity unless mixed deployments are common.

---

## 8. Key Data Structures

### 8.1 `all_layers` -- Layer Iteration Order

```python
all_layers: list[tuple[int, int, int]]
# Each tuple: (kernel_group_info_idx, local_layer_idx, global_layer_idx)
```

Built by iterating kernel groups and their layer indices, then
**sorted by `global_layer_idx`** to ensure layer-major order.
The batch loop groups consecutive entries with the same
`kernel_group_info_idx` (same kernel group) and consecutive
`local_layer_idx` values, up to `layerwise_batch_size`.

### 8.2 `batch_leader_map` -- Event Dedup

```python
batch_leader_map: dict[int, int]
# Maps global_layer_idx -> first_global_layer_idx of its batch
```

All layers in one batch share the event recorded at
`layer_events[first_gl]`.  The map tells the final export path
which event handle to reuse, avoiding redundant `export_event` calls.

### 8.3 `StreamingSink` -- Server-Side Partial Sender

```python
class StreamingSink:
    __slots__ = ("_output_queue", "_output_efd", "_prefix_frames")

    def send_partial(self, data: bytes) -> None:
        self._output_queue.put(self._prefix_frames + [_PARTIAL_MARKER, data])
        self._output_efd.notify()
```

Created per `RETRIEVE_LAYERWISE` request in `_call_blocking_handler`.
Not allocated for plain `RETRIEVE` (per-chunk path untouched).

### 8.4 `LayerwiseDeviceMessagingFuture` -- Worker-Side Consumer

Two modes:
- **Streaming** (`streaming=True`): partial handles arrive via
  `queue.Queue`; `_drain_until_layer` blocks until the target layer's
  handle is available; `_import_partial` decodes
  `(first_layer, count, handle_bytes)` and maps all layers in the
  range to the imported event.
- **Non-streaming** (`streaming=False`): all handles arrive in the
  final ZMQ response; `_on_raw_future_complete` imports and dedup-maps
  them.

---

## 9. Streaming ZMQ Protocol

### 9.1 Request Types

| Request Type | `streaming` | Used By |
|---|---|---|
| `RETRIEVE` | `False` | Per-chunk retrieve (legacy, untouched) |
| `RETRIEVE_LAYERWISE` | `True` | Layerwise retrieve (streaming) |

Both share identical `payload_classes` and `response_class`.  The
only difference is the protocol-level `streaming` flag, which
controls whether `_call_blocking_handler` allocates a `StreamingSink`.

### 9.2 Frame Formats

**Partial frame** (N per retrieve, one per batch):
```
[zmq_identity, request_uid, request_type, b'\x00', msgpack(first_layer, count, handle_bytes)]
```

**Final frame** (1 per retrieve, after handler returns):
```
[zmq_identity, request_uid, request_type, msgpack(([], True/False))]
```

**Discrimination:** partial frames have `len(b_response) >= 2` and
`b_response[0] == b'\x00'`.  Final frames have `len(b_response) == 1`
(or 0 for None response).  No collision possible.

---

## 10. Error Handling & Timeouts

- **`_drain_until_layer` timeout:** if a partial doesn't arrive within
  60 seconds, raises `LMCacheTimeoutError`.  This catches server
  crashes or hangs without leaving the worker stuck indefinitely.

- **Handler exception:** if `retrieve()` raises, the done-callback
  `_notify_response` logs the exception.  The raw future never gets
  `set_result`, so the worker's `wait()` will eventually time out.

- **Empty cache (no chunks):** `layer_events` is empty, no partials
  are sent.  The final response carries `([], False)`.  Worker's
  `wait()` returns `True` with `result_ = False`, no events to
  synchronize.

---

## 11. Performance Characteristics

| Metric | Per-Chunk | Layerwise (N=8) |
|---|---|---|
| Worker idle before first attn | Full transfer time | 1 batch time |
| Events per retrieve | 1 | ceil(64/8) = 8 |
| ZMQ messages per retrieve | 1 (final) | 9 (8 partial + 1 final) |
| H2D calls per batch | num_chunks * all_layers | num_chunks * 1 (merged) |
| GPU scatter kernels / batch | 1 per chunk | 1 per chunk (nl=N) |
| Pipeline overlap | None (all-at-once) | (N_batches-1) * batch_time |
