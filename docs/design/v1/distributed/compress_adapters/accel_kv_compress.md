# Accelerated KV compression (`accel_kv_compress`)

`accel_kv_compress` compresses KV chunks between L1 and an L2 adapter with
the kv_zip codec of intel-accel-for-llm (iaxl), which runs on Intel QAT, IAA
or CPU workers. It is a serde, so it plugs into any L2 adapter through
`SerdeL2AdapterWrapper` (see
[`serde_wrapper.md`](../l2_adapters/serde_wrapper.md)). The optional
[pipelined load](#pipelined-load) overlaps decompression with a layer-wise
retrieve.

Two pairings are intended:

- **`dram` L2** (`DramL2Adapter`): keeps more KV in the same host DRAM. A
  single budget splits DRAM between raw KV in L1 and compressed KV in L2.
- **`fs` L2**: writes and reads fewer bytes per chunk on SSD.

## Data path

L1 always holds raw KV. Compression happens only on the L1 → L2 leg, and
decompression only on the L2 → L1 leg.

```
dram L2:
store:    L1 obj (raw) ──serialize──► L2 bytes (compressed)
prefetch: L2 bytes (compressed) ──deserialize──► L1 obj (raw)

other L2 adapters, such as fs:
store:    L1 obj (raw) ──serialize──► L1 temp (compressed) ──► L2 adapter
prefetch: L2 adapter ──► L1 temp (compressed) ──deserialize──► L1 obj (raw)
```

- **Store.** With `dram` L2, the serde compresses into a new `bytes` object
  of exactly the compressed size, and the adapter keeps that object as it
  is (see [`dram` L2 adapter](#dram-l2-adapter)). With other adapters, the
  wrapper reserves an L1 temp buffer of `estimate_serialized_size()` bytes,
  compresses into it, and hands the compressed bytes to the inner adapter.
  The source object is not modified. With `--l2-store-policy skip_l1`, the
  L1 copy is dropped after the store.
- **Prefetch.** With `dram` L2, each object is decompressed straight from
  the adapter's stored bytes into its L1 destination. With other adapters,
  the inner adapter first loads the compressed bytes into an L1 temp buffer.
  Retrieval to the GPU then reads raw KV from L1 as usual. This includes
  layer-wise retrieval of N layers at a time. With `pipelined_load`, the
  prefetch completes as soon as decompression starts, and decompression
  continues in the background (see [Pipelined load](#pipelined-load)).

## Configuration

The serde is configured in the `"serde"` sub-dict of an `--l2-adapter` spec:

```bash
lmcache server \
    --l1-size-gb 6 --eviction-policy LRU --chunk-size 256 \
    --l2-store-policy skip_l1 \
    --l2-adapter '{"type": "dram", "max_size_gb": 14,
                   "serde": {"type": "accel_kv_compress", "backend": "iaxl",
                             "dtype": "bfloat16", "max_workers": 4}}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `"iaxl"` | Compression backend. `iaxl` is the only one. |
| `dtype` | `"bfloat16"` | torch dtype name of the KV cache. iaxl uses it to select BF16 data shuffling and the lossy truncation mask. |
| `num_groups` | `1` | Independently decodable groups per object (see [Blocked container](#blocked-container)). |
| `pipelined_load` | `false` | Report loads done before decompressing them, and decompress group by group across a load's objects (see [Pipelined load](#pipelined-load)). |
| `max_workers` | `1` | Threads in the `AsyncSerdeProcessor` pool. Each issues one batched iaxl call per object with the GIL released. iaxl spreads a call across its own workers, and concurrent calls take turns on one pipeline mutex, so extra threads mainly overlap the Python-side work. |

iaxl reads its settings from environment variables at load time. The ones
that matter here (defaults from iaxl's `envs.c`):

| Variable | Default | Effect |
| --- | --- | --- |
| `IAXL_QAT_ZIP_ENABLE`, `IAXL_IAA_ZIP_ENABLE`, `IAXL_CPU_ZIP_ENABLE` | `1`, `0`, `0` | Which workers compress. At least one must be on. |
| `IAXL_ZIP_LEVEL` | `1` | Compression level, 1–9. |
| `IAXL_KV_DATA_SHUFFLE` | `0` | Byte-shuffles BF16 data so exponent bytes sit together. Lossless. |
| `IAXL_ZIP_EXPONENT_ONLY` | `IAXL_KV_DATA_SHUFFLE` | With shuffling, compresses only the exponent half and stores the mantissa half raw. Lossless. |
| `IAXL_KV_LOSSY_TRUNC` | `0` | Clears the low N bits of each element before compressing. **Lossy** when > 0. |
| `IAXL_ZIP_SEGMENT_SIZE_KB` | `0` | Splits each block into segments of this size; `0` keeps one segment per block. |
| `IAXL_ZIP_SRC_CAP` | `262144` | Largest segment, in bytes. |

A segment larger than `IAXL_ZIP_SRC_CAP` fails with `ValueError`. Because a
KV chunk (tens of MB) is far larger than the 256 KiB default, set
`IAXL_ZIP_SEGMENT_SIZE_KB` (for example `64`) whenever compression is on.
iaxl's own `doc/design/kv-zip.md` documents the rest, including worker and
thread placement.

## Components

All code lives in `lmcache/v1/distributed/`.

| File | Role |
| --- | --- |
| `compress_adapters/__init__.py` | Registers the `"accel_kv_compress"` serde factory. |
| `compress_adapters/backend.py` | `AccelCompressBackend` ABC: `compress`, `decompress`, `max_compressed_length`, batched `compress_blocks` / `decompress_blocks`, and an optional `max_block_size`. |
| `compress_adapters/iaxl_backend.py` | `IaxlBackend`: calls `zip_compress_into`, `zip_decompress_into` and `zip_max_block_size` from `iaxl.torch_ext`. |
| `compress_adapters/blocked.py` | Blocked container format. |
| `compress_adapters/serde.py` | `AccelCompressSerializer`, `AccelCompressDeserializer` (whole object, or one group with `deserialize_group()`), `GroupwiseDecode`, and `AccelCompressSerdeProcessor` with the pipelined load. |
| `pending_fill.py` | `FillTracker`: how far a pipelined load has written its L1 objects; a registry readers query by object. |
| `l2_adapters/dram_l2_adapter.py` | `DramL2Adapter`, L2 type `"dram"`; `submit_store_bytes()` and `load_bytes()` pass `bytes` objects without copying. |
| `l2_adapters/serde_wrapper.py` | `SerdeL2AdapterWrapper`; skips the L1 temps when the inner adapter is `DramL2Adapter` and the serde `supports_bytes_io`. |
| `l2_adapters/fs_l2_adapter.py` | Accepts short reads when the serde is `accel_kv_compress`. |
| `dram_partition.py` | `DramPartitionCoordinator`: splits one DRAM budget between L1 and `dram` L2. |

### Backend

`IaxlBackend` hands iaxl caller-owned buffers: the L1 source and the
destination, which is the L1 temp or, with `dram` L2, a per-thread scratch
buffer. iaxl stages segments in the destination's payload area,
so the source stays intact even with lossy truncation on. Each output block is a self-describing kv_zip block, so
decompression reads the segment layout and flags from the block rather than
from the environment.

Importing iaxl is deferred to `IaxlBackend()`. Configuring the serde
without iaxl installed raises `ImportError`; the rest of LMCache does not
need iaxl.

### Blocked container

Compressed objects use a self-describing container so that each of
`num_groups` byte ranges decompresses independently:

```
[header][block 0][block 1] ... [block N-1]

header (little-endian, 28 B + 4 B per block):
  magic "LKCB" | version u16 | num_groups u16 | total_size u64
  group_size u32 | block_size u32 | num_blocks u32 | sizes u32 * num_blocks
```

Groups are a plain byte-range partition of the raw object. Blocks never
straddle a group boundary, and the block partition is derived from
`(total_size, group_size, block_size)`, so only compressed sizes are
stored. `IaxlBackend` sets no `max_block_size`, so each group is exactly one
kv_zip block, and iaxl segments it internally.

Under a layer-major layout such as `[L, 2, T, D]`, setting `num_groups` to
the number of layers divided by the retrieve's layer-group size N makes one
group one N-layer slice. `AccelCompressDeserializer.deserialize_group()`
decodes that slice into its own offset of a full-size destination.

By default prefetch calls `deserialize()` on the whole object, and the
scheduler sees the hit only after every chunk is decompressed. Attention for
layer group g needs group g of every chunk, so per-chunk decompression
cannot overlap the forward pass. The groups make the pipelined load
possible.

### Pipelined load

With `"pipelined_load": true`, decompression overlaps the layer-wise
retrieve instead of preceding it:

1. `SerdeL2AdapterWrapper` asks the serde for
   `submit_pipelined_deserialize()` once the inner adapter has loaded the
   compressed bytes. `AccelCompressSerdeProcessor` validates every header,
   registers a `FillTracker` on the destination L1 objects, and starts a
   `GroupwiseDecode`: group 0 of every object in one backend call, then
   group 1, and so on, advancing the tracker's byte watermark after each.
2. The wrapper reports the load done right away, and hands the tracker to
   `PrefetchController` through `take_load_fill()`. The controller commits
   the objects with one extra read lock, which it drops when the fill
   succeeds, so the objects outlive the decompression even if the request
   is aborted. The compressed sources (L1 temps, or references to the
   `dram` L2 bytes) are released when decompression ends.
3. Readers wait for the bytes they need. `StorageManager.unsafe_read()` and
   `read_prefetched_results()` wait for whole objects. A cache context that
   sets `reads_filling_objects` (the layer-wise plugin's
   `GPULayerwiseCacheContext`) gets the objects at once; its
   `transfer_kv_layerwise()` waits for each layer batch's bytes before
   enqueuing that batch, so group g+1 decompresses while group g is copied
   and its attention runs.

Requirements and behavior:

- Set `num_groups` to the number of layers divided by the layer-wise
  plugin's `batch_layers`, so one group is one batch. With fewer groups a
  batch waits for a larger group; with `num_groups: 1` nothing overlaps.
- Objects must be layer-major, as the layer-wise plugin stores them. Other
  retrieves still work: they wait for whole objects.
- **Failure.** A header that does not parse fails the whole load before it is
  reported, as today. A decode that fails later leaves the tracker
  registered as failed: readers waiting past the failure raise
  `FillFailedError` (the retrieve fails) or skip the object. The controller
  deletes the objects once no one else reads them, then unregisters the
  tracker. A layer batch whose bytes were written before the failure is
  still copied.

Decompressing in place lets decompression run ahead into the gap between
prefetch and the forward pass, and keeps the raw KV in L1 for repeat hits.
Decompression stays hidden only while one group across all chunks
decompresses no slower than its H2D copy plus attention.

### `dram` L2 adapter

`DramL2Adapter` stores each object's bytes in an `OrderedDict` with LRU
ordering and a `max_size_gb` capacity. Operations complete synchronously
and signal their event fds immediately. It is meant to sit under a serde.
Without one it only duplicates L1.

Under `accel_kv_compress`, the wrapper uses no L1 temps with this adapter:

- **Store.** `AccelCompressSerializer.serialize_to_bytes()` compresses
  into a per-worker scratch buffer sized for the worst case, then copies the
  header and blocks once into a new `bytes` object of exactly the compressed
  size. `submit_store_bytes()` keeps that object as it is. A chunk is held
  raw in L1 and compressed in L2; with `skip_l1` only the L2 bytes remain.
  Each serde worker keeps one scratch buffer of about one raw chunk.
- **Load.** `load_bytes()` returns the stored objects without copying, and
  each one is decompressed straight into its L1 destination. A key evicted
  while it is being decompressed stays readable until decompression ends;
  for that time its bytes are no longer counted in the adapter's usage.

The serde opts in through `SerdeProcessor.supports_bytes_io`. Serdes that
do not, and other inner adapters, keep the L1 temp path.

### `fs` L2 short reads

Compressed files are smaller than the prefetch buffer, which is sized by
`estimate_serialized_size()`. When the adapter's serde is
`accel_kv_compress`, a short read calls `set_used_size()` on the buffer and
succeeds. For every other serde, a short read is still rejected as possible
corruption.

### DRAM budget

With `dram` L2, one budget can size both tiers:

| Flag | Default | Effect |
| --- | --- | --- |
| `--total-memory-budget-gb` | `0` (off) | Total DRAM for L1 + `dram` L2. Overrides `--l1-size-gb` and the first `dram` adapter's `max_size_gb`. |
| `--l1-fraction` | `0.3` | Share for raw KV in L1; the rest goes to `dram` L2. |
| `--l1-high-watermark`, `--l2-high-watermark` | `0.8` | Eviction triggers while the budget is active. |
| `--enable-pressure-eviction` | off | Requires `--total-memory-budget-gb`. Runs one L1 eviction pass on every `L1_ALLOCATION_FAILED` event, regardless of the watermark. Only store-side reservations (`StorageManager.reserve_write`) publish that event. |

`--total-memory-budget-gb` sizes the single legacy L1, so it is rejected
together with `--l1-manager`.

### Sizing L1

LMCache does not check the L1 size against the model at startup. The chunk
size in bytes depends on the KV layout, which is known only when the engine
registers its KV cache. Size L1 yourself:

- **Chunk size.** One chunk is `2 × layers × chunk_size tokens × KV heads ×
  head_dim × element size` per rank. For example, a model with 32 layers,
  8 KV heads, `head_dim` 128 and bf16 KV has 32 MiB chunks at 256 tokens.
- **Every chunk in flight needs about its raw size in L1 with `dram` L2,
  and about 2× with other adapters**: the raw object, plus, without `dram`,
  a compressed temp buffer of `estimate_serialized_size()` bytes. iaxl's
  bound is the raw size plus headers, because incompressible data is stored
  raw.
- **One prefetch covers every chunk of the hit prefix.** L1 must hold the
  raw size (2× without `dram`) of the longest prefix loaded at once, times
  the number of concurrent loads, on top of whatever L1 keeps cached. With
  `--l2-store-policy skip_l1`, stored chunks leave L1, so L1 is mostly
  in-flight space.
- **Running short does not fail requests.** A store whose reservation fails
  skips those chunks; a prefetch whose reservation fails returns fewer hits,
  or nothing with the `full` fetching policy. Store failures count in
  `lmcache_mp.l1_allocation_failure`, prefetch failures in
  `lmcache_mp.l2_prefetch_failure{reason="l1_oom"}`.
  `--enable-pressure-eviction` reacts only to store-side failures.

## When it pays off

On the load path the effective KV bandwidth is roughly

$$\min(\text{L2 read bandwidth} \times \text{ratio},\ \text{decompress throughput})$$

So compression speeds up loads from `fs` only if iaxl decompresses faster
than the disk delivers raw KV. In practice that means QAT/IAA or many CPU
workers on a fast NVMe. With `dram` L2 the gain is capacity, not speed:
each prefetch pays a decompression that a plain L1 hit would not.

For TTFT, compare against the alternative:

- **Against an uncompressed L1 hit**, decompression is pure overhead. Without
  `pipelined_load` it adds the whole decompression time. With it, a load
  costs about $\max(\text{decompress}, \text{H2D} + \text{attention})$ per
  layer group instead of the sum, plus the first group's decompression.
- **Against a miss**, the extra capacity turns recomputed prefill into hits,
  which is usually far cheaper than recomputing.

## Tests

| File | Covers |
| --- | --- |
| `tests/v1/distributed/test_compress_iaxl.py` | Factory and dtype handling; serde round trips with 1 and 3 groups; StorageManager store with `skip_l1` then prefetch through `dram` L2. |
| `tests/v1/distributed/test_compress_blocked.py` | Container header, block planning, group decode, `GroupwiseDecode` rounds and watermarks, the pipelined processor, `serialize_to_bytes()`. |
| `tests/v1/distributed/test_pending_fill.py` | `FillTracker` waits, settling, registry. |
| `tests/v1/distributed/test_prefetch_pipelined_fill.py` | The controller's extra read lock, and deleting a failed fill's objects. |
| `tests/v1/distributed/l2_adapters/test_serde_wrapper.py` | Pipelined loads in the wrapper, among its other paths; the zero-copy path over `DramL2Adapter`. |
| `tests/v1/distributed/test_multi_l1.py` | `StorageManager.unsafe_read()` waiting for a fill. |
| `tests/v1/distributed/test_dram_l2_adapter.py` | `DramL2Adapter` on its own, including `submit_store_bytes()` and `load_bytes()`. |
| `tests/v1/distributed/test_dram_l2_serde_flow.py` | Serde wrapper over `DramL2Adapter`. |
| `tests/v1/distributed/test_dram_partition.py` | Budget split, CLI wiring. |

The tests use a zlib stand-in for `iaxl.torch_ext`. The `real` variants run
against iaxl when it is installed, for example on CPU workers:

```bash
IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=1 \
IAXL_ZIP_SEGMENT_SIZE_KB=64 IAXL_KV_DATA_SHUFFLE=1 \
    pytest tests/v1/distributed/test_compress_iaxl.py -k real
```
