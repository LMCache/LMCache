# `lmcache.v1.distributed.serde.turboquant` — TurboQuant Serde Backend

## Scope

This document describes the TurboQuant serde backend for LMCache L2 adapters.
TurboQuant serde is a storage-layer transform. It compresses LMCache KV tensors
before they are written to L2 storage and reconstructs KV tensors after L2
prefetch. It does not implement an attention backend and does not change the
StorageManager or L2 adapter public APIs.

The serde is intended to be used through the generic serde framework described
in [`docs/design/v1/distributed/serde/README.md`](README.md) and the L2 adapter
wrapper described in
[`docs/design/v1/distributed/l2_adapters/serde_wrapper.md`](../l2_adapters/serde_wrapper.md).

## Motivation

KV cache tensors can be large, especially for long-context workloads and
multi-layer models. Storing raw fp16/bf16 KV tensors in L2 increases storage
capacity requirements and L2 transfer volume.

TurboQuant serde reduces the serialized KV size by applying low-bit KV
compression before L2 store. During L2 load / prefetch, the compressed bytes are
decompressed back into the original LMCache KV tensor layout.

## Data Path

### Store Path

```text
MemoryObj(KV tensor)
  -> TurboQuantSerializer.serialize(src, dst)
  -> TurboQuant store Triton kernel
  -> MemoryObj(uint8 compressed bytes)
  -> inner L2 adapter store
```

### Load / Prefetch Path

```text
inner L2 adapter load
  -> MemoryObj(uint8 compressed bytes)
  -> TurboQuantDeserializer.deserialize(src, dst)
  -> TurboQuant decode Triton kernel
  -> MemoryObj(restored KV tensor)
```

The caller only observes the normal LMCache L2 store / load behavior. Temporary
byte buffers, serde task scheduling, eventfd signaling, and lock lifecycle are
handled by SerdeL2AdapterWrapper and AsyncSerdeProcessor.

## Public Interfaces

TurboQuant serde provides the synchronous serde interfaces required by the
generic serde framework:

* `TurboQuantSerializer.serialize(src, dst) -> int`
* `TurboQuantSerializer.estimate_serialized_size(layout_desc) -> int`
* `TurboQuantDeserializer.deserialize(src, dst) -> None`

It is registered under the serde type name:

```json
{
  "type": "turboquant"
}
```

The factory accepts TurboQuant-specific kwargs and constructs an
AsyncSerdeProcessor wrapping the serializer and deserializer.

## Configuration

`TurboQuantSerdeConfig` controls the compression preset and layout parameters.

Supported presets:

| Preset | Key path | Value path | Norm correction |
| --- | --- | --- | --- |
| `turboquant_k8v4` | FP8 key | 4-bit value quantization | No |
| `turboquant_4bit_nc` | 4-bit MSE key | 4-bit value quantization | Yes |
| `turboquant_k3v4_nc` | 3-bit MSE key | 4-bit value quantization | Yes |
| `turboquant_3bit_nc` | 3-bit MSE key | 3-bit value quantization | Yes |

Other config fields:

* `head_dim`: per-head hidden dimension.
* `block_size`: token block size used by the compressed layout.

Invalid presets are rejected with `ValueError`.

## Tensor Layout

TurboQuant serde expects LMCache KV tensors in this layout:

```text
[2, num_layers, num_tokens, hidden_dim]
```

The first dimension separates key and value tensors:

* `src[0]`: key cache
* `src[1]`: value cache

The compressed byte layout is:

```text
[num_layers, num_blocks, block_size, num_heads, slot_size]
```

where:

* `num_blocks = ceil(num_tokens / block_size)`
* `num_heads = hidden_dim / head_dim`
* `slot_size = key_packed_size + value_packed_size`
* `slot_size_aligned` is used for the serialized byte layout

`estimate_serialized_size()` computes the number of bytes required for this
compressed layout from `MemoryLayoutDesc`.

## Compression Path

`TurboQuantSerializer` launches Triton store kernels to compress each layer.

The store path performs:

1. KV layout validation.
2. Temporary CUDA staging when StorageManager provides CPU / pinned-memory
   `MemoryObj` tensors.
3. Key compression:
   * FP8 key path for `turboquant_k8v4`.
   * MSE / centroid low-bit key path for low-bit presets.
4. Value uniform quantization.
5. 3-bit or 4-bit bit-packing into a uint8 byte buffer.
6. Metadata storage, including scale / zero for values and norm metadata when
   required by the preset.

The compressed output is written into the destination uint8 `MemoryObj`.

## Decompression Path

`TurboQuantDeserializer` launches Triton decode kernels to reconstruct KV
tensors from compressed bytes.

The load path performs:

1. Serialized byte buffer validation.
2. Temporary CUDA staging when StorageManager provides CPU / pinned-memory
   `MemoryObj` tensors.
3. Key unpacking and dequantization:
   * FP8 key decode for FP8 presets.
   * MSE / centroid decode for low-bit presets.
4. Value unpacking and scale / zero dequantization.
5. Restoration of the LMCache KV tensor layout:
   `[2, num_layers, num_tokens, hidden_dim]`.

The deserializer reconstructs KV tensors for storage reuse. It does not compute
attention outputs.

## Device Handling

TurboQuant uses Triton kernels, so tensors participating in one kernel launch
must be on the same CUDA device.

Device selection follows these rules:

1. If any source or destination tensor is already on CUDA, all CUDA tensors in
   the same serde operation must be on the same device. Otherwise, TurboQuant
   serde raises `ValueError`.
2. If `cuda_device` is configured, that device is used as the staging device.
   If CUDA tensors already exist, the configured device must match them.
3. If all source and destination tensors are CPU tensors and `cuda_device` is
   not configured, TurboQuant serde selects a CUDA device with sufficient free
   memory and the lowest GPU utilization.
4. CPU / pinned-memory tensors are staged to the selected CUDA working device
   before Triton kernel execution and copied back afterward.

This backend does not change LMCache runtime placement policy; the automatic
selection only applies to CPU-only serde staging.

## Relationship to vLLM TurboQuant

The TurboQuant store/decode Triton kernel logic follows the implementation
approach used by the vLLM TurboQuant PR. The integration target is different:

* vLLM integrates TurboQuant into the attention backend.
* LMCache integrates TurboQuant into the L2 serde path.

In LMCache, TurboQuant is a storage transform only: it compresses objects before
L2 store and reconstructs objects after L2 load / prefetch.

## Performance Snapshot

The following numbers are from a local H20 serde microbenchmark. They are
intended as a sanity check for compression ratio, latency, and reconstruction
error rather than full serving benchmark results.

### Serde microbenchmark

| Serde | Preset | Raw MB | Serialized MB | Compression ratio | Encode ms | Decode ms | Corr | Mean abs err | Max abs err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | `float8_e4m3fn` | 8.00 | 4.00 | 2.00 | 0.024 | 0.040 | 0.999645 | 0.017948 | 0.250000 |
| turboquant | `turboquant_k8v4` | 8.00 | 3.06 | 2.61 | 0.375 | 0.523 | 0.997342 | 0.051538 | 0.273438 |
| turboquant | `turboquant_4bit_nc` | 8.00 | 2.09 | 3.82 | 0.554 | 0.642 | 0.995225 | 0.080693 | 0.505249 |
| turboquant | `turboquant_k3v4_nc` | 8.00 | 1.84 | 4.34 | 0.555 | 0.642 | 0.989075 | 0.115782 | 0.970703 |
| turboquant | `turboquant_3bit_nc` | 8.00 | 1.59 | 5.02 | 0.557 | 0.643 | 0.980405 | 0.164546 | 0.970703 |

## Limitations

This backend currently focuses on correctness and LMCache L2 integration.

Known limitations and follow-up work:

* Serving-level benchmark scripts are not included in the core backend PR.
* Task-level quality evaluation, such as NIAH or GSM8K, is not included.
* Multi-GPU device placement should be kept explicit and conservative.
* Reusable GPU buffers should be cached or moved out of per-layer loops where
  possible.
* Attention-specific kernels from vLLM should not be included unless they are
  used by the LMCache serde path.

## Tests

The TurboQuant serde tests cover:

* serde factory / processor creation
* preset config parsing
* invalid preset rejection
* serialized size estimation
* direct CUDA serialize / deserialize roundtrip
* StorageManager roundtrip through serde-wrapped L2
* filesystem-backed L2 roundtrip
* reconstruction quality checks using correlation and error thresholds

These tests validate the core LMCache L2 serde path. They do not claim
serving-level performance speedups or task-level quality preservation.
