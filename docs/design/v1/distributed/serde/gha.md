# `lmcache.v1.distributed.serde.gha` — GHA Serde Backend

## Scope

GHA is a 4/8-bit KV quantization codec with native CPU and XPU kernels.
This serde applies it on the **L1 → L2 path**: L1 keeps full-precision KV, the
store controller hands a chunk to a serde-wrapped L2 adapter, the serde
quantizes it while serializing, and the load path dequantizes it back into the
L1 destination chunk. Only L2 is lossy, so nothing has to guard against two
lossy layers.

It plugs into the generic serde framework
([`README.md`](README.md)) and the L2 wrapper
([`../l2_adapters/serde_wrapper.md`](../l2_adapters/serde_wrapper.md)); it does
not touch `StorageManager` or the L2 adapter APIs.

## Configuration

```json
{"type": "fs", "base_path": "/cache",
 "serde": {"type": "gha", "qbit": 4, "device": "xpu"}}
```

| Key | Default | Meaning |
|---|---|---|
| `qbit` | `4` | `4` or `8` bits per value |
| `per_tensor` | `false` | one scale per plane for every chunk (see *Scaling*) |
| `rh`, `asym`, `precond` | `true`, `true`, `true` | randomized Hadamard, asymmetric range, preconditioner |
| `device` | `cpu` | `cpu` or `xpu`; XPU must be available |
| `num_threads` | `4` | OpenMP threads of the CPU kernels |
| `kv_fp8_dtype` | none | format of an fp8 (uint8) KV cache; fused layout, `per_tensor=false` only |
| `groups`, `default_group` | none | static per-group facts (see below) |
| `max_workers` | `1` | serde thread-pool size |

`scaling_method`, `num_kv_heads` and `head_dim` are rejected: the scaling is
fixed per layout and needs no head geometry.

## Scaling

A chunk only carries `num_kv_heads * head_dim` as one hidden width, so the
serde uses only scalings whose grouping does not depend on the head split:

| `per_tensor` | split `[2,L,T,H]` | fused `[L,T,H]` |
|---|---|---|
| `false` | `per_token`: one scale per token over the whole hidden axis; RH over `H` | `per_channel`: one scale per hidden column; RH over the chunk's tokens |
| `true` | `per_tensor`: one scale per plane, no RH | `per_tensor`: one scale, no RH |

The quantized values are bit-identical to configuring the true head geometry
for these combinations.

## What a chunk is: per-group routing

What the codec must do depends on what the chunk holds, so every key is routed
by its `object_group_id`:

| Group holds | Chunk shape | Handling |
|---|---|---|
| attention, split K/V | `[2, L, T, H]` | quantize both planes |
| attention, fused-packed K/V | `[L, T, H]` | fused codec |
| MLA, unclassified, or not smaller | any | bytes copied through unchanged |

An object group's chunk is one memory object holding **one tensor per kernel
group** (a *part*). A multi-part chunk is stored as `[int32 length | payload]`
per part; a single-part chunk is the bare payload.

### Where the facts come from

Quantizing needs the group's category and K/V packing. `groups` maps an
`object_group_id` to one spec per part (`cache_category`,
`attention_plane_layout`), and `default_group` describes every other group. By
default it is attention with layout `auto`: a `[2, L, T, H]` chunk is split K/V,
a `[L, T, H]` chunk fused K/V, and any other shape is copied through.

A shape cannot tell fused K/V from other single-plane formats (an MLA latent or
an indexer cache are also `[L, T, H]`): with `auto` they would be quantized as
fused K/V. Declare such groups explicitly, e.g.
`"groups": {"1": [{"cache_category": "attention", "attention_plane_layout": "mla"}]}`.

## Size contract

The temp buffer is allocated **before** serializing, so the size must be known
from the layout alone. GHA's payload size depends only on shape, `qbit` and
the scaling (`per_tensor`) — not on the data, `rh` or `asym` — so `estimate_serialized_size`
is an exact formula, not a margin:

```
split  [2,L,T,H]: 28 + 2*(4 + 12*S + 4*L) + 2*ceil(L*T*H*qbit/8)
fused  [L,T,H]  : 25 + 12*S + 4*native_layers + ceil(L*T*H*qbit/8)
```

`S` is the scale count of the scaling in use. The tests assert the estimate
equals the bytes actually written, on CPU and XPU. Because the size can differ
between groups that share a shape, the wrapper sizes each temp buffer through
`estimate_serialized_size_for_key` (see the wrapper doc).

## Decision per chunk

`gha_l2._PlanResolver` decides once per `(group spec, part shape, dtype)` and
caches the result. A chunk is copied through (never quantized) when its group is
MLA/unclassified, its rank does not match the declared layout, the
dtype is unsupported (a uint8 fp8 cache needs the fused layout and
`kv_fp8_dtype`), or quantizing would not make it smaller. The native RH kernel
needs a power-of-two transform length; a shape that cannot satisfy it is encoded
without RH (the header records the choice, and the size is unchanged).

Payloads carry a magic number, so decoding a buffer that is not a GHA payload
fails the deserialize task instead of returning garbage. The group
configuration must be the same on store and load: a pass-through chunk has no
header to check. That includes `device`: for 4-bit per_channel (fused) chunks
the XPU kernels pack two consecutive tokens of a channel per byte while the CPU
kernels pack two adjacent channels of a token, so CPU- and XPU-written payloads
are not interchangeable.

## Module layout

```
serde/gha/
  gha_config.py   # AttentionPlaneLayout, GHACodecConfig
  gha_serde.py    # GHACodec: the native-kernel attention codec
  gha_l2.py       # per-group routing, Serializer/Deserializer, factory registration
csrc/sycl/gha/
  cpu/                # CPU fallback native kernels/bindings
  xpu/                # SYCL/XPU native kernels/bindings
```

GHA is Intel-only. The native modules are built only by the SYCL (Intel
XPU) build profile, and only when ``BUILD_GHA=1`` is set at install time
(e.g. ``BUILD_GHA=1 pip install -e .`` with oneAPI's ``icpx`` on ``PATH``):
``lmcache.gha_quant`` for the x86-64 host CPU (AVX2) and
``lmcache.gha_quant_xpu``. Other platforms never build them, and the
``gha`` serde raises an error when it is configured without them; the rest
of LMCache is unaffected.
