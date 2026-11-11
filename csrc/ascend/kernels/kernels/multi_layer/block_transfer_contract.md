# Block-Level KV Transfer Kernel Contract

This document is the host/device contract for the MP block-level KV transfer:

- device kernel: `multi_layer_block_mem_kernels.cpp` 
- host parser/validator: `csrc/ascend/mp_mem_kernels.cpp`
  (`prepare_group` + `validate_launch` are the single admission authority)

The kernel is a pure byte mover: every geometry question (format, plane
packing, strides, offsets) is resolved host-side and passed in as a flat
`BlockTransferLayout` POD. Everything below must hold before launch; the
host layer enforces it and rejects violations with `TORCH_CHECK`.

## 1. Supported engine KV formats

| Format | Enum | `separate_plane` | Work item | LMC layout |
|---|---|---|---|---|
| 2 | `NL_X_NB_TWO_BS_NH_HS` | `true` | (layer, plane, block) | same LMC layout as format 16. The pointer table is one base `data_ptr()` per layer; plane `p` is `base + engine_base_offset_bytes[p]` (derived by `prepare_group`, section 6) |
| 13 | `NL_X_NB_BS_NH_CS` | `false` | (layer, block) | `[L, T, NH*CS]`, one fused K/V plane |
| 16 | `NL_X_TWO_X_NB_BS_NH_HS` | `true` | (layer, plane, block) | `[2, L, T, NH*HS]`, K slab then V slab |
| 17 | `NL_X_NP_X_NB_BS_ONE_HS` | `false` | (layer, block), all planes moved by one core | `[L, T, sum(payload)]` packed byte rows |

Any other format is rejected by `prepare_group`. With
`separate_plane == false`, every plane of a page is moved by ONE core so no
two cores ever share a 32B LMC line (thin planes such as a 2B scale share
cache lines with their row neighbours).

## 2. Host-side validation (launch gate)

Geometry (`prepare_group`, resolved once per group):

- `PageBufferShapeDesc` is the shared type from `csrc/kv_transfer_plan_types.h`,
  registered by `lmcache_native`: the server hands the kernels that module's
  instance, and the by-value copy at the module boundary copies exactly that
  layout
- `nl, nb, bs, nh, hs > 0`; `element_size in {1, 2, 4}`
- `kv_size == 2` for formats 2 and 16, `== 1` for formats 13 and 17
- format 13 with `num_planes == 0` derives its single-plane geometry in
  `prepare_group`: `payload_bytes = nh * hs * element_size`, and
  `engine_block_stride_bytes = block_stride_elems * element_size` when an
  explicit block stride is present, otherwise `bs * payload_bytes`; explicitly
  supplied geometry is preserved
- after geometry resolution, `1 <= num_planes <= 4`; exactly `1` for format
  13, exactly `2` for formats 2 and 16; `nh == 1` for format 17
- per plane: `payload_bytes > 0` and
  `AlignUp32(payload_bytes) <= ub_segment` (one aligned row of every plane
  fits one queue slot), where `ub_segment` is the host-queried UB budget
  divided by the queue depth (see section 3)
- per plane: `engine_block_stride_bytes >= bs * payload_bytes` (no block
  overlap) and `engine_block_stride_bytes % 32 == 0`
- address-range probe `(nb - 1) * block_stride + bs * payload` fits int64
- formats 2, 13 and 16: `payload_bytes == nh * hs * element_size` per plane
- format 2: both plane base offsets are derived from the scalar descriptor
  (section 6), and `offset + bs * payload <= block_stride` for each plane. No
  format reads a base offset from the descriptor
- format 17: `sum(payload_bytes) == nh * hs * element_size`
  (the packed row width)
- `bs * lmc_row_bytes % 32 == 0` (adjacent LMC pages do not share a 32B line)
- `lmc_row_bytes - payload_bytes <= uint32 max`
  (DataCopyPad GM-side gap field is uint32)

Launch variables (`validate_launch`, per transfer):

- `1 <= num_objects <= 4`; `total_blocks % num_objects == 0`
- `blocks_per_object * bs == slots_per_object == lmcache_chunk_size`
- `0 <= skip_prefix_n_blocks <= blocks_per_object`
- the `block_ids` slice fits the provided capacity; counts fit int32

Producer contract (runtime, not launch-checkable):

- every `block_ids[i]` is in `[0, nb)`. The kernel defensively skips
  out-of-range ids -- a data-dependent check that, unlike `skip`, cannot be
  folded into the launch because the id values live in GM.

## 3. UB budget and its relation to block size

`kBlockTransferUbBytes` (128KB) bounds a single token ROW, not a whole
block. The kernel derives its segment size as:

```
tokens_per_segment = min(bs,
                         kBlockTransferUbBytes
                             / (kBlockTransferQueueDepth
                                * AlignUp32(max payload_bytes)),
                         kMaxRowsPerDataCopyPad)
```

- `bs` (tokens per block) is UNCONSTRAINED: rows are moved in segments of
  `tokens_per_segment` and one plane block may span multiple segments
- The only hard geometry requirement is that one aligned row of the widest
  plane fits a single queue slot (validated in `prepare_group`).
- The host queries the real size at launch time via
  `PlatformAscendCManager::GetCoreMemSize(CoreMemType::UB)` and passes it
  down verbatim as `ub_bytes`. `ub_bytes == 0` (query unavailable) falls 
  back to the built-in 128KB floor. 

## 4. DataCopyPad limits

Per the AscendC API reference for `DataCopyExtParams`:

| Field | Range | Kernel usage |
|---|---|---|
| `blockCount` | [1, 4095] | rows per segment (clamped to `kMaxRowsPerDataCopyPad`) |
| `blockLen` | [1, 2097151] bytes | `payload_bytes` per row |
| `srcStride` / `dstStride` | uint32 | GM side in bytes, UB side in 32B blocks |

Alignment: `DataCopyPad` has NO GM-side address alignment constraint; the
UB-side `LocalTensor` start must be 32B aligned (guaranteed by the queue
allocator). The contiguous fast path uses plain `DataCopy` instead and
therefore additionally requires 32B-aligned GM bases and a 32B-multiple
payload; the kernel checks this per work item and falls back to the row
path otherwise.

## 5. Work distribution and skip semantics

- `block_dim = min(AIV cores, ACTIVE work items)`
  `= min(cores, nl * plane_slots * (blocks - skip))`.
- Work items span ACTIVE blocks only: `block = skip + (w % active_blocks)`.
  No core iterates over skipped blocks; there are no dead work items.
- `skip == blocks` (a legal input) means nothing to transfer: the host
  skips the launch entirely (`<<<0>>>` would be an invalid launch) and the
  kernel-side loop is empty.
- Out-of-range engine block ids are skipped at runtime (data-dependent, see
  the producer contract above).

## 6. Pointer table

**One pointer per plane (default):** `ptrs[layer * num_planes + plane]`.
Each entry is a plane tensor view's `data_ptr()`, already including that
view's `storage_offset`. The table contains `nl * num_planes` int64 entries.
This is the usual layout when each plane is its own tensor view (for example
per-layer `(K, V)` pairs, enum format **16**, or multiple single-head planes
indexed separately while a single core moves every plane of a page, enum
format **17**).

**One pointer per layer with multiple planes:** `ptrs[layer]` only (`nl`
entries). Plane `p` is reached at
`ptrs[layer] + engine_base_offset_bytes[p]`. For format **2**,
`prepare_group` derives those offsets from the scalar descriptor: plane 0
is `0`, plane 1 is
`bs * nh * hs * element_size`, and both planes share
`block_stride_elems * element_size` (or the tight
`2 * bs * nh * hs * element_size` when `block_stride_elems` is 0). A
nonzero `storage_offset` on the layer tensor is already folded into the
base pointer. The planes are equal-width slabs inside one engine tensor
(axis-1 halves of a 5D cache). They may be semantic K and V, or the first
and second halves of an opaque logical page after a sub-paged attention
edit; address calculation is the same.

The table must live on the NPU.

## 7. Bridge: one pointer per layer while `num_planes > 1`

Normally, `num_planes > 1` implies one `data_ptr()` per plane (section 6).
The Ascend MP bridge is the opposite: **`one_pointer_per_layer` with
`separate_plane` work items**—one base pointer per layer, but the kernel
still copies multiple planes using `engine_base_offset_bytes`.

That path is used today to move **hybrid interleaved attention** by
registering engine enum format **2**: LMCache presents a synthetic 5D view
and plane offsets, even though axis 1 is **not** a faithful description of
the bytes in vLLM-Ascend's real interleaved page. Semantic K/V fused caches
can use the same mechanism when a single tensor owns both slabs.

On vLLM-Ascend v0.30 hybrid stacks, the runner still owns the physical
layout. See `vllm_ascend/worker/model_runner_v1.py`,
`_reshape_kv_cache_tensors` (from ~4351): when `use_hybrid_blocks` is set
(~3856, ~4559), attention layers use kernel block sizes from
`get_supported_kernel_block_sizes` / `get_kv_cache_shape`, then expose
`(k_cache, v_cache)` tuple views (~4680) over the same raw allocation—not
one 5D tensor whose axis-1 halves are independent K and V planes.

**Direction:** hybrid interleaved attention should get a dedicated engine
format (proposed enum **18**) instead of relying on format **2** plus
offsets forever.

**Proposed format 18 (sketch):**

- Single-plane registration (`kv_size == 1`, `num_planes == 1`), one
  `data_ptr()` per layer (`nl` pointers), `separate_plane == false` so one
  work item moves the whole page (same scheduling idea as enum format **17**).
- One opaque byte span per logical page (interleaved K/V as the runner laid
  it out), copied in one shot—no plane base offsets, no
  `one_pointer_per_layer` branch in the device kernel.
- LMCache would register the true page width from the runner instead of
  splitting a format-**2** tensor at axis 1.

Until then, the one-pointer-multi-plane bridge stays supported via host
`prepare_group` checks (section 2), the `one_pointer_per_layer` launch
flag, and `engine_base_offset_bytes` on each plane in the layout POD.
