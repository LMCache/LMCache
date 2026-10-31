# Spec-owned transfer regions

## Registration contract

`LMCacheMPConnector.register_kv_caches` obtains each layer's spec from the existing
`KVCacheConfig`, after applying the existing group edits. It calls
`get_kv_cache_transfer_regions(native_cache)` when the producer implements it.
The proposed upstream API returns named `[blocks, heads, states, content]` views
or `None`. This work uses `key` and `value` for unquantized dense attention.
No connector registration hook or dimension-semantic descriptor is added.

The producer must return views of the actual registered storage. LMCache checks
rank, matching geometry, dtype, device, and shared storage before accepting the
regions. Copies are rejected: retrieve must write back to attention's cache.
Uniform per-layer specs are unwrapped before invoking the exporter. Exact tensor
aliases omitted from allocation specs reuse their producer's regions; a remaining
custom six-dimensional cache without a declaration fails rather than guessing.

```text
native RBLN cache + per-layer spec in KVCacheConfig
    |
    | get_kv_cache_transfer_regions(native)
    v
named key/value BHNC regions (original storage + strides + offsets)
    |
    | apply_spec_transfer_regions: permute to internal NHD plane views
    v
per-layer [K_plane, V_plane], with kv_list_layout="unified"
    |
    +-- engine_driven: worker gather/scatter -> canonical chunks -> server
    |                                              | retrieve -> native views
    |
    +-- lmcache_driven: native owner IPC handles + per-layer view recipes
             -> server imports native owners -> restores NHD planes
             -> stride-aware pointer gather/scatter -> canonical chunks
             -> retrieve writes through imported views -> engine native storage
```

## Physical layouts

For native `[B,2,H,1,N,D]`, the producer selects each K/V plane on axis 1;
for `[2,B,H,1,N,D]`, it selects on axis 0. Both yield `[B,H,N,D]` regions
without concatenation or allocation. The producer captures its kernel's block
axis in the spec before allocation, so registration does not read ambient device
configuration or try to infer semantics when `B == 2`.

LMCache then permutes each view to `[B,N,H,D]` for its existing per-layer K/V
format. The internal `spec_transfer_regions` flag is significant: stride sorting would undo this logical
order for physical HND storage. Normalization preserves declared component views.

Blocks-first K/V planes have a gap between consecutive blocks, occupied by the
other plane. Tensor-based block ops consume those real strides directly. Native
CUDA pointer ops receive independent logical `[B,N,H,D]` K/V stride vectors in
`PageBufferShapeDesc`. Layers with different component strides are assigned to
different kernel groups. The native region branch uses byte addressing, including
non-unit head-dimension strides and unaligned rows; legacy vector kernels remain
on their existing routes. The stride map must be nonnegative and separably
non-overlapping (dense, permuted, sliced and padded views are supported).

### LMCache-driven registration and ownership

`LMCacheMPConnector` retains the original native tensors as well as the exported
planes. The worker registers the original tensor IPC handles once, plus versioned
data-only recipes in the existing `LayoutHints`: each plane's shape, stride and
offset **relative to its native view's first element**. No RPC signature or vLLM
connector registration signature changes. CPU SHM and all three CUDA owner
wrappers opt in; unsupported wrapper implementations fail explicitly.

The server opens native owners, adds each relative offset to the imported native
view's storage offset, reconstructs `as_strided` views, then normalizes and groups
those planes. This handles CUDA Raw/VMM imports whose storage origins are rebased.
Bounds and separable-stride checks run before view construction. Reconstructed
planes retain storage ownership through the cache context's imported wrappers.

CPU SHM migration rebinds the engine's original tensor to shared storage. The
worker refreshes the existing exported plane objects after that migration, so
heartbeat re-registration uses the same backing storage rather than stale views.
The adapter retains both native owners and planes for recovery. Wrapping K and V
separately would detach them from the engine's native owner and is deliberately
avoided.

The CPU pointer fallback reconstructs views from interleaved `[K0,V0,K1,V1,...]`
pointers and the descriptor strides. CUDA requires a native extension advertising
`supports_strided_kv_regions`; stale extensions are rejected before transfer.
`PageBufferShapeDesc` is shared ABI: rebuild **both** `lmcache_native` and
`cuda_ops` from the same revision. A CPU-only `NO_GPU_EXT=1` build is not a valid
CUDA deployment build.

RBLN routes component views through shared torch transfer operations. Its current
platform has no LMCache-driven IPC/event implementation and still selects
engine-driven transfer. This existing RBLN limitation does not restrict the
CPU/CUDA LMCache-driven region path; no unsupported RBLN handle mode is advertised.
NPU performance has not been validated against the tuned native-layout fast path.

## Scope and compatibility

Specs without an exporter, or returning `None`, retain legacy normalization.
Both MP transfer modes consume these exports on capable platforms. RBLN kernel block
subdivision is rejected: native states per block must match the spec block size,
because this integration does not translate scheduler IDs to kernel IDs. The vendor's separate
in-process `lmcache_rbln` connector and generic in-process GPU connectors are not
changed. Compressed/quantized codecs, heterogeneous TP transfer, and arbitrary
Mamba component exports are not implemented here.

The region-export API complements the unmerged vLLM `slice_for_tp_transfer`
work: region construction maps native storage; TP slicing narrows those standard
regions. This change performs local-partition transfers and does not substitute a
reference model for that upstream TP API.

## Validation

`tests/v1/multiprocess/test_spec_transfer_regions.py` exercises real
`EngineDrivenTransferContext.register`, `submit_store`, and `submit_retrieve`,
including the production gather/scatter and tensor wire codec. Only the network
client is mocked. The producer fixture is an external contract boundary.

The main matrix covers both source and destination layout orders (`B=2`),
float32/float16, two layers, reversed source blocks, and a block-aligned skipped
retrieve prefix. It checks exact wire values and writes to original native
storage, not merely a self-consistent roundtrip. Additional tests cover legacy
registration, rejected copied regions, missing native IPC owners, shared
layer aliases, and rejected meta storage. The RBLN device-ops tests additionally
check exact canonical bytes and native writeback through both component layouts. No arbitrary unaligned token-range support is claimed: the
existing MP path transfers whole blocks.

RBLN's companion spec tests exercise all three attention spec classes, both
kernel layouts, and preservation of the captured layout across spec merging.
Source-isolated CPU checks do not substitute for the normal SDK-loaded suite,
real NPU inference, or performance validation.

`tests/v1/platform/test_transfer_regions.py` exercises real LMCache-driven worker
registration and the production tagged handle/recipe serializer. A spawned server
process imports POSIX SHM into `CPUCacheContext`, performs pointer-based D2H/H2D,
and writes back to the original engine tensor. Both B=2 native orders, nonzero
owner offsets, per-layer padding/strides and untouched gaps are checked. Recipe
validation covers invalid geometry, overlap, bounds and origin rebasing.

`tests/v1/platform/test_cuda_transfer_regions.py` calls the actual native CUDA
pointer kernel with both native orders, FP16/BF16, skipped block prefixes, padded
rows, non-unit D strides and multiple kernel groups. It checks exact canonical
bytes, native writeback and untouched holes. These tests skip without CUDA or the
new native capability; CPU checks do not establish CUDA compiler/runtime support.
