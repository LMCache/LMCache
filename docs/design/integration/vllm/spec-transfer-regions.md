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
    | existing engine-driven registration + canonical chunk geometry
    v
EngineDrivenTransferContext.submit_store -> gather_paged_kv_to_cpu
    -> tensor wire/SHM -> server
server -> prepare_retrieve -> scatter_cpu_to_paged_kv
    -> original native cache through the retained views
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
other plane. Tensor-based block ops consume those real strides directly. The
block-stride resolver and layer-group manager only permit this when explicitly
enabled by the spec-region engine-driven path. Pointer-only transfer ops and
LMCache-driven wrappers are rejected, because reconstructing a tight tensor from
a pointer would lose that information. RBLN routes component views through its
shared torch transfer implementation; this has not been performance validated on
an NPU and does not claim the tuned native-layout fast path's performance.

## Scope and compatibility

Specs without an exporter, or returning `None`, retain legacy normalization.
Only the MP engine-driven path consumes these exports. RBLN kernel block
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
registration, rejected copied regions, rejected handle transport, shared
layer aliases, and rejected meta storage. The RBLN device-ops tests additionally
check exact canonical bytes and native writeback through both component layouts. No arbitrary unaligned token-range support is claimed: the
existing MP path transfers whole blocks.

RBLN's companion spec tests exercise all three attention spec classes, both
kernel layouts, and preservation of the captured layout across spec merging.
Source-isolated CPU checks do not substitute for the normal SDK-loaded suite,
real NPU inference, or performance validation.
