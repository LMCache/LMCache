# Producer-declared tensor axes

The consumer API is staged against [vLLM #63699](https://github.com/vllm-project/vllm/pull/63699).
It must track that PR's final merged registration contract. Older vLLM versions
continue to call `register_kv_caches`, which retains legacy detection.

## Registration and transfer

```text
vLLM allocator / device backend
  -> register_kv_caches_with_layouts(layer tensors, per-layer descriptors)
  -> encode_tensor_layouts: enum names in registered dict order
     -> in-process: GPU connector.layout_hints, before pointer initialization
     -> MP: worker adapter -> existing REGISTER_KV_CACHE layout_hints
        -> IPC view preservation -> server cache context
  -> normalize_and_discover_per_layer_formats
  -> normalize_declared_tensor: tensor permutation + identical axes permutation
  -> existing EngineKVFormat / KVFormatSpec
  -> existing transfer kernels
```

The data-only envelope is `{"version": 1, "axes": [["BLOCK", "KV", ...], null]}`.
The list follows the tensor registration order, including unspecified layers.
The MP worker adapter retains this snapshot for heartbeat re-registration.
Subsetting a group must subset and reorder declarations identically; equal
shape/dtype does not imply equal axis semantics. Excluded aliases/scratch layers
retain their tensor and are not discovered.

## Why shape and stride are insufficient

For RBLN with two blocks, both `[B, 2, H, 1, N, D]` and `[2, B, H, 1, N, D]`
have identical shapes and contiguous strides. The producer's `BLOCK` and `KV`
roles select the existing blocks-first and K/V-first transfer formats. An
ambient `NHD`/`HND` setting cannot override a per-tensor declaration.

Axes describe the exact registered view, before normalization. The declared
path preserves named singleton axes. Structural `SINGLETON` axes are squeezed
with their labels removed, giving RBLN a five-dimensional zero-copy view for
existing kernels. For permuted views it permutes axes with the tensor, and
never copies data. Only
contiguous storage or supported outer BLOCK-row padding is accepted. The
legacy stride-ratio reshape must not run first: it can move singleton extents
between named dimensions and invalidate the declaration.

IPC factories may normalize views while exporting storage handles. For declared
layers `wrap_kv_caches(preserve_views=...)` snapshots the source view, restores
its indexing metadata after CPU SHM migration, and exports that original
shape/stride/offset. Both worker and server then see the same view. Wrapper
validation and resource rollback remain in place; unspecified layers retain
the old behavior. Concrete wrappers opt in via `supports_view_metadata`:
CPU SHM and the CUDA wrappers support independent shape/stride metadata. Raw
CUDA/VMM retain their own offset origin, since `alloc_offset` already contains
the data pointer displacement. Other handle wrappers reject declared mode;
engine-driven transfer avoids that handle restriction. MUSA handle view
restoration is not implemented by this change.

In-process connectors retain the original declaration separately from the
current normalized declaration. Every request rebuilds views and their labels
from that snapshot, including after a layerwise allocator has been initialized.

## Supported contracts and limits

- Single-tensor split K/V: K/V-first or blocks-first, NHD or HND.
- RBLN's two HND forms with a structural singleton between heads and states.
- Existing per-layer three-dimensional MLA planes and four-dimensional opaque
  CONTENT planes. CONTENT describes bytes/elements, not their encoding. The
  existing uint8/132-column DSA convention still selects blocked indexer pages;
  this is encoding-specific logic, not axis inference.
- Partial declarations: only unspecified entries use legacy detection.
- Mixed declared formats use the per-layer API. APIs requiring a single format
  reject mixed formats rather than silently selecting the first layer.
- Engine-driven multi-group transfer selects declarations per kernel group,
  including its initial geometry query and subsequent gather/scatter.

Invalid declarations, unknown schema versions, unsupported role orders and
unsupported strides fail explicitly. In-process device connectors without a
layout-hint consumer reject this hook. Axis metadata does not add a new kernel,
make an arbitrary stride layout transferable, or describe tuple component roles.

Opaque Mamba/subpage group edits are deliberately rejected when they replace a
declared tensor: the original roles cannot describe their synthetic byte planes.
Supporting those edits requires spec-aware conversion, including recurrent-state
granularity. Merely labeling synthetic planes `KV` would be incorrect.

This is the runtime producer/consumer bridge tracked by
[LMCache #5866](https://github.com/LMCache/LMCache/issues/5866).
[LMCache #4592](https://github.com/LMCache/LMCache/pull/4592) separately proposes
an internal general descriptor/bijection. This implementation reuses existing
EngineKVFormat/specs and does not depend on that unmerged descriptor proposal.
[RBLN #1297](https://github.com/RBLN-SW/vllm-rbln/pull/1297) supplies backend axes;
it also depends on the upstream registration hook. CPU tests of these tensor
layouts do not establish accelerator SDK or inference correctness.
