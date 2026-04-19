# GPU KV Cache Layout — Single Source of Truth

## Invariant

> **`discover_gpu_kv_format` is the only place that parses KV-cache layout.**
> Every other module queries KV-cache information via helpers in
> `lmcache/v1/gpu_connector/utils.py` that accept a `GPUKVFormat` argument.

Layout parsing means: list-nesting depth, tensor-dimension ordering,
distinguishing HND from NHD, MLA from MHA, per-layer from cross-layer. All
of this is encoded in the `GPUKVFormat` enum; downstream code must not
re-derive it from raw shapes.

## Why this matters

Supporting a new serving engine or KV-cache layout (e.g. TRT-LLM's cross-layer
single-tensor format) should require:

1. Adding a new `GPUKVFormat` enum value.
2. Extending the `gpu_connector/utils.py` helpers to branch on that value.

Nothing else should need to change. The alternative — scattering
`isinstance(kv_cache, (tuple, list))`, `len(shape) == 5`, or `kv_caches[0][0]`
across call sites — forces every new format to touch every consumer and
tempts "canonicalization" shims that rewrite the inputs to the shape callers
expect.

## Allowed patterns (inside `gpu_connector/utils.py`)

Helpers in `utils.py` dispatch on `GPUKVFormat` using a flat
if/elif chain. This is deliberate: every supported format is visible in one
place, and adding a new format's row is a mechanical change.

```python
# lmcache/v1/gpu_connector/utils.py — allowed
if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
    return kv_caches.shape[1]
elif gpu_kv_format in (
    lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
    lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ...
):
    return len(kv_caches)
```

## Forbidden patterns (outside `gpu_connector/utils.py`)

Any of the following in a module that is not `utils.py` or `discover_gpu_kv_format`:

- `isinstance(kv_cache, (tuple, list))` to distinguish layouts.
- Indexing raw tensor shapes (`kv_cache.shape[3]`, `len(shape) == 5`) to
  derive dimensions.
- Wrapping a tensor with `[tensor]` to adapt to a helper's list-depth
  expectation. Use `get_layer_kv_caches` instead.
- "Canonicalize" functions that rewrite `kv_caches` into a uniform shape
  before passing to helpers. The helpers already canonicalize by accepting
  `GPUKVFormat`.

## Helper surface

The following per-layer helpers are the only way for code outside `utils.py`
to reach layer-specific KV-cache data:

| Helper | Purpose |
|---|---|
| `get_layer_kv_caches(kv_caches, fmt, layer_idx)` | Returns the sub-structure (list or narrowed tensor) representing a single layer, in a shape other helpers can accept. |
| `get_layer_data_ptrs(kv_caches, fmt, layer_idx)` | Returns the device pointer(s) for one layer, used to build per-group GPU pointer tensors. |
| `get_layer_dtype(kv_caches, fmt, layer_idx)` | Returns the dtype of a layer's KV tensor. |
| `get_layer_shape_signature(kv_caches, fmt, layer_idx)` | Returns a hashable `(kv_size, num_heads, head_size)` tuple used as a grouping key. |
| `make_page_buffer_shape_desc(kv_caches, fmt, layer_idx, num_layers_in_group, num_blocks, block_size)` | Builds a complete `PageBufferShapeDesc` for the layer's group. |

Existing extractors (`get_num_layers`, `get_num_blocks`, `get_block_size`,
`get_num_heads`, `get_head_size`, `get_hidden_dim_size`, `get_dtype`,
`is_mla`, `is_hnd`) remain the canonical accessors for whole-`kv_caches`
queries.

## Consumers

- `lmcache/v1/kv_layer_groups.py::KVLayerGroupsManager.__init__` uses
  `get_layer_shape_signature` + `get_layer_dtype` to partition layers, then
  calls `make_page_buffer_shape_desc` per group. The per-group
  `PageBufferShapeDesc` is stored on the `KVLayerGroupInfo`. The manager
  is a proper constructor, not a two-phase "build" — a classmethod
  `from_layer_groups` exists for test fixtures that already hold groups.
- `lmcache/v1/multiprocess/gpu_context.py::GPUCacheContext` constructs the
  manager directly, delegates `get_shape_desc(group_idx)` to it, and uses
  `get_layer_data_ptrs` to assemble per-group GPU pointer tensors. It
  holds no parallel `shape_descs_` / `hidden_dim_sizes_` lists.
- `lmcache/integration/vllm/vllm_v1_adapter.py::_build_kv_layer_groups`
  calls `discover_gpu_kv_format`, `get_num_blocks`, `get_block_size` at
  register time, then constructs
  `KVLayerGroupsManager(kv_list, gpu_kv_format=..., num_blocks=..., ...)`
  and assigns it to `metadata.kv_layer_groups_manager`.

## Extending to a new format

1. Add the enum value in `csrc/mem_kernels.cuh` and `csrc/pybind.cpp`.
2. Add a branch for the new value in every helper in `utils.py` that raises
   "Unknown GPU KV Format". The compiler-style exhaustiveness ensures
   nothing is missed.
3. Extend `discover_gpu_kv_format` to recognize and return the new value.
4. Add a row in the tests at `tests/v1/gpu_connector/test_utils_shape_desc.py`
   covering `make_page_buffer_shape_desc` and the layer-level helpers.

No other Python module should need edits. If you find yourself editing
`kv_layer_groups.py`, `gpu_context.py`, or any consumer of `KVLayerGroupInfo`
to support a new layout, pause — the branching probably belongs in `utils.py`.
