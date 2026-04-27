# GPU KV Cache Layout — Single Source of Truth

## Invariant

> **`discover_gpu_kv_format` is the only place that parses KV-cache layout.**
> Every other module queries KV-cache information via helpers in
> `lmcache/v1/gpu_connector/utils.py` that accept a `GPUKVFormat`
> argument.

"Layout parsing" means: list-nesting depth, tensor-dimension ordering,
HND vs NHD, MLA vs MHA, per-layer vs cross-layer. All of that is
encoded in `GPUKVFormat`; downstream code must never re-derive it from
raw shapes.

## Canonical type

```python
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]
```

Every KV-cache value in LMCache is one of these shapes:

- a single `torch.Tensor` (vLLM cross-layer, TRT-LLM),
- a flat `list[torch.Tensor]` (vLLM per-layer, SGLang MLA),
- a nested `list[list[torch.Tensor]]` (SGLang MHA's `[K_list, V_list]`).

Engine adapters that hand us other containers (vLLM's `dict[str, Tensor]`)
are responsible for unwrapping to this form before calling any helper.

## Adding a new format

1. Add the enum value in `csrc/mem_kernels.cuh` and `csrc/pybind.cpp`.
2. Extend `discover_gpu_kv_format` to detect it. The dispatch keys off
   `(list_depth, tensor_ndim)` — both are computed once in a single
   descent via the private `_list_depth_tensor_dim` probe, so new
   detection branches only need to add a shape check, not re-walk the
   structure.
3. Add a branch in every `utils.py` helper that raises "Unknown GPU KV
   Format" — the exhaustive chain makes it mechanical.
4. Add a row in `tests/v1/gpu_connector/test_utils_shape_desc.py`.

No other Python module should need edits. If you're editing
`kv_layer_groups.py`, `gpu_context.py`, or any `KVLayerGroupInfo`
consumer for a new layout — the branching belongs in `utils.py`.

## Helper surface

Every helper below takes `DiscoverableKVCache` and (where layout matters)
a `GPUKVFormat`. Nothing else may index raw shapes.

### Discovery

| Helper | Returns |
|---|---|
| `discover_gpu_kv_format(kv_caches, engine, layout_hints)` | `GPUKVFormat` — the one parser. |

### Scalar accessors

All of these dispatch on `GPUKVFormat`. The ones that can vary per layer
take an optional `layer_idx: int = 0`; passing an explicit index enables
per-layer queries (for heterogeneous groups) without any intermediate
helper.

| Helper | Per-layer? | Notes |
|---|---|---|
| `get_num_layers(kv, fmt)` | no | Total layer count. |
| `get_num_blocks(kv, fmt)` | no | Paged block count (group-level). |
| `get_block_size(kv, fmt)` | no | Tokens per block. |
| `get_page_buffer_size(kv, fmt)` | no | |
| `get_tokens_per_layer(kv, fmt)` | no | |
| `get_elements_per_layer(kv, fmt)` | no | |
| `get_num_heads(kv, fmt, layer_idx=0)` | yes | |
| `get_head_size(kv, fmt, layer_idx=0)` | yes | |
| `get_hidden_dim_size(kv, fmt, layer_idx=0)` | yes | |
| `get_dtype(kv, fmt, layer_idx=0)` | yes | |
| `is_mla(fmt)`, `is_hnd(fmt)` | — | Format predicates. |
| `get_device(kv)` | — | Format-agnostic (descends to any leaf). |

### Pointer and descriptor builders

| Helper | Returns | Notes |
|---|---|---|
| `get_group_data_ptrs(kv, fmt, layer_indices)` | `list[int]` | Pointer array in **kernel-expected order**: `[base]` for cross-layer (`layer_indices` ignored), `[K_0…K_N, V_0…V_N]` for SGLang MHA, per-layer flat elsewhere. Matches the dispatch in `csrc/mp_mem_kernels.cu:161-169`. The pointer-array shape is a property of the format — callers never ask "does this format have per-layer pointers?". |
| `make_page_buffer_shape_desc(kv, fmt, layer_idx, num_layers_in_group, num_blocks, block_size)` | `PageBufferShapeDesc` | The kernel-facing shape struct. |

### Contiguity

| Helper | Returns | Notes |
|---|---|---|
| `attempt_permute_to_contiguous_view(kv)` | `DiscoverableKVCache` | Recursive, metadata-only. No-op if already contiguous; raises `ValueError` for non-permutation-recoverable cases (slicing, `as_strided`). **Never copies.** Walks the full structure and permutes every tensor leaf. |

## Forbidden outside `utils.py`

- `isinstance(kv_cache, (tuple, list))` to distinguish layouts.
- Indexing raw shapes (`tensor.shape[3]`, `len(shape) == 5`) to derive
  dimensions.
- Hand-rolled list-depth probing (`while isinstance(x, list): depth +=
  1; x = x[0]`). There is no public depth helper and there shouldn't
  be one — `discover_gpu_kv_format` encapsulates the descent, and
  downstream code only ever needs the resulting `GPUKVFormat`.
- Wrapping a tensor with `[tensor]` to adapt to a helper's list-depth
  expectation — the accessors take `layer_idx` directly.
- Hand-rolled pointer assembly (`[t.data_ptr() for t in kv_caches]`) —
  use `get_group_data_ptrs`.
- Hand-rolled device discovery (`kv_caches[0][0].device`) — use
  `get_device`.
- Hand-rolled contiguity fixes (`tensor.contiguous()`, `.clone()`) —
  use `attempt_permute_to_contiguous_view` which refuses to copy.
- "Canonicalize" functions that rewrite `kv_caches` to a uniform shape
  before passing to helpers. The helpers already canonicalize by
  accepting `GPUKVFormat`.

## Consumers

- **`lmcache/v1/kv_layer_groups.py::KVLayerGroupsManager.__init__`** —
  partitions layers by the 4-tuple `(kv_size, num_heads, head_size,
  dtype)` using `is_mla`, `get_num_heads`, `get_head_size`, and
  `get_dtype` with each layer's index. Builds a `PageBufferShapeDesc`
  per group via `make_page_buffer_shape_desc`. The real constructor is
  the only way in — no test-only shortcuts, no cached topology fields;
  the manager exposes only `kv_layer_groups`, `num_groups`, and
  `get_shape_desc`.
- **`lmcache/v1/multiprocess/gpu_context.py::GPUCacheContext`** —
  constructs the manager directly at init, delegates
  `get_shape_desc(group_idx)` to it, assembles per-group GPU pointer
  tensors via `get_group_data_ptrs`. No parallel `shape_descs_` /
  `hidden_dim_sizes_` state.
- **`lmcache/v1/gpu_connector/gpu_connectors.py::VLLMPagedMemGPUConnectorV3._initialize_kv_cache_pointers`**
  — for the in-process vLLM path, discovers format (after one call to
  `attempt_permute_to_contiguous_view` for HND support) and constructs
  `metadata.kv_layer_groups_manager` lazily on first store/retrieve.
  The adapter (`vllm_v1_adapter.py`) does not participate in format
  discovery — it only stores `self.kv_caches` at register time.

Only `discover_gpu_kv_format` consumes `layout_hints`.
`attempt_permute_to_contiguous_view` infers the permutation from
strides and needs no hints.

## Implementation note: mypy and the recursive union

`utils.py` sets `# mypy: disable-error-code="union-attr,call-overload"`
at the file level. This is the **one module** that does format-
dispatched raw indexing on `DiscoverableKVCache` (`kv_caches.shape[i]`,
`kv_caches[0][j]`) — the `gpu_kv_format` argument is the proof the
indexing is well-defined, but mypy can't carry that proof through a
recursive Union without per-line casts. The file-level directive
replaces 50+ `# type: ignore` comments scattered through the
accessors. All other type checks remain live.
