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
2. Extend `discover_gpu_kv_format` to detect it.
3. Add a branch in every `utils.py` helper that raises "Unknown GPU KV
   Format" — the exhaustive chain makes it mechanical.
4. Add a row in `tests/v1/gpu_connector/test_utils_shape_desc.py`.

No other Python module should need edits. If you're editing
`kv_layer_groups.py`, `gpu_context.py`, or any `KVLayerGroupInfo`
consumer for a new layout — the branching belongs in `utils.py`.

## Helper surface

Every helper below takes `DiscoverableKVCache` and (where layout matters)
a `GPUKVFormat`. Nothing else may index raw shapes.

### Whole-structure queries

| Helper | Returns | Notes |
|---|---|---|
| `discover_gpu_kv_format(kv_caches, engine, layout_hints)` | `GPUKVFormat` | The one parser. |
| `get_num_layers`, `get_num_blocks`, `get_block_size`, `get_num_heads`, `get_head_size`, `get_hidden_dim_size`, `get_dtype`, `get_page_buffer_size`, `get_tokens_per_layer`, `get_elements_per_layer` | `int` / `dtype` | Format-dispatched scalar accessors. |
| `is_mla(fmt)`, `is_hnd(fmt)` | `bool` | Format predicates. |
| `get_device(kv_caches)` | `torch.device` | Format-agnostic (descends to any leaf). |

### Per-layer queries

| Helper | Returns | Notes |
|---|---|---|
| `get_layer_kv_caches(kv_caches, fmt, layer_idx)` | `DiscoverableKVCache` | Sub-structure for one layer, reusable as input to other helpers. |
| `get_layer_dtype(kv_caches, fmt, layer_idx)` | `torch.dtype` | |
| `get_layer_shape_signature(kv_caches, fmt, layer_idx)` | `tuple[int, ...]` | `(kv_size, num_heads, head_size)` grouping key. |
| `get_layer_data_ptrs(kv_caches, fmt, layer_idx)` | `list[int]` | Per-layer pointer(s). **Raises** for cross-layer (no per-layer pointer exists). |

### Group-level + builders

| Helper | Returns | Notes |
|---|---|---|
| `get_group_data_ptrs(kv_caches, fmt, layer_indices)` | `list[int]` | Pointer array in **kernel-expected order**: `[base]` for cross-layer, `[K_0…K_N, V_0…V_N]` for SGLang MHA, per-layer flat elsewhere. Matches the dispatch in `csrc/mp_mem_kernels.cu:161-169`. |
| `make_page_buffer_shape_desc(kv_caches, fmt, layer_idx, num_layers_in_group, num_blocks, block_size)` | `PageBufferShapeDesc` | The kernel-facing shape struct. |

### Contiguity

| Helper | Returns | Notes |
|---|---|---|
| `any_non_contiguous(kv_caches)` | `bool` | Recursive check. |
| `attempt_permute_to_contiguous_view(kv_caches)` | `DiscoverableKVCache` | Recursive, metadata-only. No-op if already contiguous; raises `ValueError` for non-permutation-recoverable cases (slicing, `as_strided`). **Never copies.** |
| `ensure_contiguous_kv_caches(kv_caches, kv_layout)` | `DiscoverableKVCache` | Thin wrapper adding HND-vs-other logging. |

## Forbidden outside `utils.py`

- `isinstance(kv_cache, (tuple, list))` to distinguish layouts.
- Indexing raw shapes (`tensor.shape[3]`, `len(shape) == 5`) to derive
  dimensions.
- Wrapping a tensor with `[tensor]` to adapt to a helper's list-depth
  expectation — use `get_layer_kv_caches`.
- Hand-rolled pointer assembly (`[t.data_ptr() for t in kv_caches]`) —
  use `get_group_data_ptrs`.
- Hand-rolled device discovery (`kv_caches[0][0].device`) — use
  `get_device`.
- "Canonicalize" functions that rewrite `kv_caches` to a uniform shape
  before passing to helpers. The helpers already canonicalize by
  accepting `GPUKVFormat`.

## Consumers

- **`lmcache/v1/kv_layer_groups.py::KVLayerGroupsManager.__init__`** —
  partitions layers via `get_layer_shape_signature` + `get_layer_dtype`,
  builds a `PageBufferShapeDesc` per group via
  `make_page_buffer_shape_desc`. Proper constructor; no side-effectful
  `build_*` method. Classmethod `from_layer_groups` exists for test
  fixtures.
- **`lmcache/v1/multiprocess/gpu_context.py::GPUCacheContext`** —
  constructs the manager directly at init, delegates
  `get_shape_desc(group_idx)` to it, assembles per-group GPU pointer
  tensors via `get_group_data_ptrs`. No parallel `shape_descs_` /
  `hidden_dim_sizes_` state.
- **`lmcache/v1/gpu_connector/gpu_connectors.py::VLLMPagedMemGPUConnectorV3._initialize_kv_cache_pointers`**
  — for the in-process vLLM path, discovers format and constructs
  `metadata.kv_layer_groups_manager` lazily on first store/retrieve.
  The adapter (`vllm_v1_adapter.py`) does not participate in format
  discovery — it only stores `self.kv_caches` at register time.

## Implementation note: mypy and the recursive union

`utils.py` sets `# mypy: disable-error-code="union-attr,call-overload"`
at the file level. This is the **one module** that does format-
dispatched raw indexing on `DiscoverableKVCache` (`kv_caches.shape[i]`,
`kv_caches[0][j]`) — the `gpu_kv_format` argument is the proof the
indexing is well-defined, but mypy can't carry that proof through a
recursive Union without per-line casts. The file-level directive
replaces 50+ `# type: ignore` comments scattered through the
accessors. All other type checks remain live.
