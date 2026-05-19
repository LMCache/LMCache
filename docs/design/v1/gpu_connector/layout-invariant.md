# GPU KV Cache Layout — Single Source of Truth

## Invariant

> **The `lmcache.v1.gpu_connector.kv_format` package is the only place
> that parses KV-cache layout.** Format discovery goes through
> `detect_format(kv_caches, engine, layout_hints)` (a per-engine
> `EngineDetector` strategy); format-keyed shape access goes through
> `get_spec(kv_caches, gpu_kv_format)` (a per-format `KVFormatSpec`
> strategy). `lmcache/v1/gpu_connector/utils.py` is a thin
> backward-compatibility facade that delegates to these two entry
> points; **no module other than `kv_format` (or `utils.py` while it
> still exists as a facade) may parse list-nesting / tensor shapes /
> HND vs NHD on its own.**

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

## Package layout

```
lmcache/v1/gpu_connector/kv_format/
├── __init__.py            # public surface: detect_format, get_spec, ...
├── base.py                # KVFormatSpec ABC (per-format strategy)
├── families.py            # PerLayer5DSpec, CrossLayer6DSpec, SGLangFusedPBSSpec
├── detection_base.py      # EngineDetector ABC (per-engine strategy)
├── registry.py            # auto-registration + lazy module discovery
├── types.py               # DiscoverableKVCache, LayoutHints
├── specs/                 # one file per concrete format
│   ├── vllm_cross_layer.py
│   ├── vllm_flash_attn.py     # NHD + HND variants
│   ├── vllm_flash_infer.py    # NHD + HND variants
│   ├── vllm_mla.py
│   ├── trtllm_cross_layer.py
│   ├── sglang_mha.py
│   └── sglang_mla.py
└── detectors/             # one file per engine
    ├── vllm.py
    ├── trtllm.py
    └── sglang.py
```

`registry.py` walks `specs/` and `detectors/` once on first lookup
(`pkgutil.iter_modules` + `importlib.import_module`); concrete classes
self-register through `__init_subclass__`. Failures from individual
modules are logged at WARNING and recorded in
`failed_module_reports()`, never raised — a broken third-party plugin
cannot take the registry down.

## Adding a new format

Pure additive. **No existing file needs to be edited.**

1. Add the enum value in `csrc/mem_kernels.cuh` and `csrc/pybind.cpp`.
2. Drop a new file under `lmcache/v1/gpu_connector/kv_format/specs/`
   with one `KVFormatSpec` subclass. If the axis layout matches an
   existing family, inherit from `PerLayer5DSpec` /
   `CrossLayer6DSpec` / `SGLangFusedPBSSpec` and just declare the
   axis-index `ClassVar`s; otherwise subclass `KVFormatSpec` directly
   and implement the abstract methods. Set `abstract = False` on the
   leaf class. Registration is automatic via
   `KVFormatSpec.__init_subclass__`; `format_id` defaults to the
   class name (sans trailing `Spec`).
3. If detection of the new format requires new logic for an existing
   engine, edit only that engine's detector file under `detectors/`.
   Adding a brand-new engine = drop a new file under `detectors/`
   with one `EngineDetector` subclass.
4. Add a row in `tests/v1/gpu_connector/test_kv_format_specs.py`
   covering the round-trip (detect + every shape accessor).

No other Python module should need edits. If you find yourself
editing `kv_layer_groups.py`, `gpu_context.py`, or `utils.py` to
support a new layout — the branching belongs in a new spec file.

## Helper surface

All shape access flows through `KVFormatSpec`. The legacy top-level
helpers in `utils.py` are kept as a backward-compatibility facade —
new code should grab a spec once and reuse it.

### Discovery

| Helper | Returns |
|---|---|
| `detect_format(kv_caches, engine, layout_hints)` | `tuple[GPUKVFormat, DiscoverableKVCache]` — runs the engine's `EngineDetector.normalize` then `.detect`, logs the discovered shape and selected format, returns the canonical kv_caches alongside the format. |
| `normalize_kv_and_discover_format(kv_caches, engine, layout_hints)` | Facade alias kept for callers that have not migrated. First runs `attempt_permute_to_contiguous_view` then delegates to `detect_format`. |

### Per-format spec object

```python
spec = get_spec(kv_caches, gpu_kv_format)
spec.num_layers()
spec.num_heads(layer_idx=3)
spec.is_mla        # ClassVar — queryable from spec class too
spec.shape_desc    # ClassVar — symbolic shape skeleton
```

`get_spec_class(gpu_kv_format)` returns the class for static-only
queries (`is_mla`, `is_hnd`, `shape_desc`, `backend_label`,
`is_cross_layer`) without needing a `kv_caches` value.

### Format → engine map

| `GPUKVFormat` | Spec class | Engine | Layout | Structure |
|---|---|---|---|---|
| `NB_NL_TWO_BS_NH_HS` | `VLLMCrossLayer` | vLLM cross-layer | NHD | bare 6-D tensor `[NB, NL, 2, BS, NH, HS]` |
| `NB_NL_TWO_NH_BS_HS` | `TRTLLMCrossLayer` | TRT-LLM cross-layer | HND | bare 6-D tensor `[NB, NL, 2, NH, BS, HS]` |
| `NL_X_TWO_NB_BS_NH_HS` | `VLLMFlashAttnNHD` | vLLM flash-attn | NHD | `NL × [2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | `VLLMFlashInferNHD` | vLLM flash-infer | NHD | `NL × [NB, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | `VLLMFlashAttnHND` | vLLM flash-attn | HND | `NL × [2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | `VLLMFlashInferHND` | vLLM flash-infer | HND | `NL × [NB, 2, NH, BS, HS]` |
| `NL_X_NB_BS_HS` | `VLLMMLA` | vLLM MLA | — | `NL × [NB, BS, HS]` |
| `TWO_X_NL_X_NBBS_NH_HS` | `SGLangMHA` | SGLang MHA | NHD | `[K_list, V_list]`, each `NL × [PBS, NH, HS]` |
| `TWO_X_NL_X_NB_BS_NH_HS` | `SGLangMHAMP` | SGLang MHA via MP daemon | NHD | `[K_list, V_list]`, each `NL × [NB, BS, NH, HS]` |
| `NL_X_NBBS_ONE_HS` | `SGLangMLA` | SGLang MLA | — | `NL × [PBS, 1, HS]` |

The cross-layer formats share a single base pointer, the kernel walks
layers internally via `shape_desc.nl`. Use `spec.is_cross_layer` for
that dispatch and `spec.is_hnd` to detect head-major within-block
layouts.

### Reshape-via-hints (TRT-LLM)

TRT-LLM hands LMCache a 4-D pool tensor
`[NB, NL, 2, num_kv_heads * tokens_per_block * head_dim]` (HND, K and V
interleaved on dim 2). `TRTLLMDetector.normalize` reshapes it to
canonical 6-D form using
`layout_hints["num_kv_heads" | "tokens_per_block" | "head_dim"]`. The
detector also collapses a 1-element list of a 6-D tensor down to the
bare 6-D tensor so detection lands on `list_depth == 0`. Adapters pass
either the 4-D bare tensor or `[4-D]`; the detector handles both.

### vLLM CPU-attention safeguard

`VLLMDetector.detect` forces `kv_layout = "HND"` whenever
`lmcache.torch_device_type == "cpu"`. vLLM's CPU attention backend
stores KV cache in HND but `get_kv_cache_layout` does not return the
correct value for that backend. The proper fix should land on the
vLLM side; this hardcode is a safeguard.

### Scalar accessors (facade)

Top-level helpers in `utils.py` delegate to `get_spec(...)` and remain
as a backward-compatibility surface. The ones that can vary per layer
take an optional `layer_idx: int = 0`.

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
| `spec.data_ptrs(layer_indices)` / `get_group_data_ptrs(kv, fmt, layer_indices)` | `list[int]` | Pointer array in **kernel-expected order**: `[base]` for cross-layer (`layer_indices` ignored), `[K_0…K_N, V_0…V_N]` for SGLang MHA, per-layer flat elsewhere. Matches the dispatch in `csrc/mp_mem_kernels.cu:161-169`. The pointer-array shape is a property of the format — callers never ask "does this format have per-layer pointers?". |
| `make_page_buffer_shape_desc(kv, fmt, layer_idx, num_layers_in_group, num_blocks, block_size, block_stride_elems)` | `PageBufferShapeDesc` | The kernel-facing shape struct. ``block_stride_elems`` carries the per-block dim-0 element stride; pass the value returned by `resolve_block_stride_and_log_layout` so groups with different physical block sizes (e.g. a compressed DeepSeek V4 indexer group alongside dense layers) share a single GPU pool. |

### Contiguity

| Helper | Returns | Notes |
|---|---|---|
| `attempt_permute_to_contiguous_view(kv)` | `DiscoverableKVCache` | Recursive, metadata-only. No-op if already contiguous; raises `ValueError` for non-permutation-recoverable cases (slicing, `as_strided`). **Never copies.** Walks the full structure and permutes every tensor leaf. Called internally by `normalize_kv_and_discover_format`; remains public only for callers that handle a tensor *outside* the discover flow (`GPUConnectorInterface.initialize_kvcaches_ptr`, `CudaIPCWrapper.__init__`). |

## Forbidden outside `kv_format`

- `isinstance(kv_cache, (tuple, list))` to distinguish layouts.
- Indexing raw shapes (`tensor.shape[3]`, `len(shape) == 5`) to derive
  dimensions.
- Hand-rolled list-depth probing (`while isinstance(x, list): depth +=
  1; x = x[0]`). `kv_format.list_depth_tensor_dim` exists for the
  detectors and the facade only — downstream code only ever needs the
  resulting `GPUKVFormat`.
- Wrapping a tensor with `[tensor]` to adapt to a helper's list-depth
  expectation — the spec accessors take `layer_idx` directly.
- Hand-rolled pointer assembly (`[t.data_ptr() for t in kv_caches]`) —
  use `spec.data_ptrs(...)` or `get_group_data_ptrs(...)`.
- Hand-rolled device discovery (`kv_caches[0][0].device`) — use
  `get_device`.
- Hand-rolled contiguity fixes (`tensor.contiguous()`, `.clone()`) —
  use `attempt_permute_to_contiguous_view` which refuses to copy.
- "Canonicalize" functions that rewrite `kv_caches` to a uniform shape
  before passing to helpers. The detectors already canonicalize via
  `EngineDetector.normalize`; callers receive the canonical form back
  from `detect_format`.

## Consumers

- **`lmcache/v1/kv_layer_groups.py::KVLayerGroupsManager.__init__`** —
  partitions layers by the 5-tuple `(kv_size, num_heads, head_size,
  block_size, dtype)` using `is_mla`, `get_num_heads`, `get_head_size`,
  `get_block_size`, and `get_dtype` with each layer's index. Including
  `block_size` in the identity lets compressed groups (e.g. a DeepSeek
  V4 indexer with a smaller physical slot count) sit alongside
  non-compressed groups under a single `GPUCacheContext`. Builds a
  `PageBufferShapeDesc` per group via `make_page_buffer_shape_desc`,
  passing the `block_stride_elems` resolved by
  `resolve_block_stride_and_log_layout`. The real constructor is the
  only way in — no test-only shortcuts, no cached topology fields; the
  manager exposes only `kv_layer_groups`, `num_groups`, and
  `get_shape_desc`.
- **`lmcache/v1/multiprocess/gpu_context.py::GPUCacheContext`** —
  constructs the manager directly at init, delegates
  `get_shape_desc(group_idx)` to it, assembles per-group GPU pointer
  tensors via `get_group_data_ptrs`. No parallel `shape_descs_` /
  `hidden_dim_sizes_` state.
- **`lmcache/v1/gpu_connector/gpu_connectors.py::VLLMPagedMemGPUConnectorV3._initialize_kv_cache_pointers`**
  — for the in-process vLLM path, calls
  `normalize_kv_and_discover_format` (facade over `detect_format`)
  and constructs `metadata.kv_layer_groups_manager` lazily on first
  store/retrieve. The adapter (`vllm_v1_adapter.py`) does not
  participate in format discovery — it only stores `self.kv_caches`
  at register time.

Only `EngineDetector` implementations consume `layout_hints`.
`attempt_permute_to_contiguous_view` (called internally) infers the
permutation from strides and needs no hints.

## Implementation note: mypy and the recursive union

`utils.py` sets `# mypy: disable-error-code="union-attr,call-overload"`
at the file level. This is the **one module** that does format-
dispatched raw indexing on `DiscoverableKVCache` while it still acts
as a facade — the `gpu_kv_format` argument is the proof the indexing
is well-defined, but mypy can't carry that proof through a recursive
Union without per-line casts. Inside `kv_format` itself the spec
classes use `cast(...)` helpers (`_as_tensor`, `_as_layer_list`,
`_as_kv_layer_list`) so type checking remains fully live there.
