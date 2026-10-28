# KV Cache Group Edits

## Summary

`lmcache/integration/vllm/kv_cache_group_edits.py` is the single place where
vLLM KV cache groups are re-presented ("edited") before LMCache registration.
The connector calls `apply_kv_cache_group_edits(kv_cache_config, kv_caches)`
once in `register_kv_caches`, and the edited dict feeds both engine-group-info
creation (`kv_cache_groups.py`) and transfer registration.

LMCache derives each group's transfer metadata (block size, page layout,
dtype) from the registered tensors, and interprets store/retrieve block IDs in
vLLM's scheduler-side block-id space (`kv_cache_spec.block_size` units). The
edits exist to restore one invariant the raw tensors can violate:

> **The registered tensor's paging granularity must equal the block-id
> granularity.**

## Structure

Each case is one `KVCacheGroupEdit` rule in the module's `_EDITS` registry:
`matches(spec, kv_cache)` decides **structurally** — from the vLLM spec kind
(`get_kv_cache_spec_kind`, which also unwraps `UniformTypeKVCacheSpecs`) and
the registered tensor, never from model name or architecture — and
`apply(spec, kv_cache)` produces a view over the same storage. First matching
rule wins; unmatched layers pass through. Covering a new group kind means
adding one rule.

Model name/arch is deliberately not an input: the same architecture yields
different group structures depending on runtime decisions
(`mamba_cache_mode`, attention backend's kernel block size, TP), all of which
the config + tensors already resolve.

## Edits

Both rules apply only to Mamba-hybrid models (the registry is only consulted
when `kv_cache_config.has_mamba_layers`).

### 1. Mamba state pages

A Mamba / linear-attention layer (e.g. Qwen3.5 GDN) registers `[conv_state,
ssm_state]` — two tensors with different shapes and dtypes, laid out
contiguously in one padded page (`conv | ssm | pad`). The raw pair trips
format discovery (the SSM view starts mid-page). The edit reinterprets each
page as one bf16 tensor shaped `(num_blocks, 2, block_size, 1, head_size)`
over the same storage, where `head_size` is derived so the bytes fill the page
exactly. `matches` requires the conv state's per-block stride to equal
`page_size_bytes` (one scheduler block == one full padded page in memory).

On some backends (e.g. vLLM-Ascend ≤ v0.29) conv and SSM are still one
logical page per block, but they register as **two separate tensors** carved
from a shared pool. Each tensor's per-block stride is only **that slice's**
byte length (conv-only or ssm-only), not `page_size_bytes`, so the packed
page reinterpretation above does not apply.

A second rule then registers **one uint8 plane per slice** (conv and ssm
stay two LMCache tensors). For each slice it keeps the same storage and only
changes metadata: dtype `uint8` and shape
``[num_blocks, block_size, 1, width]``, where ``block_size`` is the
scheduler's ``kv_cache_spec.block_size`` and ``width`` is how many bytes sit
in each "row" of that synthetic layout (`bytes_per_block / block_size`).
That shape is what LMCache's format detector recognizes as format 17
(opaque sub-planes). The edit does not copy or pad bytes; the transfer kernel
checks alignment, row size, and plane count when store/load runs.

### 2. Sub-paged full attention

vLLM unifies page sizes across hybrid groups by inflating the attention
*logical* block size (`vllm/platforms/interface.py:_align_hybrid_block_size`;
Qwen3.5-0.8B: 544), while the attention backend re-pages the physical tensor
at its own *kernel* block size (`vllm/v1/worker/utils.py:
prepare_kernel_block_sizes`). Logical block `n` then occupies the
`k = logical/kernel` contiguous kernel pages `n*k .. n*k+k-1` (vLLM expands
the worker-side block table the same way, `BlockTable.map_to_kernel_blocks`);
the scheduler-side block IDs LMCache receives stay logical.

Whether `k > 1` is a property of the backend, not the model:
`select_common_block_size` returns the largest *advertised* kernel size
dividing the logical one, so backends advertising fixed ints sub-page
(FlashInfer `[16, 32, 64]`, ROCm AITER FA `[16, 32]`), while `MultipleOf(16)`
(FlashAttention) pages at the logical size and never needs the edit.

Registering the raw kernel-paged tensor makes LMCache discover
`block_size == kernel < logical`, and `_derive_compression_metadata`
(`lmcache/v1/kv_layer_groups.py`) misclassifies the group as compressed:
only `1/k` of each chunk's KV is transferred, addressed against the kernel
page space. The edit re-views the tensor as
`(num_kernel_pages / k, 2, logical_block_size, 1, head_size)` — a pure
`view()`, valid because `k` kernel pages tile each logical page's bytes
exactly (enforced; see Invariants).

Two registered layouts must be recognized, distinguished only by rank (the
block dim is index 2 in both):

| vLLM | shape | rank |
|---|---|---|
| `<= 0.25.x` | `(num_blocks, 2, block_size, num_heads, head_size)` | 5 |
| `>= 0.26.0` | `(num_blocks, num_heads, block_size, 2 * head_size)` | 4 |

0.26.0 packs K and V into the content dim across the mainline non-MLA backends
(`flash_attn.py`, `flashinfer.py`, `triton_attn.py`, `flex_attention.py`,
`rocm_aiter_fa.py`) — but not all: `hpc_attn.py` is unchanged and still rank 5,
so both ranks stay reachable and both must be handled. Gating the rule on rank
5 alone silently disabled it on 0.26.0 and corrupted store/retrieve with no
error. The re-view only reinterprets byte ranges, so it is otherwise
indifferent to which layout it is handed.

The registered tensor is also a `permute` view of the backend's physical
layout (`get_kv_cache_stride_order`, `vllm/v1/worker/gpu/attn_utils.py`), so it
is generally *not* contiguous in logical dim order — under NHD the 0.26.0
rank-4 stride order `(0, 2, 1, 3)` is a transpose. Pages only tile by byte
range in memory order, so the **rank-4** branch of `_SubpagedAttentionViewEdit`
re-views in descending-stride order. That path is mainline vLLM only; the
vLLM-Ascend `(K, V)` plane edits below do not use it.

**Rank 5 deliberately keeps the stricter logical-order contiguity guard.** The
edited view is NHD-shaped by construction, so under an HND hint the detector
selects `NL_X_NB_TWO_NH_BS_HS`, whose `block_size()` reads the synthetic
`num_heads` axis and resolves 1 — extending the rank-4 memory-order
normalization to rank 5 would replace a loud startup failure with exactly that
silent wrong block size. This is a **general limitation** of the edits, not a
rank-5 one: under an explicit `VLLM_KV_CACHE_LAYOUT=HND` every edited group,
Mamba included, resolves `block_size` to 1. Not reached by default
(`get_required_kvcache_layout` returns `None`, so vLLM falls back to NHD);
tracked separately.

vLLM-Ascend registers the same scheduler-versus-kernel mismatch as a `(K, V)`
pair of rank-4 planes at the kernel block size. Both releases use the same
registered shape; they differ in how K and V bytes are laid out in memory:

| vLLM-Ascend | shape (each of `K`, `V`) | layout |
|---|---|---|
| `<= v0.29` | `(num_kernel_pages, kernel_block_size, num_kv_heads, head_size)` | rank 4; dense K kernel pages, then dense V (one pool or two allocations) |
| `v0.30` | `(num_kernel_pages, kernel_block_size, num_kv_heads, head_size)` | rank 4; `[K0][V0][K1][V1]…` in one pool (`stride(0)` spans one K row plus the following V row) |

On v0.30 the input planes may carry one or more TP-local KV heads on axis 2;
all of those heads are folded into each kernel row (`row_elems =
kernel_block_size * num_kv_heads * head_size`). The interleaved matcher
requires a dense trailing NHD slice on axes 1..3, K/V in the same underlying
storage with V one row after K, and dim-0 stride `2 * row_bytes`. Whole-tensor
`is_contiguous()` is intentionally not required because dim 0 skips the
interleaved partner row.

Through v0.29, `_SubpagedAttentionContiguousPlaneViewEdit` re-views each plane
to `(num_logical_blocks, scheduler_block_size, num_kv_heads, head_size)`
(format 16, `NL_X_TWO_X_NB_BS_NH_HS`). The tile check uses
`real_page_size_bytes`, because hybrid `page_size_bytes` includes Mamba conv
padding absent from the K/V views. v0.30's
`_SubpagedAttentionInterleavedPlaneViewEdit` re-views the contiguous K+V span
from K, in the original dtype, as
`(num_logical_blocks, 2, scheduler_block_size, 1, head_size)`
(format 2, `NL_X_NB_TWO_BS_NH_HS`). That is the same opaque rank-5 page as
`_SubpagedAttentionViewEdit`. Axis 1 is the first and second half of each
logical page, not semantic K and V; physical heads are folded into
`head_size`. The fused rank-4/5 rule is registered before the plane rules; among
those, the interleaved rule is registered first. A pair that matches neither
layout is left unchanged. Matching uses strides and contiguity, not the
vLLM-Ascend version string — the table rows are the layouts those releases
emit today.

### 3. Sub-paged MLA

The same re-paging for MLA-style caches: rank-3 `[NB, states, C]` (Kimi K3)
or rank-4 with the single head slot on either side of the states,
`[NB, 1, states, C]` (head-major layouts) or `[NB, states, 1, C]`
(states-major). The order is decided by the attention backends, not the
model, so the rule finds the states dim from the tensor and keeps the dim
order in the view. The target is `spec.num_states`
(`block_size / tokens_per_state`; `block_size / compress_ratio` on older
vLLM) rather than `block_size`, so a declared slot compression survives the
view and the compression path still derives it from
`tokens_per_block / slots_per_block`. Each layer is matched against its own
leaf spec: GLM-5.3-Flash's MLA and indexer layers share one
`UniformTypeKVCacheSpecs` group with different `num_states` (sparse MLA at 64
rows and the kpool indexer at 32 under a 1152-token block).

## Startup validation

`validate_kv_cache_groups` (called at connector init and again at
registration) rejects group specs the transfer path cannot serve correctly,
with one aggregated error: `CrossAttentionSpec`, and Mamba with
`mamba_cache_mode != "align"` (no reusable snapshots). Declared slot
compression (`compress_ratio > 1` / `tq_slot_size > 0`, e.g. DeepSeek-V4) is
*not* rejected — those groups are served by the compression path in
`lmcache/v1/kv_layer_groups.py` and only skipped by the edits here. Note the
compression path still derives per-group ratios from the unified vLLM block
size; switching it to per-group block sizes is pending in a separate PR.

Reference: vLLM PR #42828 (Mooncake store HMA support) uses the same
validate-and-reject-up-front pattern, and is the reference design for the
deferred follow-ups (per-group store/load masks; manager-mirroring hit
computation). Caveat for the latter: LMCache's lookup doubles as prefetch, so
vLLM's lookup-first-then-trim flow does not map directly. See the module
docstring for the full check-when list.

## Non-edit: declared compression

Groups whose spec *declares* slot compression — `tokens_per_state > 1`
(DeepSeek-V4 slot packing; `compress_ratio` on older vLLM) or
`TQFullAttentionSpec.tq_slot_size > 0` — genuinely store fewer physical
slots than logical tokens. They must reach the compression path in
`lmcache/v1/kv_layer_groups.py` with their slot count intact; only the MLA
rule (3) views them, and it preserves that count. (DeepSeek-V3.2's `fp8_ds_mla` cache
packs *bytes per slot*, not slots per block: its specs keep
`block_size == scheduler block size` and `compress_ratio == 1`, so it never
needs an edit either.)

The sub-paged rule's `matches` excludes declared-compression specs by their
own fields (`compress_ratio` / `tq_slot_size`), and its `apply` additionally
verifies by byte accounting that `k` kernel pages tile the logical page's
bytes exactly — any *undeclared* packed layout fails with a loud `ValueError`
rather than being transferred wrongly.

## The opaque-page contract

An edited view's dims are addressing metadata only (block id → byte range).
The named dims are **not** semantic: a Mamba view's "K plane" is conv/ssm
bytes, and a sub-paged attention view's "K plane" interleaves true K and V at
kernel-page granularity (true K is not contiguous across kernel pages, so no
logical-block view can have a pure-K plane). The synthetic head shape
`(1, page_bytes / (2 * block_size * elem))` signals this deliberately.

vLLM-Ascend adds two sub-paged attention edits on top of that contract. A
v0.30 interleaved `(K, V)` pair is re-paged as the same original-dtype rank-5
opaque page as the fused edit (format 2, `NL_X_NB_TWO_BS_NH_HS`). Its axis 1
mixes true K and V at kernel-page granularity: each synthetic plane is one
half of the logical page, not a pure K or V plane. The v0.29 contiguous
reshape keeps K, V, dtype, and heads, and only changes the block axis
(format 16). The synthetic output is NHD-shaped, like
`_SubpagedAttentionViewEdit`, and is not an HND layout.

Byte transport round-trips correctly because store and retrieve share the same
bijective block-id → bytes mapping. Consequences:

- **Valid**: store/retrieve through the MP transfer path on the same engine
  configuration.
- **Not valid** for edited groups: content-aware processing (serde
  compression, blending, head resharding, layout conversion), and sharing
  cache entries across engines whose attention backends choose different
  kernel block sizes (the byte order inside a logical page is
  backend-dependent).

## Invariants

- Edits are pure tensor views over the registered storage — never copies.
- A sub-paged view is only produced when `kernel_page_bytes * k ==
  spec.page_size_bytes`; any mismatch raises `ValueError` (fail loudly rather
  than silently transfer a compressed layout).
- On vLLM-Ascend, a sub-paged view is also produced when `k` kernel pages tile
  the logical page: one fused or interleaved span of `spec.page_size_bytes`,
  or the two contiguous halves added together equaling `real_page_size_bytes`
  (falling back to `page_size_bytes` when the spec has no unpadded size). A
  matched layout whose bytes do not tile raises `ValueError` rather than
  transferring a compressed layout. A pair that is not one of the recognized
  layouts is left unchanged.
- After edits, every registered tensor's block dim equals its group's
  `kv_cache_spec.block_size`, so the server derives `compress_ratio == 1` for
  these groups.

## Code map

| Area | File |
|---|---|
| Edits (this doc) | `lmcache/integration/vllm/kv_cache_group_edits.py` |
| Caller | `lmcache/integration/vllm/lmcache_mp_connector.py` (`register_kv_caches`) |
| Compression-ratio derivation (downstream consumer) | `lmcache/v1/kv_layer_groups.py` |
| vLLM block-size inflation | `vllm/platforms/interface.py` (`_align_hybrid_block_size`) |
| vLLM kernel-page split + block-table expansion | `vllm/v1/worker/utils.py`, `vllm/v1/worker/block_table.py` |
| Unit tests | `tests/v1/test_vllm_kv_cache_group_edits.py` |
| End-to-end test | `.buildkite/k3_tests/multiprocess/scripts/run-single-test.sh` (`hma_lm_eval_qwen3_5`) |

The end-to-end `hma_lm_eval_qwen3_5` gsm8k store-vs-retrieve check pins the
observable contract — faithful retrieve — but needs a GPU, so it cannot gate
*which rules fire* for a given vLLM version; the 0.26.0 regression was exactly
that. `tests/v1/test_vllm_kv_cache_group_edits.py` covers the gap on CPU: it
pins the matching gate per registered layout and the block-granularity
invariant, not internal view shapes.
