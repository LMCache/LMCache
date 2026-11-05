# Spec-driven KV slicing: CPU contract experiment

This example evaluates the direction proposed in
[vLLM RFC #42082](https://github.com/vllm-project/vllm/issues/42082): the producer's
cache spec returns matching tensor views, and the consumer copies those views
without detecting K/V axis semantics. It uses LMCache's real `TensorMemoryObj`
and `MemoryFormat.KV_2LTD`; it does not modify a production connector.

At vLLM main `ee9ba37199338cf31306a2ec27ec9afa41efc09e`,
`vllm/v1/kv_cache_interface.py` has no `slice_for_tp_transfer` method. The
`SliceSpec`, `PackedHeadSpec`, and `PackedContentSpec` classes here are executable
**reference models of the proposed contract**, not imported vLLM implementations
or proof of upstream integration. This experiment is independent of
[vLLM #63699](https://github.com/vllm-project/vllm/pull/63699) and
[LMCache #6072](https://github.com/LMCache/LMCache/pull/6072).

## Question and result

Can spec-owned slicing handle both TP head partitioning and same-TP token-range
store/load without teaching the consumer a backend's K/V arrangement? For the
dense, evenly sharded cases modeled here, yes. This does **not** settle how every
native allocation becomes an accepted registration representation.

```text
producer-owned cache + producer spec
           |
           | slice_for_tp_transfer(cache, TP/rank, peer spec, peer TP/rank)
           v
ordered component views [L, B, H, N, D] -- no allocation or concatenation
           |
           +--> pair matching source/destination views --> TP transfer copy
           |
           +--> logical token range + block table --> store_views
                                                        |
                                            real LMCache TensorMemoryObj
                                            [2, L, tokens, H * D]
                                                        |
                           destination block table --> load_views
                                                        |
                                              producer-owned cache
```

Here, L is layers, B is physical blocks, H is real local KV heads, N is states
per block, and D is head dimension. The example assumes one state per token.
The two returned components have matching K-then-V order; the consumer relies
on that order and common component geometry, not arbitrary opaque content.
This remains a dense GQA experiment, not a generic Mamba/MLA/quantization codec.

## Two producer specs

- `PackedHeadSpec`: `[L, B, 2H, N, D]`, with slots `[K heads | V heads]`.
  `num_kv_heads` records real heads separately from the 2H storage extent.
  For H=4, TP=1 -> TP=2 rank 0 receives slots `[0:2]` and `[4:6]`, not `[0:4]`.
- `PackedContentSpec`: `[L, B, H, N, 2D]`, with contiguous `[K | V]` content.
  It returns corresponding K/V views even though their storage strides differ.

Neither class concatenates component slices. Disjoint rank pairs produce empty
views. The protocol deliberately omits upstream spec inheritance, registration,
serialization, compressed states, and head replication. Packing into 2H heads
still needs upstream agreement on storage slots versus actual head semantics.

## RBLN registration boundary

The [RBLN producer](https://github.com/RBLN-SW/vllm-rbln/blob/3cfd759ec1f037b9553bdbb2bcec2503af757d5d/vllm_rbln/v1/attention/backends/flash_attention.py#L106)
uses `[B, 2, H, 1, N, D]` or `[2, B, H, 1, N, D]`.

| Native representation | Direct registration as `[B, 2H, N, D]` |
| --- | --- |
| Blocks first | `squeeze` + `view` shares storage |
| K/V first, non-singleton B/H/N/D | `permute` + `view` fails; `reshape` copies |

Tests verify both values and storage sharing at B=2 and B=3. This is a finding
about those concrete transformations, not a proof that every zero-copy adapter
is impossible. Separate K/V projections of K/V-first storage are zero-copy
`[1, B, H, N, D]` views and can roundtrip through the reference store/load code.
**That component test starts after projection; it does not demonstrate that
upstream registration accepts separate components for one attention layer.**
Likewise, the content-packed test fixture explicitly constructs a separate
allocation; it is not a zero-copy conversion from native RBLN storage.

## Tests and why they matter

Run `tests/v1/gpu_connector/test_spec_slicing_poc.py`:

| Test | Cases | Evidence |
| --- | ---: | --- |
| `test_spec_views_share_storage_and_preserve_components` | 2 | Both specs preserve exact component values and share registered storage |
| `test_tp_transfer_pairs_spec_views` | 16 | Four encoding pairs and TP 1->1, 1->2, 2->1, 2->4; exact values, including disjoint ranks |
| `test_naive_head_split_loses_matching_values` | 1 | Reproduces the K-only/V-only contiguous-split error |
| `test_token_range_roundtrip_through_lmcache_memory` | 32 | Four encoding pairs, four token ranges, int64/float16, two layers, remapped pages, untouched-token sentinels |
| `test_rbln_native_registration_copy_boundary` | 2 | Blocks-first aliases; K/V-first direct head packing copies; checks resulting values |
| `test_rbln_kv_first_component_views_are_zero_copy` | 1 | Separate K/V projections alias original storage |
| `test_rbln_kv_first_components_roundtrip_without_registration_repack` | 1 | Projected native components support partial cross-block store/load |

Unique values encode layer/block/component/head/state/content identity; shape
checks alone would miss the bug. Storage assertions apply to registration and
slice creation only. Offloading and loading intentionally copy data.

## Reproduce

Prerequisites: Python 3.12, Git, and a C++17 compiler (Apple Command Line Tools
on macOS). No CUDA, vLLM, RBLN SDK, or accelerator is needed. Start at the exact
LMCache commit recorded in the PR description, then from the repository root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch pytest packaging wheel 'setuptools>=77,<81' setuptools_scm
NO_GPU_EXT=1 python -m pip install -e . --no-build-isolation
python -m pytest --noconftest tests/v1/gpu_connector/test_spec_slicing_poc.py -q
```

The editable install builds the common native extensions imported by
`memory_management`; a source-only `NO_NATIVE_EXT=1` install is insufficient.
`--noconftest` isolates this CPU contract experiment from accelerator allocator
fixtures in the root conftest. It is not a substitute for normal accelerator CI.
If setuptools-scm rejects a local tag, use the development-only version override
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0` on the install command.

## Remaining integration work

1. Agree on the producer-owned spec representation and real-head/storage-slot
   distinction, including the K/V-first registration route.
2. Implement and expose the upstream slicing interface; this example does not
   add a vLLM registration API or transport specs across process boundaries.
3. Adapt production LMCache transfers to consume views efficiently. The CPU
   per-token reference is intentionally not a performance implementation.
4. Cover actual RBLN/MUSA hardware, async lifetime, stream synchronization,
   head replication, heterogeneous block sizes, hybrid models, and state codecs.

These tests do not justify declaring #63699 unnecessary for every backend, or
claiming production LMCache no longer needs format detection.
