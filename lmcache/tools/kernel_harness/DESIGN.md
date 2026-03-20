# Design: Block Transfer Kernel Test Harness

## Motivation

The existing `multi_layer_kv_transfer` kernel operates at token granularity via `slot_mapping`. The new `multi_layer_block_kv_transfer` kernel operates at **block granularity** via `block_ids`, which is more natural for the MP (multi-process) mode where entire blocks of KV cache are transferred between GPU paged buffers and LMCache memory objects.

## Architecture

```
__main__.py          CLI entry point, dispatches to correctness/benchmark
    |
    +-- config.py          TestConfig dataclass, 12 predefined test configs (6 formats x 2 dtypes)
    |
    +-- tensor_factory.py  Creates vLLM tensors, LMCache memory objects, block_ids
    |
    +-- reference.py       Pure Python reference implementation (correctness oracle)
    |
    +-- correctness.py     D2H -> H2D roundtrip verification
    |
    +-- benchmark.py       CUDA-event-based timing, throughput reporting
    |
    +-- csrc/              Standalone CUDA extension
        +-- multi_layer_block_kv_transfer.cuh   Header: enums, structs, declarations
        +-- multi_layer_block_kv_transfer.cu    Kernel implementation
        +-- pybind.cpp                          Python bindings
```

## Memory Layouts

### LMCache Memory Objects

Memory objects can reside on GPU (default) or pinned CPU memory (via `--mem-device cpu`).

- Non-MLA: `[2, num_layers, tokens_per_object, num_heads * head_size]`
  - Dimension 0: K/V (0=key, 1=value)
- MLA: `[1, num_layers, tokens_per_object, head_size]`
  - No K/V split (compressed latent representation)

### vLLM/SGLang Paged Buffers

Six GPU KV formats are supported:

**NORMAL (`NL_X_TWO_NB_BS_NH_HS`)**: L separate tensors
- Each: `[2, num_blocks, block_size, num_heads, head_size]`
- Dimension 0: K/V

**CROSS_LAYER (`NB_NL_TWO_BS_NH_HS`)**: Single tensor
- Shape: `[num_blocks, num_layers, 2, block_size, num_heads, head_size]`
- All layers packed into one allocation

**FLASH_INFER (`NL_X_NB_TWO_BS_NH_HS`)**: L separate tensors
- Each: `[num_blocks, 2, block_size, num_heads, head_size]`
- K/V interleaved per block (dim 1)

**MLA (`NL_X_NB_BS_HS`)**: L separate tensors
- Each: `[num_blocks, block_size, head_size]`
- No K/V split, single head

**SGLANG_MHA (`TWO_X_NL_X_NBBS_NH_HS`)**: 2L separate tensors
- First L tensors = K per layer, next L tensors = V per layer
- Each: `[NB*BS, num_heads, head_size]` (flat token indexing)

**SGLANG_MLA (`NL_X_NBBS_ONE_HS`)**: L separate tensors
- Each: `[NB*BS, 1, head_size]` (flat token indexing)
- No K/V split

## Kernel Interface

The kernel receives data pointers rather than tensor objects:

- **`paged_buffer_ptrs_tensor`**: GPU int64 tensor of raw data pointers, one per vLLM tensor. Cross-layer has 1 pointer; SGLang MHA has 2L pointers; all others have L pointers.
- **`lmcache_objects_ptrs`**: List of raw int64 data pointers to LMCache memory objects (1-4 objects).
- **`PageBufferShapeDesc`**: Struct with `kv_size, nl, nb, bs, nh, hs, element_size`.

### Kernel Dispatch

- Grid: `dim3(kv_size, num_blocks_per_object, num_layers)`
- Block: `dim3(min(scalars_per_head, 32), num_heads)`
- Each thread block handles one (block, layer, k_or_v) region
- Each `threadIdx.y` handles one head
- Inner loop over tokens within the block

### Offset Calculations

`calculate_engine_global_offset` computes the byte offset to the start of a block within the paged buffer tensor, accounting for format-specific dimension ordering. `calculate_engine_local_offset` computes the per-token, per-head offset within the block.

For SGLang MHA, the K/V distinction is handled by **pointer selection** (`paged_buffer_ptrs[k_or_v * nl + layer_idx]`) rather than offset arithmetic, since K and V reside in separate tensors.

### PTX Streaming Copy

The `warp_copy` function uses PTX `ld.global.cs` / `st.global.cs` instructions for `uint4` loads and stores, bypassing the L2 cache to avoid cache pollution during bulk transfers.

## Block-Level Mapping

Given `block_ids = [b0, b1, ..., b63]` and 4 memory objects with 256 tokens each (16 blocks per object):

```
memory_objects[0] <-> block_ids[0:16]   (tokens 0-255)
memory_objects[1] <-> block_ids[16:32]  (tokens 256-511)
memory_objects[2] <-> block_ids[32:48]  (tokens 512-767)
memory_objects[3] <-> block_ids[48:64]  (tokens 768-1023)
```

For block `block_ids[j]` mapping to memory object `i` at local block index `k`:
- `i = j // blocks_per_object`
- `k = j % blocks_per_object`
- Token range in memory object: `[k * block_size, (k+1) * block_size)`

## Skip Prefix

The `skip_prefix_n_blocks` parameter skips the first N blocks **globally** (not per-object). The kernel computes `global_block_idx = start_block_idx + block_idx_in_batch` and returns early if it is below the skip threshold.

## Correctness Strategy

Using **different** block IDs for D2H and H2D proves the data actually flows through the memory objects:

1. D2H reads from source blocks `B_src = {5, 42, 100, ...}` into memory objects
2. H2D writes from memory objects to target blocks `B_dst = {7, 88, 200, ...}`
3. If `target[B_dst[i]] == source[B_src[i]]`, the transfer is correct

This catches bugs where the kernel might accidentally do a direct GPU-GPU copy instead of going through the memory objects.

## Standalone Build

The `csrc/` directory has its own `setup.py` so the kernel can be developed independently of the main LMCache build. This allows rapid iteration on the kernel code without rebuilding the entire project.
