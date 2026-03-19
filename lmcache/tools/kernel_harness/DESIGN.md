# Design: Block Transfer Kernel Test Harness

## Motivation

The existing `multi_layer_kv_transfer` kernel operates at token granularity via `slot_mapping`. The new `multi_layer_block_kv_transfer` kernel operates at **block granularity** via `block_ids`, which is more natural for the MP (multi-process) mode where entire blocks of KV cache are transferred between GPU and CPU.

## Architecture

```
__main__.py          CLI entry point, dispatches to correctness/benchmark
    |
    +-- config.py          TestConfig dataclass, 6 predefined test configs
    |
    +-- tensor_factory.py  Creates vLLM tensors, LMCache memory objects, block_ids
    |
    +-- reference.py       Pure Python reference implementation (correctness oracle)
    |
    +-- correctness.py     D2H -> H2D roundtrip verification
    |
    +-- benchmark.py       CUDA-event-based timing, throughput reporting
    |
    +-- csrc/              Standalone CUDA extension (stub kernel + pybind)
```

## Memory Layouts

### LMCache Memory Objects

- Non-MLA: `[2, num_layers, tokens_per_object, num_heads * head_size]`
  - Dimension 0: K/V (0=key, 1=value)
- MLA: `[1, num_layers, tokens_per_object, head_size]`
  - No K/V split (compressed latent representation)

### vLLM Paged Buffers

**NORMAL (`NL_X_TWO_NB_BS_NH_HS`)**: L separate tensors
- Each: `[2, num_blocks, block_size, num_heads, head_size]`
- Dimension 0: K/V

**CROSS_LAYER (`NB_NL_TWO_BS_NH_HS`)**: Single tensor
- Shape: `[num_blocks, num_layers, 2, block_size, num_heads, head_size]`
- All layers packed into one allocation

**MLA (`NL_X_NB_BS_HS`)**: L separate tensors
- Each: `[num_blocks, block_size, head_size]`
- No K/V split, single head

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

## Correctness Strategy

Using **different** block IDs for D2H and H2D proves the data actually flows through the memory objects:

1. D2H reads from source blocks `B_src = {5, 42, 100, ...}` into memory objects
2. H2D writes from memory objects to target blocks `B_dst = {7, 88, 200, ...}`
3. If `target[B_dst[i]] == source[B_src[i]]`, the transfer is correct

This catches bugs where the kernel might accidentally do a direct GPU-GPU copy instead of going through CPU memory.

## Standalone Build

The `csrc/` directory has its own `setup.py` so the kernel can be developed independently of the main LMCache build. This allows rapid iteration on the kernel code without rebuilding the entire project.
