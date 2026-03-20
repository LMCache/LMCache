# LMCache Block Transfer Kernel Test Harness

Test harness for developing and benchmarking the `multi_layer_block_kv_transfer` CUDA kernel, which copies KV cache data between vLLM/SGLang paged buffers (GPU) and LMCache contiguous memory objects at block granularity.

## Quick Start

### Run with Python reference (no CUDA build needed)

```bash
# All tests (correctness + benchmark)
python -m lmcache.tools.kernel_harness --use-reference

# Correctness only
python -m lmcache.tools.kernel_harness --mode correctness --use-reference
```

### Build and run with CUDA kernel

```bash
# Build the kernel extension
cd lmcache/tools/kernel_harness
python setup.py build_ext --inplace
cd ../../..

# Run all tests
python -m lmcache.tools.kernel_harness

# Correctness only
python -m lmcache.tools.kernel_harness --mode correctness

# Benchmark only
python -m lmcache.tools.kernel_harness --mode benchmark --num-bench-iters 100
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `all` | What to run: `correctness`, `benchmark`, or `all` |
| `--use-reference` | `false` | Use Python reference instead of CUDA kernel |
| `--format` | `all` | vLLM format: `normal`, `cross_layer`, `mla`, `flash_infer`, `sglang_mha`, `sglang_mla`, or `all` |
| `--dtype` | `all` | Data type: `bf16`, `fp8`, or `all` |
| `--mem-device` | `gpu` | Device for LMCache memory objects: `gpu` or `cpu` |
| `--num-bench-iters` | `100` | Number of benchmark iterations |
| `--num-warmup-iters` | `10` | Number of warmup iterations |
| `--skip-prefix-n-blocks` | `0` | Number of prefix blocks to skip |

## Test Configurations

The harness tests 12 combinations (6 formats x 2 dtypes):

| Case | Layers | NH | HS | vLLM Tensor Shape | GPUKVFormat |
|------|--------|----|----|-------------------|-------------|
| Normal | 64 | 8 | 128 | L x [2, NB, BS, NH, HS] | `NL_X_TWO_NB_BS_NH_HS` |
| Cross-layer | 64 | 8 | 128 | [NB, NL, 2, BS, NH, HS] | `NB_NL_TWO_BS_NH_HS` |
| Flash Infer | 64 | 8 | 128 | L x [NB, 2, BS, NH, HS] | `NL_X_NB_TWO_BS_NH_HS` |
| MLA | 104 | 1 | 576 | L x [NB, BS, HS] | `NL_X_NB_BS_HS` |
| SGLang MHA | 64 | 8 | 128 | 2L x [NBBS, NH, HS] | `TWO_X_NL_X_NBBS_NH_HS` |
| SGLang MLA | 104 | 1 | 576 | L x [NBBS, 1, HS] | `NL_X_NBBS_ONE_HS` |

All cases use: NB=1000, BS=16, 4 memory objects with 256 tokens each (64 total blocks).

## Kernel Function Signature

```cpp
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,  // GPU int64 tensor of data pointers
    std::vector<int64_t> lmcache_objects_ptrs,       // raw pointers to memory objects
    const torch::Tensor& block_ids,                  // block indices into paged buffer
    const torch::Device& device,
    TransferDirection direction,
    PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size,                          // tokens per memory object
    GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks);
```

## Correctness Test

The test performs a D2H -> H2D roundtrip with **different** block IDs:

1. Fill source vLLM tensors with random data
2. D2H: copy blocks `B_src` from source -> memory objects
3. H2D: copy memory objects -> target vLLM at blocks `B_dst` (disjoint from `B_src`)
4. Verify: `target[B_dst[i]] == source[B_src[i]]` for all i, all layers
5. Verify: untouched blocks in target remain zero

A skip-prefix test additionally verifies `skip_prefix_n_blocks` behavior.

## Examples

```bash
# Test only the normal format with bf16
python -m lmcache.tools.kernel_harness --mode correctness --format normal --dtype bf16

# Benchmark with memory objects on CPU (pinned)
python -m lmcache.tools.kernel_harness --mode benchmark --mem-device cpu

# Test all formats with fp8
python -m lmcache.tools.kernel_harness --mode correctness --dtype fp8
```
