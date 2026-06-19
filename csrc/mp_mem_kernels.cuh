// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mem_kernels.cuh"  // TransferDirection, EngineKVFormat

#include <c10/cuda/CUDAGuard.h>
#include <vector>

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)
  // Physical per-block stride in source-dtype element units, used by
  // formats whose dim-0 is the block axis to step over padding bytes
  // (e.g. DeepSeek V4 compressor / indexer caches sharing a vLLM KV
  // pool with larger attn groups, whose rows are padded up to the
  // pool's max row width). 0 means "unset — fall back to the
  // format-specific tight stride".
  //
  // CONTRACT: pass ``tensor.stride(0)`` verbatim. PyTorch stride
  // semantics already absorb every inner-dim extent (including
  // ``kv_size``), so DO NOT pre-multiply by any inner dim.
  //
  // Honoured today only by NL_X_NB_BS_HS (per-layer [NB, BS, HS],
  // MLA). NL_X_NB_TWO_BS_NH_HS is restricted to the tight form
  // upstream and leaves this field at 0; all other formats either
  // pack non-block info into dim-0 or do not support dim-0 padding,
  // and ignore this field.
  int block_stride_elems;

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  // Per (K or V) block step along dim-0, expressed in ``ScalarType``
  // element units (the kernel's working dtype, e.g. uint4 / uint32_t /
  // uint16_t). Returns the tight ``bs * nh * hs`` by default, or the
  // physical ``block_stride_elems`` when dim-0 carries padding (today
  // only NL_X_NB_BS_HS, see ``block_stride_elems`` above). Every
  // ``calculate_engine_global_offset`` branch uses this as the dim-0
  // step, so honouring padding here propagates to all formats without
  // per-branch changes.
  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0 - 4
};

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects.
 *
 * @param paged_buffer_ptrs_tensor  GPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor)
 * @param lmcache_objects_ptrs      Raw pointers to LMCache memory objects
 * @param block_ids                 GPU int64 tensor of block indices in vLLM
 *                                  paged buffer
 * @param device                    CUDA device of vLLM tensors
 * @param direction                 H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param shape_desc                Shape descriptor for the paged buffer
 * @param lmcache_chunk_size        Tokens per LMCache memory object
 * @param engine_kv_format             EngineKVFormat identifier
 * @param skip_prefix_n_blocks      Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);
