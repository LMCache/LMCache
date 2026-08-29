// SPDX-License-Identifier: Apache-2.0
//
// Entry point for the layer-wise multi-layer block KV transfer kernels.
// Defined in mp_mem_kernels_layerwise.cu, which is the only translation unit
// that instantiates the layer-wise kernels.

#pragma once

#include "mp_mem_kernels.cuh"

/**
 * Layer-wise variant of multi_layer_block_kv_transfer.
 *
 * Reads the LMCache object pointers from a device-side array instead of an
 * in-register MemoryObj4, so a single launch can cover an arbitrary number
 * of objects (one per layer batch) rather than at most four.
 *
 * @param paged_buffer_ptrs_tensor GPU tensor of engine paged-buffer pointers
 * @param lmcache_objects_ptrs Host array of LMCache object pointers
 * @param num_objects Number of entries in lmcache_objects_ptrs
 * @param block_ids GPU int64 tensor of engine block ids
 * @param device Device the transfer runs on
 * @param direction H2D or D2H
 * @param shape_desc Paged-buffer shape descriptor
 * @param lmcache_chunk_size Tokens per LMCache chunk
 * @param engine_kv_format Engine KV memory format
 * @param skip_prefix_n_blocks Leading blocks to leave untouched
 */
void multi_layer_block_kv_transfer_layerwise(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const int64_t* lmcache_objects_ptrs, int num_objects,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks);
