#pragma once
#include <torch/torch.h>
#include <torch/extension.h>
#include "managed_mem.h"
#include "kernels/types.h"

namespace lmc {
void multi_layer_kv_transfer_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType, uint32_t blockDim, 
                                    void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, 
                                    uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs, 
                                    const int32_t numLayers, const int64_t pageBuffSize, const int32_t numTokensChunk, 
                                    const bool page2L);

void single_layer_kv_transfer_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType, 
                                     uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, 
                                     uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                                     uint8_t *slotmappings, const int64_t hiddenDims, const int32_t numTokens, 
                                     const bool page2L, const bool tokenMajor, const bool isMLA);

void load_and_reshape_flash_kernel(vllm_ascend::AscendType type, vllm_ascend::AscendType slotType, 
                                  uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, uint8_t *keyCachePtr, 
                                  uint8_t *valueCachePtr, uint8_t *slotmappings, const int64_t hiddenDims, 
                                  const int64_t numPages, const int32_t pagedSize, const int32_t numTokens, 
                                  const int32_t numLayers, const int32_t layerIdx, const bool page2L);
}


void multi_layer_kv_transfer(torch::Tensor& key_value, // [kv, num_layer, num_tokens, hidden]
                             const torch::Tensor& key_value_ptrs, // [num_layers]
                             const torch::Tensor& slot_mapping, // [num_tokens]
                             const torch::Device& paged_memory_device,
                             const int page_buffer_size, const bool direction,
                             const bool use_mla);

void multi_layer_kv_transfer_unilateral(torch::Tensor& key_value,
                                        const torch::Tensor& key_ptrs,
                                        const torch::Tensor& value_ptrs,
                                        const torch::Tensor& slot_mapping,
                                        const torch::Device& paged_memory_device,
                                        const int page_buffer_size,
                                        const bool direction);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_cache,
                              torch::Tensor& vllm_value_cache,
                              torch::Tensor& slot_mapping,
                              const bool direction,
                              const bool token_major = false);

void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);