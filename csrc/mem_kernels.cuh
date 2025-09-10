// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

// #ifndef MEM_KERNELS_CUH
// #define MEM_KERNELS_CUH

void multi_layer_kv_transfer(torch::Tensor& key_value,
                             const torch::Tensor& key_value_ptrs,
                             const torch::Tensor& slot_mapping,
                             const torch::Device& paged_memory_device,
                             const int page_buffer_size, const bool direction,
                             const bool use_mla);

void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const bool direction, const bool use_mla);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping, const bool direction,
                              const bool token_major = false,
                              const bool vllm_two_major = false);

void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);
