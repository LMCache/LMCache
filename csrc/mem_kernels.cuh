/*
 * Copyright 2024-2025 LMCache Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    torch::Tensor& key_value, const torch::Tensor& key_ptrs,
    const torch::Tensor& value_ptrs, const torch::Tensor& slot_mapping,
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const bool direction);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_cache,
                              torch::Tensor& vllm_value_cache,
                              torch::Tensor& slot_mapping, const bool direction,
                              const bool token_major = false);

void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);
