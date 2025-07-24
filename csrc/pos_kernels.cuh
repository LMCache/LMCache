// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox);