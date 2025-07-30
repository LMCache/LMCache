#pragma once
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox);