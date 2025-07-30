#pragma once
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>

void encode_cuda_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths);

void decode_cuda_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output);

void decode_cuda_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output);

at::Tensor calculate_cdf(const at::Tensor& input, const int max_bins);