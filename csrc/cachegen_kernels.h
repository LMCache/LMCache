// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <torch/torch.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#include <cmath>

#include <torch/extension.h>

#include <tuple>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iterator>

#include <bitset>

#ifndef USE_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

const int precision = 16;
const int N = 1;
using cdf_t = uint16_t;
const int PRECISION = 16;
const int RENORMALIZATION_FACTOR = 2 << (PRECISION - 1);
const int STRIDE = 1;

#ifndef USE_XPU
void encode_cuda_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths);

void decode_cuda_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output);

void decode_cuda_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output);

const struct cdf_ptr get_cdf_ptr_cuda(const at::Tensor& cdf);
#else
void encode_xpu_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths);

void decode_xpu_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output);

void decode_xpu_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output);

const struct cdf_ptr get_cdf_ptr_xpu(const at::Tensor& cdf);

#endif

at::Tensor calculate_cdf(const at::Tensor& input, const int max_bins);
