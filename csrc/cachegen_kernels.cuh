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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

const int precision = 16;
const int N = 1;
using cdf_t = uint16_t;
const int PRECISION = 16;
const int RENORMALIZATION_FACTOR = 2 << (PRECISION - 1);
const int STRIDE = 1;

void encode_cuda_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths);

void decode_cuda_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output);

void decode_cuda_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output);

const struct cdf_ptr get_cdf_ptr_cuda(const at::Tensor& cdf);

at::Tensor calculate_cdf(const at::Tensor& input, const int max_bins);