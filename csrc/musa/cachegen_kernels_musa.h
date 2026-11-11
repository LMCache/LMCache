// SPDX-License-Identifier: Apache-2.0
// MUSA port of csrc/cuda/cachegen_kernels.cuh.
//
// Host-side declarations shared by the MUSA codec kernels. The Python-visible
// names bound by ``lmcache.musa_ops`` mirror ``lmcache.cuda_ops`` so
// ``MusaDeviceOps`` can dispatch to them directly.

#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

int get_block_size_musa(int nchannels);

void encode_musa_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths);

void decode_musa_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output);

void decode_musa_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths_prefsum, at::Tensor& output);

at::Tensor calculate_cdf_musa(const at::Tensor& input, const int max_bins);
