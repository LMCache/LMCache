// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>
#include <tuple>

namespace lmcache {

// Fused amax + scale + FP8 e4m3 cast.
//
// Returns (fp8_tensor, scales) where `scales` holds the dequant
// multiplier for the chosen granularity:
//   scale_mode 0 -> per-tensor : scales[0]
//   scale_mode 1 -> rowwise    : scales[rows], one per x.size(-1) slice
//   scale_mode 2 -> blockwise  : scales[ceil(numel/block_size)]
//
// Replaces a scale-free `x.to(torch.float8_e4m3fn).contiguous()`, which
// saturates at +-448 regardless of tensor magnitude and costs two passes.
std::tuple<at::Tensor, at::Tensor> fp8QuantizeScaled(
    const at::Tensor& x, at::Tensor& fp8_out, at::Tensor& scales,
    int64_t scale_mode, int64_t block_size, double amax_ceiling);

}  // namespace lmcache