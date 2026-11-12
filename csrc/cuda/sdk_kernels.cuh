// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * Copy each (src_ptrs[i] -> dst_ptrs[i], sizes[i] bytes) range through the
 * copy engine: one cudaMemcpyBatchAsync on the current stream of `device`.
 * Host operands must be pinned. The caller synchronizes.
 *
 * @throws c10::Error if batch_memcpy_supported() is false or the CUDA call
 *         fails.
 */
void batched_memcpy_async(const std::vector<uintptr_t>& src_ptrs,
                          const std::vector<uintptr_t>& dst_ptrs,
                          const std::vector<size_t>& sizes,
                          const torch::Device& device);
