// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/cuda_transfer.h"
#include "lmcache_mp_cpp/native_transfer_kernel.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lmcache::mp {

enum class TensorLayout {
  kTwoNbBsNhHs,
  kNbTwoBsNhHs,
  kTwoNbNhBsHs,
  kNbTwoNhBsHs,
  kNbNlTwoBsNhHs,
  kNbNlTwoNhBsHs,
  kMlaNbBsHs,
};

struct LayoutInfo {
  TensorLayout layout = TensorLayout::kTwoNbBsNhHs;
  std::uint64_t kv_size = 0;
  std::uint64_t num_layers = 1;
  std::uint64_t num_blocks = 0;
  std::uint64_t block_size = 0;
  std::uint64_t num_heads = 1;
  std::uint64_t head_size = 0;
  std::uint64_t hidden_dim = 0;
  std::uint64_t element_size = 0;
  std::uint64_t layer_bytes = 0;
};

struct CudaTrtLlmLayoutHints {
  bool present = false;
  std::uint64_t num_kv_heads = 0;
  std::uint64_t tokens_per_block = 0;
  std::uint64_t head_dim = 0;
};

struct TransferGroup {
  LayoutInfo layout;
  std::vector<std::size_t> tensor_indices;
  std::uint64_t chunk_tokens = 0;
  std::uint64_t chunk_offset_bytes = 0;
  std::uint64_t chunk_bytes = 0;
};

std::optional<LayoutInfo> InferLayout(
    const KvTensorMetadata& tensor, const std::string& kv_layout,
    const CudaTrtLlmLayoutHints& trt_hints, std::string* error);

std::uint64_t DeviceTokenOffset(const LayoutInfo& layout, std::uint64_t kv,
                                std::uint64_t block_id,
                                std::uint64_t token_in_block,
                                std::uint64_t layer_in_tensor = 0,
                                std::uint64_t head = 0);

bool IsHndLayout(TensorLayout layout);
bool SupportsContiguousBlockRun(TensorLayout layout);
std::optional<int> NativeKernelGpuKvFormat(TensorLayout layout);
bool SupportsNativeKernelFastPath(const std::vector<TransferGroup>& groups);
NativePageBufferShapeDesc NativeShapeDesc(const LayoutInfo& layout);

bool ChunkBlockIdsAreContiguous(const KvTransferRequest& request,
                                std::size_t chunk_idx,
                                std::uint64_t blocks_per_chunk);

std::uint64_t ChunkByteOffset(const LayoutInfo& layout,
                              std::uint64_t group_chunk_offset,
                              std::uint64_t group_layer_count,
                              std::uint64_t chunk_size,
                              std::uint64_t group_layer_index, std::uint64_t kv,
                              std::uint64_t token,
                              std::uint64_t hidden_offset = 0);

bool ComputeGroupChunkBytes(const LayoutInfo& layout,
                            std::uint64_t group_layer_count,
                            std::uint64_t chunk_size,
                            std::uint64_t* group_bytes);

bool BuildTransferGroups(const KvTransferRequest& request,
                         const std::vector<LayoutInfo>& layouts,
                         std::vector<TransferGroup>* groups,
                         std::uint64_t* chunk_bytes, std::string* error);

bool ValidateTransferLayouts(const KvTransferRequest& request,
                             std::vector<LayoutInfo>* layouts,
                             std::vector<TransferGroup>* groups,
                             std::uint64_t* chunk_bytes, std::string* error);

bool ChecksumSupportsLayout(TensorLayout layout);
std::string ChecksumLayoutName(TensorLayout layout);

}  // namespace lmcache::mp
