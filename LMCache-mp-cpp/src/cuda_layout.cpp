// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_layout.h"

#include "lmcache_mp_cpp/cuda_metadata.h"

#include <limits>

namespace lmcache::mp {
namespace {

bool CheckedMul(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    return false;
  }
  *out = a * b;
  return true;
}

bool SameTransferGroupIdentity(const LayoutInfo& a, const LayoutInfo& b) {
  return a.layout == b.layout && a.kv_size == b.kv_size &&
         a.block_size == b.block_size && a.hidden_dim == b.hidden_dim &&
         a.element_size == b.element_size;
}

}  // namespace

std::optional<LayoutInfo> InferLayout(
    const KvTensorMetadata& tensor, const std::string& kv_layout,
    const CudaTrtLlmLayoutHints& trt_hints, std::string* error) {
  auto element_size = DtypeElementSize(tensor.dtype);
  if (!element_size) {
    *error =
        "unsupported KV tensor dtype for native CUDA transfer: " + tensor.dtype;
    return std::nullopt;
  }

  LayoutInfo info;
  info.element_size = *element_size;
  if (tensor.shape.size() == 6 && tensor.shape[2] == 2) {
    info.layout = kv_layout == "HND" ? TensorLayout::kNbNlTwoNhBsHs
                                     : TensorLayout::kNbNlTwoBsNhHs;
    info.kv_size = 2;
    info.num_layers = tensor.shape[1];
    info.num_blocks = tensor.shape[0];
    info.block_size = kv_layout == "HND" ? tensor.shape[4] : tensor.shape[3];
    info.num_heads = kv_layout == "HND" ? tensor.shape[3] : tensor.shape[4];
    info.head_size = tensor.shape[5];
  } else if (tensor.shape.size() == 5 && tensor.shape[0] == 2) {
    info.layout = kv_layout == "HND" ? TensorLayout::kTwoNbNhBsHs
                                     : TensorLayout::kTwoNbBsNhHs;
    info.kv_size = 2;
    info.num_layers = 1;
    info.num_blocks = tensor.shape[1];
    info.block_size = kv_layout == "HND" ? tensor.shape[3] : tensor.shape[2];
    info.num_heads = kv_layout == "HND" ? tensor.shape[2] : tensor.shape[3];
    info.head_size = tensor.shape[4];
  } else if (tensor.shape.size() == 5 && tensor.shape[1] == 2) {
    info.layout = kv_layout == "HND" ? TensorLayout::kNbTwoNhBsHs
                                     : TensorLayout::kNbTwoBsNhHs;
    info.kv_size = 2;
    info.num_layers = 1;
    info.num_blocks = tensor.shape[0];
    info.block_size = kv_layout == "HND" ? tensor.shape[3] : tensor.shape[2];
    info.num_heads = kv_layout == "HND" ? tensor.shape[2] : tensor.shape[3];
    info.head_size = tensor.shape[4];
  } else if (tensor.shape.size() == 4 && tensor.shape[2] == 2 &&
             trt_hints.present) {
    const std::uint64_t flat = tensor.shape[3];
    if (trt_hints.num_kv_heads == 0 || trt_hints.tokens_per_block == 0 ||
        trt_hints.head_dim == 0 ||
        trt_hints.num_kv_heads > std::numeric_limits<std::uint64_t>::max() /
                                     trt_hints.tokens_per_block ||
        trt_hints.num_kv_heads * trt_hints.tokens_per_block >
            std::numeric_limits<std::uint64_t>::max() / trt_hints.head_dim ||
        flat != trt_hints.num_kv_heads * trt_hints.tokens_per_block *
                    trt_hints.head_dim) {
      *error = "TRT-LLM KV tensor flat dimension does not match layout hints";
      return std::nullopt;
    }
    info.layout = TensorLayout::kNbNlTwoNhBsHs;
    info.kv_size = 2;
    info.num_layers = tensor.shape[1];
    info.num_blocks = tensor.shape[0];
    info.block_size = trt_hints.tokens_per_block;
    info.num_heads = trt_hints.num_kv_heads;
    info.head_size = trt_hints.head_dim;
  } else if (tensor.shape.size() == 4 && tensor.shape[0] == 2) {
    info.layout = TensorLayout::kTwoNbBsNhHs;
    info.kv_size = 2;
    info.num_layers = 1;
    info.num_blocks = tensor.shape[1];
    info.block_size = tensor.shape[2];
    info.num_heads = 1;
    info.head_size = tensor.shape[3];
  } else if (tensor.shape.size() == 3) {
    info.layout = TensorLayout::kMlaNbBsHs;
    info.kv_size = 1;
    info.num_layers = 1;
    info.num_blocks = tensor.shape[0];
    info.block_size = tensor.shape[1];
    info.num_heads = 1;
    info.head_size = tensor.shape[2];
  } else {
    *error =
        "native CUDA transfer supports per-layer vLLM NHD/HND, "
        "cross-layer NHD/HND, TRT-LLM 4D, and MLA KV tensors only";
    return std::nullopt;
  }

  if (info.num_layers == 0 || info.num_blocks == 0 || info.block_size == 0 ||
      info.num_heads == 0 || info.head_size == 0) {
    *error = "invalid zero-sized KV tensor dimension for native CUDA transfer";
    return std::nullopt;
  }
  info.hidden_dim = info.num_heads * info.head_size;
  if (info.hidden_dim >
      std::numeric_limits<std::uint64_t>::max() / info.block_size) {
    *error = "KV tensor shape overflows native transfer size arithmetic";
    return std::nullopt;
  }
  info.layer_bytes = info.kv_size * info.num_layers * info.num_blocks *
                     info.block_size * info.hidden_dim * info.element_size;
  return info;
}

std::uint64_t DeviceTokenOffset(const LayoutInfo& layout, std::uint64_t kv,
                                std::uint64_t block_id,
                                std::uint64_t token_in_block,
                                std::uint64_t layer_in_tensor,
                                std::uint64_t head) {
  switch (layout.layout) {
    case TensorLayout::kTwoNbBsNhHs:
      return (((kv * layout.num_blocks + block_id) * layout.block_size +
               token_in_block) *
              layout.hidden_dim) *
             layout.element_size;
    case TensorLayout::kNbTwoBsNhHs:
      return (((block_id * layout.kv_size + kv) * layout.block_size +
               token_in_block) *
              layout.hidden_dim) *
             layout.element_size;
    case TensorLayout::kTwoNbNhBsHs:
      return ((((kv * layout.num_blocks + block_id) * layout.num_heads + head) *
                   layout.block_size +
               token_in_block) *
              layout.head_size) *
             layout.element_size;
    case TensorLayout::kNbTwoNhBsHs:
      return ((((block_id * layout.kv_size + kv) * layout.num_heads + head) *
                   layout.block_size +
               token_in_block) *
              layout.head_size) *
             layout.element_size;
    case TensorLayout::kNbNlTwoBsNhHs:
      return (
          ((((block_id * layout.num_layers + layer_in_tensor) * layout.kv_size +
             kv) *
                layout.block_size +
            token_in_block) *
           layout.hidden_dim) *
          layout.element_size);
    case TensorLayout::kNbNlTwoNhBsHs:
      return (((((block_id * layout.num_layers + layer_in_tensor) *
                     layout.kv_size +
                 kv) *
                    layout.num_heads +
                head) *
                   layout.block_size +
               token_in_block) *
              layout.head_size) *
             layout.element_size;
    case TensorLayout::kMlaNbBsHs:
      return ((block_id * layout.block_size + token_in_block) *
              layout.hidden_dim) *
             layout.element_size;
  }
  return 0;
}

bool IsHndLayout(TensorLayout layout) {
  return layout == TensorLayout::kTwoNbNhBsHs ||
         layout == TensorLayout::kNbTwoNhBsHs ||
         layout == TensorLayout::kNbNlTwoNhBsHs;
}

bool SupportsContiguousBlockRun(TensorLayout layout) {
  return layout == TensorLayout::kTwoNbBsNhHs ||
         layout == TensorLayout::kMlaNbBsHs;
}

std::optional<int> NativeKernelGpuKvFormat(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::kNbNlTwoBsNhHs:
      return 0;
    case TensorLayout::kTwoNbBsNhHs:
      return 1;
    case TensorLayout::kNbTwoBsNhHs:
      return 2;
    case TensorLayout::kMlaNbBsHs:
      return 3;
    case TensorLayout::kTwoNbNhBsHs:
      return 6;
    case TensorLayout::kNbTwoNhBsHs:
      return 7;
    case TensorLayout::kNbNlTwoNhBsHs:
      return 8;
  }
  return std::nullopt;
}

bool SupportsNativeKernelFastPath(const std::vector<TransferGroup>& groups) {
  return groups.size() == 1 && NativeKernelGpuKvFormat(groups[0].layout.layout);
}

NativePageBufferShapeDesc NativeShapeDesc(const LayoutInfo& layout) {
  return {.kv_size = static_cast<int>(layout.kv_size),
          .nl = static_cast<int>(layout.num_layers),
          .nb = static_cast<int>(layout.num_blocks),
          .bs = static_cast<int>(layout.block_size),
          .nh = static_cast<int>(layout.num_heads),
          .hs = static_cast<int>(layout.head_size),
          .element_size = static_cast<int>(layout.element_size),
          .block_stride_elems = 0};
}

bool ChunkBlockIdsAreContiguous(const KvTransferRequest& request,
                                std::size_t chunk_idx,
                                std::uint64_t blocks_per_chunk) {
  if (blocks_per_chunk == 0) {
    return false;
  }
  const std::size_t begin = chunk_idx * blocks_per_chunk;
  if (begin + blocks_per_chunk > request.gpu_block_ids.size()) {
    return false;
  }
  const std::uint64_t first = request.gpu_block_ids[begin];
  for (std::uint64_t offset = 1; offset < blocks_per_chunk; ++offset) {
    if (request.gpu_block_ids[begin + offset] != first + offset) {
      return false;
    }
  }
  return true;
}

std::uint64_t ChunkByteOffset(const LayoutInfo& layout,
                              std::uint64_t group_chunk_offset,
                              std::uint64_t group_layer_count,
                              std::uint64_t chunk_size,
                              std::uint64_t group_layer_index, std::uint64_t kv,
                              std::uint64_t token,
                              std::uint64_t hidden_offset) {
  return group_chunk_offset +
         ((((kv * group_layer_count + group_layer_index) * chunk_size + token) *
               layout.hidden_dim +
           hidden_offset) *
          layout.element_size);
}

bool ComputeGroupChunkBytes(const LayoutInfo& layout,
                            std::uint64_t group_layer_count,
                            std::uint64_t chunk_size,
                            std::uint64_t* group_bytes) {
  std::uint64_t value = 0;
  return CheckedMul(layout.kv_size, group_layer_count, &value) &&
         CheckedMul(value, chunk_size, &value) &&
         CheckedMul(value, layout.hidden_dim, &value) &&
         CheckedMul(value, layout.element_size, group_bytes);
}

bool BuildTransferGroups(const KvTransferRequest& request,
                         const std::vector<LayoutInfo>& layouts,
                         std::vector<TransferGroup>* groups,
                         std::uint64_t* chunk_bytes, std::string* error) {
  groups->clear();
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    const LayoutInfo& layout = layouts[i];
    if (layout.block_size > request.logical_block_size ||
        request.logical_block_size % layout.block_size != 0) {
      *error =
          "native CUDA transfer requires registered block size to divide "
          "logical block size";
      return false;
    }

    TransferGroup* group = nullptr;
    for (TransferGroup& candidate : *groups) {
      if (SameTransferGroupIdentity(candidate.layout, layout)) {
        group = &candidate;
        break;
      }
    }
    if (group == nullptr) {
      groups->push_back({.layout = layout, .tensor_indices = {i}});
      continue;
    }
    if (layout.num_layers >
        std::numeric_limits<std::uint64_t>::max() - group->layout.num_layers) {
      *error = "KV tensor layer count overflows native transfer arithmetic";
      return false;
    }
    group->layout.num_layers += layout.num_layers;
    group->tensor_indices.push_back(i);
  }

  std::uint64_t total = 0;
  const std::uint64_t blocks_per_chunk =
      request.chunk_size / request.logical_block_size;
  for (TransferGroup& group : *groups) {
    std::uint64_t group_bytes = 0;
    if (!CheckedMul(blocks_per_chunk, group.layout.block_size,
                    &group.chunk_tokens)) {
      *error = "KV tensor shape overflows native transfer size arithmetic";
      return false;
    }
    if (!ComputeGroupChunkBytes(group.layout, group.layout.num_layers,
                                group.chunk_tokens, &group_bytes)) {
      *error = "KV tensor shape overflows native transfer size arithmetic";
      return false;
    }
    group.chunk_offset_bytes = total;
    group.chunk_bytes = group_bytes;
    if (group_bytes > std::numeric_limits<std::uint64_t>::max() - total) {
      *error = "KV tensor shape overflows native transfer size arithmetic";
      return false;
    }
    total += group_bytes;
  }
  if (total == 0) {
    *error = "native CUDA transfer computed a zero-byte KV chunk";
    return false;
  }
  *chunk_bytes = total;
  return true;
}

bool ValidateTransferLayouts(const KvTransferRequest& request,
                             std::vector<LayoutInfo>* layouts,
                             std::vector<TransferGroup>* groups,
                             std::uint64_t* chunk_bytes, std::string* error) {
  if (request.tensors.empty()) {
    *error = "registered context has no CUDA KV tensor metadata";
    return false;
  }
  layouts->clear();
  layouts->reserve(request.tensors.size());
  const CudaTrtLlmLayoutHints trt_hints = {
      .present = request.trt_llm_layout_hints,
      .num_kv_heads = request.trt_llm_num_kv_heads,
      .tokens_per_block = request.trt_llm_tokens_per_block,
      .head_dim = request.trt_llm_head_dim,
  };
  for (const KvTensorMetadata& tensor : request.tensors) {
    if (tensor.ipc_handle.empty()) {
      *error = "registered KV tensor is missing CUDA IPC memory handle bytes";
      return false;
    }
    auto layout = InferLayout(tensor, request.kv_layout, trt_hints, error);
    if (!layout) {
      return false;
    }
    layouts->push_back(*layout);
  }

  return BuildTransferGroups(request, *layouts, groups, chunk_bytes, error);
}

bool ChecksumSupportsLayout(TensorLayout layout) {
  return layout == TensorLayout::kTwoNbBsNhHs ||
         layout == TensorLayout::kNbTwoBsNhHs ||
         layout == TensorLayout::kTwoNbNhBsHs ||
         layout == TensorLayout::kNbTwoNhBsHs ||
         layout == TensorLayout::kMlaNbBsHs;
}

std::string ChecksumLayoutName(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::kTwoNbBsNhHs:
      return "NL_X_TWO_NB_BS_NH_HS";
    case TensorLayout::kNbTwoBsNhHs:
      return "NL_X_NB_TWO_BS_NH_HS";
    case TensorLayout::kTwoNbNhBsHs:
      return "NL_X_TWO_NB_NH_BS_HS";
    case TensorLayout::kNbTwoNhBsHs:
      return "NL_X_NB_TWO_NH_BS_HS";
    case TensorLayout::kMlaNbBsHs:
      return "NL_X_NB_BS_HS";
    case TensorLayout::kNbNlTwoBsNhHs:
      return "NB_NL_TWO_BS_NH_HS";
    case TensorLayout::kNbNlTwoNhBsHs:
      return "NB_NL_TWO_NH_BS_HS";
  }
  return "UNKNOWN";
}

}  // namespace lmcache::mp
