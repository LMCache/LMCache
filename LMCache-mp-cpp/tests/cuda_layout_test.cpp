// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_layout.h"

#include <cassert>
#include <optional>
#include <string>
#include <vector>

namespace {

lmcache::mp::KvTensorMetadata Tensor(std::vector<std::uint64_t> shape) {
  lmcache::mp::KvTensorMetadata tensor;
  tensor.dtype = "torch.float16";
  tensor.shape = std::move(shape);
  tensor.ipc_handle = {1};
  return tensor;
}

}  // namespace

int main() {
  using lmcache::mp::ChecksumLayoutName;
  using lmcache::mp::ChecksumSupportsLayout;
  using lmcache::mp::ChunkBlockIdsAreContiguous;
  using lmcache::mp::CudaTrtLlmLayoutHints;
  using lmcache::mp::DeviceTokenOffset;
  using lmcache::mp::InferLayout;
  using lmcache::mp::KvTransferRequest;
  using lmcache::mp::LayoutInfo;
  using lmcache::mp::NativeKernelGpuKvFormat;
  using lmcache::mp::NativeShapeDesc;
  using lmcache::mp::TensorLayout;
  using lmcache::mp::TransferGroup;
  using lmcache::mp::ValidateTransferLayouts;

  std::string error;
  auto layout = InferLayout(Tensor({2, 10, 16, 32}), "", {}, &error);
  assert(layout);
  assert(layout->layout == TensorLayout::kTwoNbBsNhHs);
  assert(layout->kv_size == 2);
  assert(layout->num_blocks == 10);
  assert(layout->block_size == 16);
  assert(layout->hidden_dim == 32);
  assert(layout->element_size == 2);
  assert(DeviceTokenOffset(*layout, 1, 2, 3) ==
         (((1 * 10 + 2) * 16 + 3) * 32) * 2);
  assert(NativeKernelGpuKvFormat(layout->layout) == 1);
  assert(ChecksumSupportsLayout(layout->layout));
  assert(ChecksumLayoutName(layout->layout) == "NL_X_TWO_NB_BS_NH_HS");
  const auto shape_desc = NativeShapeDesc(*layout);
  assert(shape_desc.kv_size == 2);
  assert(shape_desc.nb == 10);
  assert(shape_desc.bs == 16);

  CudaTrtLlmLayoutHints trt_hints{
      .present = true, .num_kv_heads = 2, .tokens_per_block = 16, .head_dim = 8};
  auto trt_layout = InferLayout(Tensor({4, 3, 2, 256}), "HND", trt_hints,
                                &error);
  assert(trt_layout);
  assert(trt_layout->layout == TensorLayout::kNbNlTwoNhBsHs);
  assert(trt_layout->num_layers == 3);
  assert(trt_layout->block_size == 16);
  assert(!ChecksumSupportsLayout(trt_layout->layout));
  assert(ChecksumLayoutName(trt_layout->layout) == "NB_NL_TWO_NH_BS_HS");

  KvTransferRequest request;
  request.tensors = {Tensor({2, 10, 16, 32}), Tensor({2, 10, 16, 32})};
  request.gpu_block_ids = {4, 5, 8, 9};
  request.chunk_size = 32;
  request.logical_block_size = 16;
  std::vector<LayoutInfo> layouts;
  std::vector<TransferGroup> groups;
  std::uint64_t chunk_bytes = 0;
  assert(ValidateTransferLayouts(request, &layouts, &groups, &chunk_bytes,
                                 &error));
  assert(layouts.size() == 2);
  assert(groups.size() == 1);
  assert(groups[0].layout.num_layers == 2);
  assert(groups[0].tensor_indices == std::vector<std::size_t>({0, 1}));
  assert(chunk_bytes == 2 * 2 * 32 * 32 * 2);
  assert(ChunkBlockIdsAreContiguous(request, 0, 2));
  assert(ChunkBlockIdsAreContiguous(request, 1, 2));
  assert(!ChunkBlockIdsAreContiguous(request, 0, 0));

  request.tensors[0].ipc_handle.clear();
  assert(!ValidateTransferLayouts(request, &layouts, &groups, &chunk_bytes,
                                  &error));
  assert(error == "registered KV tensor is missing CUDA IPC memory handle bytes");

  return 0;
}

