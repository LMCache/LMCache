// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_transfer.h"
#include "lmcache_mp_cpp/cuda_layout.h"
#include "lmcache_mp_cpp/cuda_runtime_state.h"
#include "lmcache_mp_cpp/native_transfer_kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#if LMCACHE_ENABLE_CUDA
  #include <cuda_runtime_api.h>
#endif

namespace lmcache::mp {
namespace {

KvTransferResult ErrorResult(std::string error) {
  return {.success = false,
          .completion_event_handle = {},
          .error = std::move(error),
          .stats = {}};
}

#if LMCACHE_ENABLE_CUDA

using Clock = std::chrono::steady_clock;

std::uint64_t ElapsedUs(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                            start)
          .count());
}

struct OpenTensor {
  void* base = nullptr;
  const KvTensorMetadata* metadata = nullptr;
  LayoutInfo layout;
  std::uint64_t base_offset_bytes = 0;
  bool close_on_finish = true;
};

class HostChunkBuffer {
 public:
  explicit HostChunkBuffer(std::uint64_t size) : size_(size) {
    if (size_ == 0) {
      return;
    }
    void* ptr = nullptr;
    if (cudaHostAlloc(&ptr, size_, cudaHostAllocMapped) == cudaSuccess) {
      pinned_ = static_cast<std::uint8_t*>(ptr);
      void* device_ptr = nullptr;
      if (cudaHostGetDevicePointer(&device_ptr, pinned_, 0) == cudaSuccess) {
        device_ptr_ = static_cast<std::uint8_t*>(device_ptr);
      }
      return;
    }
    fallback_.resize(static_cast<std::size_t>(size_));
  }

  ~HostChunkBuffer() {
    if (pinned_ != nullptr) {
      (void)cudaFreeHost(pinned_);
    }
  }

  HostChunkBuffer(const HostChunkBuffer&) = delete;
  HostChunkBuffer& operator=(const HostChunkBuffer&) = delete;

  std::uint8_t* data() {
    return pinned_ != nullptr ? pinned_ : fallback_.data();
  }

  const std::uint8_t* data() const {
    return pinned_ != nullptr ? pinned_ : fallback_.data();
  }

  std::uint64_t size() const { return size_; }

  std::uint8_t* device_data() { return device_ptr_; }

  bool device_accessible() const { return device_ptr_ != nullptr; }

 private:
  std::uint64_t size_ = 0;
  std::uint8_t* pinned_ = nullptr;
  std::uint8_t* device_ptr_ = nullptr;
  std::vector<std::uint8_t> fallback_;
};

void RecordCudaCopy(KvTransferStats* stats, std::uint64_t nbytes) {
  if (stats == nullptr) {
    return;
  }
  stats->bytes += nbytes;
  ++stats->cuda_memcpy_calls;
}

std::uint32_t Md5LeftRotate(std::uint32_t value, std::uint32_t bits) {
  return (value << bits) | (value >> (32 - bits));
}

std::array<std::uint8_t, 16> Md5Digest(const std::uint8_t* data,
                                       std::size_t size) {
  static constexpr std::uint32_t kShift[64] = {
      7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
      5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  };
  static constexpr std::uint32_t kTable[64] = {
      0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
      0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
      0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
      0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
      0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
      0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
      0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
      0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
      0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
      0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
      0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
  };

  std::vector<std::uint8_t> message;
  if (size != 0) {
    message.assign(data, data + size);
  }
  const std::uint64_t bit_len = static_cast<std::uint64_t>(size) * 8;
  message.push_back(0x80);
  while (message.size() % 64 != 56) {
    message.push_back(0);
  }
  for (int i = 0; i < 8; ++i) {
    message.push_back(static_cast<std::uint8_t>((bit_len >> (8 * i)) & 0xff));
  }

  std::uint32_t a0 = 0x67452301;
  std::uint32_t b0 = 0xefcdab89;
  std::uint32_t c0 = 0x98badcfe;
  std::uint32_t d0 = 0x10325476;

  for (std::size_t offset = 0; offset < message.size(); offset += 64) {
    std::uint32_t words[16];
    for (int i = 0; i < 16; ++i) {
      const std::size_t j = offset + static_cast<std::size_t>(i) * 4;
      words[i] = static_cast<std::uint32_t>(message[j]) |
                 (static_cast<std::uint32_t>(message[j + 1]) << 8) |
                 (static_cast<std::uint32_t>(message[j + 2]) << 16) |
                 (static_cast<std::uint32_t>(message[j + 3]) << 24);
    }

    std::uint32_t a = a0;
    std::uint32_t b = b0;
    std::uint32_t c = c0;
    std::uint32_t d = d0;
    for (int i = 0; i < 64; ++i) {
      std::uint32_t f = 0;
      std::uint32_t g = 0;
      if (i < 16) {
        f = (b & c) | ((~b) & d);
        g = static_cast<std::uint32_t>(i);
      } else if (i < 32) {
        f = (d & b) | ((~d) & c);
        g = static_cast<std::uint32_t>((5 * i + 1) % 16);
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = static_cast<std::uint32_t>((3 * i + 5) % 16);
      } else {
        f = c ^ (b | (~d));
        g = static_cast<std::uint32_t>((7 * i) % 16);
      }
      const std::uint32_t temp = d;
      d = c;
      c = b;
      b = b + Md5LeftRotate(a + f + kTable[i] + words[g], kShift[i]);
      a = temp;
    }
    a0 += a;
    b0 += b;
    c0 += c;
    d0 += d;
  }

  const std::uint32_t state[4] = {a0, b0, c0, d0};
  std::array<std::uint8_t, 16> digest{};
  for (int i = 0; i < 4; ++i) {
    digest[static_cast<std::size_t>(i) * 4] =
        static_cast<std::uint8_t>(state[i] & 0xff);
    digest[static_cast<std::size_t>(i) * 4 + 1] =
        static_cast<std::uint8_t>((state[i] >> 8) & 0xff);
    digest[static_cast<std::size_t>(i) * 4 + 2] =
        static_cast<std::uint8_t>((state[i] >> 16) & 0xff);
    digest[static_cast<std::size_t>(i) * 4 + 3] =
        static_cast<std::uint8_t>((state[i] >> 24) & 0xff);
  }
  return digest;
}

std::string Md5Hex(const std::vector<std::uint8_t>& data) {
  const auto digest = Md5Digest(data.data(), data.size());
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::uint8_t byte : digest) {
    out << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return out.str();
}

std::vector<void*> PagedBufferPtrsForGroup(
    const std::vector<OpenTensor>& tensors, const TransferGroup& group) {
  std::vector<void*> ptrs;
  ptrs.reserve(group.tensor_indices.size());
  for (std::size_t tensor_index : group.tensor_indices) {
    const OpenTensor& tensor = tensors[tensor_index];
    ptrs.push_back(static_cast<std::uint8_t*>(tensor.base) +
                   tensor.base_offset_bytes);
  }
  return ptrs;
}

bool OpenTensors(const KvTransferRequest& request,
                 const std::vector<LayoutInfo>& layouts,
                 std::vector<OpenTensor>* tensors, std::string* error) {
  tensors->clear();
  tensors->reserve(request.tensors.size());
  for (std::size_t i = 0; i < request.tensors.size(); ++i) {
    const KvTensorMetadata& metadata = request.tensors[i];
    void* ptr = nullptr;
    if (!OpenCudaTensorMemory(metadata, &ptr, error)) {
      return false;
    }
    std::uint64_t base_offset = 0;
    if (metadata.kind != "raw") {
      base_offset = metadata.storage_offset_bytes +
                    metadata.storage_offset * layouts[i].element_size;
    }
    tensors->push_back({.base = ptr,
                        .metadata = &metadata,
                        .layout = layouts[i],
                        .base_offset_bytes = base_offset,
                        .close_on_finish = false});
  }
  return true;
}

void CloseTensors(std::vector<OpenTensor>* tensors) {
  for (OpenTensor& tensor : *tensors) {
    if (tensor.base != nullptr && tensor.close_on_finish) {
      (void)cudaIpcCloseMemHandle(tensor.base);
    }
    tensor.base = nullptr;
  }
}

bool CopyTokenD2H(const OpenTensor& tensor, const TransferGroup& group,
                  std::uint64_t block_id, std::uint64_t token_in_block,
                  std::uint64_t group_layer_index,
                  std::uint64_t layer_in_tensor, std::uint64_t token_in_chunk,
                  std::uint8_t* out, std::string* error, KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (block_id >= layout.num_blocks) {
    *error = "CUDA D2H transfer block id is outside tensor range";
    return false;
  }
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    if (IsHndLayout(layout.layout)) {
      for (std::uint64_t head = 0; head < layout.num_heads; ++head) {
        const std::uint64_t src = DeviceTokenOffset(
            layout, kv, block_id, token_in_block, layer_in_tensor, head);
        const std::uint64_t dst = ChunkByteOffset(
            group.layout, group.chunk_offset_bytes, group.layout.num_layers,
            group.chunk_tokens, group_layer_index, kv, token_in_chunk,
            head * layout.head_size);
        const std::uint64_t nbytes = layout.head_size * layout.element_size;
        RecordCudaCopy(stats, nbytes);
        if (!CheckCuda(cudaMemcpy(out + dst,
                                  static_cast<std::uint8_t*>(tensor.base) +
                                      tensor.base_offset_bytes + src,
                                  nbytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H", error)) {
          return false;
        }
      }
    } else {
      const std::uint64_t src = DeviceTokenOffset(
          layout, kv, block_id, token_in_block, layer_in_tensor);
      const std::uint64_t dst = ChunkByteOffset(
          group.layout, group.chunk_offset_bytes, group.layout.num_layers,
          group.chunk_tokens, group_layer_index, kv, token_in_chunk);
      const std::uint64_t nbytes = layout.hidden_dim * layout.element_size;
      RecordCudaCopy(stats, nbytes);
      if (!CheckCuda(cudaMemcpy(out + dst,
                                static_cast<std::uint8_t*>(tensor.base) +
                                    tensor.base_offset_bytes + src,
                                nbytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H", error)) {
        return false;
      }
    }
  }
  return true;
}

bool CopyBlockD2H(const OpenTensor& tensor, const TransferGroup& group,
                  std::uint64_t block_id, std::uint64_t block_offset,
                  std::uint64_t group_layer_index,
                  std::uint64_t layer_in_tensor,
                  std::uint8_t* out, std::string* error,
                  KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (block_id >= layout.num_blocks) {
    *error = "CUDA D2H transfer block id is outside tensor range";
    return false;
  }
  const std::uint64_t token_in_chunk = block_offset * group.layout.block_size;
  const std::uint64_t nbytes =
      layout.block_size * layout.hidden_dim * layout.element_size;
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    const std::uint64_t src =
        DeviceTokenOffset(layout, kv, block_id, 0, layer_in_tensor);
    const std::uint64_t dst = ChunkByteOffset(
        group.layout, group.chunk_offset_bytes, group.layout.num_layers,
        group.chunk_tokens, group_layer_index, kv, token_in_chunk);
    RecordCudaCopy(stats, nbytes);
    if (!CheckCuda(cudaMemcpy(out + dst,
                              static_cast<std::uint8_t*>(tensor.base) +
                                  tensor.base_offset_bytes + src,
                              nbytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H", error)) {
      return false;
    }
  }
  return true;
}

bool CopyBlockRunD2H(const OpenTensor& tensor, const TransferGroup& group,
                     std::uint64_t first_block_id,
                     std::uint64_t blocks_per_chunk,
                     std::uint64_t group_layer_index,
                     std::uint64_t layer_in_tensor, std::uint8_t* out,
                     std::string* error, KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (first_block_id + blocks_per_chunk > layout.num_blocks) {
    *error = "CUDA D2H transfer block run is outside tensor range";
    return false;
  }
  const std::uint64_t nbytes =
      blocks_per_chunk * layout.block_size * layout.hidden_dim *
      layout.element_size;
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    const std::uint64_t src =
        DeviceTokenOffset(layout, kv, first_block_id, 0, layer_in_tensor);
    const std::uint64_t dst =
        ChunkByteOffset(group.layout, group.chunk_offset_bytes,
                        group.layout.num_layers, group.chunk_tokens,
                        group_layer_index, kv, 0);
    RecordCudaCopy(stats, nbytes);
    if (!CheckCuda(cudaMemcpy(out + dst,
                              static_cast<std::uint8_t*>(tensor.base) +
                                  tensor.base_offset_bytes + src,
                              nbytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H", error)) {
      return false;
    }
  }
  return true;
}

bool CopyTokenH2D(const std::uint8_t* in, OpenTensor& tensor,
                  const TransferGroup& group, std::uint64_t block_id,
                  std::uint64_t token_in_block, std::uint64_t group_layer_index,
                  std::uint64_t layer_in_tensor, std::uint64_t token_in_chunk,
                  std::string* error, KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (block_id >= layout.num_blocks) {
    *error = "CUDA H2D transfer block id is outside tensor range";
    return false;
  }
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    if (IsHndLayout(layout.layout)) {
      for (std::uint64_t head = 0; head < layout.num_heads; ++head) {
        const std::uint64_t src = ChunkByteOffset(
            group.layout, group.chunk_offset_bytes, group.layout.num_layers,
            group.chunk_tokens, group_layer_index, kv, token_in_chunk,
            head * layout.head_size);
        const std::uint64_t dst = DeviceTokenOffset(
            layout, kv, block_id, token_in_block, layer_in_tensor, head);
        const std::uint64_t nbytes = layout.head_size * layout.element_size;
        RecordCudaCopy(stats, nbytes);
        if (!CheckCuda(
                cudaMemcpy(static_cast<std::uint8_t*>(tensor.base) +
                               tensor.base_offset_bytes + dst,
                           in + src, nbytes, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D", error)) {
          return false;
        }
      }
    } else {
      const std::uint64_t src = ChunkByteOffset(
          group.layout, group.chunk_offset_bytes, group.layout.num_layers,
          group.chunk_tokens, group_layer_index, kv, token_in_chunk);
      const std::uint64_t dst = DeviceTokenOffset(
          layout, kv, block_id, token_in_block, layer_in_tensor);
      const std::uint64_t nbytes = layout.hidden_dim * layout.element_size;
      RecordCudaCopy(stats, nbytes);
      if (!CheckCuda(
              cudaMemcpy(static_cast<std::uint8_t*>(tensor.base) +
                             tensor.base_offset_bytes + dst,
                         in + src, nbytes, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D", error)) {
        return false;
      }
    }
  }
  return true;
}

bool CopyBlockH2D(const std::uint8_t* in, OpenTensor& tensor,
                  const TransferGroup& group, std::uint64_t block_id,
                  std::uint64_t block_offset,
                  std::uint64_t group_layer_index,
                  std::uint64_t layer_in_tensor, std::string* error,
                  KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (block_id >= layout.num_blocks) {
    *error = "CUDA H2D transfer block id is outside tensor range";
    return false;
  }
  const std::uint64_t token_in_chunk = block_offset * group.layout.block_size;
  const std::uint64_t nbytes =
      layout.block_size * layout.hidden_dim * layout.element_size;
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    const std::uint64_t src = ChunkByteOffset(
        group.layout, group.chunk_offset_bytes, group.layout.num_layers,
        group.chunk_tokens, group_layer_index, kv, token_in_chunk);
    const std::uint64_t dst =
        DeviceTokenOffset(layout, kv, block_id, 0, layer_in_tensor);
    RecordCudaCopy(stats, nbytes);
    if (!CheckCuda(cudaMemcpy(static_cast<std::uint8_t*>(tensor.base) +
                                  tensor.base_offset_bytes + dst,
                              in + src, nbytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D", error)) {
      return false;
    }
  }
  return true;
}

bool CopyBlockRunH2D(const std::uint8_t* in, OpenTensor& tensor,
                     const TransferGroup& group, std::uint64_t first_block_id,
                     std::uint64_t blocks_per_chunk,
                     std::uint64_t group_layer_index,
                     std::uint64_t layer_in_tensor, std::string* error,
                     KvTransferStats* stats) {
  const LayoutInfo& layout = tensor.layout;
  if (first_block_id + blocks_per_chunk > layout.num_blocks) {
    *error = "CUDA H2D transfer block run is outside tensor range";
    return false;
  }
  const std::uint64_t nbytes =
      blocks_per_chunk * layout.block_size * layout.hidden_dim *
      layout.element_size;
  for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
    const std::uint64_t src =
        ChunkByteOffset(group.layout, group.chunk_offset_bytes,
                        group.layout.num_layers, group.chunk_tokens,
                        group_layer_index, kv, 0);
    const std::uint64_t dst =
        DeviceTokenOffset(layout, kv, first_block_id, 0, layer_in_tensor);
    RecordCudaCopy(stats, nbytes);
    if (!CheckCuda(cudaMemcpy(static_cast<std::uint8_t*>(tensor.base) +
                                  tensor.base_offset_bytes + dst,
                              in + src, nbytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D", error)) {
      return false;
    }
  }
  return true;
}

bool AppendCudaBytes(const OpenTensor& tensor, std::uint64_t offset,
                     std::uint64_t nbytes, std::vector<std::uint8_t>* out,
                     std::string* error) {
  const std::size_t old_size = out->size();
  out->resize(old_size + nbytes);
  if (!CheckCuda(cudaMemcpy(out->data() + old_size,
                            static_cast<std::uint8_t*>(tensor.base) +
                                tensor.base_offset_bytes + offset,
                            nbytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H", error)) {
    return false;
  }
  return true;
}

bool AppendTensorChunkPythonOrder(const OpenTensor& tensor,
                                  const std::vector<std::uint64_t>& block_ids,
                                  std::size_t begin, std::size_t end,
                                  std::vector<std::uint8_t>* out,
                                  std::string* error) {
  const LayoutInfo& layout = tensor.layout;
  for (std::size_t index = begin; index < end; ++index) {
    if (block_ids[index] >= layout.num_blocks) {
      *error = "checksum block id is outside registered KV cache block range";
      return false;
    }
  }

  switch (layout.layout) {
    case TensorLayout::kTwoNbBsNhHs:
    case TensorLayout::kTwoNbNhBsHs:
      for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
        for (std::size_t index = begin; index < end; ++index) {
          const std::uint64_t src =
              DeviceTokenOffset(layout, kv, block_ids[index], 0, 0, 0);
          const std::uint64_t nbytes =
              layout.block_size * layout.hidden_dim * layout.element_size;
          if (!AppendCudaBytes(tensor, src, nbytes, out, error)) {
            return false;
          }
        }
      }
      return true;
    case TensorLayout::kNbTwoBsNhHs:
    case TensorLayout::kNbTwoNhBsHs:
      for (std::size_t index = begin; index < end; ++index) {
        for (std::uint64_t kv = 0; kv < layout.kv_size; ++kv) {
          const std::uint64_t src =
              DeviceTokenOffset(layout, kv, block_ids[index], 0, 0, 0);
          const std::uint64_t nbytes =
              layout.block_size * layout.hidden_dim * layout.element_size;
          if (!AppendCudaBytes(tensor, src, nbytes, out, error)) {
            return false;
          }
        }
      }
      return true;
    case TensorLayout::kMlaNbBsHs:
      for (std::size_t index = begin; index < end; ++index) {
        const std::uint64_t src =
            DeviceTokenOffset(layout, 0, block_ids[index], 0, 0, 0);
        const std::uint64_t nbytes =
            layout.block_size * layout.hidden_dim * layout.element_size;
        if (!AppendCudaBytes(tensor, src, nbytes, out, error)) {
          return false;
        }
      }
      return true;
    case TensorLayout::kNbNlTwoBsNhHs:
    case TensorLayout::kNbNlTwoNhBsHs:
      *error = "checksum not supported for GPU KV format " +
               ChecksumLayoutName(layout.layout);
      return false;
  }
  *error = "checksum not supported for GPU KV format UNKNOWN";
  return false;
}

KvChecksumResult ChecksumWithCuda(const KvChecksumRequest& request) {
  std::string error;
  if (request.chunk_blocks == 0) {
    return {.success = false,
            .num_chunks = 0,
            .chunk_checksums = {},
            .layerwise_chunk_checksums = {},
            .error = "chunk_size must be positive"};
  }
  if (request.gpu_block_ids.empty()) {
    return {.success = true,
            .num_chunks = 0,
            .chunk_checksums = {},
            .layerwise_chunk_checksums = {},
            .error = {}};
  }

  std::vector<LayoutInfo> layouts;
  layouts.reserve(request.tensors.size());
  const CudaTrtLlmLayoutHints trt_hints = {
      .present = request.trt_llm_layout_hints,
      .num_kv_heads = request.trt_llm_num_kv_heads,
      .tokens_per_block = request.trt_llm_tokens_per_block,
      .head_dim = request.trt_llm_head_dim,
  };
  for (const KvTensorMetadata& tensor : request.tensors) {
    if (tensor.ipc_handle.empty()) {
      return {
          .success = false,
          .num_chunks = 0,
          .chunk_checksums = {},
          .layerwise_chunk_checksums = {},
          .error =
              "registered KV tensor is missing CUDA IPC memory handle bytes"};
    }
    auto layout = InferLayout(tensor, request.kv_layout, trt_hints, &error);
    if (!layout) {
      return {.success = false,
              .num_chunks = 0,
              .chunk_checksums = {},
              .layerwise_chunk_checksums = {},
              .error = error};
    }
    if (!ChecksumSupportsLayout(layout->layout)) {
      return {.success = false,
              .num_chunks = 0,
              .chunk_checksums = {},
              .layerwise_chunk_checksums = {},
              .error = "checksum not supported for GPU KV format " +
                       ChecksumLayoutName(layout->layout)};
    }
    layouts.push_back(*layout);
  }

  if (!WaitForCudaInputEvent({}, &error)) {
    return {.success = false,
            .num_chunks = 0,
            .chunk_checksums = {},
            .layerwise_chunk_checksums = {},
            .error = error};
  }

  KvTransferRequest open_request;
  open_request.tensors = request.tensors;
  open_request.kv_layout = request.kv_layout;
  open_request.trt_llm_layout_hints = request.trt_llm_layout_hints;
  open_request.trt_llm_num_kv_heads = request.trt_llm_num_kv_heads;
  open_request.trt_llm_tokens_per_block = request.trt_llm_tokens_per_block;
  open_request.trt_llm_head_dim = request.trt_llm_head_dim;
  std::vector<OpenTensor> tensors;
  if (!OpenTensors(open_request, layouts, &tensors, &error)) {
    CloseTensors(&tensors);
    return {.success = false,
            .num_chunks = 0,
            .chunk_checksums = {},
            .layerwise_chunk_checksums = {},
            .error = error};
  }

  const std::uint64_t num_chunks =
      (request.gpu_block_ids.size() + request.chunk_blocks - 1) /
      request.chunk_blocks;
  std::vector<std::string> aggregated;
  aggregated.reserve(num_chunks);
  std::vector<std::vector<std::string>> per_layer(tensors.size(),
                                                  std::vector<std::string>{});
  for (auto& layer : per_layer) {
    layer.reserve(num_chunks);
  }

  for (std::uint64_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    const std::size_t begin =
        static_cast<std::size_t>(chunk_index) * request.chunk_blocks;
    const std::size_t end = std::min<std::size_t>(begin + request.chunk_blocks,
                                                  request.gpu_block_ids.size());
    std::vector<std::uint8_t> aggregate_input;
    for (std::size_t tensor_index = 0; tensor_index < tensors.size();
         ++tensor_index) {
      std::vector<std::uint8_t> chunk_bytes;
      if (!AppendTensorChunkPythonOrder(tensors[tensor_index],
                                        request.gpu_block_ids, begin, end,
                                        &chunk_bytes, &error)) {
        CloseTensors(&tensors);
        return {.success = false,
                .num_chunks = 0,
                .chunk_checksums = {},
                .layerwise_chunk_checksums = {},
                .error = error};
      }
      const std::string digest = Md5Hex(chunk_bytes);
      per_layer[tensor_index].push_back(digest);
      aggregate_input.insert(aggregate_input.end(), digest.begin(),
                             digest.end());
    }
    aggregated.push_back(Md5Hex(aggregate_input));
  }

  CloseTensors(&tensors);
  return {.success = true,
          .num_chunks = num_chunks,
          .chunk_checksums = std::move(aggregated),
          .layerwise_chunk_checksums =
              request.layerwise ? std::move(per_layer)
                                : std::vector<std::vector<std::string>>{},
          .error = {}};
}

bool TryStoreWithNativeKernel(const KvTransferRequest& request,
                              const std::vector<OpenTensor>& tensors,
                              const std::vector<TransferGroup>& groups,
                              std::uint64_t blocks_per_chunk,
                              std::uint64_t chunk_bytes,
                              LmcacheMpCppCache* cache,
                              KvTransferStats* stats, std::string* error,
                              bool* handled) {
  *handled = false;
  if (!SupportsNativeKernelFastPath(groups) ||
      request.gpu_block_ids.size() !=
          request.object_keys.size() * blocks_per_chunk) {
    return true;
  }

  const TransferGroup& group = groups.front();
  const std::optional<int> gpu_kv_format =
      NativeKernelGpuKvFormat(group.layout.layout);
  const std::vector<void*> paged_ptrs = PagedBufferPtrsForGroup(tensors, group);
  const NativePageBufferShapeDesc shape_desc = NativeShapeDesc(group.layout);
  std::vector<std::string> written;
  for (std::size_t batch_begin = 0; batch_begin < request.object_keys.size();
       batch_begin += 4) {
    const std::size_t batch_count =
        std::min<std::size_t>(4, request.object_keys.size() - batch_begin);
    std::vector<std::unique_ptr<HostChunkBuffer>> chunks;
    std::vector<void*> object_ptrs;
    chunks.reserve(batch_count);
    object_ptrs.reserve(batch_count);
    for (std::size_t i = 0; i < batch_count; ++i) {
      auto chunk = std::make_unique<HostChunkBuffer>(chunk_bytes);
      if (!chunk->device_accessible()) {
        return true;
      }
      object_ptrs.push_back(chunk->device_data());
      chunks.push_back(std::move(chunk));
    }

    const std::size_t block_begin = batch_begin * blocks_per_chunk;
    const std::size_t block_end = block_begin + batch_count * blocks_per_chunk;
    std::vector<std::int64_t> block_ids;
    block_ids.reserve(block_end - block_begin);
    for (std::size_t i = block_begin; i < block_end; ++i) {
      block_ids.push_back(static_cast<std::int64_t>(request.gpu_block_ids[i]));
    }

    std::vector<std::string> device_keys;
    if (request.enable_gpu_hot_cache) {
      std::vector<void*> device_object_ptrs;
      device_object_ptrs.reserve(batch_count);
      device_keys.reserve(batch_count);
      bool device_cache_ready = true;
      std::string device_cache_error;
      for (std::size_t i = 0; i < batch_count; ++i) {
        const std::string& key = request.object_keys[batch_begin + i];
        void* ptr = nullptr;
        if (!EnsureCudaDeviceChunk(key, chunk_bytes, &ptr,
                                   &device_cache_error)) {
          device_cache_ready = false;
          break;
        }
        device_object_ptrs.push_back(ptr);
        device_keys.push_back(key);
      }
      if (!device_cache_ready) {
        EraseCudaDeviceChunks(device_keys);
      } else {
        auto device_phase_start = Clock::now();
        if (!NativeCudaBlockTransfer(paged_ptrs, device_object_ptrs, block_ids,
                                     false, shape_desc,
                                     static_cast<int>(request.chunk_size),
                                     *gpu_kv_format, 0, error)) {
          stats->copy_us += ElapsedUs(device_phase_start);
          std::vector<std::string> rollback = written;
          rollback.insert(rollback.end(), device_keys.begin(),
                          device_keys.end());
          EraseCudaDeviceChunks(rollback);
          for (const std::string& key : written) {
            (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
          }
          return false;
        }
        stats->copy_us += ElapsedUs(device_phase_start);
        stats->bytes += static_cast<std::uint64_t>(batch_count) * chunk_bytes;
        ++stats->cuda_kernel_calls;
      }
    }

    auto phase_start = Clock::now();
    if (!NativeCudaBlockTransfer(paged_ptrs, object_ptrs, block_ids, false,
                                 shape_desc,
                                 static_cast<int>(request.chunk_size),
                                 *gpu_kv_format, 0, error)) {
      stats->copy_us += ElapsedUs(phase_start);
      for (const std::string& key : written) {
        (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
      }
      return false;
    }
    stats->copy_us += ElapsedUs(phase_start);
    stats->bytes += static_cast<std::uint64_t>(batch_count) * chunk_bytes;
    ++stats->cuda_kernel_calls;

    phase_start = Clock::now();
    for (std::size_t i = 0; i < batch_count; ++i) {
      const std::string& key = request.object_keys[batch_begin + i];
      if (lmcache_mp_cpp_cache_put(cache, key.c_str(), chunks[i]->data(),
                                   chunks[i]->size()) != 1) {
        stats->cache_us += ElapsedUs(phase_start);
        const char* cache_error = lmcache_mp_cpp_cache_last_error(cache);
        *error = std::string("native cache put failed during CUDA store: ") +
                 (cache_error == nullptr ? "" : cache_error);
        std::vector<std::string> rollback = written;
        rollback.insert(rollback.end(), device_keys.begin(), device_keys.end());
        EraseCudaDeviceChunks(rollback);
        for (const std::string& written_key : written) {
          (void)lmcache_mp_cpp_cache_remove(cache, written_key.c_str());
        }
        return false;
      }
      written.push_back(key);
    }
    stats->cache_us += ElapsedUs(phase_start);
  }

  *handled = true;
  return true;
}

bool TryRetrieveWithNativeKernel(const KvTransferRequest& request,
                                 const std::vector<OpenTensor>& tensors,
                                 const std::vector<TransferGroup>& groups,
                                 std::uint64_t blocks_per_chunk,
                                 std::uint64_t chunk_bytes,
                                 LmcacheMpCppCache* cache,
                                 KvTransferStats* stats, std::string* error,
                                 bool* handled, void* cuda_stream,
                                 bool synchronize) {
  *handled = false;
  if (request.skip_first_n_tokens != 0 ||
      !SupportsNativeKernelFastPath(groups) ||
      request.gpu_block_ids.size() !=
          request.object_keys.size() * blocks_per_chunk) {
    return true;
  }

  const TransferGroup& group = groups.front();
  const std::optional<int> gpu_kv_format =
      NativeKernelGpuKvFormat(group.layout.layout);
  const std::vector<void*> paged_ptrs = PagedBufferPtrsForGroup(tensors, group);
  const NativePageBufferShapeDesc shape_desc = NativeShapeDesc(group.layout);
  bool device_cache_available = request.enable_gpu_hot_cache;
  if (device_cache_available) {
    for (const std::string& key : request.object_keys) {
      void* ptr = nullptr;
      if (!FindCudaDeviceChunk(key, chunk_bytes, &ptr)) {
        device_cache_available = false;
        break;
      }
    }
  }
  if (device_cache_available) {
    for (std::size_t batch_begin = 0; batch_begin < request.object_keys.size();
         batch_begin += 4) {
      const std::size_t batch_count =
          std::min<std::size_t>(4, request.object_keys.size() - batch_begin);
      std::vector<void*> object_ptrs;
      object_ptrs.reserve(batch_count);
      for (std::size_t i = 0; i < batch_count; ++i) {
        void* ptr = nullptr;
        (void)FindCudaDeviceChunk(request.object_keys[batch_begin + i],
                                  chunk_bytes, &ptr);
        object_ptrs.push_back(ptr);
      }

      const std::size_t block_begin = batch_begin * blocks_per_chunk;
      const std::size_t block_end =
          block_begin + batch_count * blocks_per_chunk;
      std::vector<std::int64_t> block_ids;
      block_ids.reserve(block_end - block_begin);
      for (std::size_t i = block_begin; i < block_end; ++i) {
        block_ids.push_back(
            static_cast<std::int64_t>(request.gpu_block_ids[i]));
      }

      auto phase_start = Clock::now();
      if (!NativeCudaBlockTransferWithStream(
              paged_ptrs, object_ptrs, block_ids, true, shape_desc,
              static_cast<int>(request.chunk_size), *gpu_kv_format, 0,
              cuda_stream, synchronize, error)) {
        stats->copy_us += ElapsedUs(phase_start);
        return false;
      }
      stats->copy_us += ElapsedUs(phase_start);
      stats->bytes += static_cast<std::uint64_t>(batch_count) * chunk_bytes;
      ++stats->cuda_kernel_calls;
    }
    *handled = true;
    return true;
  }

  for (std::size_t batch_begin = 0; batch_begin < request.object_keys.size();
       batch_begin += 4) {
    const std::size_t batch_count =
        std::min<std::size_t>(4, request.object_keys.size() - batch_begin);
    std::vector<std::unique_ptr<HostChunkBuffer>> chunks;
    std::vector<void*> object_ptrs;
    chunks.reserve(batch_count);
    object_ptrs.reserve(batch_count);
    for (std::size_t i = 0; i < batch_count; ++i) {
      auto chunk = std::make_unique<HostChunkBuffer>(chunk_bytes);
      if (!chunk->device_accessible()) {
        return true;
      }
      object_ptrs.push_back(chunk->device_data());
      chunks.push_back(std::move(chunk));
    }

    auto phase_start = Clock::now();
    for (std::size_t i = 0; i < batch_count; ++i) {
      const std::string& key = request.object_keys[batch_begin + i];
      if (lmcache_mp_cpp_cache_get(cache, key.c_str(), chunks[i]->data(),
                                   chunks[i]->size()) != 1) {
        stats->cache_us += ElapsedUs(phase_start);
        const char* cache_error = lmcache_mp_cpp_cache_last_error(cache);
        *error = std::string("native cache read failed during CUDA retrieve: ") +
                 (cache_error == nullptr ? "" : cache_error);
        return false;
      }
    }
    stats->cache_us += ElapsedUs(phase_start);

    const std::size_t block_begin = batch_begin * blocks_per_chunk;
    const std::size_t block_end = block_begin + batch_count * blocks_per_chunk;
    std::vector<std::int64_t> block_ids;
    block_ids.reserve(block_end - block_begin);
    for (std::size_t i = block_begin; i < block_end; ++i) {
      block_ids.push_back(static_cast<std::int64_t>(request.gpu_block_ids[i]));
    }

    phase_start = Clock::now();
    if (!NativeCudaBlockTransfer(paged_ptrs, object_ptrs, block_ids, true,
                                 shape_desc,
                                 static_cast<int>(request.chunk_size),
                                 *gpu_kv_format, 0, error)) {
      stats->copy_us += ElapsedUs(phase_start);
      return false;
    }
    stats->copy_us += ElapsedUs(phase_start);
    stats->bytes += static_cast<std::uint64_t>(batch_count) * chunk_bytes;
    ++stats->cuda_kernel_calls;
  }

  *handled = true;
  return true;
}

bool CanRetrieveWithNativeKernelHotCache(
    const KvTransferRequest& request, const std::vector<TransferGroup>& groups,
    std::uint64_t blocks_per_chunk, std::uint64_t chunk_bytes) {
  if (!request.enable_gpu_hot_cache || request.skip_first_n_tokens != 0 ||
      !SupportsNativeKernelFastPath(groups) ||
      request.gpu_block_ids.size() !=
          request.object_keys.size() * blocks_per_chunk) {
    return false;
  }
  for (const std::string& key : request.object_keys) {
    void* ptr = nullptr;
    if (!FindCudaDeviceChunk(key, chunk_bytes, &ptr)) {
      return false;
    }
  }
  return true;
}

KvTransferResult StoreHotCacheAsyncWithCuda(const KvTransferRequest& request) {
  KvTransferStats stats;
  std::string error;
  std::vector<LayoutInfo> layouts;
  std::vector<TransferGroup> groups;
  std::uint64_t chunk_bytes = 0;
  if (!ValidateTransferLayouts(request, &layouts, &groups, &chunk_bytes,
                               &error)) {
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }

  const std::uint64_t blocks_per_chunk =
      request.chunk_size / request.logical_block_size;
  if (!SupportsNativeKernelFastPath(groups) ||
      request.gpu_block_ids.size() !=
          request.object_keys.size() * blocks_per_chunk) {
    auto result =
        ErrorResult("native CUDA async hot-cache STORE requires kernel layout");
    result.stats = stats;
    return result;
  }

  const TransferGroup& group = groups.front();
  const std::optional<int> gpu_kv_format =
      NativeKernelGpuKvFormat(group.layout.layout);
  if (!gpu_kv_format) {
    auto result =
        ErrorResult("native CUDA async hot-cache STORE has unsupported layout");
    result.stats = stats;
    return result;
  }

  std::vector<OpenTensor> tensors;
  auto phase_start = Clock::now();
  if (!OpenTensors(request, layouts, &tensors, &error)) {
    stats.open_tensors_us += ElapsedUs(phase_start);
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.open_tensors_us += ElapsedUs(phase_start);

  void* stream = CudaTransferStream(&error);
  if (stream == nullptr) {
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }

  phase_start = Clock::now();
  if (!WaitForCudaInputEventOnStream(request.input_event_handle, stream,
                                     &error)) {
    stats.wait_event_us += ElapsedUs(phase_start);
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.wait_event_us += ElapsedUs(phase_start);

  const std::vector<void*> paged_ptrs = PagedBufferPtrsForGroup(tensors, group);
  const NativePageBufferShapeDesc shape_desc = NativeShapeDesc(group.layout);
  std::vector<std::string> staged_keys;
  staged_keys.reserve(request.object_keys.size());

  for (std::size_t batch_begin = 0; batch_begin < request.object_keys.size();
       batch_begin += 4) {
    const std::size_t batch_count =
        std::min<std::size_t>(4, request.object_keys.size() - batch_begin);
    std::vector<void*> device_object_ptrs;
    std::vector<std::string> device_keys;
    device_object_ptrs.reserve(batch_count);
    device_keys.reserve(batch_count);
    for (std::size_t i = 0; i < batch_count; ++i) {
      const std::string& key = request.object_keys[batch_begin + i];
      void* ptr = nullptr;
      if (!EnsureCudaDeviceChunk(key, chunk_bytes, &ptr, &error)) {
        EraseCudaDeviceChunks(device_keys);
        EraseCudaDeviceChunks(staged_keys);
        CloseTensors(&tensors);
        auto result = ErrorResult(error);
        result.stats = stats;
        return result;
      }
      device_object_ptrs.push_back(ptr);
      device_keys.push_back(key);
    }

    const std::size_t block_begin = batch_begin * blocks_per_chunk;
    const std::size_t block_end = block_begin + batch_count * blocks_per_chunk;
    std::vector<std::int64_t> block_ids;
    block_ids.reserve(block_end - block_begin);
    for (std::size_t i = block_begin; i < block_end; ++i) {
      block_ids.push_back(static_cast<std::int64_t>(request.gpu_block_ids[i]));
    }

    phase_start = Clock::now();
    if (!NativeCudaBlockTransferWithStream(
            paged_ptrs, device_object_ptrs, block_ids, false, shape_desc,
            static_cast<int>(request.chunk_size), *gpu_kv_format, 0, stream,
            false, &error)) {
      stats.copy_us += ElapsedUs(phase_start);
      EraseCudaDeviceChunks(device_keys);
      EraseCudaDeviceChunks(staged_keys);
      CloseTensors(&tensors);
      auto result = ErrorResult(error);
      result.stats = stats;
      return result;
    }
    stats.copy_us += ElapsedUs(phase_start);
    stats.bytes += static_cast<std::uint64_t>(batch_count) * chunk_bytes;
    ++stats.cuda_kernel_calls;
    staged_keys.insert(staged_keys.end(), device_keys.begin(),
                       device_keys.end());
  }

  CloseTensors(&tensors);
  phase_start = Clock::now();
  if (!MarkCudaDeviceChunksReadyOnStream(staged_keys, stream, &error)) {
    stats.completion_event_us += ElapsedUs(phase_start);
    EraseCudaDeviceChunks(staged_keys);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  KvTransferResult result = MakeCudaCompletionEventOnStream(stream);
  stats.completion_event_us += ElapsedUs(phase_start);
  if (!result.success) {
    EraseCudaDeviceChunks(staged_keys);
  }
  result.stats = stats;
  return result;
}

KvTransferResult StoreWithCuda(const KvTransferRequest& request,
                               LmcacheMpCppCache* cache) {
  KvTransferStats stats;
  std::string error;
  std::vector<LayoutInfo> layouts;
  std::vector<TransferGroup> groups;
  std::uint64_t chunk_bytes = 0;
  if (!ValidateTransferLayouts(request, &layouts, &groups, &chunk_bytes,
                               &error)) {
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }

  auto phase_start = Clock::now();
  if (!WaitForCudaInputEvent(request.input_event_handle, &error)) {
    stats.wait_event_us += ElapsedUs(phase_start);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.wait_event_us += ElapsedUs(phase_start);

  std::vector<OpenTensor> tensors;
  phase_start = Clock::now();
  if (!OpenTensors(request, layouts, &tensors, &error)) {
    stats.open_tensors_us += ElapsedUs(phase_start);
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.open_tensors_us += ElapsedUs(phase_start);

  const std::uint64_t blocks_per_chunk =
      request.chunk_size / request.logical_block_size;
  bool kernel_handled = false;
  if (!TryStoreWithNativeKernel(request, tensors, groups, blocks_per_chunk,
                                chunk_bytes, cache, &stats, &error,
                                &kernel_handled)) {
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  if (kernel_handled) {
    CloseTensors(&tensors);
    phase_start = Clock::now();
    KvTransferResult result = MakeCudaCompletionEvent();
    stats.completion_event_us += ElapsedUs(phase_start);
    result.stats = stats;
    return result;
  }

  std::vector<std::string> written;
  HostChunkBuffer chunk(chunk_bytes);
  for (std::size_t chunk_idx = 0; chunk_idx < request.object_keys.size();
       ++chunk_idx) {
    phase_start = Clock::now();
    bool copied_with_block_run = false;
    const bool use_block_run =
        ChunkBlockIdsAreContiguous(request, chunk_idx, blocks_per_chunk) &&
        std::all_of(groups.begin(), groups.end(),
                    [](const TransferGroup& group) {
                      return SupportsContiguousBlockRun(group.layout.layout);
                    });
    if (use_block_run) {
      const std::uint64_t first_block_id =
          request.gpu_block_ids[chunk_idx * blocks_per_chunk];
      for (const TransferGroup& group : groups) {
        std::uint64_t group_layer_index = 0;
        for (std::size_t tensor_index : group.tensor_indices) {
          for (std::uint64_t layer_in_tensor = 0;
               layer_in_tensor < tensors[tensor_index].layout.num_layers;
               ++layer_in_tensor) {
            if (!CopyBlockRunD2H(tensors[tensor_index], group, first_block_id,
                                 blocks_per_chunk, group_layer_index,
                                 layer_in_tensor, chunk.data(), &error,
                                 &stats)) {
              stats.copy_us += ElapsedUs(phase_start);
              CloseTensors(&tensors);
              for (const std::string& key : written) {
                (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
              }
              auto result = ErrorResult(error);
              result.stats = stats;
              return result;
            }
            ++group_layer_index;
          }
        }
        if (group_layer_index != group.layout.num_layers) {
          stats.copy_us += ElapsedUs(phase_start);
          CloseTensors(&tensors);
          for (const std::string& key : written) {
            (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
          }
          auto result = ErrorResult(
              "native CUDA transfer group layer count "
              "mismatch during store");
          result.stats = stats;
          return result;
        }
      }
      copied_with_block_run = true;
    }
    if (!copied_with_block_run) {
    for (std::uint64_t block_offset = 0; block_offset < blocks_per_chunk;
         ++block_offset) {
      const std::uint64_t block_id =
          request.gpu_block_ids[chunk_idx * blocks_per_chunk + block_offset];
      for (const TransferGroup& group : groups) {
        if (!IsHndLayout(group.layout.layout)) {
          std::uint64_t group_layer_index = 0;
          for (std::size_t tensor_index : group.tensor_indices) {
            for (std::uint64_t layer_in_tensor = 0;
                 layer_in_tensor < tensors[tensor_index].layout.num_layers;
                 ++layer_in_tensor) {
              if (!CopyBlockD2H(tensors[tensor_index], group, block_id,
                                block_offset, group_layer_index,
                                layer_in_tensor, chunk.data(), &error,
                                &stats)) {
                stats.copy_us += ElapsedUs(phase_start);
                CloseTensors(&tensors);
                for (const std::string& key : written) {
                  (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
                }
                auto result = ErrorResult(error);
                result.stats = stats;
                return result;
              }
              ++group_layer_index;
            }
          }
          if (group_layer_index != group.layout.num_layers) {
            stats.copy_us += ElapsedUs(phase_start);
            CloseTensors(&tensors);
            for (const std::string& key : written) {
              (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
            }
            auto result = ErrorResult(
                "native CUDA transfer group layer count "
                "mismatch during store");
            result.stats = stats;
            return result;
          }
          continue;
        }

        for (std::uint64_t token = 0; token < group.layout.block_size;
             ++token) {
          const std::uint64_t token_in_chunk =
              block_offset * group.layout.block_size + token;
          std::uint64_t group_layer_index = 0;
          for (std::size_t tensor_index : group.tensor_indices) {
            for (std::uint64_t layer_in_tensor = 0;
                 layer_in_tensor < tensors[tensor_index].layout.num_layers;
                 ++layer_in_tensor) {
              if (!CopyTokenD2H(tensors[tensor_index], group, block_id, token,
                                group_layer_index, layer_in_tensor,
                                token_in_chunk, chunk.data(), &error,
                                &stats)) {
                stats.copy_us += ElapsedUs(phase_start);
                CloseTensors(&tensors);
                for (const std::string& key : written) {
                  (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
                }
                auto result = ErrorResult(error);
                result.stats = stats;
                return result;
              }
              ++group_layer_index;
            }
          }
          if (group_layer_index != group.layout.num_layers) {
            stats.copy_us += ElapsedUs(phase_start);
            CloseTensors(&tensors);
            for (const std::string& key : written) {
              (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
            }
            auto result = ErrorResult(
                "native CUDA transfer group layer count "
                "mismatch during store");
            result.stats = stats;
            return result;
          }
        }
      }
    }
    }
    stats.copy_us += ElapsedUs(phase_start);
    phase_start = Clock::now();
    if (lmcache_mp_cpp_cache_put(cache, request.object_keys[chunk_idx].c_str(),
                                 chunk.data(), chunk.size()) != 1) {
      stats.cache_us += ElapsedUs(phase_start);
      const char* cache_error = lmcache_mp_cpp_cache_last_error(cache);
      CloseTensors(&tensors);
      for (const std::string& key : written) {
        (void)lmcache_mp_cpp_cache_remove(cache, key.c_str());
      }
      auto result = ErrorResult(std::string("native cache put failed during "
                                            "CUDA store: ") +
                                (cache_error == nullptr ? "" : cache_error));
      result.stats = stats;
      return result;
    }
    stats.cache_us += ElapsedUs(phase_start);
    written.push_back(request.object_keys[chunk_idx]);
  }

  CloseTensors(&tensors);
  phase_start = Clock::now();
  KvTransferResult result = MakeCudaCompletionEvent();
  stats.completion_event_us += ElapsedUs(phase_start);
  result.stats = stats;
  return result;
}

KvTransferResult RetrieveWithCuda(const KvTransferRequest& request,
                                  LmcacheMpCppCache* cache) {
  KvTransferStats stats;
  std::string error;
  std::vector<LayoutInfo> layouts;
  std::vector<TransferGroup> groups;
  std::uint64_t chunk_bytes = 0;
  if (!ValidateTransferLayouts(request, &layouts, &groups, &chunk_bytes,
                               &error)) {
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  const std::uint64_t blocks_per_chunk =
      request.chunk_size / request.logical_block_size;
  const bool async_hot_cache_retrieve = CanRetrieveWithNativeKernelHotCache(
      request, groups, blocks_per_chunk, chunk_bytes);
  if (async_hot_cache_retrieve) {
    std::vector<OpenTensor> tensors;
    auto phase_start = Clock::now();
    if (!OpenTensors(request, layouts, &tensors, &error)) {
      stats.open_tensors_us += ElapsedUs(phase_start);
      CloseTensors(&tensors);
      auto result = ErrorResult(error);
      result.stats = stats;
      return result;
    }
    stats.open_tensors_us += ElapsedUs(phase_start);

    void* stream = CudaTransferStream(&error);
    if (stream == nullptr) {
      CloseTensors(&tensors);
      auto result = ErrorResult(error);
      result.stats = stats;
      return result;
    }

    phase_start = Clock::now();
    if (!WaitForCudaInputEventOnStream(request.input_event_handle, stream,
                                       &error)) {
      stats.wait_event_us += ElapsedUs(phase_start);
      CloseTensors(&tensors);
      auto result = ErrorResult(error);
      result.stats = stats;
      return result;
    }
    stats.wait_event_us += ElapsedUs(phase_start);

    bool kernel_handled = false;
    if (!TryRetrieveWithNativeKernel(request, tensors, groups, blocks_per_chunk,
                                     chunk_bytes, cache, &stats, &error,
                                     &kernel_handled, stream, false)) {
      CloseTensors(&tensors);
      auto result = ErrorResult(error);
      result.stats = stats;
      return result;
    }
    CloseTensors(&tensors);
    if (!kernel_handled) {
      auto result =
          ErrorResult("native async hot-cache RETRIEVE was not handled");
      result.stats = stats;
      return result;
    }
    phase_start = Clock::now();
    KvTransferResult result = MakeCudaCompletionEventOnStream(stream);
    stats.completion_event_us += ElapsedUs(phase_start);
    result.stats = stats;
    return result;
  }

  auto phase_start = Clock::now();
  if (!WaitForCudaInputEvent(request.input_event_handle, &error)) {
    stats.wait_event_us += ElapsedUs(phase_start);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.wait_event_us += ElapsedUs(phase_start);

  std::vector<OpenTensor> tensors;
  phase_start = Clock::now();
  if (!OpenTensors(request, layouts, &tensors, &error)) {
    stats.open_tensors_us += ElapsedUs(phase_start);
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  stats.open_tensors_us += ElapsedUs(phase_start);

  bool kernel_handled = false;
  if (!TryRetrieveWithNativeKernel(request, tensors, groups, blocks_per_chunk,
                                   chunk_bytes, cache, &stats, &error,
                                   &kernel_handled, nullptr, true)) {
    CloseTensors(&tensors);
    auto result = ErrorResult(error);
    result.stats = stats;
    return result;
  }
  if (kernel_handled) {
    CloseTensors(&tensors);
    phase_start = Clock::now();
    KvTransferResult result = MakeCudaCompletionEvent();
    stats.completion_event_us += ElapsedUs(phase_start);
    result.stats = stats;
    return result;
  }

  HostChunkBuffer chunk(chunk_bytes);
  for (std::size_t chunk_idx = 0; chunk_idx < request.object_keys.size();
       ++chunk_idx) {
    phase_start = Clock::now();
    if (lmcache_mp_cpp_cache_get(cache, request.object_keys[chunk_idx].c_str(),
                                 chunk.data(), chunk.size()) != 1) {
      stats.cache_us += ElapsedUs(phase_start);
      CloseTensors(&tensors);
      const char* cache_error = lmcache_mp_cpp_cache_last_error(cache);
      auto result = ErrorResult(std::string("native cache read failed during "
                                            "CUDA retrieve: ") +
                                (cache_error == nullptr ? "" : cache_error));
      result.stats = stats;
      return result;
    }
    stats.cache_us += ElapsedUs(phase_start);

    phase_start = Clock::now();
    bool copied_with_block_run = false;
    const bool use_block_run =
        request.skip_first_n_tokens == 0 &&
        ChunkBlockIdsAreContiguous(request, chunk_idx, blocks_per_chunk) &&
        std::all_of(groups.begin(), groups.end(),
                    [](const TransferGroup& group) {
                      return SupportsContiguousBlockRun(group.layout.layout);
                    });
    if (use_block_run) {
      const std::uint64_t first_block_id =
          request.gpu_block_ids[chunk_idx * blocks_per_chunk];
      for (const TransferGroup& group : groups) {
        std::uint64_t group_layer_index = 0;
        for (std::size_t tensor_index : group.tensor_indices) {
          for (std::uint64_t layer_in_tensor = 0;
               layer_in_tensor < tensors[tensor_index].layout.num_layers;
               ++layer_in_tensor) {
            if (!CopyBlockRunH2D(chunk.data(), tensors[tensor_index], group,
                                 first_block_id, blocks_per_chunk,
                                 group_layer_index, layer_in_tensor, &error,
                                 &stats)) {
              stats.copy_us += ElapsedUs(phase_start);
              CloseTensors(&tensors);
              auto result = ErrorResult(error);
              result.stats = stats;
              return result;
            }
            ++group_layer_index;
          }
        }
        if (group_layer_index != group.layout.num_layers) {
          stats.copy_us += ElapsedUs(phase_start);
          CloseTensors(&tensors);
          auto result = ErrorResult(
              "native CUDA transfer group layer count "
              "mismatch during retrieve");
          result.stats = stats;
          return result;
        }
      }
      copied_with_block_run = true;
    }
    if (!copied_with_block_run) {
    for (std::uint64_t block_offset = 0; block_offset < blocks_per_chunk;
         ++block_offset) {
      const std::uint64_t block_id =
          request.gpu_block_ids[chunk_idx * blocks_per_chunk + block_offset];
      const std::uint64_t global_block_start =
          chunk_idx * request.chunk_size +
          block_offset * request.logical_block_size;
      if (global_block_start < request.skip_first_n_tokens) {
        continue;
      }
      for (const TransferGroup& group : groups) {
        if (!IsHndLayout(group.layout.layout)) {
          std::uint64_t group_layer_index = 0;
          for (std::size_t tensor_index : group.tensor_indices) {
            for (std::uint64_t layer_in_tensor = 0;
                 layer_in_tensor < tensors[tensor_index].layout.num_layers;
                 ++layer_in_tensor) {
              if (!CopyBlockH2D(chunk.data(), tensors[tensor_index], group,
                                block_id, block_offset, group_layer_index,
                                layer_in_tensor, &error, &stats)) {
                stats.copy_us += ElapsedUs(phase_start);
                CloseTensors(&tensors);
                auto result = ErrorResult(error);
                result.stats = stats;
                return result;
              }
              ++group_layer_index;
            }
          }
          if (group_layer_index != group.layout.num_layers) {
            stats.copy_us += ElapsedUs(phase_start);
            CloseTensors(&tensors);
            auto result = ErrorResult(
                "native CUDA transfer group layer count "
                "mismatch during retrieve");
            result.stats = stats;
            return result;
          }
          continue;
        }

        for (std::uint64_t token = 0; token < group.layout.block_size;
             ++token) {
          const std::uint64_t token_in_chunk =
              block_offset * group.layout.block_size + token;
          std::uint64_t group_layer_index = 0;
          for (std::size_t tensor_index : group.tensor_indices) {
            for (std::uint64_t layer_in_tensor = 0;
                 layer_in_tensor < tensors[tensor_index].layout.num_layers;
                 ++layer_in_tensor) {
              if (!CopyTokenH2D(chunk.data(), tensors[tensor_index], group,
                                block_id, token, group_layer_index,
                                layer_in_tensor, token_in_chunk, &error,
                                &stats)) {
                stats.copy_us += ElapsedUs(phase_start);
                CloseTensors(&tensors);
                auto result = ErrorResult(error);
                result.stats = stats;
                return result;
              }
              ++group_layer_index;
            }
          }
          if (group_layer_index != group.layout.num_layers) {
            stats.copy_us += ElapsedUs(phase_start);
            CloseTensors(&tensors);
            auto result = ErrorResult(
                "native CUDA transfer group layer count "
                "mismatch during retrieve");
            result.stats = stats;
            return result;
          }
        }
      }
    }
    }
    stats.copy_us += ElapsedUs(phase_start);
  }

  CloseTensors(&tensors);
  phase_start = Clock::now();
  KvTransferResult result = MakeCudaCompletionEvent();
  stats.completion_event_us += ElapsedUs(phase_start);
  result.stats = stats;
  return result;
}

#endif  // LMCACHE_ENABLE_CUDA

}  // namespace

bool NativeCudaTransferEnabled() {
#if LMCACHE_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

CudaTransferDeviceCacheStats GetCudaTransferDeviceCacheStats() {
  return GetCudaRuntimeDeviceCacheStats();
}

KvTransferResult StoreKvChunksFromCuda(const KvTransferRequest& request,
                                       LmcacheMpCppCache* cache) {
#if LMCACHE_ENABLE_CUDA
  // CUDA IPC handle open/copy/close and event creation touch process-global
  // CUDA runtime state; serialize them across native worker threads.
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
  return StoreWithCuda(request, cache);
#else
  (void)request;
  (void)cache;
  return ErrorResult(
      "native CUDA transfer support is disabled; rebuild with "
      "-DLMCACHE_ENABLE_CUDA=ON");
#endif
}

KvTransferResult StoreKvChunksToCudaHotCacheAsync(
    const KvTransferRequest& request) {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
  return StoreHotCacheAsyncWithCuda(request);
#else
  (void)request;
  return ErrorResult(
      "native CUDA transfer support is disabled; rebuild with "
      "-DLMCACHE_ENABLE_CUDA=ON");
#endif
}

bool WarmCudaTransferTensorHandles(
    const std::vector<KvTensorMetadata>& tensors, std::string* error) {
  if (tensors.empty()) {
    return true;
  }
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
  for (const KvTensorMetadata& tensor : tensors) {
    void* ptr = nullptr;
    if (!OpenCudaTensorMemory(tensor, &ptr, error)) {
      return false;
    }
  }
  return true;
#else
  (void)tensors;
  if (error != nullptr) {
    *error =
        "native CUDA transfer support is disabled; rebuild with "
        "-DLMCACHE_ENABLE_CUDA=ON";
  }
  return false;
#endif
}

KvTransferResult RetrieveKvChunksToCuda(const KvTransferRequest& request,
                                        LmcacheMpCppCache* cache) {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
  return RetrieveWithCuda(request, cache);
#else
  (void)request;
  (void)cache;
  return ErrorResult(
      "native CUDA transfer support is disabled; rebuild with "
      "-DLMCACHE_ENABLE_CUDA=ON");
#endif
}

KvChecksumResult ChecksumKvCacheBlocksFromCuda(
    const KvChecksumRequest& request) {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
  return ChecksumWithCuda(request);
#else
  (void)request;
  return {.success = false,
          .num_chunks = 0,
          .chunk_checksums = {},
          .layerwise_chunk_checksums = {},
          .error =
              "checksum not supported when native CUDA transfer support "
              "is disabled; rebuild with -DLMCACHE_ENABLE_CUDA=ON"};
#endif
}

void ReleaseCudaTransferEvents() {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
#endif
  ReleaseCudaRuntimeState();
}

bool CudaTransferDeviceChunkReady(const std::string& key) {
  return CudaDeviceChunkReady(key);
}

bool ClearCudaTransferDeviceCache(std::string* error) {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(CudaTransferMutex());
#endif
  return ClearCudaDeviceChunks(error);
}

}  // namespace lmcache::mp
