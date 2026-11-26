// SPDX-License-Identifier: Apache-2.0
// Bounded global readers avoid a fixed-size shared bytestream limit.
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAException.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include "cachegen_kernels_musa.h"

struct BitReader {
  const uint8_t* data;
  int64_t length, stride, position = 0;
  __device__ uint32_t next() {
    int64_t byte = position / 8;
    int shift = 7 - position % 8;
    ++position;
    // Arithmetic termination implicitly pads the stream with zero bits.
    return byte < length ? (data[byte * stride] >> shift) & 1 : 0;
  }
};

__device__ uint16_t fast_count(uint32_t value, uint32_t low, uint64_t span) {
  uint64_t a = static_cast<uint64_t>(value) - low + 1;
  uint64_t num = (a << 16) - 1;
  float estimate = static_cast<float>(a) * 65536.0f / static_cast<float>(span);
  if (!(estimate >= 0.0f) || estimate > 65536.0f) return num / span;
  uint32_t q = static_cast<uint32_t>(estimate);
  while (static_cast<uint64_t>(q) * span > num) --q;
  while ((static_cast<uint64_t>(q) + 1) * span <= num) ++q;
  return q;
}

template <bool PREFIX, typename C, typename B, typename L, typename O>
__global__ void decode_kernel(C cdf, B bytes, L lengths, O output, int* error) {
  __shared__ uint16_t table[64][128];
  int layer = blockIdx.x, channel = blockIdx.y * blockDim.x + threadIdx.x;
  int tid = threadIdx.x, lp = cdf.size(2);
  int64_t start = 0, length;
  const uint8_t* data;
  int64_t stride;
  if constexpr (PREFIX) {
    int gid = layer * cdf.size(1) + channel;
    if (gid) start = lengths[(gid - 1) / cdf.size(1)][(gid - 1) % cdf.size(1)];
    int64_t end = lengths[layer][channel];
    if (start < 0 || end < start || end > bytes.size(0)) {
      atomicExch(error, 1);
      return;
    }
    length = end - start;
    data = bytes.data() + start * bytes.stride(0);
    stride = bytes.stride(0);
  } else {
    length = lengths[layer][channel];
    if (length < 0 || length > bytes.size(2)) {
      atomicExch(error, 1);
      return;
    }
    data = bytes.data() + layer * bytes.stride(0) + channel * bytes.stride(1);
    stride = bytes.stride(2);
  }
  for (int i = 0; i < lp; ++i) table[i][tid] = cdf[layer][channel][i];
  BitReader reader{data, length, stride};
  uint32_t low = 0, high = 0xffffffffU, value = 0;
  for (int i = 0; i < 32; ++i) value = (value << 1) | reader.next();
  for (int i = 0; i < output.size(1); ++i) {
    uint64_t span = static_cast<uint64_t>(high) - low + 1;
    uint16_t count = fast_count(value, low, span);
    int left = 0, right = lp - 1;
    while (left + 1 < right) {
      int mid = (left + right) / 2;
      if (table[mid][tid] <= count)
        left = mid;
      else
        right = mid;
    }
    output[layer][i][channel] = left;
    if (i == output.size(1) - 1) break;
    uint32_t cl = table[left][tid];
    uint32_t ch = left == lp - 2 ? 65536U : table[left + 1][tid];
    if (ch <= cl) {
      atomicExch(error, 1);
      return;
    }
    high = low - 1 + ((span * ch) >> 16);
    low += (span * cl) >> 16;
    while (true) {
      if (low >= 0x80000000U || high < 0x80000000U) {
      } else if (low >= 0x40000000U && high < 0xc0000000U) {
        low -= 0x40000000U;
        high -= 0x40000000U;
        value -= 0x40000000U;
      } else
        break;
      low <<= 1;
      high = (high << 1) | 1;
      value = (value << 1) | reader.next();
    }
  }
}

template <bool PREFIX>
void decode_checked(const at::Tensor& cdf, const at::Tensor& bytes,
                    const at::Tensor& lengths, at::Tensor& output) {
  TORCH_CHECK(cdf.is_musa(), "CDF should be on MUSA");
  TORCH_CHECK(bytes.device() == cdf.device() &&
                  lengths.device() == cdf.device() &&
                  output.device() == cdf.device(),
              "All tensors must be on the same device");
  c10::musa::MUSAGuard guard(cdf.device());
  TORCH_CHECK(cdf.dim() == 3 && bytes.dim() == (PREFIX ? 1 : 3) &&
                  lengths.dim() == 2 && output.dim() == 3,
              "Invalid tensor ranks");
  int layers = cdf.size(0), channels = cdf.size(1);
  TORCH_CHECK(layers > 0 && channels > 0 && cdf.size(2) >= 2 &&
                  cdf.size(2) <= 64 && output.size(1) <= 256,
              "Invalid codec dimensions");
  TORCH_CHECK(lengths.size(0) == layers && lengths.size(1) == channels &&
                  output.size(0) == layers && output.size(2) == channels,
              "Incompatible tensor shapes");
  if constexpr (!PREFIX) {
    TORCH_CHECK(bytes.size(0) == layers && bytes.size(1) == channels,
                "Incompatible bytestream shape");
  }
  auto error = at::zeros({}, cdf.options().dtype(at::kInt));
  auto stream = c10::musa::getCurrentMUSAStream(cdf.get_device()).stream();
  int block = get_block_size_musa(channels);
  auto c = cdf.packed_accessor32<int16_t, 3, torch::RestrictPtrTraits>();
  auto o = output.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>();
  if constexpr (PREFIX) {
    decode_kernel<true><<<dim3(layers, channels / block), block, 0, stream>>>(
        c, bytes.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        lengths.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(), o,
        error.data_ptr<int>());
  } else {
    decode_kernel<false><<<dim3(layers, channels / block), block, 0, stream>>>(
        c, bytes.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
        lengths.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), o,
        error.data_ptr<int>());
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(error.item<int>() == 0,
              "CacheGen decode: invalid stream lengths or CDF");
}

void decode_musa_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output) {
  decode_checked<false>(cdf, bytestreams, lengths, output);
}

void decode_musa_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths_prefsum,
                         at::Tensor& output) {
  decode_checked<true>(cdf, bytestreams, lengths_prefsum, output);
}
