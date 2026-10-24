// SPDX-License-Identifier: Apache-2.0
// Bounded MUSA arithmetic encoder. CDF is shared; variable-size streams stay
// global.
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAException.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include "cachegen_kernels_musa.h"

struct BitWriter {
  uint8_t* data;
  int stride, capacity, offset = 0, bits = 0;
  uint8_t byte = 0;

  __device__ void put(uint32_t bit) {
    byte = (byte << 1) | bit;
    if (++bits == 8) flush();
  }
  __device__ void flush() {
    if (!bits) return;
    if (offset < capacity) data[offset * stride] = byte << (8 - bits);
    ++offset;
    bits = 0;
    byte = 0;
  }
  __device__ void append(uint32_t bit, uint32_t& pending) {
    put(bit);
    while (pending) {
      put(1 - bit);
      --pending;
    }
  }
};

template <typename C, typename S, typename B, typename L>
__global__ void encode_kernel(C cdf, S symbols, B buffer, L lengths,
                              int* error) {
  __shared__ uint16_t table[48][128];
  int layer = blockIdx.x, channel = blockIdx.y * blockDim.x + threadIdx.x;
  int tid = threadIdx.x, lp = cdf.size(2);
  for (int i = 0; i < lp; ++i) table[i][tid] = cdf[layer][channel][i];
  uint32_t low = 0, high = 0xffffffffU, pending = 0;
  BitWriter writer{&buffer[layer][channel][0], buffer.stride(2),
                   buffer.size(2)};
  for (int i = 0; i < symbols.size(1); ++i) {
    uint8_t sym = symbols[layer][i][channel];
    if (sym >= lp - 1) {
      atomicExch(error, 1);
      lengths[layer][channel] = -1;
      return;
    }
    uint32_t cl = table[sym][tid];
    uint32_t ch = sym == lp - 2 ? 65536U : table[sym + 1][tid];
    if (ch <= cl) {
      atomicExch(error, 1);
      lengths[layer][channel] = -1;
      return;
    }
    uint64_t span = static_cast<uint64_t>(high) - low + 1;
    high = low - 1 + ((span * ch) >> 16);
    low += (span * cl) >> 16;
    while (true) {
      if (high < 0x80000000U)
        writer.append(0, pending);
      else if (low >= 0x80000000U)
        writer.append(1, pending);
      else if (low >= 0x40000000U && high < 0xc0000000U) {
        ++pending;
        low -= 0x40000000U;
        high -= 0x40000000U;
      } else
        break;
      low <<= 1;
      high = (high << 1) | 1;
    }
  }
  ++pending;
  writer.append(low < 0x40000000U ? 0 : 1, pending);
  writer.flush();
  if (writer.offset > writer.capacity) {
    atomicExch(error, 1);
    lengths[layer][channel] = -1;
  } else {
    lengths[layer][channel] = writer.offset;
  }
}

int get_block_size_musa(int nchannels) {
  int factor = (nchannels ^ (nchannels - 1)) + 1;
  return std::min(factor >> 1, 128);
}

void encode_musa_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths) {
  TORCH_CHECK(cdf.is_musa(), "CDF should be on MUSA");
  TORCH_CHECK(input_sym.device() == cdf.device() &&
                  output_buffer.device() == cdf.device() &&
                  output_lengths.device() == cdf.device(),
              "All tensors must be on the same device");
  c10::musa::MUSAGuard guard(cdf.device());
  TORCH_CHECK(cdf.dim() == 3 && input_sym.dim() == 3 &&
                  output_buffer.dim() == 3 && output_lengths.dim() == 2,
              "Invalid tensor ranks");
  int layers = cdf.size(0), channels = cdf.size(1), tokens = input_sym.size(1);
  TORCH_CHECK(layers > 0 && channels > 0 && tokens >= 0 && tokens <= 256 &&
                  cdf.size(2) >= 2 && cdf.size(2) <= 48,
              "Invalid codec dimensions");
  TORCH_CHECK(input_sym.size(0) == layers && input_sym.size(2) == channels &&
                  output_buffer.size(0) == layers &&
                  output_buffer.size(1) == channels &&
                  output_buffer.size(2) > 0 &&
                  output_lengths.size(0) == layers &&
                  output_lengths.size(1) == channels,
              "Incompatible tensor shapes");
  auto error = at::zeros({}, cdf.options().dtype(at::kInt));
  int block = get_block_size_musa(channels);
  auto stream = c10::musa::getCurrentMUSAStream(cdf.get_device()).stream();
  encode_kernel<<<dim3(layers, channels / block), block, 0, stream>>>(
      cdf.packed_accessor32<int16_t, 3, torch::RestrictPtrTraits>(),
      input_sym.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(),
      output_buffer.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
      output_lengths.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
      error.data_ptr<int>());
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  // A recoverable host error, not a device assert that poisons the context.
  TORCH_CHECK(
      error.item<int>() == 0,
      "CacheGen encode: insufficient output capacity or invalid symbol/CDF");
}
