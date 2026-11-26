// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 8-bit float KV element types (vLLM --kv-cache-dtype fp8).
//
// SYCL has no native fp8 type, so these are one-byte wrappers that only convert
// to/from float: the kernels read with static_cast<float>(x) and write with
// static_cast<KVType>(f), exactly as for sycl::half / bfloat16. Encoding rounds
// to nearest even and saturates to the largest finite value (never produces
// inf/NaN from a finite input), so a dequantized value slightly above the
// format maximum stays representable. Layouts match torch.float8_e4m3fn and
// torch.float8_e5m2.
// ---------------------------------------------------------------------------
namespace gha_fp8 {

template <int EB, int MB, int BIAS, uint8_t MAX_BITS, bool HAS_INF>
inline float decode(uint8_t b) {
  const int e = (b >> MB) & ((1 << EB) - 1);
  const int m = b & ((1 << MB) - 1);
  float v;
  if (e == (1 << EB) - 1) {
    // e5m2 (HAS_INF): exp all-ones is inf (m == 0) or NaN; e4m3fn: only
    // 0x7F/0xFF are NaN.
    if (HAS_INF) {
      v = (m == 0) ? std::numeric_limits<float>::infinity()
                   : std::numeric_limits<float>::quiet_NaN();
    } else if (m == (1 << MB) - 1) {
      v = std::numeric_limits<float>::quiet_NaN();
    } else {
      v = sycl::ldexp(static_cast<float>(m | (1 << MB)), e - BIAS - MB);
    }
  } else if (e == 0) {
    v = sycl::ldexp(static_cast<float>(m), 1 - BIAS - MB);
  } else {
    v = sycl::ldexp(static_cast<float>(m | (1 << MB)), e - BIAS - MB);
  }
  return (b & 0x80) ? -v : v;
}

template <int EB, int MB, int BIAS, uint8_t MAX_BITS>
inline uint8_t encode(float f) {
  if (f != f) return 0x7F;
  const uint8_t sign = f < 0.0f ? 0x80 : 0x00;
  const float a = sycl::fabs(f);
  // Largest finite magnitude: exponent/mantissa bits of MAX_BITS.
  const float max_val = decode<EB, MB, BIAS, MAX_BITS, (EB == 5)>(MAX_BITS);
  if (a >= max_val) return static_cast<uint8_t>(sign | MAX_BITS);
  if (a == 0.0f) return sign;
  int e;
  (void)sycl::frexp(
      a, &e);  // a = frac * 2^e, frac in [0.5, 1): unbiased exponent is e - 1
  e -= 1;
  uint32_t bits;
  if (e < 1 - BIAS) {
    // Subnormal range: fixed step 2^(1 - BIAS - MB); a carry to 1<<MB is the
    // smallest normal.
    bits = static_cast<uint32_t>(sycl::rint(sycl::ldexp(a, MB + BIAS - 1)));
  } else {
    const uint32_t q = static_cast<uint32_t>(
        sycl::rint(sycl::ldexp(a, MB - e)));  // [2^MB, 2^(MB+1)]
    bits = (static_cast<uint32_t>(e + BIAS) << MB) +
           (q - (1u << MB));  // a carry bumps the exponent
  }
  if (bits > MAX_BITS) bits = MAX_BITS;
  return static_cast<uint8_t>(sign | bits);
}

}  // namespace gha_fp8

// torch.float8_e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa, no inf, max
// 448.
struct fp8_e4m3_t {
  uint8_t bits;
  fp8_e4m3_t() = default;
  explicit fp8_e4m3_t(float f) : bits(gha_fp8::encode<4, 3, 7, 0x7E>(f)) {}
  explicit operator float() const {
    return gha_fp8::decode<4, 3, 7, 0x7E, false>(bits);
  }
};

// torch.float8_e5m2: 1 sign, 5 exponent (bias 15), 2 mantissa, IEEE-like, max
// 57344.
struct fp8_e5m2_t {
  uint8_t bits;
  fp8_e5m2_t() = default;
  explicit fp8_e5m2_t(float f) : bits(gha_fp8::encode<5, 2, 15, 0x7B>(f)) {}
  explicit operator float() const {
    return gha_fp8::decode<5, 2, 15, 0x7B, true>(bits);
  }
};
static_assert(sizeof(fp8_e4m3_t) == 1 && sizeof(fp8_e5m2_t) == 1,
              "fp8 types must be one byte");

struct sycl_quantization_args {
  int qbit;
  int blocks_num;
  int block_size;
  int head_num;
  int head_dim;
  bool rh;
  bool asym;
  std::string scaling_method;

  // Number of layers stacked contiguously in src/dst (element_count spans
  // all of them). Each layer is quantized independently with its own
  // scale/zero-point. Default 1.
  int64_t num_layers = 1;

  // > 0: src is fused [num_layers / fused_src_head_num, tokens,
  // fused_src_head_num * head_dim] and is read per (layer, head) unit in
  // place; dst/scales keep the split [num_layers, tokens, head_dim] layout.
  // Quantize only; per_channel/per_token only.
  int64_t fused_src_head_num = 0;

  // Dequantize counterpart: > 0 writes dst as fused [num_layers /
  // fused_dst_head_num, tokens, fused_dst_head_num * head_dim] while src and
  // scales keep the split layout. per_channel/per_token only.
  int64_t fused_dst_head_num = 0;

  // Preconditioning: forward permutation P and sign-flip diagonal D for
  // H·D·P·x, used as-is (gather) by the global-memory path.
  const int32_t* perm = nullptr;  // device pointer, length = chunk_size
  const float* signs = nullptr;   // device pointer, length = chunk_size

  // inv_perm = argsort(perm), signs_reordered = signs[inv_perm]; used by the
  // SLM path to read/write src/dst in coalesced order and permute only into
  // SLM.
  const int32_t* inv_perm = nullptr;  // device pointer, length = chunk_size
  const float* signs_reordered =
      nullptr;  // device pointer, length = chunk_size
};

// Quantize KV cache data on XPU.
// src: device pointer to fp16/bf16/fp32 input, element_count elements
// dst: device pointer to int8 output (packed int4 for qbit=4)
// scales_out: device pointer to float[num_chunks] (written by kernel)
// zps_out: device pointer to float[num_chunks] (written by kernel, zero if
// symmetric) Returns event for synchronization.
template <typename KVType, typename QType>
sycl::event quantize_kvcache_sycl(sycl::queue& q, const KVType* src, QType* dst,
                                  float* scales_out, float* zps_out,
                                  int64_t element_count,
                                  const sycl_quantization_args& args,
                                  const std::vector<sycl::event>& deps = {});

template <typename KVType, typename QType>
sycl::event dequantize_kvcache_sycl(sycl::queue& q, const QType* src,
                                    KVType* dst, const float* scales_in,
                                    const float* zps_in, int64_t element_count,
                                    const sycl_quantization_args& args,
                                    const std::vector<sycl::event>& deps = {});

// 4-bit packing: pack int8 logical values into packed nibbles on device.
sycl::event pack_int4_sycl(sycl::queue& q, const int8_t* logical,
                           int8_t* packed, int64_t logical_count,
                           const std::vector<sycl::event>& deps = {});

// 4-bit unpacking: unpack packed nibbles into int8 logical values on device.
sycl::event unpack_int4_sycl(sycl::queue& q, const int8_t* packed,
                             int8_t* logical, int64_t logical_count,
                             const std::vector<sycl::event>& deps = {});
