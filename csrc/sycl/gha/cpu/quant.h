// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <immintrin.h>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <string>

#ifdef USE_MULTITHREADING
  #include <omp.h>
#endif

// ---------------------------------------------------------------------------
// 8-bit float KV element types (vLLM --kv-cache-dtype fp8).
//
// One-byte wrappers that only convert to/from float, so every kernel that reads
// with static_cast<float>(x) and writes with static_cast<kvtype>(f) works
// unchanged. Decoding is a 256-entry table lookup; encoding rounds to nearest
// even and saturates to the largest finite value (never inf/NaN from a finite
// input). Layouts match torch.float8_e4m3fn and torch.float8_e5m2, and the
// results match the XPU kernels.
// ---------------------------------------------------------------------------
namespace gha_fp8 {

template <int EB, int MB, int BIAS, bool HAS_INF>
inline float decode_scalar(uint8_t b) {
  const int e = (b >> MB) & ((1 << EB) - 1);
  const int m = b & ((1 << MB) - 1);
  float v;
  if (e == (1 << EB) - 1 && (HAS_INF || m == (1 << MB) - 1)) {
    // e5m2: exponent all-ones is inf (m == 0) or NaN; e4m3fn: only mantissa
    // all-ones is NaN.
    v = (HAS_INF && m == 0) ? std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::quiet_NaN();
  } else if (e == 0) {
    v = std::ldexp(static_cast<float>(m), 1 - BIAS - MB);
  } else {
    v = std::ldexp(static_cast<float>(m | (1 << MB)), e - BIAS - MB);
  }
  return (b & 0x80) ? -v : v;
}

template <int EB, int MB, int BIAS, uint8_t MAX_BITS, bool HAS_INF>
inline uint8_t encode_scalar(float f) {
  if (f != f) return 0x7F;
  const uint8_t sign = std::signbit(f) ? 0x80 : 0x00;
  const float a = std::fabs(f);
  if (a >= decode_scalar<EB, MB, BIAS, HAS_INF>(MAX_BITS)) {
    return static_cast<uint8_t>(sign | MAX_BITS);
  }
  if (a == 0.0f) return sign;
  int e;
  (void)std::frexp(
      a, &e);  // a = frac * 2^e, frac in [0.5, 1): unbiased exponent is e - 1
  e -= 1;
  uint32_t bits;
  if (e < 1 - BIAS) {
    // Subnormal range: fixed step 2^(1 - BIAS - MB); a carry to 1<<MB is the
    // smallest normal.
    bits = static_cast<uint32_t>(std::nearbyint(std::ldexp(a, MB + BIAS - 1)));
  } else {
    const uint32_t q = static_cast<uint32_t>(
        std::nearbyint(std::ldexp(a, MB - e)));  // [2^MB, 2^(MB+1)]
    bits = (static_cast<uint32_t>(e + BIAS) << MB) +
           (q - (1u << MB));  // a carry bumps the exponent
  }
  if (bits > MAX_BITS) bits = MAX_BITS;
  return static_cast<uint8_t>(sign | bits);
}

template <int EB, int MB, int BIAS, bool HAS_INF>
inline const float* decode_table() {
  static const std::vector<float> table = [] {
    std::vector<float> t(256);
    for (int i = 0; i < 256; ++i)
      t[static_cast<size_t>(i)] =
          decode_scalar<EB, MB, BIAS, HAS_INF>(static_cast<uint8_t>(i));
    return t;
  }();
  return table.data();
}

}  // namespace gha_fp8

// torch.float8_e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa, no inf, max
// 448.
struct gha_fp8_e4m3 {
  uint8_t bits;
  gha_fp8_e4m3() = default;
  explicit gha_fp8_e4m3(float f)
      : bits(gha_fp8::encode_scalar<4, 3, 7, 0x7E, false>(f)) {}
  explicit operator float() const {
    return gha_fp8::decode_table<4, 3, 7, false>()[bits];
  }
};

// torch.float8_e5m2: 1 sign, 5 exponent (bias 15), 2 mantissa, IEEE-like, max
// 57344.
struct gha_fp8_e5m2 {
  uint8_t bits;
  gha_fp8_e5m2() = default;
  explicit gha_fp8_e5m2(float f)
      : bits(gha_fp8::encode_scalar<5, 2, 15, 0x7B, true>(f)) {}
  explicit operator float() const {
    return gha_fp8::decode_table<5, 2, 15, true>()[bits];
  }
};
static_assert(sizeof(gha_fp8_e4m3) == 1 && sizeof(gha_fp8_e5m2) == 1,
              "fp8 types must be one byte");

// The hand-vectorized paths below exist only for these element types; every
// other type (the fp8 wrappers) takes the scalar loops.
template <typename T>
inline constexpr bool gha_has_simd_convert_v =
    std::is_same_v<T, float> || std::is_same_v<T, _Float16> ||
    std::is_same_v<T, __bf16>;

void fast_walsh_hadamard_transform(float* x, int64_t n);

void fast_walsh_hadamard_transform_fused(const float* in, float* out, int64_t n,
                                         const int32_t* perm,
                                         const float* signs);

void fast_walsh_hadamard_inverse_fused(const float* in, float* out, int64_t n,
                                       const int32_t* perm, const float* signs);

// Reorder [layers, tokens, head_num*head_dim] to [layers*head_num, tokens,
// head_dim] (OpenMP strided copy).
template <typename T>
void split_fused_attention_heads_native(const T* src, T* dst, int64_t layers,
                                        int64_t tokens, int64_t head_num,
                                        int64_t head_dim, int num_threads) {
  int64_t hidden = head_num * head_dim;
  int64_t total_units = layers * head_num;
#ifdef USE_MULTITHREADING
  int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
  #pragma omp parallel for schedule(static) \
      num_threads(nthreads) if (nthreads > 1 && total_units > 1)
#endif
  for (int64_t nl = 0; nl < total_units; ++nl) {
    int64_t l = nl / head_num;
    int64_t h = nl % head_num;
    const T* src_unit = src + l * tokens * hidden + h * head_dim;
    T* dst_unit = dst + nl * tokens * head_dim;
    for (int64_t t = 0; t < tokens; ++t) {
      std::memcpy(dst_unit + t * head_dim, src_unit + t * hidden,
                  static_cast<size_t>(head_dim) * sizeof(T));
    }
  }
}

// Inverse of split_fused_attention_heads_native: merges
// [layers*head_num, tokens, head_dim] back into [layers, tokens,
// head_num*head_dim].
template <typename T>
void merge_fused_attention_heads_native(const T* src, T* dst, int64_t layers,
                                        int64_t tokens, int64_t head_num,
                                        int64_t head_dim, int num_threads) {
  int64_t hidden = head_num * head_dim;
  int64_t total_units = layers * head_num;
#ifdef USE_MULTITHREADING
  int nthreads = (num_threads > 0) ? num_threads : omp_get_max_threads();
  #pragma omp parallel for schedule(static) \
      num_threads(nthreads) if (nthreads > 1 && total_units > 1)
#endif
  for (int64_t nl = 0; nl < total_units; ++nl) {
    int64_t l = nl / head_num;
    int64_t h = nl % head_num;
    const T* src_unit = src + nl * tokens * head_dim;
    T* dst_unit = dst + l * tokens * hidden + h * head_dim;
    for (int64_t t = 0; t < tokens; ++t) {
      std::memcpy(dst_unit + t * hidden, src_unit + t * head_dim,
                  static_cast<size_t>(head_dim) * sizeof(T));
    }
  }
}

struct quantization_args {
  // Per-chunk scale / zero point: quantize writes chunk i's to scales_out[i] /
  // zps_out[i]; dequantize reads them from scales_in[i] / zps_in[i].
  float* scales_out = nullptr;
  float* zps_out = nullptr;
  const float* scales_in = nullptr;
  const float* zps_in = nullptr;
  // 4-bit per_token with an even row length: quantize writes / dequantize reads
  // packed nibbles row by row (the same bytes pack_buffer produces for the
  // whole buffer), so no element-count sized int8 scratch is needed.
  bool pack4_rows = false;

  int qbit;
  int blocks_num;
  int block_size;
  int head_num;
  int head_dim;
  bool rh;
  bool asym;
  std::string scaling_method;
  int num_threads = 0;  // 0 = use OMP_NUM_THREADS / all available cores

  // Preconditioning: perm (P), signs (D) for H·D·P·x transform
  const int32_t* perm = nullptr;  // permutation indices, length = chunk_size
  const float* signs = nullptr;  // sign-flip diagonal (±1), length = chunk_size
};

namespace q4_detail {

inline int64_t packed_size(int64_t logical_count) {
  return (logical_count + 1) / 2;
}

inline void pack_value(int8_t* dst, int64_t logical_idx, int8_t value) {
  int64_t byte_idx = logical_idx / 2;
  uint8_t nibble = static_cast<uint8_t>(value) & 0x0F;
  uint8_t byte = static_cast<uint8_t>(dst[byte_idx]);
  if ((logical_idx & 1) == 0) {
    byte = static_cast<uint8_t>((byte & 0xF0) | nibble);
  } else {
    byte = static_cast<uint8_t>((byte & 0x0F) | (nibble << 4));
  }
  dst[byte_idx] = static_cast<int8_t>(byte);
}

inline int8_t unpack_value(const int8_t* src, int64_t logical_idx) {
  uint8_t byte = static_cast<uint8_t>(src[logical_idx / 2]);
  uint8_t nibble =
      ((logical_idx & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
  return static_cast<int8_t>((nibble >= 8) ? (static_cast<int>(nibble) - 16)
                                           : static_cast<int>(nibble));
}

#if defined(__SSSE3__)
inline __m128i pack_16_logical_sse(__m128i values) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i even_idx =
      _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i odd_idx =
      _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
  __m128i nibbles = _mm_and_si128(values, mask);
  __m128i even = _mm_shuffle_epi8(nibbles, even_idx);
  __m128i odd = _mm_slli_epi16(_mm_shuffle_epi8(nibbles, odd_idx), 4);
  return _mm_or_si128(even, odd);
}
#endif

inline __m128i unpack_8_packed_sse(__m128i packed) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i sign_bit = _mm_set1_epi8(0x08);
  __m128i low = _mm_and_si128(packed, mask);
  __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
  __m128i interleaved = _mm_unpacklo_epi8(low, high);
  return _mm_sub_epi8(_mm_xor_si128(interleaved, sign_bit), sign_bit);
}

inline __m128i unpack_16_low_packed_sse(__m128i packed) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i sign_bit = _mm_set1_epi8(0x08);
  __m128i low = _mm_and_si128(packed, mask);
  __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
  __m128i interleaved = _mm_unpacklo_epi8(low, high);
  return _mm_sub_epi8(_mm_xor_si128(interleaved, sign_bit), sign_bit);
}

inline __m128i unpack_16_high_packed_sse(__m128i packed) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i sign_bit = _mm_set1_epi8(0x08);
  __m128i low = _mm_and_si128(packed, mask);
  __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
  __m128i interleaved = _mm_unpackhi_epi8(low, high);
  return _mm_sub_epi8(_mm_xor_si128(interleaved, sign_bit), sign_bit);
}

inline void pack_buffer(const int8_t* logical, int8_t* packed,
                        int64_t logical_count) {
  int64_t i = 0;
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
  const __m512i mask512 = _mm512_set1_epi8(0x0F);
  const __m512i even_idx512 = _mm512_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1,
      -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1,
      14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8,
      6, 4, 2, 0);
  const __m512i odd_idx512 = _mm512_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1, -1, -1, -1, -1,
      -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1,
      15, 13, 11, 9, 7, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9,
      7, 5, 3, 1);
  for (; i + 64 <= logical_count; i += 64) {
    __m512i values =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(logical + i));
    __m512i nibbles = _mm512_and_si512(values, mask512);
    __m512i even = _mm512_shuffle_epi8(nibbles, even_idx512);
    __m512i odd =
        _mm512_slli_epi16(_mm512_shuffle_epi8(nibbles, odd_idx512), 4);
    __m512i packed_lanes = _mm512_or_si512(even, odd);
    __m128i lane0 = _mm512_extracti32x4_epi32(packed_lanes, 0);
    __m128i lane1 = _mm512_extracti32x4_epi32(packed_lanes, 1);
    __m128i lane2 = _mm512_extracti32x4_epi32(packed_lanes, 2);
    __m128i lane3 = _mm512_extracti32x4_epi32(packed_lanes, 3);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(packed + i / 2), lane0);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(packed + i / 2 + 8), lane1);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(packed + i / 2 + 16), lane2);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(packed + i / 2 + 24), lane3);
  }
#endif
#if defined(__SSSE3__)
  for (; i + 16 <= logical_count; i += 16) {
    __m128i values =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(logical + i));
    __m128i packed_values = pack_16_logical_sse(values);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(packed + i / 2), packed_values);
  }
#endif
  for (; i + 1 < logical_count; i += 2) {
    uint8_t low = static_cast<uint8_t>(logical[i]) & 0x0F;
    uint8_t high = static_cast<uint8_t>(logical[i + 1]) & 0x0F;
    packed[i / 2] = static_cast<int8_t>(low | (high << 4));
  }
  if (i < logical_count) {
    packed[i / 2] =
        static_cast<int8_t>(static_cast<uint8_t>(logical[i]) & 0x0F);
  }
}

inline void unpack_buffer(const int8_t* packed, int8_t* logical,
                          int64_t logical_count) {
  int64_t i = 0;
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
  for (; i + 64 <= logical_count; i += 64) {
    __m128i p0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + i / 2));
    __m128i p1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + i / 2 + 16));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i),
                     unpack_16_low_packed_sse(p0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i + 16),
                     unpack_16_high_packed_sse(p0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i + 32),
                     unpack_16_low_packed_sse(p1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i + 48),
                     unpack_16_high_packed_sse(p1));
  }
#endif
  for (; i + 32 <= logical_count; i += 32) {
    __m128i packed_values =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + i / 2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i),
                     unpack_16_low_packed_sse(packed_values));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i + 16),
                     unpack_16_high_packed_sse(packed_values));
  }
  for (; i + 16 <= logical_count; i += 16) {
    __m128i packed_values =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + i / 2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(logical + i),
                     unpack_8_packed_sse(packed_values));
  }
  for (; i < logical_count; ++i) {
    logical[i] = unpack_value(packed, i);
  }
}

}  // namespace q4_detail

// ipex: 2(k&v) x [block_num, block_size, head_num, head_dim](src)
// Extremes of v[0..n) with exactly the result of the sequential
// std::min / std::max scan seeded with v[0]. The scan is written to vectorize;
// lane order only changes which signed zero wins a tie, so a zero extreme is
// recomputed sequentially (an all-zero row is then quantized bit-identically).
inline void row_min_max(const float* v, int64_t n, float& mn, float& mx) {
  float a = v[0];
  float b = v[0];
  for (int64_t i = 0; i < n; ++i) {
    a = std::min(a, v[i]);
    b = std::max(b, v[i]);
  }
  if (a == 0.0f || b == 0.0f) {
    a = v[0];
    b = v[0];
#if defined(__clang__)
  #pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (int64_t i = 0; i < n; ++i) {
      a = std::min(a, v[i]);
      b = std::max(b, v[i]);
    }
  }
  mn = a;
  mx = b;
}

// std::round (half away from zero) written so it vectorizes: v - trunc(v) is
// exact, so the result equals std::round for every float.
inline float round_half_away(float v) {
  const float t = std::trunc(v);
  return t + ((std::fabs(v - t) >= 0.5f) ? std::copysign(1.0f, v) : 0.0f);
}

template <typename kvtype, typename qtype>
void quantize_block_rh(kvtype* src, qtype* dst, int64_t element_count,
                       int64_t scale_id, quantization_args& args) {
  int64_t num_chunks = 1;
  if (args.scaling_method == "per_token") {
    num_chunks = args.blocks_num * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    num_chunks = args.head_dim;
  }

  int64_t chunk_size = element_count / num_chunks;

  int64_t qmax_int = (1 << (args.qbit - 1)) - 1;
  int64_t qmin_int = -(1 << (args.qbit - 1));
  float qmax_float = static_cast<float>(qmax_int);
  float qmin_float = static_cast<float>(qmin_int);

  bool is_per_channel = (args.scaling_method == "per_channel");
  int64_t stride = is_per_channel ? args.head_dim : 1;

#ifdef USE_MULTITHREADING
  {
    std::vector<float> p_scales(static_cast<size_t>(num_chunks));
    std::vector<float> p_zps(static_cast<size_t>(num_chunks));
    int nthreads =
        (args.num_threads > 0) ? args.num_threads : omp_get_max_threads();
    // Per-channel: dst writes are strided (stride=head_dim). Cap threads so
    // each thread owns at least one cache line worth of chunks, preventing
    // false sharing.
    if (is_per_channel) {
      int cl_chunks = std::max(1, 64 / static_cast<int>(sizeof(qtype)));
      nthreads = std::min(
          nthreads, std::max(1, static_cast<int>(num_chunks) / cl_chunks));
    }
  #pragma omp parallel num_threads(nthreads) if (nthreads > 1 && num_chunks > 1)
    {
      // Per-thread scratch, reused across chunks.
      std::vector<float> local_buf(static_cast<size_t>(chunk_size));
      std::vector<float> fused_out(static_cast<size_t>(chunk_size));
      std::vector<int8_t> qrow(args.pack4_rows ? chunk_size : 0);
  #pragma omp for schedule(static)
      for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        int64_t offset = is_per_channel ? chunk : chunk * chunk_size;
        if (!is_per_channel) {
          // per_token: the chunk is one contiguous row (vectorizable loops).
          const kvtype* row = src + offset;
          float* buf = local_buf.data();
          for (int64_t i = 0; i < chunk_size; ++i) {
            buf[i] = static_cast<float>(row[i]);
          }
          fast_walsh_hadamard_transform_fused(buf, fused_out.data(), chunk_size,
                                              args.perm, args.signs);
          const float* v = fused_out.data();
          float min_val, max_val;
          row_min_max(v, chunk_size, min_val, max_val);
          float scale = 0.0f, zeropoint = 0.0f;
          if (args.asym) {
            scale = (max_val - min_val) / (qmax_float - qmin_float);
            zeropoint = min_val - qmin_float * scale;
          } else {
            scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
          }
          p_scales[static_cast<size_t>(chunk)] = scale;
          p_zps[static_cast<size_t>(chunk)] = zeropoint;
          const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
          if constexpr (std::is_same_v<qtype, int8_t>) {
            if (args.pack4_rows) {
              for (int64_t i = 0; i < chunk_size; ++i) {
                qrow[i] = static_cast<int8_t>(
                    std::clamp(round_half_away((v[i] - zeropoint) * inv_scale),
                               qmin_float, qmax_float));
              }
              q4_detail::pack_buffer(qrow.data(), dst + offset / 2, chunk_size);
              continue;
            }
          }
          qtype* out = dst + offset;
          for (int64_t i = 0; i < chunk_size; ++i) {
            out[i] = static_cast<qtype>(
                std::clamp(round_half_away((v[i] - zeropoint) * inv_scale),
                           qmin_float, qmax_float));
          }
          continue;
        }
        for (int64_t i = 0; i < chunk_size; ++i)
          local_buf[static_cast<size_t>(i)] =
              static_cast<float>(src[i * stride + offset]);
        fast_walsh_hadamard_transform_fused(local_buf.data(), fused_out.data(),
                                            chunk_size, args.perm, args.signs);
        std::copy(fused_out.begin(), fused_out.end(), local_buf.begin());
        float min_val = local_buf[0], max_val = local_buf[0];
        for (const auto& v : local_buf) {
          min_val = std::min(min_val, v);
          max_val = std::max(max_val, v);
        }
        float scale = 0.0f, zeropoint = 0.0f;
        if (args.asym) {
          scale = (max_val - min_val) / (qmax_float - qmin_float);
          zeropoint = min_val - qmin_float * scale;
        } else {
          scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
        }
        p_scales[static_cast<size_t>(chunk)] = scale;
        p_zps[static_cast<size_t>(chunk)] = zeropoint;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        for (int64_t i = 0; i < chunk_size; ++i) {
          float q_val = std::clamp(
              std::round((local_buf[static_cast<size_t>(i)] - zeropoint) *
                         inv_scale),
              qmin_float, qmax_float);
          dst[i * stride + offset] = static_cast<qtype>(q_val);
        }
      }
    }
    std::copy(p_scales.begin(), p_scales.end(), args.scales_out);
    std::copy(p_zps.begin(), p_zps.end(), args.zps_out);
  }
#else
  std::vector<float> transformed_src(static_cast<size_t>(chunk_size));
  std::vector<float> fused_out(static_cast<size_t>(chunk_size));
  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    int64_t offset = is_per_channel ? chunk : chunk * chunk_size;

    for (int64_t i = 0; i < chunk_size; ++i) {
      transformed_src[static_cast<size_t>(i)] =
          static_cast<float>(src[i * stride + offset]);
    }

    fast_walsh_hadamard_transform_fused(transformed_src.data(),
                                        fused_out.data(), chunk_size, args.perm,
                                        args.signs);
    std::copy(fused_out.begin(), fused_out.end(), transformed_src.begin());

    float min_val = transformed_src[0];
    float max_val = transformed_src[0];
    for (const auto& v : transformed_src) {
      min_val = std::min(min_val, v);
      max_val = std::max(max_val, v);
    }

    float scale = 0.0f;
    float zeropoint = 0.0f;
    if (args.asym) {
      scale = (max_val - min_val) / (qmax_float - qmin_float);
      zeropoint = min_val - qmin_float * scale;
    } else {
      scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
    }

    args.scales_out[chunk] = scale;
    args.zps_out[chunk] = zeropoint;

    float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    for (int64_t i = 0; i < chunk_size; ++i) {
      float q_val = std::clamp(
          std::round((transformed_src[static_cast<size_t>(i)] - zeropoint) *
                     inv_scale),
          qmin_float, qmax_float);
      dst[i * stride + offset] = static_cast<qtype>(q_val);
    }
  }
#endif
}

template <typename kvtype, typename qtype>
void quantize_block(kvtype* src, qtype* dst, int64_t element_count,
                    int64_t scale_id, quantization_args& args) {
  int64_t num_chunks = 1;
  if (args.scaling_method == "per_token") {
    num_chunks = args.blocks_num * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    num_chunks = args.head_dim;
  }

  int64_t chunk_size = element_count / num_chunks;

  int64_t qmax_int = (1 << (args.qbit - 1)) - 1;
  int64_t qmin_int = -(1 << (args.qbit - 1));
  float qmax_float = static_cast<float>(qmax_int);
  float qmin_float = static_cast<float>(qmin_int);

  bool is_per_channel = (args.scaling_method == "per_channel");
  int64_t stride = is_per_channel ? args.head_dim : 1;

#ifdef USE_MULTITHREADING
  {
    std::vector<float> p_scales(static_cast<size_t>(num_chunks));
    std::vector<float> p_zps(static_cast<size_t>(num_chunks));
    int nthreads =
        (args.num_threads > 0) ? args.num_threads : omp_get_max_threads();
    if (is_per_channel) {
      int cl_chunks = std::max(1, 64 / static_cast<int>(sizeof(qtype)));
      nthreads = std::min(
          nthreads, std::max(1, static_cast<int>(num_chunks) / cl_chunks));
    }
  #pragma omp parallel for schedule(static) \
      num_threads(nthreads) if (nthreads > 1 && num_chunks > 1)
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t offset = is_per_channel ? chunk : chunk * chunk_size;
      if (!is_per_channel) {
        // per_token: the chunk is one contiguous row (vectorizable loops).
        thread_local std::vector<float> row_buf;
        row_buf.resize(static_cast<size_t>(chunk_size));
        float* v = row_buf.data();
        const kvtype* row = src + offset;
        for (int64_t i = 0; i < chunk_size; ++i) {
          v[i] = static_cast<float>(row[i]);
        }
        float min_val, max_val;
        row_min_max(v, chunk_size, min_val, max_val);
        float scale = 0.0f, zeropoint = 0.0f;
        if (args.asym) {
          scale = (max_val - min_val) / (qmax_float - qmin_float);
          zeropoint = min_val - qmin_float * scale;
        } else {
          scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
        }
        p_scales[static_cast<size_t>(chunk)] = scale;
        p_zps[static_cast<size_t>(chunk)] = zeropoint;
        const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        if constexpr (std::is_same_v<qtype, int8_t>) {
          if (args.pack4_rows) {
            thread_local std::vector<int8_t> qrow;
            qrow.resize(static_cast<size_t>(chunk_size));
            for (int64_t i = 0; i < chunk_size; ++i) {
              qrow[i] = static_cast<int8_t>(
                  std::clamp(round_half_away((v[i] - zeropoint) * inv_scale),
                             qmin_float, qmax_float));
            }
            q4_detail::pack_buffer(qrow.data(), dst + offset / 2, chunk_size);
            continue;
          }
        }
        qtype* out = dst + offset;
        for (int64_t i = 0; i < chunk_size; ++i) {
          out[i] = static_cast<qtype>(
              std::clamp(round_half_away((v[i] - zeropoint) * inv_scale),
                         qmin_float, qmax_float));
        }
        continue;
      }
      float min_val = static_cast<float>(src[offset]);
      float max_val = static_cast<float>(src[offset]);
      for (int64_t i = 0; i < chunk_size; ++i) {
        float v = static_cast<float>(src[i * stride + offset]);
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
      }
      float scale = 0.0f, zeropoint = 0.0f;
      if (args.asym) {
        scale = (max_val - min_val) / (qmax_float - qmin_float);
        zeropoint = min_val - qmin_float * scale;
      } else {
        scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
      }
      p_scales[static_cast<size_t>(chunk)] = scale;
      p_zps[static_cast<size_t>(chunk)] = zeropoint;
      float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
      for (int64_t i = 0; i < chunk_size; ++i) {
        float q_val = std::clamp(
            std::round(
                (static_cast<float>(src[i * stride + offset]) - zeropoint) *
                inv_scale),
            qmin_float, qmax_float);
        dst[i * stride + offset] = static_cast<qtype>(q_val);
      }
    }
    std::copy(p_scales.begin(), p_scales.end(), args.scales_out);
    std::copy(p_zps.begin(), p_zps.end(), args.zps_out);
  }
#else
  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    int64_t offset = is_per_channel ? chunk : chunk * chunk_size;

    float min_val = static_cast<float>(src[offset]);
    float max_val = static_cast<float>(src[offset]);
    for (int64_t i = 0; i < chunk_size; ++i) {
      float v = static_cast<float>(src[i * stride + offset]);
      min_val = std::min(min_val, v);
      max_val = std::max(max_val, v);
    }

    float scale = 0.0f;
    float zeropoint = 0.0f;
    if (args.asym) {
      scale = (max_val - min_val) / (qmax_float - qmin_float);
      zeropoint = min_val - qmin_float * scale;
    } else {
      scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
    }

    args.scales_out[chunk] = scale;
    args.zps_out[chunk] = zeropoint;
    float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    for (int64_t i = 0; i < chunk_size; ++i) {
      float q_val =
          std::clamp(std::round((static_cast<float>(src[i * stride + offset]) -
                                 zeropoint) *
                                inv_scale),
                     qmin_float, qmax_float);
      dst[i * stride + offset] = static_cast<qtype>(q_val);
    }
  }
#endif
}

template <typename kvtype, typename qtype>
void dequantize_block_rh(qtype* src, kvtype* dst, int64_t element_count,
                         int64_t scale_id, quantization_args& args) {
  int64_t num_chunks = 1;
  if (args.scaling_method == "per_token") {
    num_chunks = args.blocks_num * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    num_chunks = args.head_dim;
  }

  int64_t chunk_size = element_count / num_chunks;

  bool is_per_channel = (args.scaling_method == "per_channel");
  int64_t stride = is_per_channel ? args.head_dim : 1;

#ifdef USE_MULTITHREADING
  {
    std::vector<float> p_scales(static_cast<size_t>(num_chunks));
    std::vector<float> p_zps(static_cast<size_t>(num_chunks));
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      p_scales[static_cast<size_t>(chunk)] = args.scales_in[chunk];
      p_zps[static_cast<size_t>(chunk)] = args.asym ? args.zps_in[chunk] : 0.0f;
    }
    int nthreads =
        (args.num_threads > 0) ? args.num_threads : omp_get_max_threads();
    // dst writes are strided for per_channel; cap to avoid false sharing
    if (is_per_channel) {
      int cl_chunks = std::max(1, 64 / static_cast<int>(sizeof(kvtype)));
      nthreads = std::min(
          nthreads, std::max(1, static_cast<int>(num_chunks) / cl_chunks));
    }
  #pragma omp parallel num_threads(nthreads) if (nthreads > 1 && num_chunks > 1)
    {
      // Per-thread scratch, reused across chunks.
      std::vector<float> local_buf(static_cast<size_t>(chunk_size));
      std::vector<float> inv_out(static_cast<size_t>(chunk_size));
      std::vector<int8_t> qrow(args.pack4_rows ? chunk_size : 0);
  #pragma omp for schedule(static)
      for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        int64_t offset = is_per_channel ? chunk : chunk * chunk_size;
        float scale = p_scales[static_cast<size_t>(chunk)];
        float zeropoint = p_zps[static_cast<size_t>(chunk)];
        if (!is_per_channel) {
          // per_token: the chunk is one contiguous row (vectorizable loops).
          const qtype* row = src + offset;
          if constexpr (std::is_same_v<qtype, int8_t>) {
            if (args.pack4_rows) {
              q4_detail::unpack_buffer(src + offset / 2, qrow.data(),
                                       chunk_size);
              row = qrow.data();
            }
          }
          float* buf = local_buf.data();
          for (int64_t i = 0; i < chunk_size; ++i) {
            buf[i] = static_cast<float>(row[i]) * scale + zeropoint;
          }
          fast_walsh_hadamard_inverse_fused(buf, inv_out.data(), chunk_size,
                                            args.perm, args.signs);
          const float* v = inv_out.data();
          kvtype* out = dst + offset;
          for (int64_t i = 0; i < chunk_size; ++i) {
            out[i] = static_cast<kvtype>(v[i]);
          }
          continue;
        }
        for (int64_t i = 0; i < chunk_size; ++i)
          local_buf[static_cast<size_t>(i)] =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
        fast_walsh_hadamard_inverse_fused(local_buf.data(), inv_out.data(),
                                          chunk_size, args.perm, args.signs);
        std::copy(inv_out.begin(), inv_out.end(), local_buf.begin());
        for (int64_t i = 0; i < chunk_size; ++i)
          dst[i * stride + offset] =
              static_cast<kvtype>(local_buf[static_cast<size_t>(i)]);
      }
    }
  }
#else
  std::vector<float> transformed_src(static_cast<size_t>(chunk_size));
  std::vector<float> inv_out(static_cast<size_t>(chunk_size));

  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    int64_t offset = is_per_channel ? chunk : chunk * chunk_size;
    float scale = args.scales_in[chunk];
    float zeropoint = 0.0f;
    if (args.asym) zeropoint = args.zps_in[chunk];

    for (int64_t i = 0; i < chunk_size; ++i) {
      transformed_src[static_cast<size_t>(i)] =
          static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
    }

    fast_walsh_hadamard_inverse_fused(transformed_src.data(), inv_out.data(),
                                      chunk_size, args.perm, args.signs);
    std::copy(inv_out.begin(), inv_out.end(), transformed_src.begin());
    for (int64_t i = 0; i < chunk_size; ++i) {
      dst[i * stride + offset] =
          static_cast<kvtype>(transformed_src[static_cast<size_t>(i)]);
    }
  }
#endif
}

template <typename kvtype, typename qtype>
void dequantize_block(qtype* src, kvtype* dst, int64_t element_count,
                      int64_t scale_id, quantization_args& args) {
  int64_t num_chunks = 1;
  if (args.scaling_method == "per_token") {
    num_chunks = args.blocks_num * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    num_chunks = args.head_dim;
  }

  int64_t chunk_size = element_count / num_chunks;

  bool is_per_channel = (args.scaling_method == "per_channel");
  int64_t stride = is_per_channel ? args.head_dim : 1;

#ifdef USE_MULTITHREADING
  {
    std::vector<float> p_scales(static_cast<size_t>(num_chunks));
    std::vector<float> p_zps(static_cast<size_t>(num_chunks));
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      p_scales[static_cast<size_t>(chunk)] = args.scales_in[chunk];
      p_zps[static_cast<size_t>(chunk)] = args.asym ? args.zps_in[chunk] : 0.0f;
    }
    int nthreads =
        (args.num_threads > 0) ? args.num_threads : omp_get_max_threads();
    if (is_per_channel) {
      int cl_chunks = std::max(1, 64 / static_cast<int>(sizeof(kvtype)));
      nthreads = std::min(
          nthreads, std::max(1, static_cast<int>(num_chunks) / cl_chunks));
    }
  #pragma omp parallel for schedule(static) \
      num_threads(nthreads) if (nthreads > 1 && num_chunks > 1)
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t offset = is_per_channel ? chunk : chunk * chunk_size;
      float scale = p_scales[static_cast<size_t>(chunk)];
      float zeropoint = p_zps[static_cast<size_t>(chunk)];

      if (is_per_channel) {
        for (int64_t i = 0; i < chunk_size; ++i) {
          dst[i * stride + offset] = static_cast<kvtype>(
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint);
        }

      } else {
        if constexpr (std::is_same_v<qtype, int8_t>) {
  #ifdef __AVX512F__
          __m512 scale_vec = _mm512_set1_ps(scale);
          __m512 zero_vec = _mm512_set1_ps(zeropoint);
          int64_t simd_size = chunk_size - (chunk_size % 16);

          for (int64_t i = 0; i < simd_size; i += 16) {
            __m128i data_int8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
            __m512i data_int32 = _mm512_cvtepi8_epi32(data_int8);
            __m512 data_fp32 = _mm512_cvtepi32_ps(data_int32);
            __m512 data_scaled = _mm512_mul_ps(data_fp32, scale_vec);
            data_scaled = _mm512_add_ps(data_scaled, zero_vec);

            if constexpr (std::is_same_v<kvtype, float>) {
              _mm512_storeu_ps(&dst[i * stride + offset], data_scaled);
    #ifdef __AVX512BF16__
            } else if constexpr (std::is_same_v<kvtype, __bf16>) {
              __m256bh packed = _mm512_cvtneps_pbh(data_scaled);
              _mm256_storeu_si256(
                  reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                  (__m256i)packed);
    #endif
            } else if constexpr (std::is_same_v<kvtype, _Float16>) {
              __m256i packed = _mm512_cvtps_ph(
                  data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              _mm256_storeu_si256(
                  reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                  packed);
            } else {
              for (int k = 0; k < 16; ++k) {
                float value =
                    static_cast<float>(src[(i + k) * stride + offset]) * scale +
                    zeropoint;
                dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
              }
            }
          }

          for (int64_t i = simd_size; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #elif defined(__AVX2__)
          __m256 scale_vec = _mm256_set1_ps(scale);
          __m256 zero_vec = _mm256_set1_ps(zeropoint);
          int64_t simd_size = chunk_size - (chunk_size % 8);

          for (int64_t i = 0; i < simd_size; i += 8) {
            __m128i data_int8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
            __m256i data_int32 = _mm256_cvtepi8_epi32(data_int8);
            __m256 data_fp32 = _mm256_cvtepi32_ps(data_int32);
            __m256 data_scaled = _mm256_mul_ps(data_fp32, scale_vec);
            data_scaled = _mm256_add_ps(data_scaled, zero_vec);

            if constexpr (std::is_same_v<kvtype, float>) {
              _mm256_storeu_ps(&dst[i * stride + offset], data_scaled);
            } else if constexpr (std::is_same_v<kvtype, _Float16>) {
              __m128i packed = _mm256_cvtps_ph(
                  data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              _mm_storeu_si128(
                  reinterpret_cast<__m128i*>(&dst[i * stride + offset]),
                  packed);
            } else {
              for (int k = 0; k < 8; ++k) {
                float value =
                    static_cast<float>(src[(i + k) * stride + offset]) * scale +
                    zeropoint;
                dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
              }
            }
          }

          for (int64_t i = simd_size; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #else
          for (int64_t i = 0; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #endif

        } else {
  #ifdef __AVX512F__
          __m512 scale_vec = _mm512_set1_ps(scale);
          __m512 zero_vec = _mm512_set1_ps(zeropoint);

          int64_t simd_size = chunk_size - (chunk_size % 16);

          for (int64_t i = 0; i < simd_size; i += 16) {
            __m256i data_int16 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&src[i * stride + offset]));
            __m512i data_int32 = _mm512_cvtepi16_epi32(data_int16);
            __m512 data_fp32 = _mm512_cvtepi32_ps(data_int32);
            __m512 data_scaled = _mm512_mul_ps(data_fp32, scale_vec);
            data_scaled = _mm512_add_ps(data_scaled, zero_vec);

            if constexpr (std::is_same_v<kvtype, float>) {
              _mm512_storeu_ps(&dst[i * stride + offset], data_scaled);
    #ifdef __AVX512BF16__
            } else if constexpr (std::is_same_v<kvtype, __bf16>) {
              __m256bh packed = _mm512_cvtneps_pbh(data_scaled);
              _mm256_storeu_si256(
                  reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                  (__m256i)packed);
    #endif
            } else if constexpr (std::is_same_v<kvtype, _Float16>) {
              __m256i packed = _mm512_cvtps_ph(
                  data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              _mm256_storeu_si256(
                  reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                  packed);
            } else {
              for (int k = 0; k < 16; ++k) {
                float value =
                    static_cast<float>(src[(i + k) * stride + offset]) * scale +
                    zeropoint;
                dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
              }
            }
          }

          for (int64_t i = simd_size; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #elif defined(__AVX2__)
          __m256 scale_vec = _mm256_set1_ps(scale);
          __m256 zero_vec = _mm256_set1_ps(zeropoint);

          int64_t simd_size = chunk_size - (chunk_size % 8);

          for (int64_t i = 0; i < simd_size; i += 8) {
            __m128i data_int16 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
            __m256i data_int32 = _mm256_cvtepi16_epi32(data_int16);
            __m256 data_fp32 = _mm256_cvtepi32_ps(data_int32);
            __m256 data_scaled = _mm256_mul_ps(data_fp32, scale_vec);
            data_scaled = _mm256_add_ps(data_scaled, zero_vec);

            if constexpr (std::is_same_v<kvtype, float>) {
              _mm256_storeu_ps(&dst[i * stride + offset], data_scaled);
            } else if constexpr (std::is_same_v<kvtype, _Float16>) {
              __m128i packed = _mm256_cvtps_ph(
                  data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              _mm_storeu_si128(
                  reinterpret_cast<__m128i*>(&dst[i * stride + offset]),
                  packed);
            } else {
              for (int k = 0; k < 8; ++k) {
                float value =
                    static_cast<float>(src[(i + k) * stride + offset]) * scale +
                    zeropoint;
                dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
              }
            }
          }

          for (int64_t i = simd_size; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #else
          for (int64_t i = 0; i < chunk_size; ++i) {
            float value = static_cast<float>(src[i * stride + offset]) * scale +
                          zeropoint;
            dst[i * stride + offset] = static_cast<kvtype>(value);
          }
  #endif
        }
      }
    }
  }
#else
  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    int64_t offset = is_per_channel ? chunk : chunk * chunk_size;

    float scale = args.scales_in[chunk];
    float zeropoint = 0.0f;
    if (args.asym) zeropoint = args.zps_in[chunk];

    if (is_per_channel) {
      for (int64_t i = 0; i < chunk_size; ++i) {
        dst[i * stride + offset] = static_cast<kvtype>(
            static_cast<float>(src[i * stride + offset]) * scale + zeropoint);
      }

    } else {
      if constexpr (std::is_same_v<qtype, int8_t>) {

  #ifdef __AVX512F__
        __m512 scale_vec = _mm512_set1_ps(scale);
        __m512 zero_vec = _mm512_set1_ps(zeropoint);
        int64_t simd_size = chunk_size - (chunk_size % 16);

        for (int64_t i = 0; i < simd_size; i += 16) {
          __m128i data_int8 = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
          __m512i data_int32 = _mm512_cvtepi8_epi32(data_int8);
          __m512 data_fp32 = _mm512_cvtepi32_ps(data_int32);
          __m512 data_scaled = _mm512_mul_ps(data_fp32, scale_vec);
          data_scaled = _mm512_add_ps(data_scaled, zero_vec);

          if constexpr (std::is_same_v<kvtype, float>) {
            _mm512_storeu_ps(&dst[i * stride + offset], data_scaled);
    #ifdef __AVX512BF16__
          } else if constexpr (std::is_same_v<kvtype, __bf16>) {
            __m256bh packed = _mm512_cvtneps_pbh(data_scaled);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                (__m256i)packed);
    #endif
          } else if constexpr (std::is_same_v<kvtype, _Float16>) {
            __m256i packed = _mm512_cvtps_ph(
                data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&dst[i * stride + offset]), packed);
          } else {
            for (int k = 0; k < 16; ++k) {
              float value =
                  static_cast<float>(src[(i + k) * stride + offset]) * scale +
                  zeropoint;
              dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
            }
          }
        }

        for (int64_t i = simd_size; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #elif defined(__AVX2__)
        __m256 scale_vec = _mm256_set1_ps(scale);
        __m256 zero_vec = _mm256_set1_ps(zeropoint);
        int64_t simd_size = chunk_size - (chunk_size % 8);

        for (int64_t i = 0; i < simd_size; i += 8) {
          __m128i data_int8 = _mm_loadl_epi64(
              reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
          __m256i data_int32 = _mm256_cvtepi8_epi32(data_int8);
          __m256 data_fp32 = _mm256_cvtepi32_ps(data_int32);
          __m256 data_scaled = _mm256_mul_ps(data_fp32, scale_vec);
          data_scaled = _mm256_add_ps(data_scaled, zero_vec);

          if constexpr (std::is_same_v<kvtype, float>) {
            _mm256_storeu_ps(&dst[i * stride + offset], data_scaled);
          } else if constexpr (std::is_same_v<kvtype, _Float16>) {
            __m128i packed = _mm256_cvtps_ph(
                data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(&dst[i * stride + offset]), packed);
          } else {
            for (int k = 0; k < 8; ++k) {
              float value =
                  static_cast<float>(src[(i + k) * stride + offset]) * scale +
                  zeropoint;
              dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
            }
          }
        }

        for (int64_t i = simd_size; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #else
        for (int64_t i = 0; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #endif

      } else {
  #ifdef __AVX512F__
        __m512 scale_vec = _mm512_set1_ps(scale);
        __m512 zero_vec = _mm512_set1_ps(zeropoint);

        int64_t simd_size = chunk_size - (chunk_size % 16);

        for (int64_t i = 0; i < simd_size; i += 16) {
          __m256i data_int16 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&src[i * stride + offset]));
          __m512i data_int32 = _mm512_cvtepi16_epi32(data_int16);
          __m512 data_fp32 = _mm512_cvtepi32_ps(data_int32);
          __m512 data_scaled = _mm512_mul_ps(data_fp32, scale_vec);
          data_scaled = _mm512_add_ps(data_scaled, zero_vec);

          if constexpr (std::is_same_v<kvtype, float>) {
            _mm512_storeu_ps(&dst[i * stride + offset], data_scaled);
    #ifdef __AVX512BF16__
          } else if constexpr (std::is_same_v<kvtype, __bf16>) {
            __m256bh packed = _mm512_cvtneps_pbh(data_scaled);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&dst[i * stride + offset]),
                (__m256i)packed);
    #endif
          } else if constexpr (std::is_same_v<kvtype, _Float16>) {
            __m256i packed = _mm512_cvtps_ph(
                data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&dst[i * stride + offset]), packed);
          } else {
            for (int k = 0; k < 16; ++k) {
              float value =
                  static_cast<float>(src[(i + k) * stride + offset]) * scale +
                  zeropoint;
              dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
            }
          }
        }

        for (int64_t i = simd_size; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #elif defined(__AVX2__)
        __m256 scale_vec = _mm256_set1_ps(scale);
        __m256 zero_vec = _mm256_set1_ps(zeropoint);

        int64_t simd_size = chunk_size - (chunk_size % 8);

        for (int64_t i = 0; i < simd_size; i += 8) {
          __m128i data_int16 = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(&src[i * stride + offset]));
          __m256i data_int32 = _mm256_cvtepi16_epi32(data_int16);
          __m256 data_fp32 = _mm256_cvtepi32_ps(data_int32);
          __m256 data_scaled = _mm256_mul_ps(data_fp32, scale_vec);
          data_scaled = _mm256_add_ps(data_scaled, zero_vec);

          if constexpr (std::is_same_v<kvtype, float>) {
            _mm256_storeu_ps(&dst[i * stride + offset], data_scaled);
          } else if constexpr (std::is_same_v<kvtype, _Float16>) {
            __m128i packed = _mm256_cvtps_ph(
                data_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(&dst[i * stride + offset]), packed);
          } else {
            for (int k = 0; k < 8; ++k) {
              float value =
                  static_cast<float>(src[(i + k) * stride + offset]) * scale +
                  zeropoint;
              dst[(i + k) * stride + offset] = static_cast<kvtype>(value);
            }
          }
        }

        for (int64_t i = simd_size; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #else
        for (int64_t i = 0; i < chunk_size; ++i) {
          float value =
              static_cast<float>(src[i * stride + offset]) * scale + zeropoint;
          dst[i * stride + offset] = static_cast<kvtype>(value);
        }
  #endif
      }
    }
  }
#endif
}

// Validate runtime args before dispatch; the kernels do no internal checks (OOB
// / divide-by-zero otherwise).
inline void validate_quant_args(int64_t element_count,
                                const quantization_args& args) {
  if (args.qbit < 1 || args.qbit > 16) {
    throw std::invalid_argument("quantization qbit must be in [1, 16]");
  }
  int64_t num_chunks = 1;
  if (args.scaling_method == "per_token") {
    num_chunks = static_cast<int64_t>(args.blocks_num) * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    num_chunks = args.head_dim;
  }
  if (num_chunks <= 0) {
    throw std::invalid_argument(
        "quantization num_chunks must be positive "
        "(check blocks_num/block_size/head_dim)");
  }
  if (element_count % num_chunks != 0) {
    throw std::invalid_argument(
        "element_count must be divisible by num_chunks");
  }
  if (args.rh) {
    int64_t chunk_size = element_count / num_chunks;
    if (chunk_size < 1 || (chunk_size & (chunk_size - 1)) != 0) {
      throw std::invalid_argument(
          "rh (Hadamard) requires chunk_size to be a power of 2");
    }
  }
}

// 4-bit per_token rows of even length can be packed / unpacked row by row.
inline bool packs_rows_q4(int64_t element_count,
                          const quantization_args& args) {
  if (args.scaling_method != "per_token") return false;
  const int64_t rows = static_cast<int64_t>(args.blocks_num) * args.block_size;
  return rows > 0 && (element_count / rows) % 2 == 0;
}

template <typename kvtype, typename qtype>
void quantize_kvcache(kvtype* src, qtype* dst, int64_t element_count,
                      int64_t scale_id, quantization_args& args) {
  validate_quant_args(element_count, args);
  if (args.scales_out == nullptr || args.zps_out == nullptr) {
    throw std::invalid_argument("quantize_kvcache needs scales_out / zps_out");
  }

  if constexpr (std::is_same_v<qtype, int8_t>) {
    if (args.qbit == 4 && packs_rows_q4(element_count, args)) {
      quantization_args rows = args;
      rows.pack4_rows = true;
      if (args.rh) {
        quantize_block_rh<kvtype, int8_t>(src, dst, element_count, scale_id,
                                          rows);
      } else {
        quantize_block<kvtype, int8_t>(src, dst, element_count, scale_id, rows);
      }
      return;
    }
    if (args.qbit == 4) {
      std::vector<int8_t> logical(static_cast<size_t>(element_count));
      if (args.rh) {
        quantize_block_rh<kvtype, int8_t>(src, logical.data(), element_count,
                                          scale_id, args);
      } else {
        quantize_block<kvtype, int8_t>(src, logical.data(), element_count,
                                       scale_id, args);
      }
      q4_detail::pack_buffer(logical.data(), dst, element_count);
      return;
    }
  }

  if (args.rh) {
    quantize_block_rh<kvtype, qtype>(src, dst, element_count, scale_id, args);
  } else {
    quantize_block<kvtype, qtype>(src, dst, element_count, scale_id, args);
  }
}

template <typename kvtype, typename qtype>
void dequantize_kvcache(qtype* src, kvtype* dst, int64_t element_count,
                        int64_t scale_id, quantization_args& args) {
  validate_quant_args(element_count, args);
  if (args.scales_in == nullptr || args.zps_in == nullptr) {
    throw std::invalid_argument("dequantize_kvcache needs scales_in / zps_in");
  }

  if constexpr (std::is_same_v<qtype, int8_t>) {
    if (args.qbit == 4 && args.rh && packs_rows_q4(element_count, args)) {
      quantization_args rows = args;
      rows.pack4_rows = true;
      dequantize_block_rh<kvtype, int8_t>(src, dst, element_count, scale_id,
                                          rows);
      return;
    }
    if (args.qbit == 4) {
      std::vector<int8_t> logical(static_cast<size_t>(element_count));
      q4_detail::unpack_buffer(src, logical.data(), element_count);
      if (args.rh) {
        dequantize_block_rh<kvtype, int8_t>(logical.data(), dst, element_count,
                                            scale_id, args);
      } else {
        dequantize_block<kvtype, int8_t>(logical.data(), dst, element_count,
                                         scale_id, args);
      }
      return;
    }
  }

  if (args.rh) {
    dequantize_block_rh<kvtype, qtype>(src, dst, element_count, scale_id, args);
  } else {
    dequantize_block<kvtype, qtype>(src, dst, element_count, scale_id, args);
  }
}

// Quantizes fused-K/V attention [layers, tokens, head_num*head_dim] in place of
// split_fused_attention_heads_native + per-(layer*head) quantize_kvcache.
// Output q bytes and scales are identical to that path; only the access order
// differs.
struct fused_direct_params {
  int64_t layers = 0;
  int64_t tokens = 0;
  int64_t head_num = 0;
  int64_t head_dim = 0;
  int qbit = 4;
  bool rh = false;
  bool asym = false;
  const int32_t* perm = nullptr;
  const float* signs = nullptr;
  int num_threads = 0;
};

namespace fused_direct_detail {

constexpr int64_t kMaxColumnBlock = 32;

inline int64_t column_block(int64_t head_dim) {
  for (int64_t cb = kMaxColumnBlock; cb >= 2; cb >>= 1) {
    if (head_dim % cb == 0) {
      return cb;
    }
  }
  return 1;
}

inline void store_logical(const int8_t* logical, int8_t* unit_dst,
                          int64_t logical_off, int64_t n, int qbit) {
  if (qbit == 4) {
    q4_detail::pack_buffer(logical, unit_dst + logical_off / 2, n);
  } else {
    std::memcpy(unit_dst + logical_off, logical, static_cast<size_t>(n));
  }
}

// Same expressions as quantize_block / quantize_block_rh.
inline void scale_and_zeropoint(float min_val, float max_val, bool asym,
                                float qmin_float, float qmax_float,
                                float& scale, float& zeropoint,
                                bool exact_division = false,
                                [[maybe_unused]] bool rh = false) {
  scale = 0.0f;
  zeropoint = 0.0f;
  if (exact_division) {
    // Match the reference scale arithmetic exactly (avoid reciprocal
    // substitution).
    volatile float denominator = asym ? qmax_float - qmin_float : qmax_float;
    scale = (asym ? max_val - min_val
                  : std::max(std::abs(min_val), std::abs(max_val))) /
            denominator;
#ifdef __INTEL_LLVM_COMPILER
    if (!asym && !rh) {
      volatile float reciprocal = 1.0f / qmax_float;
      scale = std::max(std::abs(min_val), std::abs(max_val)) * reciprocal;
    }
#endif
    if (asym) {
      zeropoint = min_val - qmin_float * scale;
    }
    return;
  }
  if (asym) {
    scale = (max_val - min_val) / (qmax_float - qmin_float);
    zeropoint = min_val - qmin_float * scale;
  } else {
    scale = std::max(std::abs(min_val), std::abs(max_val)) / qmax_float;
  }
}

// One (layer, head) unit, columns [c0, c0+cb): every column is its own scale
// group over tokens.
// The folded 1/sqrt(n) multiply must round on its own, exactly as the separate
// normalization pass did: forbid FMA contraction / reassociation where it is
// combined with other arithmetic (icpx defaults to a fast fp-model).
#if defined(__clang__)
  #define GHA_STRICT_FP _Pragma("clang fp contract(off) reassociate(off)")
#else
  #define GHA_STRICT_FP
#endif

// Unnormalized Walsh-Hadamard transform along the row axis of a row-major
// [rows, cb] block (rows a power of two). Stages are applied three at a time
// (eight rows held per column), so each element is loaded and stored once per
// three stages instead of once per stage. Every butterfly pairs the same rows
// in the same order as the stage-by-stage loop, so the result is bit-identical.
inline void fwht_rows_radix8(float* buf, int64_t rows, int64_t cb, int64_t h) {
  for (int64_t blk = 0; blk < rows; blk += 8 * h) {
    for (int64_t o = 0; o < h; ++o) {
      float* __restrict p0 = buf + (blk + o) * cb;
      float* __restrict p1 = p0 + h * cb;
      float* __restrict p2 = p1 + h * cb;
      float* __restrict p3 = p2 + h * cb;
      float* __restrict p4 = p3 + h * cb;
      float* __restrict p5 = p4 + h * cb;
      float* __restrict p6 = p5 + h * cb;
      float* __restrict p7 = p6 + h * cb;
      for (int64_t c = 0; c < cb; ++c) {
        // stage h
        float a0 = p0[c] + p1[c], a1 = p0[c] - p1[c];
        float a2 = p2[c] + p3[c], a3 = p2[c] - p3[c];
        float a4 = p4[c] + p5[c], a5 = p4[c] - p5[c];
        float a6 = p6[c] + p7[c], a7 = p6[c] - p7[c];
        // stage 2h
        float b0 = a0 + a2, b2 = a0 - a2, b1 = a1 + a3, b3 = a1 - a3;
        float b4 = a4 + a6, b6 = a4 - a6, b5 = a5 + a7, b7 = a5 - a7;
        // stage 4h
        p0[c] = b0 + b4;
        p4[c] = b0 - b4;
        p1[c] = b1 + b5;
        p5[c] = b1 - b5;
        p2[c] = b2 + b6;
        p6[c] = b2 - b6;
        p3[c] = b3 + b7;
        p7[c] = b3 - b7;
      }
    }
  }
}

inline void fwht_rows_radix4(float* buf, int64_t rows, int64_t cb, int64_t h) {
  for (int64_t blk = 0; blk < rows; blk += 4 * h) {
    for (int64_t o = 0; o < h; ++o) {
      float* __restrict p0 = buf + (blk + o) * cb;
      float* __restrict p1 = p0 + h * cb;
      float* __restrict p2 = p1 + h * cb;
      float* __restrict p3 = p2 + h * cb;
      for (int64_t c = 0; c < cb; ++c) {
        float a0 = p0[c] + p1[c], a1 = p0[c] - p1[c];
        float a2 = p2[c] + p3[c], a3 = p2[c] - p3[c];
        p0[c] = a0 + a2;
        p2[c] = a0 - a2;
        p1[c] = a1 + a3;
        p3[c] = a1 - a3;
      }
    }
  }
}

inline void fwht_rows_radix2(float* buf, int64_t rows, int64_t cb, int64_t h) {
  for (int64_t blk = 0; blk < rows; blk += 2 * h) {
    for (int64_t o = 0; o < h; ++o) {
      float* __restrict p0 = buf + (blk + o) * cb;
      float* __restrict p1 = p0 + h * cb;
      for (int64_t c = 0; c < cb; ++c) {
        const float x = p0[c];
        const float y = p1[c];
        p0[c] = x + y;
        p1[c] = x - y;
      }
    }
  }
}

inline void fwht_rows(float* buf, int64_t rows, int64_t cb) {
  int64_t h = 1;
  while (h < rows) {
    const int64_t left = rows / h;  // 2^(stages left)
    if (left >= 8) {
      fwht_rows_radix8(buf, rows, cb, h);
      h *= 8;
    } else if (left >= 4) {
      fwht_rows_radix4(buf, rows, cb, h);
      h *= 4;
    } else {
      fwht_rows_radix2(buf, rows, cb, h);
      h *= 2;
    }
  }
}

template <typename kvtype>
void per_channel_block(const kvtype* unit_src, int64_t row_stride, int64_t c0,
                       int64_t cb, const fused_direct_params& p,
                       float qmin_float, float qmax_float, float* buf,
                       int8_t* qrow, float* scales, float* zps,
                       int8_t* unit_dst, bool exact_division = false) {
  const int64_t tokens = p.tokens;
  const int32_t* perm = p.rh ? p.perm : nullptr;
  const float* signs = p.rh ? p.signs : nullptr;

  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t src_t = perm ? static_cast<int64_t>(perm[t]) : t;
    const kvtype* row = unit_src + src_t * row_stride + c0;
    float* out = buf + t * cb;
    for (int64_t c = 0; c < cb; ++c) {
      out[c] = static_cast<float>(row[c]);
    }
    if (signs) {
      const float s = signs[t];
      for (int64_t c = 0; c < cb; ++c) {
        out[c] *= s;
      }
    }
  }

  // The 1/sqrt(n) normalization is folded into the min/max and the quantize
  // step below (x * s is monotonic for s > 0, so min/max commute with it).
  if (p.rh) {
    fwht_rows(buf, tokens, cb);
  }
  const float norm = p.rh ? 1.0f / std::sqrt(static_cast<float>(tokens)) : 1.0f;

  float min_vals[kMaxColumnBlock];
  float max_vals[kMaxColumnBlock];
  float zero_points[kMaxColumnBlock];
  float inv_scales[kMaxColumnBlock];
  for (int64_t c = 0; c < cb; ++c) {
    min_vals[c] = buf[c];
    max_vals[c] = buf[c];
  }
  for (int64_t t = 0; t < tokens; ++t) {
    const float* row = buf + t * cb;
    for (int64_t c = 0; c < cb; ++c) {
      min_vals[c] = std::min(min_vals[c], row[c]);
      max_vals[c] = std::max(max_vals[c], row[c]);
    }
  }
  if (p.rh) {
    GHA_STRICT_FP
    for (int64_t c = 0; c < cb; ++c) {
      min_vals[c] *= norm;
      max_vals[c] *= norm;
    }
  }
  for (int64_t c = 0; c < cb; ++c) {
    float scale = 0.0f;
    float zeropoint = 0.0f;
    scale_and_zeropoint(min_vals[c], max_vals[c], p.asym, qmin_float,
                        qmax_float, scale, zeropoint, exact_division, p.rh);
    scales[c0 + c] = scale;
    zps[c0 + c] = zeropoint;
    zero_points[c] = zeropoint;
    inv_scales[c] = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
  }

  for (int64_t t = 0; t < tokens; ++t) {
    const float* row = buf + t * cb;
    if (p.rh) {
      GHA_STRICT_FP
      for (int64_t c = 0; c < cb; ++c) {
        const float v = row[c] * norm;
        qrow[c] = static_cast<int8_t>(
            std::clamp(round_half_away((v - zero_points[c]) * inv_scales[c]),
                       qmin_float, qmax_float));
      }
    } else {
      for (int64_t c = 0; c < cb; ++c) {
        qrow[c] = static_cast<int8_t>(
            std::clamp(std::round((row[c] - zero_points[c]) * inv_scales[c]),
                       qmin_float, qmax_float));
      }
    }
    store_logical(qrow, unit_dst, t * p.head_dim + c0, cb, p.qbit);
  }
}

}  // namespace fused_direct_detail

// per_channel only. q_dst: units*unit_q_bytes; scales/zps:
// [layers*head_num, head_dim].
template <typename kvtype>
void quantize_fused_direct(const kvtype* src, int8_t* q_dst, float* scales,
                           float* zps, const fused_direct_params& p) {
  using namespace fused_direct_detail;
  const int64_t tokens = p.tokens;
  const int64_t head_num = p.head_num;
  const int64_t head_dim = p.head_dim;
  const int64_t hidden = head_num * head_dim;
  const int64_t unit_elems = tokens * head_dim;
  const int64_t unit_q_bytes = (p.qbit == 4) ? unit_elems / 2 : unit_elems;
  const int64_t qmax_int = (1 << (p.qbit - 1)) - 1;
  const int64_t qmin_int = -(1 << (p.qbit - 1));
  const float qmax_float = static_cast<float>(qmax_int);
  const float qmin_float = static_cast<float>(qmin_int);
#ifdef USE_MULTITHREADING
  const int nthreads =
      (p.num_threads > 0) ? p.num_threads : omp_get_max_threads();
#endif

  const int64_t cb = column_block(head_dim);
  const int64_t blocks_per_unit = head_dim / cb;
  const int64_t tasks = p.layers * head_num * blocks_per_unit;
#ifdef USE_MULTITHREADING
  #pragma omp parallel num_threads(nthreads) if (nthreads > 1 && tasks > 1)
#endif
  {
    std::vector<float> buf(static_cast<size_t>(tokens * cb));
    std::vector<int8_t> qrow(static_cast<size_t>(cb));
#ifdef USE_MULTITHREADING
  #pragma omp for schedule(static)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
      const int64_t unit = task / blocks_per_unit;
      const int64_t c0 = (task % blocks_per_unit) * cb;
      const int64_t layer = unit / head_num;
      const int64_t h = unit % head_num;
      per_channel_block<kvtype>(src + layer * tokens * hidden + h * head_dim,
                                hidden, c0, cb, p, qmin_float, qmax_float,
                                buf.data(), qrow.data(),
                                scales + unit * head_dim, zps + unit * head_dim,
                                q_dst + unit * unit_q_bytes);
    }
  }
}

namespace fused_direct_detail {

inline void load_logical(const int8_t* unit_q, int64_t logical_off, int64_t n,
                         int qbit, int8_t* logical) {
  if (qbit == 4) {
    q4_detail::unpack_buffer(unit_q + logical_off / 2, logical, n);
  } else {
    std::memcpy(logical, unit_q + logical_off, static_cast<size_t>(n));
  }
}

// Inverse of per_channel_block: dequantize columns [c0, c0+cb) of one unit and
// write them straight into the fused [tokens, row_stride] destination.
template <typename kvtype>
void dequant_per_channel_block(const int8_t* unit_q, kvtype* unit_dst,
                               int64_t row_stride, int64_t c0, int64_t cb,
                               const fused_direct_params& p,
                               const float* scales, const float* zps,
                               float* buf, int8_t* qrow) {
  const int64_t tokens = p.tokens;
  const int32_t* perm = p.rh ? p.perm : nullptr;
  const float* signs = p.rh ? p.signs : nullptr;
  float col_scale[kMaxColumnBlock];
  float col_zp[kMaxColumnBlock];
  for (int64_t c = 0; c < cb; ++c) {
    col_scale[c] = scales[c0 + c];
    col_zp[c] = p.asym ? zps[c0 + c] : 0.0f;
  }
  for (int64_t t = 0; t < tokens; ++t) {
    load_logical(unit_q, t * p.head_dim + c0, cb, p.qbit, qrow);
    float* row = buf + t * cb;
    for (int64_t c = 0; c < cb; ++c) {
      row[c] = static_cast<float>(qrow[c]) * col_scale[c] + col_zp[c];
    }
  }

  if (p.rh) {
    fwht_rows(buf, tokens, cb);
  }
  // 1/sqrt(n) is applied in the write (same multiply order as before).
  const float norm = p.rh ? 1.0f / std::sqrt(static_cast<float>(tokens)) : 1.0f;

  // Forward gathered row perm[t] into slot t; the inverse scatters slot t back.
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t dst_t = perm ? static_cast<int64_t>(perm[t]) : t;
    const float s = signs ? signs[t] : 1.0f;
    const float* row = buf + t * cb;
    kvtype* out = unit_dst + dst_t * row_stride + c0;
    if (p.rh) {
      GHA_STRICT_FP
      for (int64_t c = 0; c < cb; ++c) {
        out[c] = static_cast<kvtype>((row[c] * norm) * s);
      }
    } else {
      for (int64_t c = 0; c < cb; ++c) {
        out[c] = static_cast<kvtype>(row[c] * s);
      }
    }
  }
}

}  // namespace fused_direct_detail

// Inverse of quantize_fused_direct: q_src/scales/zps use the same per-unit
// layout, dst is the fused [layers, tokens, head_num*head_dim] tensor.
template <typename kvtype>
void dequantize_fused_direct(const int8_t* q_src, kvtype* dst,
                             const float* scales, const float* zps,
                             const fused_direct_params& p) {
  using namespace fused_direct_detail;
  const int64_t tokens = p.tokens;
  const int64_t head_num = p.head_num;
  const int64_t head_dim = p.head_dim;
  const int64_t hidden = head_num * head_dim;
  const int64_t unit_elems = tokens * head_dim;
  const int64_t unit_q_bytes = (p.qbit == 4) ? unit_elems / 2 : unit_elems;
#ifdef USE_MULTITHREADING
  const int nthreads =
      (p.num_threads > 0) ? p.num_threads : omp_get_max_threads();
#endif

  const int64_t cb = column_block(head_dim);
  const int64_t blocks_per_unit = head_dim / cb;
  const int64_t tasks = p.layers * head_num * blocks_per_unit;
#ifdef USE_MULTITHREADING
  #pragma omp parallel num_threads(nthreads) if (nthreads > 1 && tasks > 1)
#endif
  {
    std::vector<float> buf(static_cast<size_t>(tokens * cb));
    std::vector<int8_t> qrow(static_cast<size_t>(cb));
#ifdef USE_MULTITHREADING
  #pragma omp for schedule(static)
#endif
    for (int64_t task = 0; task < tasks; ++task) {
      const int64_t unit = task / blocks_per_unit;
      const int64_t c0 = (task % blocks_per_unit) * cb;
      const int64_t layer = unit / head_num;
      const int64_t h = unit % head_num;
      dequant_per_channel_block<kvtype>(
          q_src + unit * unit_q_bytes,
          dst + layer * tokens * hidden + h * head_dim, hidden, c0, cb, p,
          scales + unit * head_dim, zps + unit * head_dim, buf.data(),
          qrow.data());
    }
  }
}