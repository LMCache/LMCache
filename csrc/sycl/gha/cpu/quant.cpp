// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "quant.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

// ---------------------------------------------------------------------------
// AVX-512 in-register butterfly helpers (16 lanes, 4 stages)
// ---------------------------------------------------------------------------
#ifdef __AVX512F__
namespace fwht_detail {

// XOR_IDX[k]: lane i maps to (i ^ (1<<k)).
// SIGN_VEC[k]: +1 where (i & (1<<k))==0, -1 otherwise.
static const __m512i XOR_IDX[4] = {
    _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14),
    _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13),
    _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11),
    _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7),
};

static const __m512 SIGN_VEC[4] = {
    _mm512_setr_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1),
    _mm512_setr_ps(1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1),
    _mm512_setr_ps(1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1),
    _mm512_setr_ps(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1),
};

static inline __m512 butterfly_lanes(__m512 v, int k) {
  __m512 partner = _mm512_permutexvar_ps(XOR_IDX[k], v);
  return _mm512_fmadd_ps(v, SIGN_VEC[k], partner);
}

}  // namespace fwht_detail

// ---------------------------------------------------------------------------
// AVX2 in-register butterfly helpers (4 lanes via 128-bit VEX, 2 stages)
// Uses 128-bit ops to avoid 256-bit AVX frequency throttling on PTL/E-cores.
// ---------------------------------------------------------------------------
#elif defined(__AVX2__)
namespace fwht_detail {

static const __m256 SIGN_VEC_256[3] = {
    _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1),
    _mm256_setr_ps(1, 1, -1, -1, 1, 1, -1, -1),
    _mm256_setr_ps(1, 1, 1, 1, -1, -1, -1, -1),
};

// k=0: swap adjacent pairs {1,0,3,2,5,4,7,6}
static inline __m256 butterfly_lanes_256_k0(__m256 v) {
  __m256i idx = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
  __m256 partner = _mm256_permutevar8x32_ps(v, idx);
  return _mm256_fmadd_ps(v, SIGN_VEC_256[0], partner);
}

// k=1: swap pairs of 2 {2,3,0,1,6,7,4,5}
static inline __m256 butterfly_lanes_256_k1(__m256 v) {
  __m256i idx = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
  __m256 partner = _mm256_permutevar8x32_ps(v, idx);
  return _mm256_fmadd_ps(v, SIGN_VEC_256[1], partner);
}

// k=2: swap halves {4,5,6,7,0,1,2,3}
static inline __m256 butterfly_lanes_256_k2(__m256 v) {
  __m256i idx = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
  __m256 partner = _mm256_permutevar8x32_ps(v, idx);
  return _mm256_fmadd_ps(v, SIGN_VEC_256[2], partner);
}

}  // namespace fwht_detail
#endif  // __AVX512F__ / __AVX2__

// ---------------------------------------------------------------------------
// Plain in-place FWHT (backward compatible, no perm/signs)
// ---------------------------------------------------------------------------
void fast_walsh_hadamard_transform(float* x, int64_t n) {
  if ((n & (n - 1)) != 0) {
    throw std::invalid_argument("Length must be a power of 2");
  }
#if defined(__AVX512F__) || defined(__AVX2__)
  // Delegates to the fused kernel (bit-identical; in-place safe without
  // perm/signs).
  if (n > 8) {
    fast_walsh_hadamard_transform_fused(x, x, n, nullptr, nullptr);
    return;
  }
#endif

  const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));

  for (int64_t h = 1; h < n; h <<= 1) {
    const int64_t stride = h << 1;

    for (int64_t i = 0; i < n; i += stride) {
      int64_t j = i;
      int64_t j_end = i + h;

#ifdef __AVX512F__
      for (; j + 16 <= j_end; j += 16) {
        __m512 a = _mm512_loadu_ps(&x[j]);
        __m512 b = _mm512_loadu_ps(&x[j + h]);
        __m512 sum = _mm512_add_ps(a, b);
        __m512 diff = _mm512_sub_ps(a, b);
        _mm512_storeu_ps(&x[j], sum);
        _mm512_storeu_ps(&x[j + h], diff);
      }
#elif defined(__AVX2__)
      for (; j + 8 <= j_end; j += 8) {
        __m256 a = _mm256_loadu_ps(&x[j]);
        __m256 b = _mm256_loadu_ps(&x[j + h]);
        __m256 sum = _mm256_add_ps(a, b);
        __m256 diff = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(&x[j], sum);
        _mm256_storeu_ps(&x[j + h], diff);
      }
#endif
      for (; j < j_end; ++j) {
        float a = x[j];
        float b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }

  int64_t i = 0;
#ifdef __AVX512F__
  __m512 inv_sqrt_n_vec = _mm512_set1_ps(inv_sqrt_n);
  for (; i + 16 <= n; i += 16) {
    __m512 data = _mm512_loadu_ps(&x[i]);
    data = _mm512_mul_ps(data, inv_sqrt_n_vec);
    _mm512_storeu_ps(&x[i], data);
  }
#elif defined(__AVX2__)
  __m256 inv_sqrt_n_vec = _mm256_set1_ps(inv_sqrt_n);
  for (; i + 8 <= n; i += 8) {
    __m256 data = _mm256_loadu_ps(&x[i]);
    data = _mm256_mul_ps(data, inv_sqrt_n_vec);
    _mm256_storeu_ps(&x[i], data);
  }
#endif
  for (; i < n; ++i) {
    x[i] *= inv_sqrt_n;
  }
}

// ---------------------------------------------------------------------------
// Forward FWHT with fused perm (P) and sign-flip diagonal (D):
//   out = (1/sqrt(n)) * H @ D @ P @ in
// Either perm or signs (or both) may be nullptr.
// When perm is provided, this is NOT in-place: reads from in, writes to out.
// When perm is nullptr and signs is nullptr, works in-place (out may alias in).
// ---------------------------------------------------------------------------
void fast_walsh_hadamard_transform_fused(const float* in, float* out, int64_t n,
                                         const int32_t* perm,
                                         const float* signs) {
  if ((n & (n - 1)) != 0) {
    throw std::invalid_argument("Length must be a power of 2");
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  // Only small n falls back to scalar.
  if (!perm && !signs && n <= 8) {
#else
  if (!perm && !signs) {
#endif
    // No preconditioning — fall back to in-place path if aliased, else
    // copy+transform
    if (in != out) {
      std::memcpy(out, in, static_cast<std::size_t>(n) * sizeof(float));
    }
    fast_walsh_hadamard_transform(out, n);
    return;
  }

  const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));

  // Compute log2(n)
  int log_n = 0;
  {
    int64_t tmp = n;
    while (tmp > 1) {
      tmp >>= 1;
      ++log_n;
    }
  }

  if (log_n <= 3) {
    // Small n (1,2,4,8): scalar prelude + scalar FWHT
    for (int i = 0; i < n; ++i) {
      float v = in[perm ? perm[i] : i];
      if (signs) v *= signs[i];
      out[i] = v;
    }
    for (int s = 1; s < n; s <<= 1) {
      for (int i = 0; i < n; i += 2 * s) {
        for (int j = 0; j < s; ++j) {
          float a = out[i + j];
          float b = out[i + j + s];
          out[i + j] = a + b;
          out[i + j + s] = a - b;
        }
      }
    }
  } else {
#ifdef __AVX512F__
    // Pass A: fused load + (perm + signs) + in-register butterflies (s=1,2,4,8)
    for (int64_t c = 0; c < n; c += 16) {
      __m512 v;
      if (perm) {
        __m512i idx =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(perm + c));
        v = _mm512_i32gather_ps(idx, in, 4);
      } else {
        v = _mm512_loadu_ps(in + c);
      }
      if (signs) {
        v = _mm512_mul_ps(v, _mm512_loadu_ps(signs + c));
      }
      v = fwht_detail::butterfly_lanes(v, 0);
      v = fwht_detail::butterfly_lanes(v, 1);
      v = fwht_detail::butterfly_lanes(v, 2);
      v = fwht_detail::butterfly_lanes(v, 3);
      _mm512_storeu_ps(out + c, v);
    }
    // Pass B: inter-register stages (s=16, 32, ..., n/2), in-place on out
    for (int64_t s = 16; s < n; s <<= 1) {
      for (int64_t i = 0; i < n; i += 2 * s) {
        for (int64_t j = 0; j < s; j += 16) {
          __m512 va = _mm512_loadu_ps(out + i + j);
          __m512 vb = _mm512_loadu_ps(out + i + j + s);
          _mm512_storeu_ps(out + i + j, _mm512_add_ps(va, vb));
          _mm512_storeu_ps(out + i + j + s, _mm512_sub_ps(va, vb));
        }
      }
    }
#elif defined(__AVX2__)
    // Pass A: fused load + (perm + signs) + in-register butterflies (s=1,2,4)
    // Uses 256-bit AVX2 ops (8 lanes) with 3 butterfly stages.
    for (int64_t c = 0; c < n; c += 8) {
      __m256 v;
      if (perm) {
        __m256i idx =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(perm + c));
        v = _mm256_i32gather_ps(in, idx, 4);
      } else {
        v = _mm256_loadu_ps(in + c);
      }
      if (signs) {
        v = _mm256_mul_ps(v, _mm256_loadu_ps(signs + c));
      }
      v = fwht_detail::butterfly_lanes_256_k0(v);
      v = fwht_detail::butterfly_lanes_256_k1(v);
      v = fwht_detail::butterfly_lanes_256_k2(v);
      _mm256_storeu_ps(out + c, v);
    }
    // Pass B: inter-register stages (s=8, 16, ..., n/2), in-place on out
    for (int64_t s = 8; s < n; s <<= 1) {
      for (int64_t i = 0; i < n; i += 2 * s) {
        for (int64_t j = 0; j < s; j += 8) {
          __m256 va = _mm256_loadu_ps(out + i + j);
          __m256 vb = _mm256_loadu_ps(out + i + j + s);
          _mm256_storeu_ps(out + i + j, _mm256_add_ps(va, vb));
          _mm256_storeu_ps(out + i + j + s, _mm256_sub_ps(va, vb));
        }
      }
    }
#else
  // Scalar fallback: apply perm + signs, then FWHT
  for (int64_t i = 0; i < n; ++i) {
    float v = in[perm ? perm[i] : i];
    if (signs) v *= signs[i];
    out[i] = v;
  }
  for (std::size_t h = 1; h < static_cast<std::size_t>(n); h <<= 1) {
    const std::size_t stride = h << 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); i += stride) {
      for (std::size_t j = i; j < i + h; ++j) {
        float a = out[j];
        float b = out[j + h];
        out[j] = a + b;
        out[j + h] = a - b;
      }
    }
  }
#endif
  }

  // Normalize by 1/sqrt(n)
  std::size_t i = 0;
#ifdef __AVX512F__
  __m512 inv_sqrt_n_vec = _mm512_set1_ps(inv_sqrt_n);
  for (; i + 16 <= static_cast<std::size_t>(n); i += 16) {
    __m512 data = _mm512_loadu_ps(&out[i]);
    data = _mm512_mul_ps(data, inv_sqrt_n_vec);
    _mm512_storeu_ps(&out[i], data);
  }
#elif defined(__AVX2__)
  __m256 inv_sqrt_n_vec = _mm256_set1_ps(inv_sqrt_n);
  for (; i + 8 <= static_cast<std::size_t>(n); i += 8) {
    __m256 data = _mm256_loadu_ps(&out[i]);
    data = _mm256_mul_ps(data, inv_sqrt_n_vec);
    _mm256_storeu_ps(&out[i], data);
  }
#endif
  for (; i < static_cast<std::size_t>(n); ++i) {
    out[i] *= inv_sqrt_n;
  }
}

// ---------------------------------------------------------------------------
// Inverse transform for dequantization path:
//   out = P^T @ D @ H @ in
// Steps: (1) H @ in, (2) multiply by signs (D), (3) P^T via scatter with the
// forward perm: out[perm[i]] = tmp[i] * signs[i].
// ---------------------------------------------------------------------------
void fast_walsh_hadamard_inverse_fused(const float* in, float* out, int64_t n,
                                       const int32_t* perm,
                                       const float* signs) {
  if ((n & (n - 1)) != 0) {
    throw std::invalid_argument("Length must be a power of 2");
  }

  if (!perm && !signs) {
    if (in != out) {
      std::memcpy(out, in, static_cast<std::size_t>(n) * sizeof(float));
    }
    fast_walsh_hadamard_transform(out, n);
    return;
  }

  // H into a temp buffer, then apply D and P^T.
  std::vector<float> tmp(static_cast<std::size_t>(n));
  std::memcpy(tmp.data(), in, static_cast<std::size_t>(n) * sizeof(float));
  fast_walsh_hadamard_transform(tmp.data(), n);

  // Inverse of forward pre[i] = signs[i]*in[perm[i]]: out[perm[i]] =
  // tmp[i]*signs[i].
  if (perm && signs) {
    for (int64_t i = 0; i < n; ++i) {
      out[perm[i]] = tmp[static_cast<size_t>(i)] * signs[i];
    }
  } else if (signs) {
    // No permutation, just sign flip
#ifdef __AVX512F__
    std::size_t si = 0;
    for (; si + 16 <= static_cast<std::size_t>(n); si += 16) {
      __m512 data = _mm512_loadu_ps(&tmp[si]);
      __m512 sv = _mm512_loadu_ps(&signs[si]);
      _mm512_storeu_ps(&out[si], _mm512_mul_ps(data, sv));
    }
    for (; si < static_cast<std::size_t>(n); ++si) {
      out[si] = tmp[si] * signs[si];
    }
#elif defined(__AVX2__)
    std::size_t si = 0;
    for (; si + 8 <= static_cast<std::size_t>(n); si += 8) {
      __m256 data = _mm256_loadu_ps(&tmp[si]);
      __m256 sv = _mm256_loadu_ps(&signs[si]);
      _mm256_storeu_ps(&out[si], _mm256_mul_ps(data, sv));
    }
    for (; si < static_cast<std::size_t>(n); ++si) {
      out[si] = tmp[si] * signs[si];
    }
#else
    for (int64_t i = 0; i < n; ++i) {
      out[i] = tmp[i] * signs[i];
    }
#endif
  } else {
    // Only inverse permutation (scatter), no signs
    for (int64_t i = 0; i < n; ++i) {
      out[perm[i]] = tmp[static_cast<size_t>(i)];
    }
  }
}
