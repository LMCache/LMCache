// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "quant_sycl.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <type_traits>

// ---------------------------------------------------------------------------
// Sub-group (register + shuffle) quantize path.
//
// One work-group == one 32-lane sub-group == one quantization chunk. Each lane
// owns E = chunk_size / 32 consecutive logical elements, so the first log2(E)
// Hadamard stages run in registers and the last five exchange values across
// lanes with xor shuffles: no SLM, no barriers. The butterflies are applied in
// the same bit order (0, 1, 2, ...) with the same `a + b` / `a - b` operand
// order as the SLM kernels, so the output is bit-identical to them.
// Preconditioning is a gather at load time (x[k] = src[perm[k]] * signs[k]),
// which is the same value the SLM path scatters (local[inv_perm[m]] = src[m] *
// signs_reordered[m], inv_perm = argsort(perm)). Only the 4-bit packed,
// row-contiguous (per_token / per_tensor-in-SLM) case with a power-of-two chunk
// of 64..1024 elements takes it. With RH on, 2048/4096/8192-element chunks (any
// scaling method, including per_channel) take the multi-sub-group variant
// below; everything else keeps the SLM kernels.
// ---------------------------------------------------------------------------
static bool device_supports_sub_group_32(sycl::queue& q) {
  static std::mutex mutex;
  static std::unique_ptr<sycl::device> cached_device;
  static bool cached_value = false;
  std::lock_guard<std::mutex> lock(mutex);
  if (!cached_device || *cached_device != q.get_device()) {
    cached_device = std::make_unique<sycl::device>(q.get_device());
    cached_value = false;
    for (size_t sz :
         cached_device->get_info<sycl::info::device::sub_group_sizes>()) {
      if (sz == 32) cached_value = true;
    }
  }
  return cached_value;
}

// sycl::vec<KVType, 4> loads/stores exist only for the native 16/32-bit types;
// bf16 and the one-byte fp8 wrappers take the scalar paths.
template <typename KVType>
inline constexpr bool kv_has_vec4_v =
    !std::is_same_v<KVType, sycl::ext::oneapi::bfloat16> &&
    !std::is_same_v<KVType, fp8_e4m3_t> && !std::is_same_v<KVType, fp8_e5m2_t>;

static constexpr int kSgLanes = 32;

// Mirror of sg_store_lane_row: reads `E` consecutive KVType values starting at
// `p` with wide loads (scalar loads when `p` is not 4-byte aligned). Conversion
// to float is the same static_cast the scalar path uses, so values are
// bit-identical.
template <typename KVType, int E>
static inline void sg_load_lane_row(const KVType* p, float (&x)[E]) {
  constexpr size_t kBytes = static_cast<size_t>(E) * sizeof(KVType);
  if constexpr (kBytes >= 4 && kBytes % 4 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0) {
      constexpr int kWords = static_cast<int>(kBytes / 4);
      uint32_t words[kWords];
      const uint32_t* src32 = reinterpret_cast<const uint32_t*>(p);
      constexpr int kVec = (kWords % 4 == 0) ? 4 : ((kWords % 2 == 0) ? 2 : 1);
      bool wide = false;
      if constexpr (kVec == 4) {
        if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
#pragma unroll
          for (int w = 0; w < kWords; w += 4) {
            sycl::vec<uint32_t, 4> v;
            v.load(static_cast<size_t>(w / 4),
                   sycl::address_space_cast<
                       sycl::access::address_space::global_space,
                       sycl::access::decorated::no>(src32));
            words[w] = v[0];
            words[w + 1] = v[1];
            words[w + 2] = v[2];
            words[w + 3] = v[3];
          }
          wide = true;
        }
      }
      if (!wide) {
#pragma unroll
        for (int w = 0; w < kWords; ++w) words[w] = src32[w];
      }
      KVType tmp[E];
      __builtin_memcpy(tmp, words, kBytes);
#pragma unroll
      for (int j = 0; j < E; ++j) x[j] = static_cast<float>(tmp[j]);
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < E; ++j) x[j] = static_cast<float>(p[j]);
}

// Writes `E` consecutive KVType values starting at `p` with 4/8/16-byte stores.
// Used by the sub-group dequantize kernel so a lane's contiguous slice of the
// row goes out as one wide store instead of E scalar 2-byte stores (which are
// strided across lanes and measured ~1.5-2x slower than the SLM kernel's
// coalesced vector stores). Falls back to scalar stores when `p` is not 4-byte
// aligned.
template <typename KVType, int E>
static inline void sg_store_lane_row(KVType* p, const float (&x)[E]) {
  constexpr size_t kBytes = static_cast<size_t>(E) * sizeof(KVType);
  if constexpr (kBytes >= 4 && kBytes % 4 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0) {
      KVType tmp[E];
#pragma unroll
      for (int j = 0; j < E; ++j) tmp[j] = static_cast<KVType>(x[j]);
      constexpr int kWords = static_cast<int>(kBytes / 4);
      uint32_t words[kWords];
#pragma unroll
      for (int w = 0; w < kWords; ++w) {
        uint32_t v;
        __builtin_memcpy(&v, reinterpret_cast<const char*>(tmp) + w * 4, 4);
        words[w] = v;
      }
      uint32_t* dst32 = reinterpret_cast<uint32_t*>(p);
      constexpr int kVec = (kWords % 4 == 0) ? 4 : ((kWords % 2 == 0) ? 2 : 1);
      if constexpr (kVec == 4) {
        if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
#pragma unroll
          for (int w = 0; w < kWords; w += 4) {
            sycl::vec<uint32_t, 4> v(words[w], words[w + 1], words[w + 2],
                                     words[w + 3]);
            v.store(static_cast<size_t>(w / 4),
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(dst32));
          }
          return;
        }
      }
#pragma unroll
      for (int w = 0; w < kWords; ++w) dst32[w] = words[w];
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < E; ++j) p[j] = static_cast<KVType>(x[j]);
}

// Chunks of 2048..8192 elements: one work-group of S = chunk/1024 sub-groups,
// each lane owning E = 32 elements. The in-tile (1024-element) stages run in
// registers / xor shuffles exactly as in the single-sub-group kernel; the last
// log2(S) stages (the sub-group index bits) cross sub-groups through one SLM
// exchange. Stage and operand order are unchanged, so the result stays
// bit-identical to the SLM kernels.
static bool sg_multi_chunk_ok(int64_t chunk_size) {
  return chunk_size == 2048 || chunk_size == 4096 || chunk_size == 8192;
}

// Local accessors for the cross-sub-group exchange exist only when S > 1. A
// single sub-group kernel must not declare any SLM: an unused 4-byte accessor
// still puts the launch in SLM mode, which measured ~25% slower for the small
// per_token quantize kernel on Arc Pro B70.
struct SgNoLocal {};
template <int S>
using SgLocal =
    std::conditional_t<(S > 1), sycl::local_accessor<float, 1>, SgNoLocal>;

template <int S>
static inline SgLocal<S> make_sg_local(sycl::handler& cgh, size_t floats) {
  if constexpr (S > 1) {
    return sycl::local_accessor<float, 1>(sycl::range<1>(floats), cgh);
  } else {
    (void)cgh;
    (void)floats;
    return SgNoLocal{};
  }
}

// Butterflies over the sub-group-index bits. `ex` holds one chunk of floats
// laid out [j][sub-group][lane] so every access is a contiguous run of lanes
// (no bank conflicts).
template <int E, int S>
static inline void sg_cross_subgroup_hadamard(sycl::nd_item<1>& item, float* ex,
                                              float (&x)[E], int sgi,
                                              int lane) {
#pragma unroll
  for (int j = 0; j < E; ++j) ex[(j * S + sgi) * kSgLanes + lane] = x[j];
  sycl::group_barrier(item.get_group());
#pragma unroll
  for (int j = 0; j < E; ++j) {
    float v[S];
#pragma unroll
    for (int u = 0; u < S; ++u) v[u] = ex[(j * S + u) * kSgLanes + lane];
#pragma unroll
    for (int m = 1; m < S; m <<= 1) {
#pragma unroll
      for (int u = 0; u < S; ++u) {
        if ((u & m) == 0) {
          const float a = v[u], b = v[u | m];
          v[u] = a + b;
          v[u | m] = a - b;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < S; ++u) {
      if (u == sgi) x[j] = v[u];
    }
  }
}

static bool sg_quantize_chunk_ok(int64_t chunk_size) {
  return chunk_size >= 2 * kSgLanes && chunk_size <= 32 * kSgLanes &&
         (chunk_size & (chunk_size - 1)) == 0;
}

// Work-group size for the multi-kernel / elementwise quantization kernels.
static constexpr int WG_SIZE = 256;

// Work-group size for the single-kernel SLM path, which launches one work-group
// per quantization chunk (quantize_block_sycl_impl /
// dequantize_block_sycl_impl). A fixed 256 is wrong for almost every shape: the
// group wants roughly one lane per 16 elements of the row, since the int4
// pack/unpack phases hand each lane 8 of them and extra lanes just idle at the
// barriers. Measured on the Xe3 iGPU (16384 rows, bf16; output is bit-identical
// at every size):
// ("row width" is the kernel's chunk_size: head_size for fused K/V attention,
// whose K and V halves are quantized as separate heads.)
//   row width         64  128  256  512  1024  2048+
//   best WG           32   32   32   64   128    256
//   best WG, rh quant 32   32   64   64   256    256   (rh quantize wants ~2x)
// per_channel instead reads each chunk with a stride of `hidden`, so it needs
// enough lanes to hide latency and tracks the chunk size (~chunk/8, floor 32,
// cap 256); with a 16-lane group chunk 2048 is 2.7x slower than with 256.
// Chunks too large for SLM never reach this and keep WG_SIZE.
enum class SlmDirection { Quantize, Dequantize };

static size_t slm_work_group_size(bool per_channel, int64_t chunk_size, bool rh,
                                  SlmDirection dir) {
  constexpr int64_t kMinSlmWg = 32;
  if (per_channel) {
    int64_t wg = kMinSlmWg;
    while (wg * 2 <= WG_SIZE && wg * 2 * 8 <= chunk_size) wg *= 2;
    return static_cast<size_t>(wg);
  }
  // Row-contiguous (per_token / per_tensor): indexed by
  // floor(log2(chunk_size)), capped at 12 (4096). Widths such as 6144 land on
  // the cap's bucket.
  static constexpr int kRowWg[13] = {32, 32, 32, 32,  32,  32, 32,
                                     32, 32, 64, 128, 256, 256};
  static constexpr int kRowWgRhQuant[13] = {32, 32, 32, 32,  32,  32, 32,
                                            32, 64, 64, 256, 256, 256};
  int lg = 0;
  while (lg < 12 && (int64_t(1) << (lg + 1)) <= chunk_size) ++lg;
  const bool rh_quantize = (dir == SlmDirection::Quantize) && rh;
  return static_cast<size_t>(rh_quantize ? kRowWgRhQuant[lg] : kRowWg[lg]);
}

namespace {

// Returns log4(chunk_size) (the number of radix-4 register-tile +
// one-time-transpose rounds needed) when chunk_size is an exact power of 4,
// else -1. Used to gate the SLM Hadamard path's transpose-based
// decomposition, which only applies cleanly when every round can process
// whole groups of 4 contiguous elements.
int64_t compute_log4_rounds(int64_t chunk_size) {
  int64_t c = chunk_size;
  int64_t rounds = 0;
  while (c > 1 && (c % 4) == 0) {
    c /= 4;
    ++rounds;
  }
  return (c == 1) ? rounds : -1;
}

// Computes the largest power-of-two chunk_size that can safely use the SLM/
// fused-kernel path on `dev`, derived from its actual reported local memory
// capacity rather than a value tuned to one specific GPU.
int64_t compute_max_slm_chunk(sycl::device& dev) {
  const int64_t local_mem_bytes =
      static_cast<int64_t>(dev.get_info<sycl::info::device::local_mem_size>());

  // Stay clear of the exact SLM capacity (can hang the driver):
  // budget = min(90% of nominal, nominal - 4KB).
  constexpr double kPercentBudget = 0.90;
  constexpr int64_t kAbsoluteReserveBytes = 4096;
  const int64_t budget_bytes =
      std::min(static_cast<int64_t>(static_cast<double>(local_mem_bytes) *
                                    kPercentBudget),
               local_mem_bytes - kAbsoluteReserveBytes);

  // Reserve reduce_mem (2*WG_SIZE floats); a threshold safe for quantize is
  // safe for dequantize.
  const int64_t reduce_mem_floats = 2 * WG_SIZE;
  int64_t max_data_floats =
      budget_bytes / static_cast<int64_t>(sizeof(float)) - reduce_mem_floats;

  // Reserve a second chunk_size scratch for the radix-4 Hadamard ping-pong.
  max_data_floats /= 2;

  // Degenerate query: disable the SLM path (0 forces the large-chunk path).
  if (max_data_floats < WG_SIZE) return 0;

  int64_t chunk = 1;
  while (chunk * 2 <= max_data_floats) chunk *= 2;
  return chunk;
}

// Caches compute_max_slm_chunk() per device so the query only runs once per
// distinct device for the process lifetime (this is on the hot path).
int64_t get_max_slm_chunk(sycl::queue& q) {
  static std::mutex mutex;
  static std::unique_ptr<sycl::device> cached_device;
  static int64_t cached_value = -1;

  std::lock_guard<std::mutex> lock(mutex);
  if (!cached_device || *cached_device != q.get_device()) {
    cached_device = std::make_unique<sycl::device>(q.get_device());
    cached_value = compute_max_slm_chunk(*cached_device);
  }
  return cached_value;
}

}  // namespace

// RAII guard for a SYCL device scratch allocation.  On normal completion the
// caller calls release() and hands the pointer to an async host_task free
// chained on the final kernel.  If a submit throws before that hand-off, the
// guard drains the queue and frees the buffer, so scratch is never leaked on
// the exception path.
template <typename T>
class DeviceScratch {
 public:
  DeviceScratch(T* ptr, sycl::queue& q) : ptr_(ptr), q_(q) {}
  ~DeviceScratch() {
    if (ptr_) {
      try {
        q_.wait();
      } catch (...) {
      }
      sycl::free(ptr_, q_);
    }
  }
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  T* get() const { return ptr_; }
  void release() { ptr_ = nullptr; }

 private:
  T* ptr_;
  sycl::queue& q_;
};

// ---------------------------------------------------------------------------
// vec4 contiguous load/write-back: stride==1, no permutation,
// element_count % 4 == 0.
// ---------------------------------------------------------------------------
template <typename KVType>
class LoadGatherVec4Kernel;

template <typename KVType>
static sycl::event load_gather_contiguous_vec4(
    sycl::queue& q, const KVType* src, float* scratch, int64_t element_count,
    const std::vector<sycl::event>& deps) {
  const int64_t num_groups = element_count / 4;
  const int64_t global_size = ((num_groups + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<LoadGatherVec4Kernel<KVType>>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t g = static_cast<int64_t>(item.get_global_id(0));
          if (g >= num_groups) return;
          auto src_mptr = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(src);
          auto dst_mptr = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(scratch);
          sycl::vec<KVType, 4> sv;
          sv.load(static_cast<size_t>(g), src_mptr);
          sv.template convert<float>().store(static_cast<size_t>(g), dst_mptr);
        });
  });
}

template <typename KVType>
class PrecondWritebackVec4Kernel;

template <typename KVType>
static sycl::event precond_writeback_contiguous_vec4(
    sycl::queue& q, const float* scratch, KVType* dst, int64_t element_count,
    const std::vector<sycl::event>& deps) {
  const int64_t num_groups = element_count / 4;
  const int64_t global_size = ((num_groups + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<PrecondWritebackVec4Kernel<KVType>>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t g = static_cast<int64_t>(item.get_global_id(0));
          if (g >= num_groups) return;
          auto src_mptr = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(scratch);
          auto dst_mptr = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(dst);
          sycl::vec<float, 4> fv;
          fv.load(static_cast<size_t>(g), src_mptr);
          fv.template convert<KVType>().store(static_cast<size_t>(g), dst_mptr);
        });
  });
}

// ---------------------------------------------------------------------------
// Quantize kernel: one work-group per chunk
// Small chunks use SLM; large chunks use global scratch with multi-kernel
// Hadamard.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Global-memory Hadamard: one butterfly stage over all chunks per launch;
// log2(chunk_size) launches total.
// ---------------------------------------------------------------------------
class HadamardStageBatchedKernel;

static sycl::event hadamard_stage_global_batched(
    sycl::queue& q,
    float* base,         // scratch: num_chunks contiguous chunks of chunk_size
    int64_t chunk_size,  // n, power of two, same for every chunk
    int64_t num_chunks,
    int64_t h,  // butterfly half-stride for this stage
    const std::vector<sycl::event>& deps) {
  const int64_t num_pairs = chunk_size / 2;
  const int64_t total_pairs = num_chunks * num_pairs;
  const int64_t global_size = ((total_pairs + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<HadamardStageBatchedKernel>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
          if (gid >= total_pairs) return;
          const int64_t chunk_id = gid / num_pairs;
          const int64_t local_idx = gid % num_pairs;
          float* buf = base + chunk_id * chunk_size;
          const int64_t stride = h << 1;
          int64_t block = local_idx / h;
          int64_t off_in_block = local_idx % h;
          int64_t i = block * stride + off_in_block;
          float a = buf[i];
          float b = buf[i + h];
          buf[i] = a + b;
          buf[i + h] = a - b;
        });
  });
}

// Vec4 variant for h >= 4 (contiguous on both sides).
class HadamardStageBatchedVec4Kernel;

static sycl::event hadamard_stage_global_batched_vec4(
    sycl::queue& q, float* base, int64_t chunk_size, int64_t num_chunks,
    int64_t h,  // butterfly half-stride; caller guarantees h >= 4
    const std::vector<sycl::event>& deps) {
  const int64_t groups_per_h = h / 4;
  const int64_t groups_per_chunk = chunk_size / 8;  // (chunk_size/2)/4
  const int64_t total_groups = num_chunks * groups_per_chunk;
  const int64_t global_size =
      ((total_groups + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<HadamardStageBatchedVec4Kernel>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
          if (gid >= total_groups) return;
          const int64_t chunk_id = gid / groups_per_chunk;
          const int64_t group_idx = gid % groups_per_chunk;
          float* buf = base + chunk_id * chunk_size;
          const int64_t stride = h << 1;
          const int64_t block = group_idx / groups_per_h;
          const int64_t group_in_block = group_idx % groups_per_h;
          const int64_t i = block * stride + group_in_block * 4;

          auto mptr = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(buf);
          const size_t vidx_a = static_cast<size_t>(i / 4);
          const size_t vidx_b = static_cast<size_t>((i + h) / 4);
          sycl::vec<float, 4> va, vb;
          va.load(vidx_a, mptr);
          vb.load(vidx_b, mptr);
          const sycl::vec<float, 4> sum = va + vb;
          const sycl::vec<float, 4> diff = va - vb;
          sum.store(vidx_a, mptr);
          diff.store(vidx_b, mptr);
        });
  });
}

// Fuses stages h=1..block_size/2 into one launch (pairs stay within a block
// when 2h <= block_size). Plain per-stage SLM loop.
class HadamardBlockLocalKernel;

static sycl::event hadamard_block_local_batched(
    sycl::queue& q, float* base,
    int64_t block_size,  // B, power of two, always >= WG_SIZE when > 1
    int64_t num_blocks,  // total blocks across all chunks
    const std::vector<sycl::event>& deps) {
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    auto local_block = sycl::local_accessor<float, 1>(
        sycl::range<1>(static_cast<size_t>(block_size)), cgh);
    cgh.parallel_for<HadamardBlockLocalKernel>(
        sycl::nd_range<1>(static_cast<size_t>(num_blocks) * WG_SIZE, WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t block_id = static_cast<int64_t>(item.get_group(0));
          const int lid = static_cast<int>(item.get_local_id(0));
          const int wg_size = static_cast<int>(item.get_local_range(0));
          float* blk = base + block_id * block_size;
          float* local_buf =
              local_block.get_multi_ptr<sycl::access::decorated::no>()
                  .get_raw();

          for (int64_t i = lid; i < block_size; i += wg_size)
            local_buf[i] = blk[i];
          sycl::group_barrier(item.get_group());

          for (int64_t h = 1; h < block_size; h <<= 1) {
            const int64_t bfly_stride = h << 1;
            for (int64_t p = lid; p < block_size / 2; p += wg_size) {
              int64_t blk_idx = p / h;
              int64_t off_in_block = p % h;
              int64_t idx = blk_idx * bfly_stride + off_in_block;
              float a = local_buf[idx];
              float b = local_buf[idx + h];
              local_buf[idx] = a + b;
              local_buf[idx + h] = a - b;
            }
            sycl::group_barrier(item.get_group());
          }

          for (int64_t i = lid; i < block_size; i += wg_size)
            blk[i] = local_buf[i];
        });
  });
}

// Multiply buffer by a constant (vec4 body + scalar tail).
class ScaleKernelVec4;
class ScaleKernelTail;

static sycl::event scale_buffer(sycl::queue& q, float* buf, int64_t n,
                                float factor,
                                const std::vector<sycl::event>& deps) {
  const int64_t num_groups = n / 4;
  const int64_t tail = n - num_groups * 4;

  sycl::event main_event;
  if (num_groups > 0) {
    const int64_t global_size =
        ((num_groups + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
    main_event = q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(deps);
      cgh.parallel_for<ScaleKernelVec4>(
          sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
          [=](sycl::nd_item<1> item) {
            const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
            if (idx >= num_groups) return;
            auto mptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(buf);
            const size_t vidx = static_cast<size_t>(idx);
            sycl::vec<float, 4> v;
            v.load(vidx, mptr);
            v *= factor;
            v.store(vidx, mptr);
          });
    });
  }

  if (tail == 0) {
    return main_event;
  }

  std::vector<sycl::event> tail_deps = deps;
  if (num_groups > 0) tail_deps.push_back(main_event);
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(tail_deps);
    cgh.parallel_for<ScaleKernelTail>(
        sycl::range<1>(static_cast<size_t>(tail)), [=](sycl::item<1> item) {
          const int64_t idx =
              num_groups * 4 + static_cast<int64_t>(item.get_id(0));
          buf[idx] *= factor;
        });
  });
}

template <typename KVType, typename QType, bool Pack4Bit>
class QuantizeKernel;

template <typename KVType, typename QType, bool Pack4Bit>
class QuantizeKernelLarge;

template <typename KVType, typename QType, bool Pack4Bit>
class MinMaxPartialKernelLarge;

// Number of work-groups to split each chunk across in the large-chunk
// (global scratch) kernels. With few chunks (e.g. per_tensor: one chunk) a
// work-group-per-chunk launch leaves nearly the whole GPU idle, so split
// each chunk into `splits` equal slices. Returns 1 (legacy behaviour) when
// there are already enough chunks or the chunk doesn't divide evenly.
// Slices stay a multiple of 16 elements so every vector/packed path divides.
static int64_t large_kernel_splits(int64_t num_chunks, int64_t chunk_size) {
  constexpr int64_t kTargetGroups = 256;
  constexpr int64_t kMaxSplits = 128;
  constexpr int64_t kMinSlice = 4096;
  int64_t splits = 1;
  while (num_chunks * splits < kTargetGroups && splits < kMaxSplits &&
         chunk_size % (splits * 2 * 16) == 0 &&
         chunk_size / (splits * 2) >= kMinSlice) {
    splits *= 2;
  }
  return splits;
}

// Single sub-group per_token quantize kernel (original form, see
// quantize_block_sycl_impl).
template <typename KVType, int E>
class QuantizeSgSingleKernel;

// Register/sub-group variant of the SLM quantize kernel (see
// quantize_block_sycl_impl).
template <typename KVType, int E, int S>
class QuantizeSgKernel;

// ... and of the SLM dequantize kernel (see dequantize_block_sycl_impl).
template <typename KVType, int E, int S>
class DequantizeSgKernel;

// Pack4Bit: write two 4-bit values per byte, chunk-contiguous layout
// (private to quantize/dequantize_kvcache_sycl; only instantiated with
// int8_t).
template <typename KVType, typename QType, bool Pack4Bit = false>
sycl::event quantize_block_sycl_impl(sycl::queue& q, const KVType* src,
                                     QType* dst, float* scales_out,
                                     float* zps_out, int64_t element_count,
                                     const sycl_quantization_args& args,
                                     const std::vector<sycl::event>& deps) {
  static_assert(!Pack4Bit || std::is_same_v<QType, int8_t>,
                "Pack4Bit is only valid for QType=int8_t");

  int64_t chunks_per_layer = 1;
  if (args.scaling_method == "per_token") {
    chunks_per_layer = static_cast<int64_t>(args.blocks_num) * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    chunks_per_layer = args.head_dim;
  }

  const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
  const int64_t num_chunks = num_layers * chunks_per_layer;
  const int64_t chunk_size = element_count / num_chunks;
  // Per-layer element count; for per_channel, chunk_id/stride is the layer and
  // chunk_id%stride the channel.
  const int64_t single_layer_numel = element_count / num_layers;
  const bool is_per_channel = (args.scaling_method == "per_channel");
  const int64_t stride = is_per_channel ? args.head_dim : 1;

  // Fused-source addressing (SLM path only; quantize_kvcache_sycl routes
  // fused calls that can't take the SLM path through a split gather first).
  const bool fused_src = args.fused_src_head_num > 0;
  const int64_t f_head_num = fused_src ? args.fused_src_head_num : 1;
  const int64_t f_head_dim = args.head_dim;
  const int64_t f_hidden = f_head_num * f_head_dim;
  const int64_t f_tokens = fused_src ? single_layer_numel / f_head_dim : 0;
  const int64_t src_stride =
      fused_src ? (is_per_channel ? f_hidden : 1) : stride;

  const int qbit = args.qbit;
  const bool asym = args.asym;
  const bool rh = args.rh;
  // Large-chunk path: forward gather.
  const int32_t* perm = args.perm;
  const float* signs = args.signs;
  // SLM path: inverse-permutation scatter.
  const int32_t* inv_perm = args.inv_perm;
  const float* signs_reordered = args.signs_reordered;

  const int64_t qmax_int = (1 << (qbit - 1)) - 1;
  const int64_t qmin_int = -(1 << (qbit - 1));
  const float qmax_float = static_cast<float>(qmax_int);
  const float qmin_float = static_cast<float>(qmin_int);
  const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(chunk_size));

  const int64_t max_slm_chunk = get_max_slm_chunk(q);
  const bool use_local_mem = chunk_size <= max_slm_chunk;

  if constexpr (Pack4Bit) {
    // Row-contiguous (per_token / per_tensor-in-SLM) chunks of 64..1024
    // elements use the original single sub-group kernel below, kept separate
    // from the per_channel / multi-sub-group kernel: sharing one kernel body
    // raised its register spills from 2 KB to 9 KB and made per_token quantize
    // ~25% slower.
    if (use_local_mem && !is_per_channel && sg_quantize_chunk_ok(chunk_size) &&
        device_supports_sub_group_32(q)) {
      auto launch_sg = [&](auto e_tag) -> sycl::event {
        constexpr int E = decltype(e_tag)::value;
        constexpr int PB = E / 2;  // packed bytes per lane
        return q.submit([&](sycl::handler& cgh) {
          cgh.depends_on(deps);
          cgh.parallel_for<QuantizeSgSingleKernel<KVType, E>>(
              sycl::nd_range<1>(static_cast<size_t>(num_chunks) * kSgLanes,
                                kSgLanes),
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int chunk_id = static_cast<int>(item.get_group(0));
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_id()[0]);

                // Addressing: identical to the SLM kernel (per_token => stride
                // 1).
                int64_t src_offset =
                    static_cast<int64_t>(chunk_id) * chunk_size;
                const KVType* srcp = src;
                if (fused_src) {
                  const int64_t unit = chunk_id / chunks_per_layer;
                  const int64_t in_unit = chunk_id % chunks_per_layer;
                  src_offset = (unit / f_head_num) * f_tokens * f_hidden +
                               (unit % f_head_num) * f_head_dim +
                               (is_per_channel ? in_unit : in_unit * f_hidden);
                }

                // Gather-load this lane's E logical elements k = lane * E + j.
                float x[E];
                if (src_stride == 1 && !(rh && perm)) {
                  // Contiguous slice of the row: one wide load per lane.
                  sg_load_lane_row<KVType, E>(
                      srcp + src_offset + static_cast<int64_t>(lane) * E, x);
                  if (rh && signs) {
#pragma unroll
                    for (int j = 0; j < E; ++j)
                      x[j] *= signs[static_cast<int64_t>(lane) * E + j];
                  }
                } else {
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    const int64_t k = static_cast<int64_t>(lane) * E + j;
                    const int64_t m =
                        (rh && perm) ? static_cast<int64_t>(perm[k]) : k;
                    float v =
                        static_cast<float>(srcp[m * src_stride + src_offset]);
                    if (rh && signs) v *= signs[k];
                    x[j] = v;
                  }
                }

                float lmin = std::numeric_limits<float>::max();
                float lmax = std::numeric_limits<float>::lowest();
                if (rh) {
// Stages over the register-index bits (lowest bits first).
#pragma unroll
                  for (int h = 1; h < E; h <<= 1) {
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      if ((j & h) == 0) {
                        const float a = x[j], b = x[j | h];
                        x[j] = a + b;
                        x[j | h] = a - b;
                      }
                    }
                  }
// Stages over the lane-index bits: the lane with the bit clear
// holds `a`, the lane with it set holds `b`; new a = a + b,
// new b = a - b (operand order as in the SLM butterflies).
#pragma unroll
                  for (int mask = 1; mask < kSgLanes; mask <<= 1) {
                    const bool upper = (lane & mask) != 0;
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      const float u = sycl::permute_group_by_xor(
                          sg, x[j], static_cast<uint32_t>(mask));
                      x[j] = upper ? (u - x[j]) : (x[j] + u);
                    }
                  }
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    x[j] *= inv_sqrt_n;
                    lmin = sycl::fmin(lmin, x[j]);
                    lmax = sycl::fmax(lmax, x[j]);
                  }
                } else {
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    lmin = sycl::fmin(lmin, x[j]);
                    lmax = sycl::fmax(lmax, x[j]);
                  }
                }
                const float min_val =
                    sycl::reduce_over_group(sg, lmin, sycl::minimum<float>());
                const float max_val =
                    sycl::reduce_over_group(sg, lmax, sycl::maximum<float>());

                float scale, zeropoint = 0.0f;
                if (asym) {
                  scale = (max_val - min_val) / (qmax_float - qmin_float);
                  zeropoint = min_val - qmin_float * scale;
                } else {
                  scale = sycl::fmax(sycl::fabs(min_val), sycl::fabs(max_val)) /
                          qmax_float;
                }
                if (lane == 0) {
                  scales_out[chunk_id] = scale;
                  zps_out[chunk_id] = zeropoint;
                }
                const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

                // Quantize and pack two adjacent logical elements per byte.
                uint8_t pk[PB];
#pragma unroll
                for (int t = 0; t < PB; ++t) {
                  const float q0 = sycl::clamp(
                      sycl::round((x[2 * t] - zeropoint) * inv_scale),
                      qmin_float, qmax_float);
                  const float q1 = sycl::clamp(
                      sycl::round((x[2 * t + 1] - zeropoint) * inv_scale),
                      qmin_float, qmax_float);
                  const uint8_t lo = static_cast<uint8_t>(static_cast<int8_t>(
                                         static_cast<int>(q0))) &
                                     0x0F;
                  const uint8_t hi = static_cast<uint8_t>(static_cast<int8_t>(
                                         static_cast<int>(q1))) &
                                     0x0F;
                  pk[t] = static_cast<uint8_t>(lo | (hi << 4));
                }
                uint8_t* out_bytes =
                    reinterpret_cast<uint8_t*>(dst) +
                    static_cast<int64_t>(chunk_id) * (chunk_size / 2) +
                    lane * PB;
#pragma unroll
                for (int t = 0; t < PB; ++t) out_bytes[t] = pk[t];
              });
        });
      };
      switch (chunk_size / kSgLanes) {
        case 2:
          return launch_sg(std::integral_constant<int, 2>{});
        case 4:
          return launch_sg(std::integral_constant<int, 4>{});
        case 8:
          return launch_sg(std::integral_constant<int, 8>{});
        case 16:
          return launch_sg(std::integral_constant<int, 16>{});
        default:
          return launch_sg(std::integral_constant<int, 32>{});
      }
    }
    // per_channel (chunk <= 1024) and multi-sub-group (2048..8192) variants.
    const bool sg_single = use_local_mem && is_per_channel && rh &&
                           sg_quantize_chunk_ok(chunk_size);
    const bool sg_multi = use_local_mem && rh && sg_multi_chunk_ok(chunk_size);
    if ((sg_single || sg_multi) && device_supports_sub_group_32(q)) {
      auto launch_sg = [&](auto e_tag, auto s_tag) -> sycl::event {
        constexpr int E = decltype(e_tag)::value;
        constexpr int S = decltype(s_tag)::value;  // sub-groups per work-group
        constexpr int PB = E / 2;                  // packed bytes per lane
        return q.submit([&](sycl::handler& cgh) {
          cgh.depends_on(deps);
          // Cross-sub-group exchange / min-max partials (S > 1 only).
          auto exchange =
              make_sg_local<S>(cgh, static_cast<size_t>(E) * S * kSgLanes);
          auto red_mem = make_sg_local<S>(cgh, 2 * S);
          cgh.parallel_for<QuantizeSgKernel<KVType, E, S>>(
              sycl::nd_range<1>(static_cast<size_t>(num_chunks) * S * kSgLanes,
                                S * kSgLanes),
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = item.get_sub_group();
                const int chunk_id = static_cast<int>(item.get_group(0));
                const int lane = static_cast<int>(sg.get_local_id()[0]);
                const int sgi =
                    S > 1 ? static_cast<int>(sg.get_group_id()[0]) : 0;
                // First logical element owned by this lane.
                const int64_t k0 = static_cast<int64_t>(sgi) * (kSgLanes * E) +
                                   static_cast<int64_t>(lane) * E;

                // Addressing: identical to the SLM kernel.
                int64_t src_offset =
                    is_per_channel
                        ? (chunk_id / stride) * single_layer_numel +
                              (chunk_id % stride)
                        : static_cast<int64_t>(chunk_id) * chunk_size;
                const KVType* srcp = src;
                if (fused_src) {
                  const int64_t unit = chunk_id / chunks_per_layer;
                  const int64_t in_unit = chunk_id % chunks_per_layer;
                  src_offset = (unit / f_head_num) * f_tokens * f_hidden +
                               (unit % f_head_num) * f_head_dim +
                               (is_per_channel ? in_unit : in_unit * f_hidden);
                }

                // Gather-load this lane's E logical elements k = lane * E + j.
                float x[E];
                if (src_stride == 1 && !(rh && perm)) {
                  // Contiguous slice of the row: one wide load per lane.
                  sg_load_lane_row<KVType, E>(srcp + src_offset + k0, x);
                  if (rh && signs) {
#pragma unroll
                    for (int j = 0; j < E; ++j) x[j] *= signs[k0 + j];
                  }
                } else {
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    const int64_t k = k0 + j;
                    const int64_t m =
                        (rh && perm) ? static_cast<int64_t>(perm[k]) : k;
                    float v =
                        static_cast<float>(srcp[m * src_stride + src_offset]);
                    if (rh && signs) v *= signs[k];
                    x[j] = v;
                  }
                }

                float lmin = std::numeric_limits<float>::max();
                float lmax = std::numeric_limits<float>::lowest();
                if (rh) {
// Stages over the register-index bits (lowest bits first).
#pragma unroll
                  for (int h = 1; h < E; h <<= 1) {
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      if ((j & h) == 0) {
                        const float a = x[j], b = x[j | h];
                        x[j] = a + b;
                        x[j | h] = a - b;
                      }
                    }
                  }
// Stages over the lane-index bits: the lane with the bit clear
// holds `a`, the lane with it set holds `b`; new a = a + b,
// new b = a - b (operand order as in the SLM butterflies).
#pragma unroll
                  for (int mask = 1; mask < kSgLanes; mask <<= 1) {
                    const bool upper = (lane & mask) != 0;
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      const float u = sycl::permute_group_by_xor(
                          sg, x[j], static_cast<uint32_t>(mask));
                      x[j] = upper ? (u - x[j]) : (x[j] + u);
                    }
                  }
                  if constexpr (S > 1) {
                    sg_cross_subgroup_hadamard<E, S>(
                        item,
                        exchange
                            .template get_multi_ptr<
                                sycl::access::decorated::no>()
                            .get_raw(),
                        x, sgi, lane);
                  }
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    x[j] *= inv_sqrt_n;
                    lmin = sycl::fmin(lmin, x[j]);
                    lmax = sycl::fmax(lmax, x[j]);
                  }
                } else {
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    lmin = sycl::fmin(lmin, x[j]);
                    lmax = sycl::fmax(lmax, x[j]);
                  }
                }
                float min_val =
                    sycl::reduce_over_group(sg, lmin, sycl::minimum<float>());
                float max_val =
                    sycl::reduce_over_group(sg, lmax, sycl::maximum<float>());
                if constexpr (S > 1) {
                  float* red =
                      red_mem
                          .template get_multi_ptr<sycl::access::decorated::no>()
                          .get_raw();
                  if (lane == 0) {
                    red[sgi] = min_val;
                    red[S + sgi] = max_val;
                  }
                  sycl::group_barrier(item.get_group());
#pragma unroll
                  for (int u = 0; u < S; ++u) {
                    min_val = sycl::fmin(min_val, red[u]);
                    max_val = sycl::fmax(max_val, red[S + u]);
                  }
                }

                float scale, zeropoint = 0.0f;
                if (asym) {
                  scale = (max_val - min_val) / (qmax_float - qmin_float);
                  zeropoint = min_val - qmin_float * scale;
                } else {
                  scale = sycl::fmax(sycl::fabs(min_val), sycl::fabs(max_val)) /
                          qmax_float;
                }
                if (lane == 0 && sgi == 0) {
                  scales_out[chunk_id] = scale;
                  zps_out[chunk_id] = zeropoint;
                }
                const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

                // Quantize and pack two adjacent logical elements per byte.
                uint8_t pk[PB];
#pragma unroll
                for (int t = 0; t < PB; ++t) {
                  const float q0 = sycl::clamp(
                      sycl::round((x[2 * t] - zeropoint) * inv_scale),
                      qmin_float, qmax_float);
                  const float q1 = sycl::clamp(
                      sycl::round((x[2 * t + 1] - zeropoint) * inv_scale),
                      qmin_float, qmax_float);
                  const uint8_t lo = static_cast<uint8_t>(static_cast<int8_t>(
                                         static_cast<int>(q0))) &
                                     0x0F;
                  const uint8_t hi = static_cast<uint8_t>(static_cast<int8_t>(
                                         static_cast<int>(q1))) &
                                     0x0F;
                  pk[t] = static_cast<uint8_t>(lo | (hi << 4));
                }
                uint8_t* out_bytes =
                    reinterpret_cast<uint8_t*>(dst) +
                    static_cast<int64_t>(chunk_id) * (chunk_size / 2) + k0 / 2;
#pragma unroll
                for (int t = 0; t < PB; ++t) out_bytes[t] = pk[t];
              });
        });
      };
      using std::integral_constant;
      switch (chunk_size) {
        case 2048:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 2>{});
        case 4096:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 4>{});
        case 8192:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 8>{});
        default:
          break;
      }
      switch (chunk_size / kSgLanes) {
        case 2:
          return launch_sg(integral_constant<int, 2>{},
                           integral_constant<int, 1>{});
        case 4:
          return launch_sg(integral_constant<int, 4>{},
                           integral_constant<int, 1>{});
        case 8:
          return launch_sg(integral_constant<int, 8>{},
                           integral_constant<int, 1>{});
        case 16:
          return launch_sg(integral_constant<int, 16>{},
                           integral_constant<int, 1>{});
        default:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 1>{});
      }
    }
  }

  if (use_local_mem) {
    // Small-chunk path: local memory for data + Hadamard
    const int64_t rounds_L = rh ? compute_log4_rounds(chunk_size) : -1;
    const bool use_transpose_scheme = rounds_L >= 1;
    const size_t slm_wg = slm_work_group_size(is_per_channel, chunk_size, rh,
                                              SlmDirection::Quantize);
    return q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(deps);
      auto local_data = sycl::local_accessor<float, 1>(
          sycl::range<1>(static_cast<size_t>(chunk_size)), cgh);
      auto reduce_mem =
          sycl::local_accessor<float, 1>(sycl::range<1>(2 * slm_wg), cgh);
      auto local_scratch = sycl::local_accessor<float, 1>(
          sycl::range<1>(
              static_cast<size_t>(use_transpose_scheme ? chunk_size : 1)),
          cgh);

      cgh.parallel_for<QuantizeKernel<KVType, QType, Pack4Bit>>(
          sycl::nd_range<1>(static_cast<size_t>(num_chunks) * slm_wg, slm_wg),
          [=](sycl::nd_item<1> item) {
            const int chunk_id = static_cast<int>(item.get_group(0));
            const int lid = static_cast<int>(item.get_local_id(0));
            const int wg_size = static_cast<int>(item.get_local_range(0));
            float* local_buf =
                local_data.get_multi_ptr<sycl::access::decorated::no>()
                    .get_raw();
            float* red_buf =
                reduce_mem.get_multi_ptr<sycl::access::decorated::no>()
                    .get_raw();

            const int64_t offset =
                is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                     (chunk_id % stride)
                               : chunk_id * chunk_size;
            int64_t src_offset = offset;
            const KVType* srcp = src;
            if (fused_src) {
              const int64_t unit = chunk_id / chunks_per_layer;
              const int64_t in_unit = chunk_id % chunks_per_layer;
              src_offset = (unit / f_head_num) * f_tokens * f_hidden +
                           (unit % f_head_num) * f_head_dim +
                           (is_per_channel ? in_unit : in_unit * f_hidden);
            }

            // Load into local memory with optional precond. Iterate
            // over `m`, src's physical/natural index, so the HBM read
            // stays sequential; push the permutation indirection onto
            // the local_buf (SLM) write instead, via inv_perm[m] =
            // the logical slot that physical position m belongs to.
            if (rh && (inv_perm || signs_reordered)) {
              for (int64_t m = lid; m < chunk_size; m += wg_size) {
                float v = static_cast<float>(srcp[m * src_stride + src_offset]);
                if (signs_reordered) v *= signs_reordered[m];
                int64_t k = inv_perm ? static_cast<int64_t>(inv_perm[m]) : m;
                local_buf[k] = v;
              }
            } else if constexpr (kv_has_vec4_v<KVType>) {
              // vec4 load when stride==1 (no vec for bfloat16).
              if (src_stride == 1 && (!fused_src || chunk_size % 4 == 0)) {
                const int64_t num_vec_groups = chunk_size / 4;
                auto src_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(srcp + src_offset);
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                for (int64_t g = lid; g < num_vec_groups; g += wg_size) {
                  sycl::vec<KVType, 4> sv;
                  sv.load(static_cast<size_t>(g), src_mptr);
                  sv.template convert<float>().store(static_cast<size_t>(g),
                                                     local_mptr);
                }
              } else {
                for (int64_t i = lid; i < chunk_size; i += wg_size) {
                  local_buf[i] =
                      static_cast<float>(srcp[i * src_stride + src_offset]);
                }
              }
            } else {
              for (int64_t i = lid; i < chunk_size; i += wg_size) {
                local_buf[i] =
                    static_cast<float>(srcp[i * src_stride + src_offset]);
              }
            }
            sycl::group_barrier(item.get_group());

            // Min/max accumulators, populated below either by the
            // fused Hadamard-normalize pass (rh) or a plain scan
            // (!rh); reduced via subgroup shuffle further down.
            float lmin = std::numeric_limits<float>::max();
            float lmax = std::numeric_limits<float>::lowest();

            // Hadamard transform
            if (rh) {
              if (rounds_L >= 1) {
                // Radix-4: per round, stages h=1,2 in registers, then an SLM
                // transpose (ping-pong via scratch_buf) rotates the next base-4
                // digit.
                constexpr int64_t kR = 4;
                const int64_t num_groups = chunk_size / kR;
                float* scratch_buf =
                    local_scratch.get_multi_ptr<sycl::access::decorated::no>()
                        .get_raw();

                for (int64_t round = 0; round < rounds_L; ++round) {
                  for (int64_t g = lid; g < num_groups; g += wg_size) {
                    const int64_t base = g * kR;
                    const float r0 = local_buf[base + 0];
                    const float r1 = local_buf[base + 1];
                    const float r2 = local_buf[base + 2];
                    const float r3 = local_buf[base + 3];
                    const float a0 = r0 + r1, a1 = r0 - r1;
                    const float a2 = r2 + r3, a3 = r2 - r3;
                    local_buf[base + 0] = a0 + a2;
                    local_buf[base + 1] = a1 + a3;
                    local_buf[base + 2] = a0 - a2;
                    local_buf[base + 3] = a1 - a3;
                  }
                  sycl::group_barrier(item.get_group());

                  if (round + 1 < rounds_L) {
                    // Swap the base-4 digit at position 0
                    // with the digit at position (round+1),
                    // so that digit becomes fast/contiguous
                    // for the next round.
                    const int shift1 = static_cast<int>((round + 1) * 2);
                    for (int64_t a = lid; a < chunk_size; a += wg_size) {
                      const int64_t d0 = a & 0x3LL;
                      const int64_t d1 = (a >> shift1) & 0x3LL;
                      const int64_t new_a = (a & ~0x3LL & ~(0x3LL << shift1)) |
                                            d1 | (d0 << shift1);
                      scratch_buf[new_a] = local_buf[a];
                    }
                    sycl::group_barrier(item.get_group());
                    float* tmp = local_buf;
                    local_buf = scratch_buf;
                    scratch_buf = tmp;
                  }
                }

                // Final un-transpose; write into the buffer that is not
                // local_buf to avoid aliasing.
                {
                  const int top_shift = static_cast<int>((rounds_L - 1) * 2);
                  for (int64_t a = lid; a < chunk_size; a += wg_size) {
                    const int64_t bottom = a & 0x3LL;
                    const int64_t rest = a >> 2;
                    const int64_t n = rest | (bottom << top_shift);
                    scratch_buf[n] = local_buf[a];
                  }
                  sycl::group_barrier(item.get_group());
                  local_buf = scratch_buf;
                }
              } else {
                // chunk_size is a power of two here (Hadamard invariant), so
                // every stride h is too: index with 32-bit shifts/masks rather
                // than a 64-bit div/mod per butterfly (the latter dominated the
                // transform for non-power-of-4 sizes such as 128 and 512).
                const int n_half = static_cast<int>(chunk_size / 2);
                for (int lg = 0; (int64_t(1) << lg) < chunk_size; ++lg) {
                  const int h = 1 << lg;
                  const int hmask = h - 1;
                  for (int base = lid; base < n_half; base += wg_size) {
                    const int idx = ((base >> lg) << (lg + 1)) + (base & hmask);
                    float a = local_buf[idx];
                    float b = local_buf[idx + h];
                    local_buf[idx] = a + b;
                    local_buf[idx + h] = a - b;
                  }
                  sycl::group_barrier(item.get_group());
                }
              }
              // Normalize and accumulate min/max in one pass; no extra barrier
              // needed (each lane reads only its own writes).
              if (chunk_size % 4 == 0) {
                sycl::vec<float, 4> vmin(std::numeric_limits<float>::max());
                sycl::vec<float, 4> vmax(std::numeric_limits<float>::lowest());
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                const int64_t num_vec_groups = chunk_size / 4;
                for (int64_t g = lid; g < num_vec_groups; g += wg_size) {
                  sycl::vec<float, 4> v;
                  v.load(static_cast<size_t>(g), local_mptr);
                  v *= inv_sqrt_n;
                  v.store(static_cast<size_t>(g), local_mptr);
                  vmin = sycl::fmin(vmin, v);
                  vmax = sycl::fmax(vmax, v);
                }
                lmin = sycl::fmin(sycl::fmin(vmin[0], vmin[1]),
                                  sycl::fmin(vmin[2], vmin[3]));
                lmax = sycl::fmax(sycl::fmax(vmax[0], vmax[1]),
                                  sycl::fmax(vmax[2], vmax[3]));
              } else {
                for (int64_t i = lid; i < chunk_size; i += wg_size) {
                  float v = local_buf[i] * inv_sqrt_n;
                  local_buf[i] = v;
                  lmin = sycl::fmin(lmin, v);
                  lmax = sycl::fmax(lmax, v);
                }
              }
            } else {
              if (chunk_size % 4 == 0) {
                sycl::vec<float, 4> vmin(std::numeric_limits<float>::max());
                sycl::vec<float, 4> vmax(std::numeric_limits<float>::lowest());
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                const int64_t num_vec_groups = chunk_size / 4;
                for (int64_t g = lid; g < num_vec_groups; g += wg_size) {
                  sycl::vec<float, 4> v;
                  v.load(static_cast<size_t>(g), local_mptr);
                  vmin = sycl::fmin(vmin, v);
                  vmax = sycl::fmax(vmax, v);
                }
                lmin = sycl::fmin(sycl::fmin(vmin[0], vmin[1]),
                                  sycl::fmin(vmin[2], vmin[3]));
                lmax = sycl::fmax(sycl::fmax(vmax[0], vmax[1]),
                                  sycl::fmax(vmax[2], vmax[3]));
              } else {
                for (int64_t i = lid; i < chunk_size; i += wg_size) {
                  float v = local_buf[i];
                  lmin = sycl::fmin(lmin, v);
                  lmax = sycl::fmax(lmax, v);
                }
              }
            }

            // Subgroup shuffle reduce, then combine per-subgroup partials via
            // SLM.
            auto sg = item.get_sub_group();
            const int sg_id = static_cast<int>(sg.get_group_id()[0]);
            const int sg_lid = static_cast<int>(sg.get_local_id()[0]);
            const int num_sg = static_cast<int>(sg.get_group_range()[0]);
            lmin = sycl::reduce_over_group(sg, lmin, sycl::minimum<float>());
            lmax = sycl::reduce_over_group(sg, lmax, sycl::maximum<float>());
            if (sg_lid == 0) {
              red_buf[sg_id] = lmin;
              red_buf[wg_size + sg_id] = lmax;
            }
            // Barrier: publish local_buf and subgroup partials.
            sycl::group_barrier(item.get_group());

            // Compute scale
            float scale = 0.0f, zeropoint = 0.0f;
            if (lid == 0) {
              float min_val = red_buf[0], max_val = red_buf[wg_size];
              for (int s = 1; s < num_sg; ++s) {
                min_val = sycl::fmin(min_val, red_buf[s]);
                max_val = sycl::fmax(max_val, red_buf[wg_size + s]);
              }
              if (asym) {
                scale = (max_val - min_val) / (qmax_float - qmin_float);
                zeropoint = min_val - qmin_float * scale;
              } else {
                scale = sycl::fmax(sycl::fabs(min_val), sycl::fabs(max_val)) /
                        qmax_float;
              }
              red_buf[0] = scale;
              red_buf[1] = zeropoint;
              scales_out[chunk_id] = scale;
              zps_out[chunk_id] = zeropoint;
            }
            sycl::group_barrier(item.get_group());
            scale = red_buf[0];
            zeropoint = red_buf[1];
            float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize
            if constexpr (Pack4Bit) {
              // Fused int4 packing: chunk-contiguous layout (chunk_size is
              // even).
              const int64_t packed_chunk_size = chunk_size / 2;
              QType* packed_dst =
                  dst + static_cast<int64_t>(chunk_id) * packed_chunk_size;
              if (packed_chunk_size % 4 == 0) {
                // Vec8 batch: 4 packed bytes (8 logical elements)
                // per work-item. packed_chunk_size = chunk_size/2
                // is a multiple of 4 whenever chunk_size is a
                // power of two >= 8 (always true in practice).
                const int64_t num_groups = packed_chunk_size / 4;
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                auto packed_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(packed_dst);
                for (int64_t g = lid; g < num_groups; g += wg_size) {
                  sycl::vec<float, 8> v;
                  v.load(static_cast<size_t>(g), local_mptr);
                  sycl::vec<float, 8> qf = sycl::clamp(
                      sycl::round((v - sycl::vec<float, 8>(zeropoint)) *
                                  inv_scale),
                      sycl::vec<float, 8>(qmin_float),
                      sycl::vec<float, 8>(qmax_float));
                  sycl::vec<int8_t, 4> packed;
                  for (int k = 0; k < 4; ++k) {
                    const uint8_t lo = static_cast<uint8_t>(static_cast<int8_t>(
                                           static_cast<int>(qf[k * 2]))) &
                                       0x0F;
                    const uint8_t hi = static_cast<uint8_t>(static_cast<int8_t>(
                                           static_cast<int>(qf[k * 2 + 1]))) &
                                       0x0F;
                    packed[k] = static_cast<int8_t>(lo | (hi << 4));
                  }
                  packed.store(static_cast<size_t>(g), packed_mptr);
                }
              } else {
                for (int64_t p = lid; p < packed_chunk_size; p += wg_size) {
                  const int64_t i0 = p * 2, i1 = i0 + 1;
                  const int q0 = static_cast<int>(sycl::clamp(
                      sycl::round((local_buf[i0] - zeropoint) * inv_scale),
                      qmin_float, qmax_float));
                  const int q1 = static_cast<int>(sycl::clamp(
                      sycl::round((local_buf[i1] - zeropoint) * inv_scale),
                      qmin_float, qmax_float));
                  const uint8_t lo =
                      static_cast<uint8_t>(static_cast<int8_t>(q0)) & 0x0F;
                  const uint8_t hi =
                      static_cast<uint8_t>(static_cast<int8_t>(q1)) & 0x0F;
                  packed_dst[p] =
                      static_cast<QType>(static_cast<int8_t>(lo | (hi << 4)));
                }
              }
            } else {
              for (int64_t i = lid; i < chunk_size; i += wg_size) {
                float val = local_buf[i];
                float q_val =
                    sycl::clamp(sycl::round((val - zeropoint) * inv_scale),
                                qmin_float, qmax_float);
                dst[i * stride + offset] =
                    static_cast<QType>(static_cast<int>(q_val));
              }
            }
          });
    });
  } else {
    // Large-chunk path (global scratch).
    sycl::queue& kq = q;

    // Allocate a float scratch buffer for the entire element_count.
    float* scratch =
        sycl::malloc_device<float>(static_cast<size_t>(element_count), kq);
    DeviceScratch<float> scratch_guard(scratch, kq);

    // Step 1: Load into scratch (vec4 fast path when contiguous).
    const bool load_fast_contig =
        (stride == 1) && !(rh && (perm || signs)) && (element_count % 4 == 0);
    sycl::event load_event;
    if constexpr (kv_has_vec4_v<KVType>) {
      if (load_fast_contig) {
        load_event = load_gather_contiguous_vec4<KVType>(kq, src, scratch,
                                                         element_count, deps);
      } else {
        load_event = kq.submit([&](sycl::handler& cgh) {
          cgh.depends_on(deps);
          const int64_t global_size =
              ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
          cgh.parallel_for(
              sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
              [=](sycl::nd_item<1> item) {
                const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
                if (gid >= element_count) return;
                int64_t chunk_id = gid / chunk_size;
                int64_t i = gid % chunk_size;
                int64_t offset =
                    is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                         (chunk_id % stride)
                                   : chunk_id * chunk_size;

                float v;
                if (rh && (perm || signs)) {
                  int64_t src_idx = perm ? static_cast<int64_t>(perm[i]) : i;
                  v = static_cast<float>(src[src_idx * stride + offset]);
                  if (signs) v *= signs[i];
                } else {
                  v = static_cast<float>(src[i * stride + offset]);
                }
                scratch[gid] = v;
              });
        });
      }
    } else {
      load_event = kq.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const int64_t global_size =
            ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
            [=](sycl::nd_item<1> item) {
              const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
              if (gid >= element_count) return;
              int64_t chunk_id = gid / chunk_size;
              int64_t i = gid % chunk_size;
              int64_t offset = is_per_channel
                                   ? (chunk_id / stride) * single_layer_numel +
                                         (chunk_id % stride)
                                   : chunk_id * chunk_size;

              float v;
              if (rh && (perm || signs)) {
                int64_t src_idx = perm ? static_cast<int64_t>(perm[i]) : i;
                v = static_cast<float>(src[src_idx * stride + offset]);
                if (signs) v *= signs[i];
              } else {
                v = static_cast<float>(src[i * stride + offset]);
              }
              scratch[gid] = v;
            });
      });
    }

    // Step 2: Hadamard (if rh): block-local stages in one launch, then one
    // launch per cross-block stage.
    sycl::event had_event = load_event;
    if (rh) {
      sycl::event stage_event = load_event;
      std::vector<sycl::event> stage_deps = {load_event};

      const int64_t block_size = std::min(max_slm_chunk, chunk_size);
      int64_t h_start = 1;
      if (block_size > 1) {
        const int64_t num_blocks = element_count / block_size;
        stage_event = hadamard_block_local_batched(kq, scratch, block_size,
                                                   num_blocks, stage_deps);
        stage_deps = {stage_event};
        h_start = block_size;
      }

      for (int64_t h = h_start; h < chunk_size; h <<= 1) {
        stage_event =
            (h >= 4) ? hadamard_stage_global_batched_vec4(
                           kq, scratch, chunk_size, num_chunks, h, stage_deps)
                     : hadamard_stage_global_batched(kq, scratch, chunk_size,
                                                     num_chunks, h, stage_deps);
        stage_deps = {stage_event};
      }
      had_event =
          scale_buffer(kq, scratch, element_count, inv_sqrt_n, {stage_event});
    }

    // Step 3: Min/max reduction + quantize from scratch buffer. Each chunk is
    // split across `splits` work-groups: a partial min/max pass first, then
    // every quantize group folds its chunk's partials into scale/zp.
    const int64_t splits = large_kernel_splits(num_chunks, chunk_size);
    float* partials = nullptr;
    sycl::event minmax_event = had_event;
    if (splits > 1) {
      partials = sycl::malloc_device<float>(
          static_cast<size_t>(2 * num_chunks * splits), kq);
      minmax_event = kq.submit([&](sycl::handler& cgh) {
        cgh.depends_on({had_event});
        auto reduce_mem =
            sycl::local_accessor<float, 1>(sycl::range<1>(2 * WG_SIZE), cgh);
        cgh.parallel_for<MinMaxPartialKernelLarge<KVType, QType, Pack4Bit>>(
            sycl::nd_range<1>(
                static_cast<size_t>(num_chunks * splits) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              const int grp = static_cast<int>(item.get_group(0));
              const int chunk_id = grp / static_cast<int>(splits);
              const int sid = grp % static_cast<int>(splits);
              const int lid = static_cast<int>(item.get_local_id(0));
              const int wg_size = static_cast<int>(item.get_local_range(0));
              float* red_buf =
                  reduce_mem.get_multi_ptr<sycl::access::decorated::no>()
                      .get_raw();
              const int64_t slice = chunk_size / splits;
              const float* slice_data =
                  scratch + chunk_id * chunk_size + sid * slice;

              sycl::vec<float, 4> vmin(std::numeric_limits<float>::max());
              sycl::vec<float, 4> vmax(std::numeric_limits<float>::lowest());
              auto data_mptr = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(slice_data);
              for (int64_t g = lid; g < slice / 4; g += wg_size) {
                sycl::vec<float, 4> v;
                v.load(static_cast<size_t>(g), data_mptr);
                vmin = sycl::fmin(vmin, v);
                vmax = sycl::fmax(vmax, v);
              }
              float lmin = sycl::fmin(sycl::fmin(vmin[0], vmin[1]),
                                      sycl::fmin(vmin[2], vmin[3]));
              float lmax = sycl::fmax(sycl::fmax(vmax[0], vmax[1]),
                                      sycl::fmax(vmax[2], vmax[3]));
              auto sg = item.get_sub_group();
              const int sg_id = static_cast<int>(sg.get_group_id()[0]);
              const int sg_lid = static_cast<int>(sg.get_local_id()[0]);
              const int num_sg = static_cast<int>(sg.get_group_range()[0]);
              lmin = sycl::reduce_over_group(sg, lmin, sycl::minimum<float>());
              lmax = sycl::reduce_over_group(sg, lmax, sycl::maximum<float>());
              if (sg_lid == 0) {
                red_buf[sg_id] = lmin;
                red_buf[wg_size + sg_id] = lmax;
              }
              sycl::group_barrier(item.get_group());
              if (lid == 0) {
                float mn = red_buf[0], mx = red_buf[wg_size];
                for (int s = 1; s < num_sg; ++s) {
                  mn = sycl::fmin(mn, red_buf[s]);
                  mx = sycl::fmax(mx, red_buf[wg_size + s]);
                }
                partials[2 * grp] = mn;
                partials[2 * grp + 1] = mx;
              }
            });
      });
    }
    auto quant_event = kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on({minmax_event});
      auto reduce_mem =
          sycl::local_accessor<float, 1>(sycl::range<1>(2 * WG_SIZE), cgh);

      cgh.parallel_for<QuantizeKernelLarge<KVType, QType, Pack4Bit>>(
          sycl::nd_range<1>(static_cast<size_t>(num_chunks * splits) * WG_SIZE,
                            WG_SIZE),
          [=](sycl::nd_item<1> item) {
            const int grp = static_cast<int>(item.get_group(0));
            const int chunk_id = grp / static_cast<int>(splits);
            const int sid = grp % static_cast<int>(splits);
            const int lid = static_cast<int>(item.get_local_id(0));
            const int wg_size = static_cast<int>(item.get_local_range(0));
            float* red_buf =
                reduce_mem.get_multi_ptr<sycl::access::decorated::no>()
                    .get_raw();

            const int64_t offset =
                is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                     (chunk_id % stride)
                               : chunk_id * chunk_size;
            const float* chunk_data = scratch + chunk_id * chunk_size;

            // Min/max reduction (subgroup shuffle + SLM partials). When the
            // chunk is split, the partial pass already did this.
            float lmin = std::numeric_limits<float>::max();
            float lmax = std::numeric_limits<float>::lowest();
            if (splits > 1) {
              // fall through: scale is folded from `partials` below
            } else if (chunk_size % 4 == 0) {
              sycl::vec<float, 4> vmin(std::numeric_limits<float>::max());
              sycl::vec<float, 4> vmax(std::numeric_limits<float>::lowest());
              auto data_mptr = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(chunk_data);
              const int64_t num_vec_groups = chunk_size / 4;
              for (int64_t g = lid; g < num_vec_groups; g += wg_size) {
                sycl::vec<float, 4> v;
                v.load(static_cast<size_t>(g), data_mptr);
                vmin = sycl::fmin(vmin, v);
                vmax = sycl::fmax(vmax, v);
              }
              lmin = sycl::fmin(sycl::fmin(vmin[0], vmin[1]),
                                sycl::fmin(vmin[2], vmin[3]));
              lmax = sycl::fmax(sycl::fmax(vmax[0], vmax[1]),
                                sycl::fmax(vmax[2], vmax[3]));
            } else {
              for (int64_t i = lid; i < chunk_size; i += wg_size) {
                float v = chunk_data[i];
                lmin = sycl::fmin(lmin, v);
                lmax = sycl::fmax(lmax, v);
              }
            }
            auto sg = item.get_sub_group();
            const int sg_id = static_cast<int>(sg.get_group_id()[0]);
            const int sg_lid = static_cast<int>(sg.get_local_id()[0]);
            const int num_sg = static_cast<int>(sg.get_group_range()[0]);
            if (splits == 1) {
              lmin = sycl::reduce_over_group(sg, lmin, sycl::minimum<float>());
              lmax = sycl::reduce_over_group(sg, lmax, sycl::maximum<float>());
              if (sg_lid == 0) {
                red_buf[sg_id] = lmin;
                red_buf[wg_size + sg_id] = lmax;
              }
              sycl::group_barrier(item.get_group());
            }

            // Compute scale
            float scale = 0.0f, zeropoint = 0.0f;
            if (lid == 0) {
              float min_val, max_val;
              if (splits > 1) {
                const float* pp =
                    partials + 2 * static_cast<int64_t>(chunk_id) * splits;
                min_val = pp[0];
                max_val = pp[1];
                for (int s = 1; s < static_cast<int>(splits); ++s) {
                  min_val = sycl::fmin(min_val, pp[2 * s]);
                  max_val = sycl::fmax(max_val, pp[2 * s + 1]);
                }
              } else {
                min_val = red_buf[0];
                max_val = red_buf[wg_size];
                for (int s = 1; s < num_sg; ++s) {
                  min_val = sycl::fmin(min_val, red_buf[s]);
                  max_val = sycl::fmax(max_val, red_buf[wg_size + s]);
                }
              }
              if (asym) {
                scale = (max_val - min_val) / (qmax_float - qmin_float);
                zeropoint = min_val - qmin_float * scale;
              } else {
                scale = sycl::fmax(sycl::fabs(min_val), sycl::fabs(max_val)) /
                        qmax_float;
              }
              red_buf[0] = scale;
              red_buf[1] = zeropoint;
              if (sid == 0) {
                scales_out[chunk_id] = scale;
                zps_out[chunk_id] = zeropoint;
              }
            }
            sycl::group_barrier(item.get_group());
            scale = red_buf[0];
            zeropoint = red_buf[1];
            float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize this work-group's slice from scratch buffer, write to
            // dst
            if constexpr (Pack4Bit) {
              const int64_t packed_chunk_size = chunk_size / 2;
              QType* packed_dst =
                  dst + static_cast<int64_t>(chunk_id) * packed_chunk_size;
              if (packed_chunk_size % 4 == 0) {
                const int64_t slice_groups = packed_chunk_size / 4 / splits;
                const int64_t g_begin = sid * slice_groups;
                const int64_t g_end = g_begin + slice_groups;
                auto data_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(chunk_data);
                auto packed_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(packed_dst);
                for (int64_t g = g_begin + lid; g < g_end; g += wg_size) {
                  sycl::vec<float, 8> v;
                  v.load(static_cast<size_t>(g), data_mptr);
                  sycl::vec<float, 8> qf = sycl::clamp(
                      sycl::round((v - sycl::vec<float, 8>(zeropoint)) *
                                  inv_scale),
                      sycl::vec<float, 8>(qmin_float),
                      sycl::vec<float, 8>(qmax_float));
                  sycl::vec<int8_t, 4> packed;
                  for (int k = 0; k < 4; ++k) {
                    const uint8_t lo = static_cast<uint8_t>(static_cast<int8_t>(
                                           static_cast<int>(qf[k * 2]))) &
                                       0x0F;
                    const uint8_t hi = static_cast<uint8_t>(static_cast<int8_t>(
                                           static_cast<int>(qf[k * 2 + 1]))) &
                                       0x0F;
                    packed[k] = static_cast<int8_t>(lo | (hi << 4));
                  }
                  packed.store(static_cast<size_t>(g), packed_mptr);
                }
              } else {
                for (int64_t p = sid * (packed_chunk_size / splits) + lid;
                     p < (sid + 1) * (packed_chunk_size / splits);
                     p += wg_size) {
                  const int64_t i0 = p * 2, i1 = i0 + 1;
                  const int q0 = static_cast<int>(sycl::clamp(
                      sycl::round((chunk_data[i0] - zeropoint) * inv_scale),
                      qmin_float, qmax_float));
                  const int q1 = static_cast<int>(sycl::clamp(
                      sycl::round((chunk_data[i1] - zeropoint) * inv_scale),
                      qmin_float, qmax_float));
                  const uint8_t lo =
                      static_cast<uint8_t>(static_cast<int8_t>(q0)) & 0x0F;
                  const uint8_t hi =
                      static_cast<uint8_t>(static_cast<int8_t>(q1)) & 0x0F;
                  packed_dst[p] =
                      static_cast<QType>(static_cast<int8_t>(lo | (hi << 4)));
                }
              }
            } else {
              for (int64_t i = sid * (chunk_size / splits) + lid;
                   i < (sid + 1) * (chunk_size / splits); i += wg_size) {
                float val = chunk_data[i];
                float q_val =
                    sycl::clamp(sycl::round((val - zeropoint) * inv_scale),
                                qmin_float, qmax_float);
                dst[i * stride + offset] =
                    static_cast<QType>(static_cast<int>(q_val));
              }
            }
          });
    });

    // Success: hand scratch off to the async free (chained on quant_event).
    scratch_guard.release();
    kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on(quant_event);
      cgh.host_task([=, &kq]() {
        sycl::free(scratch, kq);
        if (partials) sycl::free(partials, kq);
      });
    });

    return quant_event;
  }
}

// ---------------------------------------------------------------------------
// Dequantize kernel: one work-group per chunk, with Hadamard support
// ---------------------------------------------------------------------------
template <typename KVType, typename QType, bool Pack4Bit>
class DequantizeKernel;

template <typename KVType, typename QType, bool Pack4Bit>
class DequantizeKernelLarge;

// Pack4Bit: src is chunk-contiguous packed nibbles.
template <typename KVType, typename QType, bool Pack4Bit = false>
sycl::event dequantize_block_sycl_impl(sycl::queue& q, const QType* src,
                                       KVType* dst, const float* scales_in,
                                       const float* zps_in,
                                       int64_t element_count,
                                       const sycl_quantization_args& args,
                                       const std::vector<sycl::event>& deps) {
  static_assert(!Pack4Bit || std::is_same_v<QType, int8_t>,
                "Pack4Bit is only valid for QType=int8_t");

  int64_t chunks_per_layer = 1;
  if (args.scaling_method == "per_token") {
    chunks_per_layer = static_cast<int64_t>(args.blocks_num) * args.block_size;
  } else if (args.scaling_method == "per_channel") {
    chunks_per_layer = args.head_dim;
  }

  const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
  const int64_t num_chunks = num_layers * chunks_per_layer;
  const int64_t chunk_size = element_count / num_chunks;
  // Per-layer element count (per_channel: chunk_id/stride is the layer).
  const int64_t single_layer_numel = element_count / num_layers;
  const bool is_per_channel = (args.scaling_method == "per_channel");
  const int64_t stride = is_per_channel ? args.head_dim : 1;
  const bool rh = args.rh;
  // Large-chunk path: forward scatter.
  const int32_t* perm = args.perm;
  const float* signs = args.signs;
  // SLM path: inverse-permutation gather.
  const int32_t* inv_perm = args.inv_perm;
  const float* signs_reordered = args.signs_reordered;
  const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(chunk_size));

  // Fused-destination addressing (SLM path only; dequantize_kvcache_sycl
  // routes fused calls that can't take the SLM path through a split scratch).
  const bool fused_dst = args.fused_dst_head_num > 0;
  const int64_t f_head_num = fused_dst ? args.fused_dst_head_num : 1;
  const int64_t f_head_dim = args.head_dim;
  const int64_t f_hidden = f_head_num * f_head_dim;
  const int64_t f_tokens = fused_dst ? single_layer_numel / f_head_dim : 0;
  const int64_t dst_stride =
      fused_dst ? (is_per_channel ? f_hidden : 1) : stride;

  const int64_t max_slm_chunk = get_max_slm_chunk(q);
  const bool use_local_mem = chunk_size <= max_slm_chunk;

  if constexpr (Pack4Bit) {
    // per_channel stays on the SLM kernel for chunks <= 1024: measured slower
    // here than SLM (its strided write-back gains nothing from the register
    // Hadamard).
    const bool sg_single =
        use_local_mem && !is_per_channel && sg_quantize_chunk_ok(chunk_size);
    // With preconditioning the multi-sub-group dequantize adds a second
    // work-group barrier and doubles the SLM (exchange + inverse-permutation
    // staging); it measured slower than the SLM kernel on Arc Pro B70, so only
    // the plain RH inverse takes it.
    const bool sg_multi = use_local_mem && rh &&
                          !(inv_perm || signs_reordered) &&
                          sg_multi_chunk_ok(chunk_size);
    if ((sg_single || sg_multi) && device_supports_sub_group_32(q)) {
      auto launch_sg = [&](auto e_tag, auto s_tag) -> sycl::event {
        constexpr int E = decltype(e_tag)::value;
        constexpr int S = decltype(s_tag)::value;  // sub-groups per work-group
        constexpr int PB = E / 2;                  // packed bytes per lane
        return q.submit([&](sycl::handler& cgh) {
          cgh.depends_on(deps);
          // One chunk of floats per work-group (== one sub-group): used only to
          // transpose the preconditioning permutation on the way out.
          auto local_data = sycl::local_accessor<float, 1>(
              sycl::range<1>(static_cast<size_t>(chunk_size)), cgh);
          auto exchange =
              make_sg_local<S>(cgh, static_cast<size_t>(E) * S * kSgLanes);
          cgh.parallel_for<DequantizeSgKernel<KVType, E, S>>(
              sycl::nd_range<1>(static_cast<size_t>(num_chunks) * S * kSgLanes,
                                S * kSgLanes),
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int chunk_id = static_cast<int>(item.get_group(0));
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_id()[0]);
                const int sgi =
                    S > 1 ? static_cast<int>(sg.get_group_id()[0]) : 0;
                // First logical element owned by this lane.
                const int64_t k0 = static_cast<int64_t>(sgi) * (kSgLanes * E) +
                                   static_cast<int64_t>(lane) * E;

                // Addressing: identical to the SLM kernel.
                int64_t dst_offset =
                    is_per_channel
                        ? (chunk_id / stride) * single_layer_numel +
                              (chunk_id % stride)
                        : static_cast<int64_t>(chunk_id) * chunk_size;
                KVType* dstp = dst;
                if (fused_dst) {
                  const int64_t unit = chunk_id / chunks_per_layer;
                  const int64_t in_unit = chunk_id % chunks_per_layer;
                  dst_offset = (unit / f_head_num) * f_tokens * f_hidden +
                               (unit % f_head_num) * f_head_dim +
                               (is_per_channel ? in_unit : in_unit * f_hidden);
                }
                const float scale = scales_in[chunk_id];
                const float zeropoint = zps_in[chunk_id];

                // Unpack this lane's E consecutive logical elements.
                const QType* packed_src =
                    src + static_cast<int64_t>(chunk_id) * (chunk_size / 2) +
                    k0 / 2;
                float x[E];
#pragma unroll
                for (int t = 0; t < PB; ++t) {
                  const uint8_t byte = static_cast<uint8_t>(packed_src[t]);
                  const uint8_t lo = byte & 0x0F;
                  const uint8_t hi = (byte >> 4) & 0x0F;
                  const int8_t qlo =
                      (lo >= 8) ? static_cast<int8_t>(static_cast<int>(lo) - 16)
                                : static_cast<int8_t>(lo);
                  const int8_t qhi =
                      (hi >= 8) ? static_cast<int8_t>(static_cast<int>(hi) - 16)
                                : static_cast<int8_t>(hi);
                  x[2 * t] = static_cast<float>(qlo) * scale + zeropoint;
                  x[2 * t + 1] = static_cast<float>(qhi) * scale + zeropoint;
                }

                if (rh) {
// Same bit order and operand order as the SLM butterflies.
#pragma unroll
                  for (int h = 1; h < E; h <<= 1) {
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      if ((j & h) == 0) {
                        const float a = x[j], b = x[j | h];
                        x[j] = a + b;
                        x[j | h] = a - b;
                      }
                    }
                  }
#pragma unroll
                  for (int mask = 1; mask < kSgLanes; mask <<= 1) {
                    const bool upper = (lane & mask) != 0;
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      const float u = sycl::permute_group_by_xor(
                          sg, x[j], static_cast<uint32_t>(mask));
                      x[j] = upper ? (u - x[j]) : (x[j] + u);
                    }
                  }
                  if constexpr (S > 1) {
                    sg_cross_subgroup_hadamard<E, S>(
                        item,
                        exchange
                            .template get_multi_ptr<
                                sycl::access::decorated::no>()
                            .get_raw(),
                        x, sgi, lane);
                  }
#pragma unroll
                  for (int j = 0; j < E; ++j) x[j] *= inv_sqrt_n;
                }

                // Inverse precondition + write, same formulation as the SLM
                // kernel: dst[m] = local[inv_perm[m]] * signs_reordered[m]. The
                // values are parked in SLM by logical index k (one
                // sub-group-local pass, no work-group barrier) so the global
                // stores below stay contiguous per lane instead of scattering
                // 2-byte writes across the row.
                const bool precond = rh && (inv_perm || signs_reordered);
                if (precond) {
                  float* lb =
                      local_data.get_multi_ptr<sycl::access::decorated::no>()
                          .get_raw();
#pragma unroll
                  for (int j = 0; j < E; ++j) lb[k0 + j] = x[j];
                  // One sub-group owns the whole chunk when S == 1: a sub-group
                  // sync is enough (a work-group barrier is a real barrier
                  // instruction).
                  if constexpr (S > 1) {
                    sycl::group_barrier(item.get_group());
                  } else {
                    sycl::group_barrier(sg);
                  }
#pragma unroll
                  for (int j = 0; j < E; ++j) {
                    const int64_t m = k0 + j;
                    const int64_t k =
                        inv_perm ? static_cast<int64_t>(inv_perm[m]) : m;
                    float v = lb[k];
                    if (signs_reordered) v *= signs_reordered[m];
                    x[j] = v;  // x is dead after being parked in SLM; reuse it
                               // for the output values
                  }
                  if (dst_stride == 1) {
                    sg_store_lane_row<KVType, E>(dstp + dst_offset + k0, x);
                  } else {
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      dstp[(k0 + j) * dst_stride + dst_offset] =
                          static_cast<KVType>(x[j]);
                    }
                  }
                } else {
                  if (dst_stride == 1) {
                    sg_store_lane_row<KVType, E>(dstp + dst_offset + k0, x);
                  } else {
#pragma unroll
                    for (int j = 0; j < E; ++j) {
                      dstp[(k0 + j) * dst_stride + dst_offset] =
                          static_cast<KVType>(x[j]);
                    }
                  }
                }
              });
        });
      };
      using std::integral_constant;
      switch (chunk_size) {
        case 2048:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 2>{});
        case 4096:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 4>{});
        case 8192:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 8>{});
        default:
          break;
      }
      switch (chunk_size / kSgLanes) {
        case 2:
          return launch_sg(integral_constant<int, 2>{},
                           integral_constant<int, 1>{});
        case 4:
          return launch_sg(integral_constant<int, 4>{},
                           integral_constant<int, 1>{});
        case 8:
          return launch_sg(integral_constant<int, 8>{},
                           integral_constant<int, 1>{});
        case 16:
          return launch_sg(integral_constant<int, 16>{},
                           integral_constant<int, 1>{});
        default:
          return launch_sg(integral_constant<int, 32>{},
                           integral_constant<int, 1>{});
      }
    }
  }

  if (use_local_mem) {
    const int64_t rounds_L = rh ? compute_log4_rounds(chunk_size) : -1;
    const bool use_transpose_scheme = rounds_L >= 1;
    const size_t slm_wg = slm_work_group_size(is_per_channel, chunk_size, rh,
                                              SlmDirection::Dequantize);
    return q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(deps);
      auto local_data = sycl::local_accessor<float, 1>(
          sycl::range<1>(static_cast<size_t>(chunk_size)), cgh);
      auto local_scratch = sycl::local_accessor<float, 1>(
          sycl::range<1>(
              static_cast<size_t>(use_transpose_scheme ? chunk_size : 1)),
          cgh);

      cgh.parallel_for<DequantizeKernel<KVType, QType, Pack4Bit>>(
          sycl::nd_range<1>(static_cast<size_t>(num_chunks) * slm_wg, slm_wg),
          [=](sycl::nd_item<1> item) {
            const int chunk_id = static_cast<int>(item.get_group(0));
            const int lid = static_cast<int>(item.get_local_id(0));
            const int wg_size = static_cast<int>(item.get_local_range(0));
            float* local_buf =
                local_data.get_multi_ptr<sycl::access::decorated::no>()
                    .get_raw();
            const int64_t offset =
                is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                     (chunk_id % stride)
                               : chunk_id * chunk_size;
            int64_t dst_offset = offset;
            KVType* dstp = dst;
            if (fused_dst) {
              const int64_t unit = chunk_id / chunks_per_layer;
              const int64_t in_unit = chunk_id % chunks_per_layer;
              dst_offset = (unit / f_head_num) * f_tokens * f_hidden +
                           (unit % f_head_num) * f_head_dim +
                           (is_per_channel ? in_unit : in_unit * f_hidden);
            }
            const float scale = scales_in[chunk_id];
            const float zeropoint = zps_in[chunk_id];

            // Each thread unpacks both nibbles of one byte (avoids redundant
            // byte reads).
            if constexpr (Pack4Bit) {
              const int64_t packed_chunk_size = chunk_size / 2;
              const QType* packed_src =
                  src + static_cast<int64_t>(chunk_id) * packed_chunk_size;
              if (packed_chunk_size % 4 == 0) {
                // Vec4 batch: 4 packed bytes (8 logical elements)
                // per work-item.
                const int64_t num_groups = packed_chunk_size / 4;
                auto packed_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(packed_src);
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                for (int64_t g = lid; g < num_groups; g += wg_size) {
                  sycl::vec<int8_t, 4> pv;
                  pv.load(static_cast<size_t>(g), packed_mptr);
                  sycl::vec<float, 8> out;
                  for (int k = 0; k < 4; ++k) {
                    const uint8_t byte = static_cast<uint8_t>(pv[k]);
                    const uint8_t lo = byte & 0x0F;
                    const uint8_t hi = (byte >> 4) & 0x0F;
                    const int8_t qlo =
                        (lo >= 8)
                            ? static_cast<int8_t>(static_cast<int>(lo) - 16)
                            : static_cast<int8_t>(lo);
                    const int8_t qhi =
                        (hi >= 8)
                            ? static_cast<int8_t>(static_cast<int>(hi) - 16)
                            : static_cast<int8_t>(hi);
                    out[k * 2] = static_cast<float>(qlo) * scale + zeropoint;
                    out[k * 2 + 1] =
                        static_cast<float>(qhi) * scale + zeropoint;
                  }
                  out.store(static_cast<size_t>(g), local_mptr);
                }
              } else {
                for (int64_t p = lid; p < packed_chunk_size; p += wg_size) {
                  const uint8_t byte = static_cast<uint8_t>(packed_src[p]);
                  const uint8_t lo = byte & 0x0F;
                  const uint8_t hi = (byte >> 4) & 0x0F;
                  const int8_t qlo =
                      (lo >= 8) ? static_cast<int8_t>(static_cast<int>(lo) - 16)
                                : static_cast<int8_t>(lo);
                  const int8_t qhi =
                      (hi >= 8) ? static_cast<int8_t>(static_cast<int>(hi) - 16)
                                : static_cast<int8_t>(hi);
                  local_buf[p * 2] =
                      static_cast<float>(qlo) * scale + zeropoint;
                  local_buf[p * 2 + 1] =
                      static_cast<float>(qhi) * scale + zeropoint;
                }
              }
            } else {
              for (int64_t i = lid; i < chunk_size; i += wg_size) {
                float qv = static_cast<float>(src[i * stride + offset]);
                local_buf[i] = qv * scale + zeropoint;
              }
            }
            sycl::group_barrier(item.get_group());

            // Inverse Hadamard
            if (rh) {
              if (rounds_L >= 1) {
                // Radix-4 Hadamard (self-inverse, same as forward path).
                constexpr int64_t kR = 4;
                const int64_t num_groups = chunk_size / kR;
                float* scratch_buf =
                    local_scratch.get_multi_ptr<sycl::access::decorated::no>()
                        .get_raw();

                for (int64_t round = 0; round < rounds_L; ++round) {
                  for (int64_t g = lid; g < num_groups; g += wg_size) {
                    const int64_t base = g * kR;
                    const float r0 = local_buf[base + 0];
                    const float r1 = local_buf[base + 1];
                    const float r2 = local_buf[base + 2];
                    const float r3 = local_buf[base + 3];
                    const float a0 = r0 + r1, a1 = r0 - r1;
                    const float a2 = r2 + r3, a3 = r2 - r3;
                    local_buf[base + 0] = a0 + a2;
                    local_buf[base + 1] = a1 + a3;
                    local_buf[base + 2] = a0 - a2;
                    local_buf[base + 3] = a1 - a3;
                  }
                  sycl::group_barrier(item.get_group());

                  if (round + 1 < rounds_L) {
                    const int shift1 = static_cast<int>((round + 1) * 2);
                    for (int64_t a = lid; a < chunk_size; a += wg_size) {
                      const int64_t d0 = a & 0x3LL;
                      const int64_t d1 = (a >> shift1) & 0x3LL;
                      const int64_t new_a = (a & ~0x3LL & ~(0x3LL << shift1)) |
                                            d1 | (d0 << shift1);
                      scratch_buf[new_a] = local_buf[a];
                    }
                    sycl::group_barrier(item.get_group());
                    float* tmp = local_buf;
                    local_buf = scratch_buf;
                    scratch_buf = tmp;
                  }
                }

                // Write into the buffer that is not local_buf.
                {
                  const int top_shift = static_cast<int>((rounds_L - 1) * 2);
                  for (int64_t a = lid; a < chunk_size; a += wg_size) {
                    const int64_t bottom = a & 0x3LL;
                    const int64_t rest = a >> 2;
                    const int64_t n = rest | (bottom << top_shift);
                    scratch_buf[n] = local_buf[a];
                  }
                  sycl::group_barrier(item.get_group());
                  local_buf = scratch_buf;
                }
              } else {
                // chunk_size is a power of two here (Hadamard invariant), so
                // every stride h is too: index with 32-bit shifts/masks rather
                // than a 64-bit div/mod per butterfly (the latter dominated the
                // transform for non-power-of-4 sizes such as 128 and 512).
                const int n_half = static_cast<int>(chunk_size / 2);
                for (int lg = 0; (int64_t(1) << lg) < chunk_size; ++lg) {
                  const int h = 1 << lg;
                  const int hmask = h - 1;
                  for (int base = lid; base < n_half; base += wg_size) {
                    const int idx = ((base >> lg) << (lg + 1)) + (base & hmask);
                    float a = local_buf[idx];
                    float b = local_buf[idx + h];
                    local_buf[idx] = a + b;
                    local_buf[idx + h] = a - b;
                  }
                  sycl::group_barrier(item.get_group());
                }
              }
              for (int64_t i = lid; i < chunk_size; i += wg_size)
                local_buf[i] *= inv_sqrt_n;
              sycl::group_barrier(item.get_group());
            }

            // Inverse precond + write. Iterate over `m`, dst's
            // physical/natural index, so the HBM write stays
            // sequential; the permutation indirection only touches
            // the local_buf (SLM) read, via inv_perm[m] = the
            // logical slot whose value belongs at physical position m.
            if (rh && (inv_perm || signs_reordered)) {
              for (int64_t m = lid; m < chunk_size; m += wg_size) {
                int64_t k = inv_perm ? static_cast<int64_t>(inv_perm[m]) : m;
                float v = local_buf[k];
                if (signs_reordered) v *= signs_reordered[m];
                dstp[m * dst_stride + dst_offset] = static_cast<KVType>(v);
              }
            } else if constexpr (kv_has_vec4_v<KVType>) {
              if (dst_stride == 1 && (!fused_dst || chunk_size % 4 == 0)) {
                const int64_t num_vec_groups = chunk_size / 4;
                auto local_mptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(local_buf);
                auto dst_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(dstp + dst_offset);
                for (int64_t g = lid; g < num_vec_groups; g += wg_size) {
                  sycl::vec<float, 4> fv;
                  fv.load(static_cast<size_t>(g), local_mptr);
                  fv.template convert<KVType>().store(static_cast<size_t>(g),
                                                      dst_mptr);
                }
              } else {
                for (int64_t i = lid; i < chunk_size; i += wg_size) {
                  dstp[i * dst_stride + dst_offset] =
                      static_cast<KVType>(local_buf[i]);
                }
              }
            } else {
              for (int64_t i = lid; i < chunk_size; i += wg_size) {
                dstp[i * dst_stride + dst_offset] =
                    static_cast<KVType>(local_buf[i]);
              }
            }
          });
    });
  } else {
    // Large-chunk path (global scratch).
    sycl::queue& kq = q;

    float* scratch =
        sycl::malloc_device<float>(static_cast<size_t>(element_count), kq);
    DeviceScratch<float> scratch_guard(scratch, kq);

    // Step 1: Dequantize into scratch buffer (each chunk split across
    // `splits` work-groups so a single large chunk still fills the GPU).
    const int64_t splits = large_kernel_splits(num_chunks, chunk_size);
    auto dequant_event = kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on(deps);
      cgh.parallel_for<DequantizeKernelLarge<KVType, QType, Pack4Bit>>(
          sycl::nd_range<1>(static_cast<size_t>(num_chunks * splits) * WG_SIZE,
                            WG_SIZE),
          [=](sycl::nd_item<1> item) {
            const int grp = static_cast<int>(item.get_group(0));
            const int chunk_id = grp / static_cast<int>(splits);
            const int sid = grp % static_cast<int>(splits);
            const int lid = static_cast<int>(item.get_local_id(0));
            const int wg_size = static_cast<int>(item.get_local_range(0));
            const int64_t offset =
                is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                     (chunk_id % stride)
                               : chunk_id * chunk_size;
            const float scale = scales_in[chunk_id];
            const float zeropoint = zps_in[chunk_id];
            float* chunk_data = scratch + chunk_id * chunk_size;

            if constexpr (Pack4Bit) {
              // Unpack both nibbles per thread.
              const int64_t packed_chunk_size = chunk_size / 2;
              const QType* packed_src =
                  src + static_cast<int64_t>(chunk_id) * packed_chunk_size;
              if (packed_chunk_size % 4 == 0) {
                const int64_t slice_groups = packed_chunk_size / 4 / splits;
                const int64_t g_begin = sid * slice_groups;
                const int64_t g_end = g_begin + slice_groups;
                auto packed_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(packed_src);
                auto data_mptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(chunk_data);
                for (int64_t g = g_begin + lid; g < g_end; g += wg_size) {
                  sycl::vec<int8_t, 4> pv;
                  pv.load(static_cast<size_t>(g), packed_mptr);
                  sycl::vec<float, 8> out;
                  for (int k = 0; k < 4; ++k) {
                    const uint8_t byte = static_cast<uint8_t>(pv[k]);
                    const uint8_t lo = byte & 0x0F;
                    const uint8_t hi = (byte >> 4) & 0x0F;
                    const int8_t qlo =
                        (lo >= 8)
                            ? static_cast<int8_t>(static_cast<int>(lo) - 16)
                            : static_cast<int8_t>(lo);
                    const int8_t qhi =
                        (hi >= 8)
                            ? static_cast<int8_t>(static_cast<int>(hi) - 16)
                            : static_cast<int8_t>(hi);
                    out[k * 2] = static_cast<float>(qlo) * scale + zeropoint;
                    out[k * 2 + 1] =
                        static_cast<float>(qhi) * scale + zeropoint;
                  }
                  out.store(static_cast<size_t>(g), data_mptr);
                }
              } else {
                for (int64_t p = sid * (packed_chunk_size / splits) + lid;
                     p < (sid + 1) * (packed_chunk_size / splits);
                     p += wg_size) {
                  const uint8_t byte = static_cast<uint8_t>(packed_src[p]);
                  const uint8_t lo = byte & 0x0F;
                  const uint8_t hi = (byte >> 4) & 0x0F;
                  const int8_t qlo =
                      (lo >= 8) ? static_cast<int8_t>(static_cast<int>(lo) - 16)
                                : static_cast<int8_t>(lo);
                  const int8_t qhi =
                      (hi >= 8) ? static_cast<int8_t>(static_cast<int>(hi) - 16)
                                : static_cast<int8_t>(hi);
                  chunk_data[p * 2] =
                      static_cast<float>(qlo) * scale + zeropoint;
                  chunk_data[p * 2 + 1] =
                      static_cast<float>(qhi) * scale + zeropoint;
                }
              }
            } else {
              for (int64_t i = sid * (chunk_size / splits) + lid;
                   i < (sid + 1) * (chunk_size / splits); i += wg_size) {
                float qv = static_cast<float>(src[i * stride + offset]);
                chunk_data[i] = qv * scale + zeropoint;
              }
            }
          });
    });

    // Step 2: Inverse Hadamard per chunk (if rh)
    sycl::event had_event = dequant_event;
    if (rh) {
      // Batched like quantize: block-local stages first, then cross-block
      // stages.
      sycl::event stage_event = dequant_event;
      std::vector<sycl::event> stage_deps = {dequant_event};

      const int64_t block_size = std::min(max_slm_chunk, chunk_size);
      int64_t h_start = 1;
      if (block_size > 1) {
        const int64_t num_blocks = element_count / block_size;
        stage_event = hadamard_block_local_batched(kq, scratch, block_size,
                                                   num_blocks, stage_deps);
        stage_deps = {stage_event};
        h_start = block_size;
      }

      for (int64_t h = h_start; h < chunk_size; h <<= 1) {
        stage_event =
            (h >= 4) ? hadamard_stage_global_batched_vec4(
                           kq, scratch, chunk_size, num_chunks, h, stage_deps)
                     : hadamard_stage_global_batched(kq, scratch, chunk_size,
                                                     num_chunks, h, stage_deps);
        stage_deps = {stage_event};
      }
      had_event =
          scale_buffer(kq, scratch, element_count, inv_sqrt_n, {stage_event});
    }

    // Step 3: Inverse precond + write to dst (vec4 fast path when contiguous).
    const bool write_fast_contig =
        (stride == 1) && !(rh && (perm || signs)) && (element_count % 4 == 0);
    sycl::event write_event;
    if constexpr (kv_has_vec4_v<KVType>) {
      if (write_fast_contig) {
        write_event = precond_writeback_contiguous_vec4<KVType>(
            kq, scratch, dst, element_count, {had_event});
      } else {
        const int64_t global_size =
            ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        write_event = kq.submit([&](sycl::handler& cgh) {
          cgh.depends_on({had_event});
          cgh.parallel_for(
              sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
              [=](sycl::nd_item<1> item) {
                const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
                if (gid >= element_count) return;
                int64_t chunk_id = gid / chunk_size;
                int64_t i = gid % chunk_size;
                int64_t offset =
                    is_per_channel ? (chunk_id / stride) * single_layer_numel +
                                         (chunk_id % stride)
                                   : chunk_id * chunk_size;

                float v = scratch[gid];
                if (rh && (perm || signs)) {
                  if (signs) v *= signs[i];
                  int64_t dst_idx = perm ? static_cast<int64_t>(perm[i]) : i;
                  dst[dst_idx * stride + offset] = static_cast<KVType>(v);
                } else {
                  dst[i * stride + offset] = static_cast<KVType>(v);
                }
              });
        });
      }
    } else {
      const int64_t global_size =
          ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
      write_event = kq.submit([&](sycl::handler& cgh) {
        cgh.depends_on({had_event});
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
            [=](sycl::nd_item<1> item) {
              const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
              if (gid >= element_count) return;
              int64_t chunk_id = gid / chunk_size;
              int64_t i = gid % chunk_size;
              int64_t offset = is_per_channel
                                   ? (chunk_id / stride) * single_layer_numel +
                                         (chunk_id % stride)
                                   : chunk_id * chunk_size;

              float v = scratch[gid];
              if (rh && (perm || signs)) {
                if (signs) v *= signs[i];
                int64_t dst_idx = perm ? static_cast<int64_t>(perm[i]) : i;
                dst[dst_idx * stride + offset] = static_cast<KVType>(v);
              } else {
                dst[i * stride + offset] = static_cast<KVType>(v);
              }
            });
      });
    }

    // Success: hand scratch off to the async free (chained on write_event).
    scratch_guard.release();
    kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on(write_event);
      cgh.host_task([=, &kq]() { sycl::free(scratch, kq); });
    });

    return write_event;
  }
}

// ---------------------------------------------------------------------------
// 4-bit packing kernel
// ---------------------------------------------------------------------------
class PackInt4Kernel;

sycl::event pack_int4_sycl(sycl::queue& q, const int8_t* logical,
                           int8_t* packed, int64_t logical_count,
                           const std::vector<sycl::event>& deps) {
  const int64_t packed_count = (logical_count + 1) / 2;
  const int64_t global_size =
      ((packed_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<PackInt4Kernel>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
          if (idx >= packed_count) return;

          int64_t lo_idx = idx * 2;
          int64_t hi_idx = lo_idx + 1;
          uint8_t lo_nibble = static_cast<uint8_t>(logical[lo_idx]) & 0x0F;
          uint8_t hi_nibble =
              (hi_idx < logical_count)
                  ? (static_cast<uint8_t>(logical[hi_idx]) & 0x0F)
                  : 0;
          packed[idx] = static_cast<int8_t>(lo_nibble | (hi_nibble << 4));
        });
  });
}

// ---------------------------------------------------------------------------
// 4-bit unpacking kernel
// ---------------------------------------------------------------------------
class UnpackInt4Kernel;

sycl::event unpack_int4_sycl(sycl::queue& q, const int8_t* packed,
                             int8_t* logical, int64_t logical_count,
                             const std::vector<sycl::event>& deps) {
  const int64_t global_size =
      ((logical_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<UnpackInt4Kernel>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
          if (idx >= logical_count) return;

          int64_t byte_idx = idx / 2;
          uint8_t byte = static_cast<uint8_t>(packed[byte_idx]);
          uint8_t nibble =
              ((idx & 1) == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
          // Sign-extend 4-bit: if bit 3 set, value is negative
          int8_t value =
              (nibble >= 8) ? static_cast<int8_t>(static_cast<int>(nibble) - 16)
                            : static_cast<int8_t>(nibble);
          logical[idx] = value;
        });
  });
}

// ---------------------------------------------------------------------------
// Fused [layers, tokens, head_num*head_dim] -> split
// [layers*head_num, tokens, head_dim] gather, used only when a fused-source
// quantize can't take the SLM path (which reads the fused layout in place).
// ---------------------------------------------------------------------------
template <typename KVType>
class SplitFusedHeadsKernel;

template <typename KVType>
static sycl::event split_fused_heads_sycl(
    sycl::queue& q, const KVType* src, KVType* dst, int64_t element_count,
    int64_t tokens, int64_t head_num, int64_t head_dim,
    const std::vector<sycl::event>& deps) {
  const int64_t unit_elems = tokens * head_dim;
  const int64_t hidden = head_num * head_dim;
  const int64_t global_size =
      ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<SplitFusedHeadsKernel<KVType>>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t i = static_cast<int64_t>(item.get_global_id(0));
          if (i >= element_count) return;
          const int64_t unit = i / unit_elems;
          const int64_t r = i % unit_elems;
          const int64_t t = r / head_dim;
          const int64_t d = r % head_dim;
          dst[i] = src[((unit / head_num) * tokens + t) * hidden +
                       (unit % head_num) * head_dim + d];
        });
  });
}

// Inverse of split_fused_heads_sycl, for fused-dst dequantize off the SLM path.
template <typename KVType>
class MergeFusedHeadsKernel;

template <typename KVType>
static sycl::event merge_fused_heads_sycl(
    sycl::queue& q, const KVType* src, KVType* dst, int64_t element_count,
    int64_t tokens, int64_t head_num, int64_t head_dim,
    const std::vector<sycl::event>& deps) {
  const int64_t unit_elems = tokens * head_dim;
  const int64_t hidden = head_num * head_dim;
  const int64_t global_size =
      ((element_count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(deps);
    cgh.parallel_for<MergeFusedHeadsKernel<KVType>>(
        sycl::nd_range<1>(static_cast<size_t>(global_size), WG_SIZE),
        [=](sycl::nd_item<1> item) {
          const int64_t i = static_cast<int64_t>(item.get_global_id(0));
          if (i >= element_count) return;
          const int64_t unit = i / unit_elems;
          const int64_t r = i % unit_elems;
          const int64_t t = r / head_dim;
          const int64_t d = r % head_dim;
          dst[((unit / head_num) * tokens + t) * hidden +
              (unit % head_num) * head_dim + d] = src[i];
        });
  });
}

// ---------------------------------------------------------------------------
// Top-level quantize dispatch (handles qbit=4 packing)
// ---------------------------------------------------------------------------
template <typename KVType, typename QType>
sycl::event quantize_kvcache_sycl(sycl::queue& q, const KVType* src, QType* dst,
                                  float* scales_out, float* zps_out,
                                  int64_t element_count,
                                  const sycl_quantization_args& args,
                                  const std::vector<sycl::event>& deps) {
  if (args.fused_src_head_num > 0) {
    const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
    const int64_t tokens = element_count / (num_layers * args.head_dim);
    const int64_t chunk_size =
        (args.scaling_method == "per_channel") ? tokens : args.head_dim;
    if (chunk_size > get_max_slm_chunk(q)) {
      KVType* split =
          sycl::malloc_device<KVType>(static_cast<size_t>(element_count), q);
      DeviceScratch<KVType> split_guard(split, q);
      auto split_event = split_fused_heads_sycl<KVType>(
          q, src, split, element_count, tokens, args.fused_src_head_num,
          args.head_dim, deps);
      sycl_quantization_args split_args = args;
      split_args.fused_src_head_num = 0;
      auto quant_event = quantize_kvcache_sycl<KVType, QType>(
          q, split, dst, scales_out, zps_out, element_count, split_args,
          {split_event});
      split_guard.release();
      q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(quant_event);
        cgh.host_task([=, &q]() { sycl::free(split, q); });
      });
      return quant_event;
    }
  }

  if (args.qbit == 4) {
    int64_t chunks_per_layer = 1;
    if (args.scaling_method == "per_token") {
      chunks_per_layer =
          static_cast<int64_t>(args.blocks_num) * args.block_size;
    } else if (args.scaling_method == "per_channel") {
      chunks_per_layer = args.head_dim;
    }
    const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
    const int64_t num_chunks = num_layers * chunks_per_layer;
    const int64_t chunk_size = element_count / num_chunks;

    if (chunk_size % 2 == 0) {
      // Fused path: the quantize kernel writes packed nibbles directly.
      return quantize_block_sycl_impl<KVType, int8_t, /*Pack4Bit=*/true>(
          q, src, reinterpret_cast<int8_t*>(dst), scales_out, zps_out,
          element_count, args, deps);
    }

    // Odd chunk_size fallback: quantize to int8, then pack.
    sycl::queue& kq = q;

    int8_t* logical_buf =
        sycl::malloc_device<int8_t>(static_cast<size_t>(element_count), kq);
    DeviceScratch<int8_t> logical_guard(logical_buf, kq);

    auto quant_event = quantize_block_sycl_impl<KVType, int8_t>(
        kq, src, logical_buf, scales_out, zps_out, element_count, args, deps);

    auto pack_event =
        pack_int4_sycl(kq, logical_buf, reinterpret_cast<int8_t*>(dst),
                       element_count, {quant_event});

    // Success: hand logical_buf off to the async free (chained on pack).
    logical_guard.release();
    kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on(pack_event);
      cgh.host_task([=, &kq]() { sycl::free(logical_buf, kq); });
    });

    return pack_event;
  }

  return quantize_block_sycl_impl<KVType, QType>(
      q, src, dst, scales_out, zps_out, element_count, args, deps);
}

// ---------------------------------------------------------------------------
// Top-level dequantize dispatch (handles qbit=4 unpacking)
// ---------------------------------------------------------------------------

template <typename KVType, typename QType>
sycl::event dequantize_kvcache_sycl(sycl::queue& q, const QType* src,
                                    KVType* dst, const float* scales_in,
                                    const float* zps_in, int64_t element_count,
                                    const sycl_quantization_args& args,
                                    const std::vector<sycl::event>& deps) {
  if (args.fused_dst_head_num > 0) {
    const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
    const int64_t tokens = element_count / (num_layers * args.head_dim);
    const bool per_channel = args.scaling_method == "per_channel";
    const int64_t chunk_size = per_channel ? tokens : args.head_dim;
    // per_channel column writes strided by the fused hidden size measured
    // slower than a split-layout dequantize plus one contiguous device merge.
    if (per_channel || chunk_size > get_max_slm_chunk(q)) {
      KVType* split =
          sycl::malloc_device<KVType>(static_cast<size_t>(element_count), q);
      DeviceScratch<KVType> split_guard(split, q);
      sycl_quantization_args split_args = args;
      split_args.fused_dst_head_num = 0;
      auto dequant_event = dequantize_kvcache_sycl<KVType, QType>(
          q, src, split, scales_in, zps_in, element_count, split_args, deps);
      auto merge_event = merge_fused_heads_sycl<KVType>(
          q, split, dst, element_count, tokens, args.fused_dst_head_num,
          args.head_dim, {dequant_event});
      split_guard.release();
      q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(merge_event);
        cgh.host_task([=, &q]() { sycl::free(split, q); });
      });
      return merge_event;
    }
  }

  if (args.qbit == 4) {
    int64_t chunks_per_layer = 1;
    if (args.scaling_method == "per_token") {
      chunks_per_layer =
          static_cast<int64_t>(args.blocks_num) * args.block_size;
    } else if (args.scaling_method == "per_channel") {
      chunks_per_layer = args.head_dim;
    }
    const int64_t num_layers = args.num_layers > 0 ? args.num_layers : 1;
    const int64_t num_chunks = num_layers * chunks_per_layer;
    const int64_t chunk_size = element_count / num_chunks;

    if (chunk_size % 2 == 0) {
      // Fused path: reads packed nibbles directly.
      return dequantize_block_sycl_impl<KVType, int8_t, /*Pack4Bit=*/true>(
          q, reinterpret_cast<const int8_t*>(src), dst, scales_in, zps_in,
          element_count, args, deps);
    }

    // Fallback for odd chunk_size: unpack, then dequantize.
    sycl::queue& kq = q;

    int8_t* logical_buf =
        sycl::malloc_device<int8_t>(static_cast<size_t>(element_count), kq);
    DeviceScratch<int8_t> logical_guard(logical_buf, kq);

    auto unpack_event =
        unpack_int4_sycl(kq, reinterpret_cast<const int8_t*>(src), logical_buf,
                         element_count, deps);

    auto dequant_event = dequantize_block_sycl_impl<KVType, int8_t>(
        kq, logical_buf, dst, scales_in, zps_in, element_count, args,
        {unpack_event});

    // Success: hand logical_buf off to the async free (chained on dequant).
    logical_guard.release();
    kq.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dequant_event);
      cgh.host_task([=, &kq]() { sycl::free(logical_buf, kq); });
    });

    return dequant_event;
  }

  return dequantize_block_sycl_impl<KVType, QType>(
      q, src, dst, scales_in, zps_in, element_count, args, deps);
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------
// fp16 (sycl::half)
template sycl::event quantize_kvcache_sycl<sycl::half, int8_t>(
    sycl::queue&, const sycl::half*, int8_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<sycl::half, int8_t>(
    sycl::queue&, const int8_t*, sycl::half*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);

// bf16 (sycl::ext::oneapi::bfloat16)
template sycl::event quantize_kvcache_sycl<sycl::ext::oneapi::bfloat16, int8_t>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*, int8_t*, float*, float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event
dequantize_kvcache_sycl<sycl::ext::oneapi::bfloat16, int8_t>(
    sycl::queue&, const int8_t*, sycl::ext::oneapi::bfloat16*, const float*,
    const float*, int64_t, const sycl_quantization_args&,
    const std::vector<sycl::event>&);

// fp32
template sycl::event quantize_kvcache_sycl<float, int8_t>(
    sycl::queue&, const float*, int8_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<float, int8_t>(
    sycl::queue&, const int8_t*, float*, const float*, const float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);

// int16 variants (for qbit > 8)
template sycl::event quantize_kvcache_sycl<sycl::half, int16_t>(
    sycl::queue&, const sycl::half*, int16_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<sycl::half, int16_t>(
    sycl::queue&, const int16_t*, sycl::half*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);

template sycl::event
quantize_kvcache_sycl<sycl::ext::oneapi::bfloat16, int16_t>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*, int16_t*, float*, float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event
dequantize_kvcache_sycl<sycl::ext::oneapi::bfloat16, int16_t>(
    sycl::queue&, const int16_t*, sycl::ext::oneapi::bfloat16*, const float*,
    const float*, int64_t, const sycl_quantization_args&,
    const std::vector<sycl::event>&);

// fp32 + int16
template sycl::event quantize_kvcache_sycl<float, int16_t>(
    sycl::queue&, const float*, int16_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<float, int16_t>(
    sycl::queue&, const int16_t*, float*, const float*, const float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);

// fp8 KV cache (torch.float8_e4m3fn / torch.float8_e5m2)
template sycl::event quantize_kvcache_sycl<fp8_e4m3_t, int8_t>(
    sycl::queue&, const fp8_e4m3_t*, int8_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<fp8_e4m3_t, int8_t>(
    sycl::queue&, const int8_t*, fp8_e4m3_t*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event quantize_kvcache_sycl<fp8_e4m3_t, int16_t>(
    sycl::queue&, const fp8_e4m3_t*, int16_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<fp8_e4m3_t, int16_t>(
    sycl::queue&, const int16_t*, fp8_e4m3_t*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event quantize_kvcache_sycl<fp8_e5m2_t, int8_t>(
    sycl::queue&, const fp8_e5m2_t*, int8_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<fp8_e5m2_t, int8_t>(
    sycl::queue&, const int8_t*, fp8_e5m2_t*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event quantize_kvcache_sycl<fp8_e5m2_t, int16_t>(
    sycl::queue&, const fp8_e5m2_t*, int16_t*, float*, float*, int64_t,
    const sycl_quantization_args&, const std::vector<sycl::event>&);
template sycl::event dequantize_kvcache_sycl<fp8_e5m2_t, int16_t>(
    sycl::queue&, const int16_t*, fp8_e5m2_t*, const float*, const float*,
    int64_t, const sycl_quantization_args&, const std::vector<sycl::event>&);
